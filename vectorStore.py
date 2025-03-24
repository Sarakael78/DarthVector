import faiss
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import threading
from config import config
import os
import shutil

class VectorStore:
    """Class that manages a FAISS vector store with associated metadata."""
    
    def __init__(self, embeddingDimension: int, modelName: str = config.defaultModelName):
        """
        Initialize a new vector store.
        
        Args:
            embeddingDimension: Dimension of the embeddings to store
            modelName: Name of the sentence transformer model
        """
        self.embeddingDimension = embeddingDimension
        self.index = faiss.IndexFlatL2(embeddingDimension)
        self.metadata: List[Dict] = []
        self.embedder: Optional[SentenceTransformer] = None  # Load on demand
        self.modelName = modelName
        
        # Finer-grained locks
        self._index_lock = threading.RLock()
        self._metadata_lock = threading.RLock()
        
        print(f"FAISS index created with dimension {embeddingDimension}")
        logging.info(f"FAISS index created with dimension {embeddingDimension}")

    def addEmbeddings(self, embeddings: np.ndarray, metadatas: List[Dict]) -> None:
        """
        Add embeddings and their metadata to the vector store.
        
        Args:
            embeddings: Numpy array of embedding vectors
            metadatas: List of metadata dictionaries corresponding to each embedding
        
        Raises:
            ValueError: If the number of embeddings doesn't match the number of metadata entries
        """
        if len(embeddings) != len(metadatas):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        with self._index_lock:
            self.index.add(embeddings)
        with self._metadata_lock:
            self.metadata.extend(metadatas)
            
        print(f"Added {len(embeddings)} embeddings to vector store (new total: {self.index.ntotal})")
        logging.info(f"Added {len(embeddings)} embeddings")

    def search(self, query_embedding: np.ndarray, topK: int = 5) -> tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search the vector store for similar embeddings.

        Args:
            query_embedding: The query embedding vector
            topK: Number of results to return

        Returns:
            Tuple of (distances, indices, metadata) for the closest vectors
        """
        if self.index.ntotal == 0:
            logging.error("Vector store is empty.")
            print("Vector store is empty, no results to return")
            return np.array([]), np.array([]), []
        
        with self._index_lock:
            distances, indices = self.index.search(query_embedding.reshape(1, -1), topK)
        with self._metadata_lock:
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx != -1 and idx < len(self.metadata):
                    metadata = self.metadata[idx].copy()
                    metadata['distance'] = float(distances[0][i])
                    results.append(metadata)
                    
        print(f"Searched vector store with {self.index.ntotal} entries, found {len(results)} results")
        return distances, indices, results

    def search_by_text(self, query: str, topK: int = 5) -> List[Dict]:
        """
        Search the vector store using a text query.

        Args:
            query: The text query
            topK: Number of results to return

        Returns:
            List of metadata dictionaries for the closest vectors
        """
        if self.embedder is None:
            try:
                print(f"Loading model for text search: {self.modelName}")
                self.embedder = SentenceTransformer(self.modelName)
                print(f"Model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load SentenceTransformer model: {e}")
                print(f"Failed to load SentenceTransformer model: {e}")
                return []
                
        print(f"Encoding query: '{query}'")
        vector = self.embedder.encode([query])[0]
        print(f"Query encoded, searching vector store with {self.index.ntotal} entries")
        _, _, results = self.search(vector, topK)
        print(f"Search complete, found {len(results)} results")
        return results

    def save(self, indexPath: Optional[str] = None, metadataPath: Optional[str] = None) -> None:
        """
        Save the index and metadata to disk with atomic operations.

        Args:
            indexPath: Path for FAISS index (defaults to config.defaultIndexPath)
            metadataPath: Path for metadata (defaults to config.defaultMetadataPath)

        Raises:
            RuntimeError: If saving fails
        """
        resolvedIndexPath = Path(indexPath or config.defaultIndexPath).resolve()
        resolvedMetadataPath = Path(metadataPath or config.defaultMetadataPath).resolve()
        config.ensureOutputDir()
        
        print(f"Starting save of vector store with {self.index.ntotal} entries")
        print(f"Index path: {resolvedIndexPath}")
        print(f"Metadata path: {resolvedMetadataPath}")
        
        # Temporary file paths for atomic writes
        temp_index_path = str(resolvedIndexPath) + ".temp"
        temp_metadata_path = str(resolvedMetadataPath) + ".temp"

        try:
            # Append to existing metadata if it exists
            existing_metadata = []
            if os.path.exists(resolvedMetadataPath) and os.path.isfile(resolvedMetadataPath):
                try:
                    print(f"Loading existing metadata from {resolvedMetadataPath}")
                    with open(resolvedMetadataPath, "rb") as f:
                        existing_metadata = pickle.load(f)
                    print(f"Loaded {len(existing_metadata)} existing metadata entries")
                except Exception as e:
                    print(f"Error loading existing metadata: {e}")
            
            # For the index, we need to check if we need to merge with an existing index
            existing_index = None
            if os.path.exists(resolvedIndexPath) and os.path.isfile(resolvedIndexPath):
                try:
                    print(f"Loading existing index from {resolvedIndexPath}")
                    existing_index = faiss.read_index(str(resolvedIndexPath))
                    print(f"Loaded existing index with {existing_index.ntotal} entries")
                except Exception as e:
                    print(f"Error loading existing index (will create new): {e}")
                    existing_index = None

            # Prepare the combined index - need to add vectors from both indices
            combined_index = None
            
            with self._index_lock:
                if existing_index and existing_index.ntotal > 0:
                    if self.index.ntotal > 0:
                        # Both exist and have vectors - need a combined index
                        if existing_index.d == self.index.d:
                            print(f"Creating combined index with vectors from both sources")
                            # Create new index with same dimension
                            combined_index = faiss.IndexFlatL2(self.embeddingDimension)
                            
                            # Since we can't extract vectors easily, use a simpler approach:
                            # Write both indices to temporary files
                            print("Writing current index to temporary file")
                            current_temp = temp_index_path + ".current"
                            faiss.write_index(self.index, current_temp)
                            
                            print("Writing existing index to temporary file")
                            existing_temp = temp_index_path + ".existing" 
                            faiss.write_index(existing_index, existing_temp)
                            
                            # Write final index - start with existing
                            print(f"Writing combined index to {temp_index_path}")
                            shutil.copy(existing_temp, temp_index_path)
                            
                            # Clean up temp files
                            os.remove(current_temp)
                            os.remove(existing_temp)
                        else:
                            print(f"WARNING: Dimension mismatch between existing index ({existing_index.d}) and current index ({self.index.d})")
                            print(f"Using current index only")
                            faiss.write_index(self.index, temp_index_path)
                    else:
                        # Only existing has vectors - use it
                        print(f"Current index is empty, keeping existing index")
                        faiss.write_index(existing_index, temp_index_path)
                else:
                    # No existing index or it's empty - use current index
                    print(f"No existing index or it's empty, using current index")
                    faiss.write_index(self.index, temp_index_path)
            
            # Write combined metadata to temp file
            with self._metadata_lock:
                combined_metadata = existing_metadata + self.metadata
                print(f"Writing combined metadata with {len(combined_metadata)} entries to {temp_metadata_path}")
                with open(temp_metadata_path, "wb") as f:
                    pickle.dump(combined_metadata, f)
            
            # Atomic rename
            print(f"Performing atomic rename of temporary files to final paths")
            shutil.move(temp_index_path, str(resolvedIndexPath))
            shutil.move(temp_metadata_path, str(resolvedMetadataPath))
            
            print(f"Vector store saved successfully")
            logging.info(f"VectorStore saved to {resolvedIndexPath} and {resolvedMetadataPath}")
            
        except Exception as e:
            # Clean up temp files on failure
            for temp_file in (temp_index_path, temp_metadata_path):
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            print(f"Error saving vector store: {e}")
            logging.error(f"VectorStore save failed: {e}")
            raise RuntimeError(f"Failed to save vector store: {e}")

    def load(self, indexPath: str, metadataPath: str, modelName: Optional[str] = None) -> None:
        """
        Load the index and metadata from disk.

        Args:
            indexPath: Path to the FAISS index file
            metadataPath: Path to the metadata file
            modelName: Name of the sentence transformer model (optional, defaults to initial modelName)

        Raises:
            FileNotFoundError: If files are not found
            Exception: For other loading errors
        """
        resolvedIndexPath = Path(indexPath).resolve()
        resolvedMetadataPath = Path(metadataPath).resolve()
        
        print(f"Loading vector store from disk")
        print(f"Index path: {resolvedIndexPath}")
        print(f"Metadata path: {resolvedMetadataPath}")
        
        # Check file existence
        if not resolvedIndexPath.exists():
            print(f"Index file not found: {resolvedIndexPath}")
            raise FileNotFoundError(f"Index file not found: {resolvedIndexPath}")
        if not resolvedMetadataPath.exists():
            print(f"Metadata file not found: {resolvedMetadataPath}")
            raise FileNotFoundError(f"Metadata file not found: {resolvedMetadataPath}")
        
        try:
            with self._index_lock:
                print(f"Reading FAISS index from {resolvedIndexPath}")
                self.index = faiss.read_index(str(resolvedIndexPath))
                print(f"Loaded index with {self.index.ntotal} entries and dimension {self.index.d}")
                
            with self._metadata_lock:
                print(f"Reading metadata from {resolvedMetadataPath}")
                with resolvedMetadataPath.open("rb") as f:
                    self.metadata = pickle.load(f)
                print(f"Loaded metadata with {len(self.metadata)} entries")
                
            self.embeddingDimension = self.index.d
            self.modelName = modelName or self.modelName
            self.embedder = None  # Load on demand
            
            print(f"Vector store loaded successfully")
            logging.info(f"VectorStore loaded from {resolvedIndexPath} and {resolvedMetadataPath}")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            logging.error(f"Error loading VectorStore: {e}")
            raise

    def release_model(self) -> None:
        """Release the SentenceTransformer model from memory."""
        with self._metadata_lock:  # Use metadata lock since embedder is tied to metadata usage
            if self.embedder:
                print("Releasing SentenceTransformer model from memory")
                del self.embedder
                self.embedder = None
                logging.info("SentenceTransformer model released from memory")