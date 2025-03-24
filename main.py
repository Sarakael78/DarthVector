import argparse
import logging
import sys
import time
import multiprocessing as mp
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import numpy as np
import gc

from config import config
from textExtractor import extractTextFromFile, FileProcessingError
from exceptions import FileSizeExceededError, UnsupportedFileTypeError
from textPreprocessor import TextPreprocessor
from embeddingGenerator import SentenceTransformerEmbedder
from vectorStore import VectorStore

# Set the multiprocessing start method globally
def initialize_multiprocessing():
    """Initialize multiprocessing with appropriate method."""
    # 'spawn' is default on Windows, we want to use it on all platforms for consistency
    if sys.platform != 'win32' and mp.get_start_method(allow_none=True) is None:
        try:
            mp.set_start_method('spawn')
            logging.info("Multiprocessing start method set to 'spawn'")
        except RuntimeError:
            logging.warning("Multiprocessing start method already set, using existing configuration")

# Initialize multiprocessing at module import time
initialize_multiprocessing()

class RtfProcessingPipeline:
    """Pipeline for processing documents into embeddings."""

    def __init__(
        self,
        modelName: Optional[str] = None,
        chunkSize: Optional[int] = None,
        chunkOverlap: Optional[int] = None,
        maxWorkers: Optional[int] = None,
        maxFileSizeMB: Optional[int] = None,
        useMultiprocessing: Optional[bool] = None,
        embeddingDimension: Optional[int] = None,
        ingestedFilesPath: Optional[str] = None,
        existingVectorStore: Optional[VectorStore] = None
    ) -> None:
        """
        Initialise the pipeline.

        Args:
            modelName: SentenceTransformer model name.
            chunkSize: Maximum words per chunk.
            chunkOverlap: Overlap between chunks.
            maxWorkers: Number of worker processes.
            maxFileSizeMB: Maximum file size in MB.
            useMultiprocessing: Whether to use multiprocessing.
            embeddingDimension: Dimension of the embeddings.
            ingestedFilesPath: Path to the file storing the list of ingested files.
            existingVectorStore: An existing VectorStore instance to use.
        """
        self.modelName = modelName or config.defaultModelName
        self.chunkSize = chunkSize or config.defaultChunkSize
        self.chunkOverlap = chunkOverlap or config.defaultChunkOverlap
        self.maxWorkers = maxWorkers or config.defaultMaxWorkers
        self.maxFileSizeMB = maxFileSizeMB or config.maxFileSizeMB
        
        if useMultiprocessing is not None:
            self.useMultiprocessing = useMultiprocessing
        else:
            self.useMultiprocessing = not config.disableMultiprocessing
            
        # Initialize embedder first to get dimensions if needed
        self.embedder = SentenceTransformerEmbedder(self.modelName)
        
        # Use provided dimension or get from model
        self.embeddingDimension = embeddingDimension or self.embedder.embeddingDim
        
        # Use existing vector store or create new one
        if existingVectorStore:
            self.vectorStore = existingVectorStore
        else:
            self.vectorStore = VectorStore(embeddingDimension=self.embeddingDimension, modelName=self.modelName)
            
        self.ingestedFilesPath = ingestedFilesPath or config.defaultIngestedFilesPath
        self.ingestedFiles = self._loadIngestedFiles() or []  # Default to empty list if None
        
        logging.info(f"Pipeline initialised with model: {self.modelName}, multiprocessing: {self.useMultiprocessing}")
    
    def _loadIngestedFiles(self) -> List[Dict[str, Any]]:
        """Load the list of ingested files (name and size) from disk."""
        try:
            ingestedFilesDir = Path(self.ingestedFilesPath).parent
            ingestedFilesDir.mkdir(exist_ok=True, parents=True)
            
            if not Path(self.ingestedFilesPath).exists():
                with open(self.ingestedFilesPath, 'w') as f:
                    json.dump([], f)
                return []
                
            with open(self.ingestedFilesPath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.info(f"Ingested files list not found at {self.ingestedFilesPath}. Creating an empty file.")
            os.makedirs(os.path.dirname(os.path.abspath(self.ingestedFilesPath)), exist_ok=True)
            with open(self.ingestedFilesPath, 'w') as f:
                json.dump([], f)
            return []
        except json.JSONDecodeError:
            logging.warning("Could not decode ingested files list, starting with an empty list.")
            return []
        except Exception as e:
            logging.error(f"Error loading ingested files list: {e}")
            return []
    
    def _saveIngestedFiles(self) -> None:
        """Save the list of ingested files (name and size) to disk atomically."""
        # Use atomic write pattern with temporary file
        temp_path = Path(self.ingestedFilesPath).with_name(f"{Path(self.ingestedFilesPath).name}.tmp")
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.ingestedFiles, f)
            
            # Atomic replacement
            os.replace(temp_path, self.ingestedFilesPath)
            logging.info(f"Saved ingested files list to {self.ingestedFilesPath}")
        except Exception as e:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            logging.error(f"Error saving ingested files list: {e}")

    def processFile(self, filePath: Path) -> int:
        """
        Process a single file and add its embeddings to the vector store.
        
        Args:
            filePath: Path to the file to process
            
        Returns:
            Number of chunks processed
        """
        try:
            # Check if file already processed
            fileName = filePath.name
            fileSize = os.path.getsize(filePath)
            if any(ingested_file.get('name') == fileName and 
                   ingested_file.get('size') == fileSize 
                   for ingested_file in self.ingestedFiles):
                logging.info(f"File already ingested: {fileName} (size: {fileSize} bytes)")
                return 0
                
            # Extract and process text
            startTime = time.time()
            rawText = extractTextFromFile(filePath)
            
            textPreprocessor = TextPreprocessor(self.chunkSize, self.chunkOverlap)
            cleanedText = textPreprocessor.cleanText(rawText)
            chunks = textPreprocessor.segmentText(cleanedText)
            
            if not chunks:
                logging.warning(f"No chunks from {filePath}")
                return 0
                
            # Generate embeddings
            embeddings = self.embedder.encodeChunks(chunks)
            
            # Create metadata
            metadatas = [
                {"file": filePath.name, "chunkIndex": idx, "chunkLength": len(chunk.split()), "chunkText": chunk}
                for idx, chunk in enumerate(chunks)]
                
            # Add to vector store
            self.vectorStore.addEmbeddings(embeddings, metadatas)
            
            # Update ingested files list
            self.ingestedFiles.append({'name': fileName, 'size': fileSize})
            self._saveIngestedFiles()
            
            processingTime = time.time() - startTime
            logging.info(f"Processed {filePath} in {processingTime:.2f}s: {len(chunks)} chunks")
            return len(chunks)
            
        except (FileProcessingError, FileSizeExceededError, UnsupportedFileTypeError) as e:
            logging.error(f"Processing error for {filePath}: {e}")
            return 0
        except Exception as e:
            logging.error(f"Unexpected error for {filePath}: {e}", exc_info=True)
            return 0

    def processDirectory(self, inputDir: Path, progress_callback=None) -> int:
        """
    Process new files in a directory and subdirectories, skipping already ingested ones.
        
    Args:
        inputDir: Directory containing files to process
        progress_callback: Optional callback function to report progress
            
    Returns the number of successfully processed files.
    """
        resolvedInputDir = inputDir.resolve()
        if not resolvedInputDir.is_dir():
            raise ValueError(f"Invalid directory: {resolvedInputDir}")

        print(f"Processing directory: {resolvedInputDir}")

        # Find files with supported extensions recursively
        supportedFiles = []
        for ext in config.supportedExtensions:
            # Use ** pattern for recursive search
            print(f"Searching for *{ext} files...")
            files = list(resolvedInputDir.glob(f"**/*{ext}"))
            supportedFiles.extend(files)
            print(f"Found {len(files)} {ext} files")
            
        supportedFiles = sorted(supportedFiles)
        if not supportedFiles:
            logging.warning(f"No supported files in {resolvedInputDir}")
            print(f"No supported files found in {resolvedInputDir}")
            return 0

        # Build list of files that need processing by checking against ingested files
        filesToProcess = []
        for filePath in supportedFiles:
            fileName = filePath.name
            fileSize = os.path.getsize(filePath)
            
            # Skip if already ingested
            if any(ingested_file.get('name') == fileName and 
                ingested_file.get('size') == fileSize 
                for ingested_file in self.ingestedFiles):
                print(f"File already ingested: {fileName} (size: {fileSize} bytes)")
                logging.info(f"File already ingested: {fileName} (size: {fileSize} bytes)")
                continue
                
            filesToProcess.append(filePath)

        if not filesToProcess:
            logging.info("All files already processed")
            print("All files already processed, nothing to add to vector store")
            # Even with no files to process, we won't delete vector store content
            return 0
            
        print(f"Processing {len(filesToProcess)} files, skipping {len(supportedFiles) - len(filesToProcess)} already ingested files")
        logging.info(f"Processing {len(filesToProcess)} files, skipping {len(supportedFiles) - len(filesToProcess)} already ingested files")
        
        successfulFiles = 0
        totalChunks = 0

        # Process files using the selected method (multiprocessing or sequential)
        if self.useMultiprocessing and self.maxWorkers > 1:
            # Group files in batches to reduce model loading overhead
            batch_size = max(1, len(filesToProcess) // self.maxWorkers)
            file_batches = [filesToProcess[i:i + batch_size] for i in range(0, len(filesToProcess), batch_size)]
            
            print(f"Using multiprocessing with {self.maxWorkers} workers, {len(file_batches)} batches")
            
            # Create a process pool safely using spawn context
            ctx = mp.get_context('spawn') 
            
            # Use context manager to ensure proper cleanup
            with ctx.Pool(processes=min(self.maxWorkers, len(file_batches))) as pool:
                try:
                    print(f"Starting parallel processing of {len(file_batches)} batches")
                    # Process batches in parallel
                    batch_results = pool.map(
                        self._process_file_batch,
                        [(batch, self.maxFileSizeMB, self.chunkSize, self.chunkOverlap, self.modelName) 
                        for batch in file_batches]
                    )
                    
                    # Handle results from all batches
                    for batch_result in batch_results:
                        for filePath, (embeddings, metadatas, chunksProcessed) in batch_result:
                            if embeddings is not None and metadatas is not None and chunksProcessed > 0:
                                print(f"Adding {chunksProcessed} chunks from {filePath.name} to vector store")
                                self.vectorStore.addEmbeddings(embeddings, metadatas)
                                successfulFiles += 1
                                totalChunks += chunksProcessed
                                
                                fileName = filePath.name
                                fileSize = os.path.getsize(filePath)
                                self.ingestedFiles.append({'name': fileName, 'size': fileSize})
                                self._saveIngestedFiles()
                                
                                print(f"Successfully processed file: {fileName}, {chunksProcessed} chunks.")
                                logging.info(f"Successfully processed file: {fileName}, {chunksProcessed} chunks.")
                                
                                # Update progress if callback provided
                                if progress_callback:
                                    progress_callback(chunksProcessed)
                                    
                except Exception as e:
                    print(f"Error in parallel processing: {e}")
                    logging.error(f"Error in parallel processing: {e}", exc_info=True)
        else:
            # Sequential processing
            print(f"Using sequential processing for {len(filesToProcess)} files")
            for filePath in filesToProcess:
                try:
                    fileName = filePath.name
                    fileSize = os.path.getsize(filePath)
                    
                    print(f"Processing file: {fileName}")
                    
                    # Extract text
                    rawText = extractTextFromFile(filePath)
                    textPreprocessor = TextPreprocessor(self.chunkSize, self.chunkOverlap)
                    cleanedText = textPreprocessor.cleanText(rawText)
                    chunks = textPreprocessor.segmentText(cleanedText)
                    
                    print(f"Generated {len(chunks)} chunks from {fileName}")
                    
                    if not chunks:
                        logging.warning(f"No chunks from {filePath}")
                        print(f"No chunks generated from {fileName}, skipping")
                        continue
                        
                    # Generate embeddings
                    print(f"Generating embeddings for {len(chunks)} chunks from {fileName}")
                    embeddings = self.embedder.encodeChunks(chunks)
                    
                    # Create metadata
                    metadatas = [
                        {"file": fileName, "chunkIndex": idx, "chunkLength": len(chunk.split()), "chunkText": chunk}
                        for idx, chunk in enumerate(chunks)]
                        
                    # Add to vector store
                    print(f"Adding {len(embeddings)} embeddings to vector store from {fileName}")
                    self.vectorStore.addEmbeddings(embeddings, metadatas)
                    
                    # Update ingested files list
                    self.ingestedFiles.append({'name': fileName, 'size': fileSize})
                    self._saveIngestedFiles()
                    
                    successfulFiles += 1
                    totalChunks += len(chunks)
                    
                    print(f"Successfully processed file: {fileName}, {len(chunks)} chunks.")
                    logging.info(f"Successfully processed file: {fileName}, {len(chunks)} chunks.")
                    
                    # Update progress if callback provided
                    if progress_callback:
                        progress_callback(len(chunks))
                        
                except Exception as e:
                    print(f"Error processing file {filePath}: {e}")
                    logging.error(f"Error processing file {filePath}: {e}", exc_info=True)
                    # Continue processing other files

            print(f"Finished processing {successfulFiles}/{len(filesToProcess)} files, {totalChunks} chunks")
            logging.info(f"Processed {successfulFiles}/{len(filesToProcess)} files, {totalChunks} chunks")
            return successfulFiles


    @staticmethod
    def _process_file_batch(args):
        """Process a batch of files with a single model instance."""
        batch, maxFileSizeMB, chunkSize, chunkOverlap, modelName = args
        results = []
        embedder = None
        try:
            # Load model once for the batch
            embedder = SentenceTransformerEmbedder(modelName)
            for filePath in batch:
                try:
                    # Extract text
                    rawText = extractTextFromFile(filePath)
                    textPreprocessor = TextPreprocessor(chunkSize, chunkOverlap)
                    cleanedText = textPreprocessor.cleanText(rawText)
                    chunks = textPreprocessor.segmentText(cleanedText)
                    
                    if not chunks:
                        logging.warning(f"No chunks from {filePath}")
                        results.append((filePath, (None, None, 0)))
                        continue
                    
                    # Generate embeddings reusing the same embedder
                    embeddings = embedder.encodeChunks(chunks)
                    
                    # Create metadata
                    metadatas = [
                        {"file": filePath.name, "chunkIndex": idx, "chunkLength": len(chunk.split()), "chunkText": chunk}
                        for idx, chunk in enumerate(chunks)
                    ]
                    
                    results.append((filePath, (embeddings, metadatas, len(chunks))))
                except Exception as e:
                    logging.error(f"Error processing {filePath}: {e}")
                    results.append((filePath, (None, None, 0)))
        finally:
            # Clean up resources
            if embedder:
                del embedder
            gc.collect()
        return results

    def saveResults(self, indexPath: Optional[str] = None, metadataPath: Optional[str] = None) -> None:
        """Save the vector store to disk."""
        try:
            self.vectorStore.save(indexPath, metadataPath)
            logging.info("Vector store saved successfully.")
            self.vectorStore.release_model() # Release the model after saving
        except Exception as e:
            logging.error(f"Error during saving vector store: {e}", exc_info=True)
            raise  # Propagate the error

def parseArguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process RTF files into embeddings for RAG.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with RTF files")
    parser.add_argument("--output_index", type=str, default=str(config.defaultIndexPath), help="FAISS index path")
    parser.add_argument("--output_metadata", type=str, default=str(config.defaultMetadataPath), help="Metadata path")
    parser.add_argument("--model_name", type=str, default=config.defaultModelName, help="SentenceTransformer model")
    parser.add_argument("--max_workers", type=int, default=config.defaultMaxWorkers, help="Number of workers")
    parser.add_argument("--chunk_size", type=int, default=config.defaultChunkSize, help="Words per chunk")
    parser.add_argument("--chunk_overlap", type=int, default=config.defaultChunkOverlap, help="Chunk overlap")
    parser.add_argument("--max_file_size_mb", type=int, default=config.maxFileSizeMB, help="Maximum file size in MB")
    parser.add_argument("--disable_multiprocessing", action='store_true', help="Disable multiprocessing")
    parser.add_argument("--ingested_files_path", type=str, default=str(config.defaultIngestedFilesPath), help="Path to the ingested files list")
    return parser.parse_args()

def main() -> None:
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.loggingLevel),
        format=config.loggingFormat,
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("rtf_processor.log")]
    )

    # Parse arguments
    args = parseArguments()

    try:
        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output_index)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_metadata)), exist_ok=True)
        
        # Create pipeline
        pipeline = RtfProcessingPipeline(
            modelName=args.model_name,
            chunkSize=args.chunk_size,
            chunkOverlap=args.chunk_overlap,
            maxWorkers=args.max_workers,
            maxFileSizeMB=args.max_file_size_mb,
            useMultiprocessing=not args.disable_multiprocessing,
            ingestedFilesPath=args.ingested_files_path
        )

        # Process files
        processedFiles = pipeline.processDirectory(Path(args.input_dir))
        
        # Save results
        pipeline.saveResults(args.output_index, args.output_metadata)

        logging.info(f"Processing complete. {processedFiles} files processed successfully.")
        logging.info(f"Vector store saved to {args.output_index} and {args.output_metadata}")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()