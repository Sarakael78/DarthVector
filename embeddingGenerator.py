import torch  # Add this import
import logging
import threading
import time
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional, Any, Dict
from config import config
import gc
import weakref

class BaseEmbedder(ABC):
    """Abstract base class for embedding generators."""

    @abstractmethod
    def encodeChunks(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        pass

    @property
    @abstractmethod
    def embeddingDim(self) -> int:
        """Return the embedding dimension."""
        pass

class SentenceTransformerEmbedder(BaseEmbedder):
    """Generates embeddings using SentenceTransformer models."""

    _model_cache = {}
    _lock = threading.RLock()
    _ref_counts: Dict[str, int] = {}  # Add this line to define _ref_counts
    
    def __init__(self, modelName: Optional[str] = None, batchSize: int = 16,  # Reduced batch size
                 maxWorkers: Optional[int] = None, existing_model: Optional[Any] = None) -> None:
        """
        Initialise the embedder.

        Args:
            modelName: Name of the SentenceTransformer model (defaults to config.modelName).
            batchSize: Batch size for encoding.
            maxWorkers: Number of threads for parallel encoding (currently unused).
            existing_model: Optional existing model instance to reuse.

        Raises:
            RuntimeError: If model loading fails.
        """
        self.modelName = modelName if modelName is not None else config.defaultModelName
        self.batchSize = batchSize
        self.maxWorkers = maxWorkers
        self.logger = logging.getLogger(__name__)
        
        try:
            # Use existing model if provided
            if existing_model is not None:
                print(f"Using existing model instance for {self.modelName}")
                self.model = existing_model
                self._embeddingDim = self.model.get_sentence_embedding_dimension()
                self.logger.info(f"Using existing model, dimension: {self._embeddingDim}")
                print(f"Model dimension: {self._embeddingDim}")
            else:
                # Get or create model with global caching
                with self._lock:
                    if self.modelName not in self._model_cache:
                        print(f"Loading model '{self.modelName}'")
                        self.logger.info(f"Loading model '{self.modelName}'")
                        startTime = time.time()
                        self._model_cache[self.modelName] = SentenceTransformer(self.modelName)
                        loadTime = time.time() - startTime
                        print(f"Model '{self.modelName}' loaded in {loadTime:.2f}s")
                        self.logger.info(f"Model '{self.modelName}' loaded in {loadTime:.2f}s")
                
                self.model = self._model_cache[self.modelName]
                self._embeddingDim = self.model.get_sentence_embedding_dimension()
                print(f"Using cached model with dimension: {self._embeddingDim}")
        except Exception as e:
            self.logger.error(f"Failed to load model '{self.modelName}': {e}", exc_info=True)
            print(f"Failed to load model '{self.modelName}': {e}")
            raise RuntimeError(f"Failed to load model '{self.modelName}': {e}")

    def _get_model_for_thread(self):
        """Get or create a model instance for the current thread with reference counting."""
        if hasattr(self._thread_local, 'model'):
            print(f"Using existing thread-local model for {self.modelName}")
            self.model = self._thread_local.model
            self._embeddingDim = self.model.get_sentence_embedding_dimension()
            return
        
        with self._lock:
            if self.modelName not in SentenceTransformerEmbedder._model_instances:
                print(f"Loading model '{self.modelName}' for thread {threading.current_thread().name}")
                self.logger.info(f"Loading model '{self.modelName}' for thread {threading.current_thread().name}")
                startTime = time.time()
                model = SentenceTransformer(self.modelName)
                SentenceTransformerEmbedder._model_instances[self.modelName] = weakref.ref(model)
                SentenceTransformerEmbedder._ref_counts[self.modelName] = 1
                loadTime = time.time() - startTime
                print(f"Model '{self.modelName}' loaded in {loadTime:.2f}s")
                self.logger.info(f"Model '{self.modelName}' loaded in {time.time() - startTime:.2f}s")
                print(f"Current model instances: {list(SentenceTransformerEmbedder._model_instances.keys())}")
                print(f"Reference counts: {SentenceTransformerEmbedder._ref_counts}")
            else:
                SentenceTransformerEmbedder._ref_counts[self.modelName] += 1
                model_ref = SentenceTransformerEmbedder._model_instances[self.modelName]
                model = model_ref()
                if model is None:
                    print(f"Model was garbage collected unexpectedly, reloading {self.modelName}")
                    model = SentenceTransformer(self.modelName)
                    SentenceTransformerEmbedder._model_instances[self.modelName] = weakref.ref(model)
                else:
                    print(f"Reusing existing model {self.modelName}, new reference count: {SentenceTransformerEmbedder._ref_counts[self.modelName]}")

            self._thread_local.model = model
            self.model = model
            self._embeddingDim = model.get_sentence_embedding_dimension()
            print(f"Model ready with dimension: {self._embeddingDim}")

    def encodeChunks(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        if not chunks:
            return np.array([], dtype=np.float32).reshape(0, self.embeddingDim)
        try:
            print(f"Encoding {len(chunks)} chunks with batch size {self.batchSize}")
            result = self.model.encode(chunks, batch_size=self.batchSize, convert_to_numpy=True)
            
            print(f"Encoding complete, generated {len(result)} embeddings of dimension {result.shape[1]}")
            return result
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}", exc_info=True)
            print(f"Encoding failed: {e}")
            raise RuntimeError(f"Failed to encode chunks: {e}")
        
    @property
    def embeddingDim(self) -> int:
        """Return the embedding dimension."""
        return self._embeddingDim
    
    @classmethod
    def release_models(cls) -> None:
        """Release all model instances from memory."""
        with cls._lock:
            print("Releasing SentenceTransformer models")
            cls._model_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("All models released")
            logging.info("Released all SentenceTransformer model instances.")    
            
    def __del__(self):
        """Decrement reference count when embedder instance is deleted."""
        if hasattr(self, 'modelName'):
            with self._lock:
                if self.modelName in SentenceTransformerEmbedder._ref_counts:
                    SentenceTransformerEmbedder._ref_counts[self.modelName] -= 1
                    print(f"Decremented reference count for {self.modelName} to {SentenceTransformerEmbedder._ref_counts[self.modelName]}")
                    if SentenceTransformerEmbedder._ref_counts[self.modelName] <= 0:
                        print(f"Reference count for {self.modelName} is zero, releasing models")
                        SentenceTransformerEmbedder.release_models()