import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from pathlib import Path
import logging
import os
import threading
import tempfile
import shutil
from config import config
from dotenv import set_key, find_dotenv
from typing import Any
from embeddingGenerator import SentenceTransformerEmbedder
from vectorStore import VectorStore
from main import RtfProcessingPipeline
import gc

class UpdateVectorStoreTab(ttk.Frame):
    """Tab for updating the existing vector store with new files."""

    def __init__(self, parent: ttk.Notebook, app: Any):
        """Initialise the Update Vector Store Tab."""
        super().__init__(parent)
        self.parent = parent
        self.app = app
        self.logger = logging.getLogger(__name__)
        self.create_widgets()

    def create_widgets(self) -> None:
        """Create and layout widgets for the Update Vector Store tab."""
        # --- Existing Vector Store Paths ---
        existingFrame = ttk.LabelFrame(self, text="Existing Vector Store Paths")
        existingFrame.pack(padx=10, pady=10, fill="x")

        ttk.Label(existingFrame, text="Index Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.updateIndexEntry = ttk.Entry(existingFrame, width=60)
        self.updateIndexEntry.insert(0, str(config.defaultIndexPath))
        self.updateIndexEntry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browseUpdateIndexButton = ttk.Button(existingFrame, text="Browse", command=self.browseUpdateIndexFile)
        self.browseUpdateIndexButton.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(existingFrame, text="Metadata Path:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.updateMetadataEntry = ttk.Entry(existingFrame, width=60)
        self.updateMetadataEntry.insert(0, str(config.defaultMetadataPath))
        self.updateMetadataEntry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.browseUpdateMetadataButton = ttk.Button(existingFrame, text="Browse", command=self.browseUpdateMetadataFile)
        self.browseUpdateMetadataButton.grid(row=1, column=2, padx=5, pady=5)

        # --- New Files Directory ---
        inputFrame = ttk.LabelFrame(self, text="New Files Directory")
        inputFrame.pack(padx=10, pady=10, fill="x")

        ttk.Label(inputFrame, text="Select Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.inputDirEntry = ttk.Entry(inputFrame, width=60)
        self.inputDirEntry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browseInputDirButton = ttk.Button(inputFrame, text="Browse", command=self.browseInputDir)
        self.browseInputDirButton.grid(row=0, column=2, padx=5, pady=5)

        # --- Multiprocessing Configuration ---
        multiprocessingFrame = ttk.LabelFrame(self, text="Multiprocessing")
        multiprocessingFrame.pack(padx=10, pady=10, fill="x")

        self.disableMultiprocessingVar = tk.BooleanVar(value=True)  # Default to disabled for memory safety
        self.disableMultiprocessingCheckbutton = ttk.Checkbutton(
            multiprocessingFrame,
            text="Disable Multiprocessing (Use single worker - recommended for large batches)",
            variable=self.disableMultiprocessingVar
        )
        self.disableMultiprocessingCheckbutton.pack(padx=5, pady=5, fill="x")

# --- Processing Parameters ---
        paramsFrame = ttk.LabelFrame(self, text="Processing Parameters")
        paramsFrame.pack(padx=10, pady=10, fill="x")

        ttk.Label(paramsFrame, text="Model Name:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.modelNameEntry = ttk.Entry(paramsFrame, width=60)
        self.modelNameEntry.insert(0, config.defaultModelName)
        self.modelNameEntry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(paramsFrame, text="Max Workers:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.maxWorkersEntry = ttk.Entry(paramsFrame, width=10)
        self.maxWorkersEntry.insert(0, str(config.defaultMaxWorkers))
        self.maxWorkersEntry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(paramsFrame, text="Chunk Size:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.chunkSizeEntry = ttk.Entry(paramsFrame, width=10)
        self.chunkSizeEntry.insert(0, str(config.defaultChunkSize))
        self.chunkSizeEntry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(paramsFrame, text="Chunk Overlap:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.chunkOverlapEntry = ttk.Entry(paramsFrame, width=10)
        self.chunkOverlapEntry.insert(0, str(config.defaultChunkOverlap))
        self.chunkOverlapEntry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        # --- Update Button ---
        self.updateButton = ttk.Button(self, text="Update Vector Store", command=self.startUpdateVectorStore)
        self.updateButton.pack(pady=20)

        # --- Progress Bar ---
        self.updateProgressBar = ttk.Progressbar(self, mode='determinate', maximum=100)
        self.updateProgressBar.pack(padx=10, pady=10, fill="x")

        # --- Status Text Area ---
        statusFrame = ttk.LabelFrame(self, text="Update Status")
        statusFrame.pack(padx=10, pady=10, fill="both", expand=True)
        self.updateStatusText = scrolledtext.ScrolledText(statusFrame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.updateStatusText.pack(fill="both", expand=True)

        # --- Save Configuration Button ---
        self.saveConfigButton = ttk.Button(self, text="Save Configuration to .env", command=self.saveConfig)
        self.saveConfigButton.pack(pady=10)

    def browseUpdateIndexFile(self) -> None:
        """Open file selection dialogue for the existing FAISS index file."""
        filePath = filedialog.askopenfilename(defaultextension=".faiss", filetypes=[("FAISS Index Files", "*.faiss"), ("All Files", "*.*")])
        if filePath:
            self.updateIndexEntry.delete(0, tk.END)
            self.updateIndexEntry.insert(0, filePath)

    def browseUpdateMetadataFile(self) -> None:
        """Open file selection dialogue for the existing metadata file."""
        filePath = filedialog.askopenfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")])
        if filePath:
            self.updateMetadataEntry.delete(0, tk.END)
            self.updateMetadataEntry.insert(0, filePath)

    def browseInputDir(self) -> None:
        """Open directory selection dialogue for new files."""
        directory = filedialog.askdirectory()
        if directory:
            self.inputDirEntry.delete(0, tk.END)
            self.inputDirEntry.insert(0, directory)

    def log_update_message(self, message: str) -> None:
        """Append message to status text area in the Update Vector Store tab."""
        self.updateStatusText.config(state=tk.NORMAL)
        self.updateStatusText.insert(tk.END, message + "\n")
        self.updateStatusText.see(tk.END)
        self.updateStatusText.config(state=tk.DISABLED)
        self.app.update_status(message)
        print(message)  # Added to print to terminal

    def validate_numeric_input(self, value, field_name, min_value=None, max_value=None):
        """Validate that a string can be converted to an integer within range."""
        try:
            num_value = int(value)
            if min_value is not None and num_value < min_value:
                raise ValueError(f"{field_name} must be at least {min_value}")
            if max_value is not None and num_value > max_value:
                raise ValueError(f"{field_name} must be less than or equal to {max_value}")
            return num_value
        except ValueError:
            if str(value).strip() == '':
                raise ValueError(f"{field_name} cannot be empty")
            raise ValueError(f"Invalid {field_name}: '{value}' is not a valid integer")          
   
    def validate_path(self, path_str, check_type="file", allow_create=False):
        """
        Validate a path string.
        
        Args:
            path_str: The path string to validate
            check_type: One of "file" or "dir" to check the path type
            allow_create: Whether to allow non-existent paths (for output files)
            
        Returns:
            The validated Path object
        
        Raises:
            ValueError: If the path is invalid
        """
        if not path_str or not path_str.strip():
            raise ValueError(f"Path cannot be empty")
            
        try:
            # Convert to absolute path and resolve to eliminate '..' components
            path = Path(path_str).resolve(strict=False)
            
            # Check if path exists
            if check_type == "file":
                if path.exists() and not path.is_file():
                    raise ValueError(f"Path exists but is not a file: {path}")
                if not allow_create and not path.exists():
                    raise ValueError(f"File does not exist: {path}")
            elif check_type == "dir":
                if path.exists() and not path.is_dir():
                    raise ValueError(f"Path exists but is not a directory: {path}")
                if not path.exists():
                    raise ValueError(f"Directory does not exist: {path}")
                    
            return path
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Invalid path: {path_str} - {e}")

    def startUpdateVectorStore(self) -> None:
        """Initiate the process of updating the vector store."""
        # Disable the update button during validation to prevent multiple clicks
        self.updateButton.config(state=tk.DISABLED)
        
        try:
            indexPath = self.updateIndexEntry.get()
            metadataPath = self.updateMetadataEntry.get()
            
            # Validate paths
            try:
                self.validate_path(indexPath, "file", allow_create=True)
                self.validate_path(metadataPath, "file", allow_create=True)
                inputDir = self.validate_path(self.inputDirEntry.get(), "dir")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid path: {e}")
                self.updateButton.config(state=tk.NORMAL)  # Re-enable button on error
                return
            
            modelName = self.modelNameEntry.get()
            
            # Input validation
            try:
                maxWorkers = self.validate_numeric_input(self.maxWorkersEntry.get(), "Max Workers", min_value=1)
                chunkSize = self.validate_numeric_input(self.chunkSizeEntry.get(), "Chunk Size", min_value=1)
                chunkOverlap = self.validate_numeric_input(self.chunkOverlapEntry.get(), "Chunk Overlap", min_value=0)
                
                # Additional validation
                if chunkOverlap >= chunkSize:
                    raise ValueError("Chunk Overlap must be less than Chunk Size")

                os.makedirs(Path(indexPath).parent, exist_ok=True)
                os.makedirs(Path(metadataPath).parent, exist_ok=True)
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
                self.updateButton.config(state=tk.NORMAL)  # Re-enable button on error
                return
            

            # Removed the confirmation dialog that asked about overwriting
            self.log_update_message("Starting vector store update...")
            self.log_update_message(f"Index Path: {indexPath}")
            self.log_update_message(f"Metadata Path: {metadataPath}")
            self.log_update_message(f"Input Directory: {inputDir}")
            self.log_update_message(f"Model Name: {modelName}")
            self.log_update_message(f"Max Workers: {maxWorkers}")
            self.log_update_message(f"Chunk Size: {chunkSize}")
            self.log_update_message(f"Chunk Overlap: {chunkOverlap}")

            # Disable the update button during processing and start progress bar
            self.updateButtonOriginalStyle = self.updateButton.cget("style")
            disabled_style = "Disabled.TButton"
            style = ttk.Style()
            style.configure(disabled_style, background="lightgrey")
            self.updateButton.config(state=tk.DISABLED, style=disabled_style)
            self.updateProgressBar['value'] = 0
            
            # Process in background thread to prevent GUI freeze
            def process_in_thread():
                try:
                    # Track chunks processed instead of files
                    total_chunks_processed = [0]
                    estimated_total_chunks = [100]  # Start with placeholder value
                    modules_loaded = {}
                    
                    def update_progress(chunks_processed):
                        total_chunks_processed[0] += chunks_processed
                        # Update progress based on chunks
                        progress = min(100, int((total_chunks_processed[0] / max(estimated_total_chunks[0], 1)) * 100))
                        self.master.after(0, lambda: self._update_progress(progress))
                    
                    # Load existing vector store
                    embedGenerator = None
                    vectorStore = None
                    
                    try:
                        # Load existing vector store
                        self.master.after(0, lambda: self.log_update_message(f"Loading embedding model: {modelName}"))
                        print(f"Loading model: {modelName}")
                        embedGenerator = SentenceTransformerEmbedder(modelName)
                        
                        # Track module loading
                        if modelName in modules_loaded:
                            modules_loaded[modelName] += 1
                        else:
                            modules_loaded[modelName] = 1
                        self.master.after(0, lambda: self.log_update_message(f"Model loaded: {modelName}"))
                        
                        vectorStore = VectorStore(embedGenerator.embeddingDim)
                        
                        # Check if files exist - if not, we'll create new ones
                        if os.path.exists(indexPath) and os.path.exists(metadataPath):
                            try:
                                self.master.after(0, lambda: self.log_update_message("Loading existing vector store..."))
                                vectorStore.load(indexPath, metadataPath)
                                self.master.after(0, lambda: self.log_update_message(
                                    f"Loaded existing vector store successfully with {vectorStore.index.ntotal} entries"))
                            except Exception as e:
                                self.master.after(0, lambda: self.log_update_message(
                                    f"Failed to load existing vector store: {e}, creating new one"))
                                # If we fail to load, we'll create a new one below
                        else:
                            self.master.after(0, lambda: self.log_update_message("Creating new vector store"))
                        
                        # Get the multiprocessing setting
                        disableMultiprocessing = self.disableMultiprocessingVar.get()
                        self.log_update_message(f"Disable Multiprocessing: {disableMultiprocessing}")

                        # Initialize pipeline with the setting
                        # # Create processing pipeline with the existing vector store to ensure we append
                        self.master.after(0, lambda: self.log_update_message("Initializing processing pipeline..."))
                        pipeline = RtfProcessingPipeline(
                            modelName=modelName,
                            chunkSize=chunkSize,
                            chunkOverlap=chunkOverlap,
                            maxWorkers=maxWorkers,
                            useMultiprocessing=not disableMultiprocessing,
                            existingVectorStore=vectorStore
                        )
                        
                        # Estimate chunks per file to give better progress indication
                        # Get one sample file to estimate chunk count
                        for ext in config.supportedExtensions:
                            sample_files = list(inputDir.glob(f"*{ext}"))
                            if sample_files:
                                try:
                                    self.master.after(0, lambda: self.log_update_message(
                                        f"Analyzing sample file to estimate workload: {sample_files[0].name}"))
                                    chunks = pipeline.processFile(sample_files[0])
                                    avg_chunks_per_file = max(1, chunks)  # At least 1
                                    # Count files to process
                                    total_files = sum(len(list(inputDir.glob(f"*{ext}"))) for ext in config.supportedExtensions)
                                    estimated_total_chunks[0] = total_files * avg_chunks_per_file
                                    self.master.after(0, lambda: self.log_update_message(
                                        f"Estimated {estimated_total_chunks[0]} chunks to process from {total_files} files"))
                                    break
                                except Exception as e:
                                    self.master.after(0, lambda: self.log_update_message(
                                        f"Error analyzing sample file: {e}"))
                        
                        # Process directory with progress tracking
                        self.master.after(0, lambda: self.log_update_message("Processing files..."))
                        
                        # Define custom progress callback with more detailed logging
                        def progress_callback(chunks_processed):
                            update_progress(chunks_processed)
                            self.master.after(0, lambda: self.log_update_message(
                                f"Processed {chunks_processed} chunks. Total: {total_chunks_processed[0]} of estimated {estimated_total_chunks[0]}"))
                        
                        # Process directory with progress callback
                        processedFiles = pipeline.processDirectory(inputDir, progress_callback)
                        
                        # Always save results (no more overwrite concern as we're ensuring append behavior)
                        self.master.after(0, lambda: self.log_update_message("Saving updated vector store..."))
                        pipeline.saveResults(indexPath, metadataPath)
                        
                        # Update progress
                        self.master.after(0, lambda: self._update_progress_final(processedFiles, total_chunks_processed[0]))
                        
                        # Report on modules loaded
                        module_report = ", ".join([f"{name}: {count}" for name, count in modules_loaded.items()])
                        self.master.after(0, lambda: self.log_update_message(f"Modules loaded: {module_report}"))
                        
                    except Exception as e:
                        self.master.after(0, lambda: self._handle_error(e))
                    finally:
                        # Clean up resources even if an error occurred
                        try:
                            if embedGenerator:
                                self.master.after(0, lambda: self.log_update_message("Cleaning up embedding generator..."))
                                del embedGenerator
                            if vectorStore:
                                self.master.after(0, lambda: self.log_update_message("Releasing vector store model..."))
                                vectorStore.release_model()
                                del vectorStore
                            self.master.after(0, lambda: self.log_update_message("Releasing all sentence transformer models..."))
                            SentenceTransformerEmbedder.release_models()
                            self.master.after(0, lambda: self.log_update_message("Running garbage collection..."))
                            gc.collect()
                        except Exception as cleanup_error:
                            logging.error(f"Error during resource cleanup: {cleanup_error}")
                except Exception as e:
                    self.master.after(0, lambda: self._handle_error(e))
            
            # Start thread
            update_thread = threading.Thread(target=process_in_thread)
            update_thread.daemon = True
            update_thread.start()
        except Exception as e:
            # For any unexpected errors during setup
            self._handle_error(e)
    
    def _update_progress(self, progress):
        """Update progress bar value."""
        self.updateProgressBar['value'] = progress
        self.master.update_idletasks()
    
    def _update_progress_final(self, processed_files, total_chunks):
        """Update progress bar to final state."""
        self.updateProgressBar['value'] = 100
        
        self.log_update_message(f"Processed {processed_files} files with {total_chunks} chunks")
        messagebox.showinfo("Success", "Vector store updated successfully!")
        self.app.update_status("Vector store updated successfully.")
        self.updateButton.config(state=tk.NORMAL, style=self.updateButtonOriginalStyle)
    
    def _handle_error(self, e):
        """Handle errors in the UI thread."""
        self.app.logger.error(f"Update failed: {e}", exc_info=True)
        self.log_update_message(f"Error during update: {e}")
        messagebox.showerror("Error", f"An error occurred during update: {e}")
        self.app.update_status("Vector store update failed.")
        self.updateButton.config(state=tk.NORMAL, style=self.updateButtonOriginalStyle)

    def saveConfig(self) -> None:
        """Save the configuration from the GUI to the .env file."""
        index_path = self.updateIndexEntry.get()
        metadata_path = self.updateMetadataEntry.get()
        output_dir = Path(self.updateIndexEntry.get()).parent.as_posix()
        index_name = Path(self.updateIndexEntry.get()).name
        metadata_name = Path(self.updateMetadataEntry.get()).name
        model_name = self.modelNameEntry.get()
        max_workers = self.maxWorkersEntry.get()
        chunk_size = self.chunkSizeEntry.get()
        chunk_overlap = self.chunkOverlapEntry.get()

        dotenv_path = find_dotenv()
        if dotenv_path:
            set_key(dotenv_path, "DEFAULT_INDEX_PATH", index_path)
            set_key(dotenv_path, "DEFAULT_METADATA_PATH", metadata_path)
            set_key(dotenv_path, "DEFAULT_OUTPUT_DIR", output_dir)
            set_key(dotenv_path, "DEFAULT_INDEX_NAME", index_name)
            set_key(dotenv_path, "DEFAULT_METADATA_NAME", metadata_name)
            set_key(dotenv_path, "MODEL_NAME", model_name)
            set_key(dotenv_path, "MAX_WORKERS", max_workers)
            set_key(dotenv_path, "CHUNK_SIZE", chunk_size)
            set_key(dotenv_path, "CHUNK_OVERLAP", chunk_overlap)
            messagebox.showinfo("Success", "Update Vector Store configuration saved to .env. Please restart the application for the changes to fully take effect.")
            self.app.update_status("Update Vector Store configuration saved to .env")
        else:
            messagebox.showerror("Error", ".env file not found.")
            self.app.update_status("Error: .env file not found.")