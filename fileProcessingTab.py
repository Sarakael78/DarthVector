# fileProcessingTab.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from pathlib import Path
import logging
import os
import threading
import json
from typing import Any
from dotenv import set_key, find_dotenv
from config import config
from main import RtfProcessingPipeline
import gc

class FileProcessingTab(ttk.Frame):
    """Tab for processing files and updating vector store."""

    def __init__(self, parent: ttk.Notebook, app: Any):
        """Initialise the File Processing Tab."""
        super().__init__(parent)
        self.parent = parent
        self.app = app
        self.logger = logging.getLogger(__name__)
        self.files_to_process = 0  # Local counter for this tab
        self.create_widgets()

    def create_widgets(self) -> None:
        """Create and layout widgets for the File Processing tab."""
        # --- Vector Store Paths Section ---
        storeFrame = ttk.LabelFrame(self, text="Vector Store Paths")
        storeFrame.pack(padx=10, pady=10, fill="x")

        ttk.Label(storeFrame, text="Index Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.indexEntry = ttk.Entry(storeFrame, width=60)
        self.indexEntry.insert(0, str(config.defaultIndexPath))
        self.indexEntry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browseIndexButton = ttk.Button(storeFrame, text="Browse", command=self.browseIndexFile)
        self.browseIndexButton.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(storeFrame, text="Metadata Path:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.metadataEntry = ttk.Entry(storeFrame, width=60)
        self.metadataEntry.insert(0, str(config.defaultMetadataPath))
        self.metadataEntry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.browseMetadataButton = ttk.Button(storeFrame, text="Browse", command=self.browseMetadataFile)
        self.browseMetadataButton.grid(row=1, column=2, padx=5, pady=5)

        # --- Input Files Section ---
        inputFrame = ttk.LabelFrame(self, text="Input Files")
        inputFrame.pack(padx=10, pady=10, fill="x")

        ttk.Label(inputFrame, text="Select Directory:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.inputDirEntry = ttk.Entry(inputFrame, width=60)
        self.inputDirEntry.insert(0, str(config.defaultInputDir))
        self.inputDirEntry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browseInputDirButton = ttk.Button(inputFrame, text="Browse", command=self.browseInputDir)
        self.browseInputDirButton.grid(row=0, column=2, padx=5, pady=5)

        # --- Processing Parameters Section ---
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

        ttk.Label(paramsFrame, text="Max File Size (MB):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.maxFileSizeEntry = ttk.Entry(paramsFrame, width=10)
        self.maxFileSizeEntry.insert(0, str(config.maxFileSizeMB))
        self.maxFileSizeEntry.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        # --- Multiprocessing Configuration ---
        multiprocessingFrame = ttk.LabelFrame(self, text="Multiprocessing")
        multiprocessingFrame.pack(padx=10, pady=10, fill="x")

        self.disableMultiprocessingVar = tk.BooleanVar(value=config.disableMultiprocessing)
        self.disableMultiprocessingCheckbutton = ttk.Checkbutton(
            multiprocessingFrame,
            text="Disable Multiprocessing (Use single worker)",
            variable=self.disableMultiprocessingVar
        )
        self.disableMultiprocessingCheckbutton.pack(padx=5, pady=5, fill="x")

        # --- Process Button ---
        buttonFrame = ttk.Frame(self)
        buttonFrame.pack(padx=10, pady=10, fill="x")
        
        self.processButton = ttk.Button(buttonFrame, text="Process Files / Update Vector Store", command=self.startProcessing)
        self.processButton.pack(pady=5, fill="x")

        # --- Progress Bar ---
        self.progressBar = ttk.Progressbar(self, mode='determinate', maximum=100)
        self.progressBar.pack(padx=10, pady=10, fill="x")

        # --- Status Text Area ---
        statusFrame = ttk.LabelFrame(self, text="Status")
        statusFrame.pack(padx=10, pady=10, fill="both", expand=True)
        self.statusText = scrolledtext.ScrolledText(statusFrame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.statusText.pack(fill="both", expand=True)

        # --- Save Configuration Button ---
        self.saveConfigButton = ttk.Button(self, text="Save Configuration to .env", command=self.saveConfig)
        self.saveConfigButton.pack(pady=10)

    def browseIndexFile(self) -> None:
        """Open file selection dialogue for the FAISS index file."""
        filePath = filedialog.askopenfilename(defaultextension=".faiss", filetypes=[("FAISS Index Files", "*.faiss"), ("All Files", "*.*")])
        if filePath:
            self.indexEntry.delete(0, tk.END)
            self.indexEntry.insert(0, filePath)

    def browseMetadataFile(self) -> None:
        """Open file selection dialogue for the metadata file."""
        filePath = filedialog.askopenfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")])
        if filePath:
            self.metadataEntry.delete(0, tk.END)
            self.metadataEntry.insert(0, filePath)

    def browseInputDir(self) -> None:
        """Open directory selection dialogue for input files."""
        directory = filedialog.askdirectory()
        if directory:
            self.inputDirEntry.delete(0, tk.END)
            self.inputDirEntry.insert(0, directory)

    def log_message(self, message: str) -> None:
        """Append message to status text area."""
        self.statusText.config(state=tk.NORMAL)
        self.statusText.insert(tk.END, message + "\n")
        self.statusText.see(tk.END)
        self.statusText.config(state=tk.DISABLED)
        self.app.update_status(message)

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

    def startProcessing(self) -> None:
        """Initiate the file processing pipeline."""
        # Disable the process button during validation to prevent multiple clicks
        self.processButton.config(state=tk.DISABLED)
        
        try:
            # Validate inputs
            indexPath = self.indexEntry.get()
            metadataPath = self.metadataEntry.get()
            
            # Validate paths
            try:
                self.validate_path(indexPath, "file", allow_create=True)
                self.validate_path(metadataPath, "file", allow_create=True)
                inputDir = self.validate_path(self.inputDirEntry.get(), "dir")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid path: {e}")
                self.processButton.config(state=tk.NORMAL)  # Re-enable button on error
                return
            
            modelName = self.modelNameEntry.get()
            
            # Input validation for numeric fields
            try:
                maxWorkers = self.validate_numeric_input(self.maxWorkersEntry.get(), "Max Workers", min_value=1)
                chunkSize = self.validate_numeric_input(self.chunkSizeEntry.get(), "Chunk Size", min_value=1)
                chunkOverlap = self.validate_numeric_input(self.chunkOverlapEntry.get(), "Chunk Overlap", min_value=0)
                maxFileSizeMB = self.validate_numeric_input(self.maxFileSizeEntry.get(), "Max File Size (MB)", min_value=1)
                
                # Additional validation
                if chunkOverlap >= chunkSize:
                    raise ValueError("Chunk Overlap must be less than Chunk Size")
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
                self.processButton.config(state=tk.NORMAL)  # Re-enable button on error
                return
            
            # Ensure output directories exist
            os.makedirs(Path(indexPath).parent, exist_ok=True)
            os.makedirs(Path(metadataPath).parent, exist_ok=True)
            
            # Check if we're updating an existing vector store
            updating = os.path.exists(indexPath) and os.path.exists(metadataPath)
            operation_type = "updating" if updating else "creating"
            
            if updating:
                message = "You are about to update the existing vector store with new files. " \
                          "New files will be appended to the store. Continue?"
            else:
                message = "You are about to create a new vector store. Continue?"
            
            if not messagebox.askyesno("Confirm Operation", message):
                self.processButton.config(state=tk.NORMAL)  # Re-enable button on cancel
                return
            
            disableMultiprocessing = self.disableMultiprocessingVar.get()

            # Log the operation we're about to perform
            self.log_message(f"Starting {operation_type} vector store...")
            self.log_message(f"Index Path: {indexPath}")
            self.log_message(f"Metadata Path: {metadataPath}")
            self.log_message(f"Input Directory: {inputDir}")
            self.log_message(f"Model Name: {modelName}")
            self.log_message(f"Max Workers: {maxWorkers}")
            self.log_message(f"Chunk Size: {chunkSize}")
            self.log_message(f"Chunk Overlap: {chunkOverlap}")
            self.log_message(f"Max File Size (MB): {maxFileSizeMB}")
            self.log_message(f"Disable Multiprocessing: {disableMultiprocessing}")

            # Disable the process button during processing
            self.processButtonOriginalStyle = self.processButton.cget("style")
            disabled_style = "Disabled.TButton"
            style = ttk.Style()
            style.configure(disabled_style, background="lightgrey")
            self.processButton.config(state=tk.DISABLED, style=disabled_style)
            self.progressBar['value'] = 0
            
            # Process in background thread to prevent GUI freeze
            def process_in_thread():
                try:
                    # Track progress
                    total_chunks_processed = [0]
                    estimated_total_chunks = [100]  # Start with placeholder value
                    
                    def update_progress(chunks_processed):
                        total_chunks_processed[0] += chunks_processed
                        # Update progress based on chunks
                        progress = min(100, int((total_chunks_processed[0] / max(estimated_total_chunks[0], 1)) * 100))
                        self.master.after(0, lambda: self._update_progress(progress))
                    
                    # Create processing pipeline
                    pipeline = RtfProcessingPipeline(
                        modelName=modelName,
                        chunkSize=chunkSize,
                        chunkOverlap=chunkOverlap,
                        maxWorkers=maxWorkers,
                        maxFileSizeMB=maxFileSizeMB,
                        useMultiprocessing=not disableMultiprocessing
                    )
                    
                    # Process directory with recursive file search
                    self.master.after(0, lambda: self.log_message("Processing files recursively..."))
                    
                    # Define custom progress callback
                    def progress_callback(chunks_processed):
                        update_progress(chunks_processed)
                    
                    # Process directory with progress callback
                    processedFiles = pipeline.processDirectory(inputDir, progress_callback)
                    
                    # Save results
                    pipeline.saveResults(indexPath, metadataPath)
                    
                    # Update progress
                    self.master.after(0, lambda: self._update_progress_final(processedFiles, total_chunks_processed[0]))
                    
                except Exception as e:
                    self.master.after(0, lambda: self._handle_error(e))
            
            # Start thread
            processing_thread = threading.Thread(target=process_in_thread)
            processing_thread.daemon = True
            processing_thread.start()
        except Exception as e:
            # For any unexpected errors during setup
            self._handle_error(e)
    
    def _update_progress(self, progress):
        """Update progress bar value."""
        self.progressBar['value'] = progress
        self.master.update_idletasks()
    
    def _update_progress_final(self, processed_files, total_chunks):
        """Update progress bar to final state."""
        self.progressBar['value'] = 100
        
        self.log_message(f"Processed {processed_files} files with {total_chunks} chunks")
        messagebox.showinfo("Success", "Vector store processing completed successfully!")
        self.app.update_status("Vector store processing completed successfully.")
        self.processButton.config(state=tk.NORMAL, style=self.processButtonOriginalStyle)
    
    def _handle_error(self, e):
        """Handle errors in the UI thread."""
        self.app.logger.error(f"Processing failed: {e}", exc_info=True)
        self.log_message(f"Error during processing: {e}")
        messagebox.showerror("Error", f"An error occurred during processing: {e}")
        self.app.update_status("Vector store processing failed.")
        self.processButton.config(state=tk.NORMAL, style=self.processButtonOriginalStyle)

    def saveConfig(self) -> None:
        """Save the configuration to the .env file."""
        index_path = self.indexEntry.get()
        metadata_path = self.metadataEntry.get()
        output_dir = Path(self.indexEntry.get()).parent.as_posix()
        index_name = Path(self.indexEntry.get()).name
        metadata_name = Path(self.metadataEntry.get()).name
        model_name = self.modelNameEntry.get()
        max_workers = self.maxWorkersEntry.get()
        chunk_size = self.chunkSizeEntry.get()
        chunk_overlap = self.chunkOverlapEntry.get()
        max_file_size = self.maxFileSizeEntry.get()
        disable_multiprocessing = "true" if self.disableMultiprocessingVar.get() else "false"

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
            set_key(dotenv_path, "MAX_FILE_SIZE_MB", max_file_size)
            set_key(dotenv_path, "DEFAULT_INPUT_DIR", str(Path(self.inputDirEntry.get()).resolve()))
            set_key(dotenv_path, "DISABLE_MULTIPROCESSING", disable_multiprocessing)
            messagebox.showinfo("Success", "Configuration saved to .env. Please restart the application for the changes to fully take effect.")
            self.app.update_status("Configuration saved to .env")
        else:
            messagebox.showerror("Error", ".env file not found.")
            self.app.update_status("Error: .env file not found.")