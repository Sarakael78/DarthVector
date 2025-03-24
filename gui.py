import tkinter as tk
from tkinter import ttk
from pathlib import Path
import logging
import sys
import faiss
import pickle
import atexit
from sentence_transformers import SentenceTransformer
from config import config
from fileProcessingTab import FileProcessingTab
from updateVectorStoreTab import UpdateVectorStoreTab
from lmStudioTab import LMStudioTab
from embeddingGenerator import SentenceTransformerEmbedder
import gc

class RtfProcessingApp:
    """A modern GUI for processing RTF files into embeddings."""

    def __init__(self, master: tk.Tk):
        """Initialise the main application window."""
        self.master = master
        self.master.title("RTF Processing Pipeline")
        self.totalFiles = 0
        
        # Don't load model at initialization - load on demand instead
        self.sentenceTransformerModel = None

        # Ensure vector and metadata files exist
        self.defaultIndexPath = Path(config.defaultIndexPath)
        self.defaultMetadataPath = Path(config.defaultMetadataPath)
        self.ensure_files_exist()

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, config.loggingLevel),
            format=config.loggingFormat,
            handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("gui.log")]
        )
        self.logger = logging.getLogger(__name__)
        self.loaded_modules = {}

        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.defaultIndex = None
        self.vectorMetadata = []
        self.create_widgets()

        # Status Bar
        self.statusBarText = tk.StringVar()
        self.statusBar = ttk.Label(master, textvariable=self.statusBarText, relief=tk.SUNKEN, anchor=tk.W)
        self.statusBar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_status("Ready")

        # Set initial full screen
        self.master.attributes('-fullscreen', True)

        # Add menu bar for full-screen toggle
        self.menu = tk.Menu(self.master)
        self.master.config(menu=self.menu)
        view_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Full Screen", command=self.toggle_fullscreen)

        # Bind F11 to toggle full screen
        self.master.bind("<F11>", lambda event: self.toggle_fullscreen())
        
        # Register cleanup handler when app closes
        atexit.register(self.cleanup_resources)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Do NOT automatically load the vector store at startup to avoid popup
        # self.lmStudioTab.loadVectorStore()
        
        print("Application started successfully")
        self.update_status("Application ready - vector stores will be loaded on demand")

    def get_model(self):
        """Get or load the sentence transformer model on demand"""
        if self.sentenceTransformerModel is None:
            self.update_status("Loading sentence transformer model...")
            print(f"Loading model: {config.defaultModelName}")
            self.sentenceTransformerModel = SentenceTransformer(config.defaultModelName)
            self.track_module_loaded(config.defaultModelName)
            self.update_status("Model loaded")
            print(f"Model loaded: {config.defaultModelName}")
        return self.sentenceTransformerModel

    def track_module_loaded(self, module_name):
        """Track when a module is loaded into memory"""
        if module_name in self.loaded_modules:
            self.loaded_modules[module_name] += 1
        else:
            self.loaded_modules[module_name] = 1
        
        module_count = len(self.loaded_modules)
        total_instances = sum(self.loaded_modules.values())
        print(f"Module tracking: {module_name} loaded. Total unique modules: {module_count}, Total instances: {total_instances}")
        print(f"Currently loaded modules: {self.loaded_modules}")

    def toggle_fullscreen(self):
        """Toggle between full-screen and windowed mode."""
        current = self.master.attributes('-fullscreen')
        self.master.attributes('-fullscreen', not current)

    def ensure_files_exist(self) -> None:
        """Ensure vector and metadata files are created if missing."""
        self.defaultIndexPath.parent.mkdir(parents=True, exist_ok=True)
        self.defaultMetadataPath.parent.mkdir(parents=True, exist_ok=True)

        # Index file will be created dynamically with correct dimensions when needed
        # Don't create it here with hardcoded dimensions
        if not self.defaultMetadataPath.exists():
            with open(self.defaultMetadataPath, 'wb') as f:
                pickle.dump([], f)
            logging.info(f"Created missing vector metadata file at {self.defaultMetadataPath}")

    def create_widgets(self) -> None:
        """Create and layout GUI widgets."""
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill='both')

    # Combined File Processing Tab
        self.fileProcessingTab = FileProcessingTab(self.notebook, self) 
        self.notebook.add(self.fileProcessingTab, text='File Processing')

        # LM Studio Integration Tab
        self.lmStudioTab = LMStudioTab(self.notebook, self)
        self.notebook.add(self.lmStudioTab, text='LM Studio Integration')

    def update_status(self, message: str) -> None:
        """Update the text in the status bar."""
        self.statusBarText.set(message)
        
    def cleanup_resources(self):
        """Clean up resources when app closes"""
        self.update_status("Cleaning up resources...")
        print("Cleaning up resources...")
        
        try:
            # Release sentence transformer model
            if self.sentenceTransformerModel is not None:
                print("Releasing SentenceTransformer model from memory")
                del self.sentenceTransformerModel
                self.sentenceTransformerModel = None
                logging.info("SentenceTransformer model released from memory")
            
            # Clear all cached models from SentenceTransformerEmbedder
            print("Releasing all SentenceTransformerEmbedder models")
            SentenceTransformerEmbedder.release_models()
            
            # Ensure any remaining tensors are moved to CPU
            print("Moving any remaining tensors to CPU")
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                for obj in gc.get_objects():
                    if isinstance(obj, torch.Tensor) and obj.is_cuda:
                        print(f"Moving tensor {obj} to CPU")
                        obj.cpu()
            
            # Force garbage collection
            print("Running garbage collection")
            gc.collect()
            
            self.update_status("Resources cleaned up")
            print("Resources cleaned up")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}", exc_info=True)
            print(f"Error during cleanup: {e}")
        
    def on_closing(self):
        """Handle window closing event"""
        try:
            self.cleanup_resources()
        finally:
            self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = RtfProcessingApp(root)
    root.mainloop()