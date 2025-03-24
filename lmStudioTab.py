# DarthVector/lmStudioTab.py
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import requests
import json
import logging
import faiss
import pickle
from config import config
from typing import Any, List, Dict, Optional
import pyperclip
import os
from dotenv import set_key, find_dotenv
from sentence_transformers import SentenceTransformer
import gc

class LMStudioTab(ttk.Frame):
    """Tab for integrating with LM Studio."""

    def __init__(self, parent: ttk.Notebook, app: Any):
        """Initialise the LM Studio Tab."""
        super().__init__(parent)
        self.parent = parent
        self.app = app
        self.logger = logging.getLogger(__name__)
        self.defaultIndex = None
        self.vectorMetadata = []
        self.sentenceTransformerModel: Optional[SentenceTransformer] = None
        self.create_widgets()

    def create_widgets(self) -> None:
        """Create and layout widgets for the LM Studio tab."""
        # --- LM Studio API Configuration ---
        apiFrame = ttk.LabelFrame(self, text="LM Studio API Configuration")
        apiFrame.pack(padx=10, pady=10, fill="x")

        ttk.Label(apiFrame, text="API URL:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.apiUrlEntry = ttk.Entry(apiFrame, width=50)
        self.apiUrlEntry.insert(0, config.lmStudioApiUrl)
        self.apiUrlEntry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(apiFrame, text="Model Name:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.modelNameEntry = ttk.Entry(apiFrame, width=50)
        self.modelNameEntry.insert(0, config.lmStudioModelName)
        self.modelNameEntry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(apiFrame, text="Max Tokens:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.maxTokensEntry = ttk.Entry(apiFrame, width=10)
        self.maxTokensEntry.insert(0, str(config.lmStudioMaxTokens))
        self.maxTokensEntry.grid(row=2, column=1, padx=5, pady=5, sticky="w")

        self.saveConfigButton = ttk.Button(apiFrame, text="Save Configuration to .env", command=self.saveLmStudioConfig)
        self.saveConfigButton.grid(row=3, column=0, columnspan=2, pady=10)

        # --- Load Vector Store Section ---
        loadFrame = ttk.LabelFrame(self, text="Vector Store")
        loadFrame.pack(padx=10, pady=10, fill="x")

        ttk.Label(loadFrame, text="Index Path:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.defaultIndexEntry = ttk.Entry(loadFrame, width=50)
        self.defaultIndexEntry.insert(0, str(config.defaultIndexPath))
        self.defaultIndexEntry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.browsedefaultIndexButton = ttk.Button(loadFrame, text="Browse", command=self.browsedefaultIndexFile)
        self.browsedefaultIndexButton.grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(loadFrame, text="Metadata Path:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.vectorMetadataEntry = ttk.Entry(loadFrame, width=50)
        self.vectorMetadataEntry.insert(0, str(config.defaultMetadataPath))
        self.vectorMetadataEntry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.browseVectorMetadataButton = ttk.Button(loadFrame, text="Browse", command=self.browseVectorMetadataFile)
        self.browseVectorMetadataButton.grid(row=1, column=2, padx=5, pady=5)

        self.loadButton = ttk.Button(loadFrame, text="Reload Vector Store", command=self.loadVectorStore)
        self.loadButton.grid(row=2, column=0, columnspan=3, pady=5)

        # --- Query Section ---
        queryFrame = ttk.LabelFrame(self, text="Query Vector Store with LM Studio")
        queryFrame.pack(padx=10, pady=10, fill="both", expand=True)

        ttk.Label(queryFrame, text="Your Query:").pack(padx=5, pady=5, anchor="w")
        self.queryEntry = ttk.Entry(queryFrame, width=60)
        self.queryEntry.pack(padx=5, pady=5, fill="x")

        ttk.Label(queryFrame, text=f"Number of Search Results (Top {config.searchResultCount}):").pack(padx=5, pady=5, anchor="w")
        self.searchResultsText = scrolledtext.ScrolledText(queryFrame, height=5, wrap=tk.WORD)
        self.searchResultsText.pack(padx=5, pady=5, fill="both", expand=True)
        self.searchResultsText.config(state=tk.DISABLED)

        self.queryButton = ttk.Button(queryFrame, text="Query", command=self.executeQuery)
        self.queryButton.pack(pady=10)

        # --- Response Section ---
        responseFrame = ttk.LabelFrame(self, text="LM Studio Response")
        responseFrame.pack(padx=10, pady=10, fill="both", expand=True)

        self.responseArea = scrolledtext.ScrolledText(responseFrame, height=15, wrap=tk.WORD)
        self.responseArea.pack(padx=5, pady=5, fill="both", expand=True)
        self.responseArea.config(state=tk.DISABLED)

        # Add a copy button for the response
        self.copyResponseButton = ttk.Button(responseFrame, text="Copy Response", command=self.copyResponse)
        self.copyResponseButton.pack(pady=5)

    def copyResponse(self) -> None:
        """Copy the response text to clipboard."""
        response_text = self.responseArea.get(1.0, tk.END).strip()
        if response_text:
            pyperclip.copy(response_text)
            self.app.update_status("Response copied to clipboard")
        else:
            self.app.update_status("No response to copy")

    def browsedefaultIndexFile(self) -> None:
        """Open file selection dialogue for the FAISS index file."""
        filePath = filedialog.askopenfilename(defaultextension=".faiss", filetypes=[("FAISS Index Files", "*.faiss"), ("All Files", "*.*")])
        if filePath:
            self.defaultIndexEntry.delete(0, tk.END)
            self.defaultIndexEntry.insert(0, filePath)

    def browseVectorMetadataFile(self) -> None:
        """Open file selection dialogue for the metadata file."""
        filePath = filedialog.askopenfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")])
        if filePath:
            self.vectorMetadataEntry.delete(0, tk.END)
            self.vectorMetadataEntry.insert(0, filePath)

    def loadVectorStore(self) -> None:
        """Load the FAISS vector store and metadata, handling dimension mismatches."""
        indexFile = self.defaultIndexEntry.get()
        metadataFile = self.vectorMetadataEntry.get()
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(os.path.abspath(indexFile)), exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(metadataFile)), exist_ok=True)
        
        try:
            # Load sentence transformer model first to get dimensions
            if not hasattr(self.app, 'sentenceTransformerModel') or self.app.sentenceTransformerModel is None:
                try:
                    self.sentenceTransformerModel = SentenceTransformer(config.defaultModelName)
                    self.app.sentenceTransformerModel = self.sentenceTransformerModel
                    model_dimension = self.sentenceTransformerModel.get_sentence_embedding_dimension()
                    self.app.update_status(f"Model loaded with dimension: {model_dimension}")
                except Exception as e:
                    messagebox.showerror("Error", f"Error loading SentenceTransformer model: {e}")
                    self.logger.error(f"Error loading SentenceTransformer model: {e}", exc_info=True)
                    return
            
            model_dimension = self.sentenceTransformerModel.get_sentence_embedding_dimension()
            
            if not os.path.exists(indexFile):
                # Create an empty FAISS index with the correct dimension
                empty_index = faiss.IndexFlatL2(model_dimension)
                faiss.write_index(empty_index, str(indexFile))
                self.logger.info(f"Created new FAISS index with dimension {model_dimension}")
                messagebox.showinfo("Info", f"Created new FAISS index with dimension {model_dimension}")
            else:
                # Check existing index dimension
                try:
                    temp_index = faiss.read_index(str(indexFile))
                    index_dimension = temp_index.d
                    if index_dimension != model_dimension:
                        if messagebox.askyesno("Dimension Mismatch", 
                            f"The existing index has dimension {index_dimension}, but the model expects {model_dimension}. "
                            "Recreate the index with the correct dimension?"):
                            empty_index = faiss.IndexFlatL2(model_dimension)
                            faiss.write_index(empty_index, str(indexFile))
                            self.logger.info(f"Recreated FAISS index with dimension {model_dimension}")
                            messagebox.showinfo("Info", f"Recreated FAISS index with dimension {model_dimension}")
                        else:
                            messagebox.showerror("Error", "Cannot proceed with dimension mismatch.")
                            return
                except Exception as e:
                    messagebox.showerror("Error", f"Error checking index dimension: {e}")
                    return
                
            # Load the vector index
            self.defaultIndex = faiss.read_index(str(indexFile))
                
            # Load or create the metadata file
            if not os.path.exists(metadataFile):
                with open(metadataFile, "wb") as f:
                    pickle.dump([], f)
                self.logger.info(f"Created empty metadata file at {metadataFile}")
                
            with open(metadataFile, "rb") as file:
                self.vectorMetadata = pickle.load(file)
                
            messagebox.showinfo("Success", "Vector store and metadata loaded successfully.")
            self.app.update_status("Vector store and metadata loaded successfully.")
            
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"File not found: {e}")
            self.app.update_status(f"Error: {e}")
            self.defaultIndex = None
            self.vectorMetadata = []
            gc.collect()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading vector store: {e}")
            self.app.update_status(f"Error loading vector store: {e}")
            self.logger.error(f"Error loading vector store: {e}", exc_info=True)
            if self.defaultIndex is not None:
                del self.defaultIndex
            self.defaultIndex = None
            self.vectorMetadata = []
            gc.collect()
 
    def executeQuery(self) -> None:
        """Execute the query against LM Studio with context from the vector store."""
        query = self.queryEntry.get()
        if not query:
            messagebox.showerror("Error", "Please enter a query.")
            return
            
        # Check if vector store is loaded
        if self.defaultIndex is None or not self.vectorMetadata:
            messagebox.showerror("Error", "Vector store not loaded. Please load the vector store first.")
            return
            
        # Check if sentence transformer model is available
        if not hasattr(self.app, 'sentenceTransformerModel') or self.app.sentenceTransformerModel is None:
            if not hasattr(self, 'sentenceTransformerModel') or self.sentenceTransformerModel is None:
                messagebox.showerror("Error", "SentenceTransformer model not loaded.")
                return
            self.app.sentenceTransformerModel = self.sentenceTransformerModel

        self.app.update_status("Searching vector store...")
        searchResults = self.performVectorSearch(query, top_n=config.searchResultCount)
        self.displaySearchResults(searchResults)
        
        if not searchResults:
            messagebox.showinfo("Information", "No relevant search results found.")
            return
            
        context = "\n\n".join([result['metadata']['chunkText'] for result in searchResults])

        prompt = f"""Based on the following context:
                {context}
                
                Answer the query: {query}"""

        self.app.update_status("Querying LM Studio...")
        try:
            max_tokens = int(self.maxTokensEntry.get())
            if max_tokens <= 0:
                raise ValueError("Max Tokens must be positive")
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid Max Tokens value: {e}")
            self.app.update_status("Error: Invalid Max Tokens value")
            return
        self.queryLMStudio(prompt, max_tokens)

    def performVectorSearch(self, query: str, top_n: int = 5) -> List[Dict]:
        """Perform a vector similarity search."""
        if self.defaultIndex is None:
            messagebox.showerror("Error", "Vector index not loaded. Please load the vector store first.")
            return []
        if not self.vectorMetadata:
            messagebox.showerror("Error", "Vector metadata is empty.")
            return []

        try:
            # Use either the app's model or this tab's model
            model = getattr(self.app, 'sentenceTransformerModel', None) or self.sentenceTransformerModel
            if model is None:
                raise ValueError("SentenceTransformer model not available")
                
            queryEmbedding = model.encode([query])[0]
            distances, indices = self.defaultIndex.search(queryEmbedding.reshape(1, -1), top_n)

            results = []
            for i in range(len(indices[0])):
                index = indices[0][i]
                if index != -1 and index < len(self.vectorMetadata):
                    results.append({
                        'metadata': self.vectorMetadata[index],
                        'distance': float(distances[0][i])
                    })
            return results
        except Exception as e:
            self.logger.error(f"Error performing vector search: {e}", exc_info=True)
            messagebox.showerror("Error", f"Error performing vector search: {e}")
            return []

    def displaySearchResults(self, searchResults: List[Dict]) -> None:
        """Display the search results in the text area."""
        self.searchResultsText.config(state=tk.NORMAL)
        self.searchResultsText.delete(1.0, tk.END)
        if searchResults:
            for result in searchResults:
                metadata = result['metadata']
                text = metadata.get('chunkText', 'No text available')
                filename = metadata.get('file', 'Unknown file')
                chunkIndex = metadata.get('chunkIndex', 'Unknown index')
                distance = result.get('distance', 0)
                self.searchResultsText.insert(tk.END, f"File: {filename}, Chunk: {chunkIndex}, Distance: {distance:.4f}\n{text}\n---\n")
        else:
            self.searchResultsText.insert(tk.END, "No relevant search results found.\n")
        self.searchResultsText.config(state=tk.DISABLED)

    def queryLMStudio(self, prompt: str, max_tokens: int) -> None:
        """Send the prompt to the LM Studio API and display the response."""
        apiUrl = self.apiUrlEntry.get()
        # Normalize apiUrl to ensure correct base URL
        base_url = apiUrl.rstrip('/').split('/v1')[0]
        endpoint = f"{base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.modelNameEntry.get(),
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "stream": False
        }

        try:
            response = requests.post(endpoint, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            responseJson = response.json()
            if 'choices' in responseJson and responseJson['choices']:
                assistantResponse = responseJson['choices'][0]['message']['content']
                self.displayResponse(assistantResponse)
            else:
                self.displayResponse("No response received from LM Studio or unexpected format.")
                self.logger.warning(f"Unexpected response format from LM Studio: {responseJson}")
        except requests.exceptions.ConnectionError as e:
            self.displayResponse(f"Error: Could not connect to LM Studio at {endpoint}. Please ensure LM Studio is running and the API URL is correct.")
            self.logger.error(f"Connection error to LM Studio at {endpoint}: {e}")
        except requests.exceptions.RequestException as e:
            self.displayResponse(f"Error querying LM Studio: {e}")
            self.logger.error(f"Error querying LM Studio: {e}")
        except json.JSONDecodeError:
            self.displayResponse("Error: Could not decode JSON response from LM Studio.")
            self.logger.error("Could not decode JSON response from LM Studio.")

    def displayResponse(self, response: str) -> None:
        """Display the LM Studio response in the response area."""
        self.responseArea.config(state=tk.NORMAL)
        self.responseArea.delete(1.0, tk.END)
        self.responseArea.insert(tk.END, response)
        self.responseArea.config(state=tk.DISABLED)
        self.app.update_status("Ready")

    def saveLmStudioConfig(self) -> None:
        """Save the LM Studio API URL, Model Name, and Max Tokens to the .env file."""
        api_url = self.apiUrlEntry.get()
        model_name = self.modelNameEntry.get()
        max_tokens = self.maxTokensEntry.get()
        dotenv_path = find_dotenv()
        if dotenv_path:
            set_key(dotenv_path, "LMSTUDIO_API_URL", api_url)
            set_key(dotenv_path, "LMSTUDIO_MODEL_NAME", model_name)
            set_key(dotenv_path, "LMSTUDIO_MAX_TOKENS", max_tokens)
            messagebox.showinfo("Success", "LM Studio configuration saved to .env. Please restart the application for the changes to fully take effect.")
            self.app.update_status("LM Studio configuration saved to .env")
        else:
            messagebox.showerror("Error", ".env file not found.")
            self.app.update_status("Error: .env file not found.")