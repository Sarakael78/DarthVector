# DarthVector

DarthVector is a powerful document processing and vector embedding application designed for semantic search and retrieval-augmented generation (RAG). It provides an intuitive GUI for processing documents (PDF, RTF, DOCX) into vector embeddings, maintaining a FAISS vector store, and integrating with LM Studio for AI-powered document queries.

![DarthVector](https://img.shields.io/badge/DarthVector-Document%20Processing%20Pipeline-blue)

## Features

- **Multiple File Format Support**: Process PDF, RTF, and DOCX documents
- **Advanced Text Processing**: Intelligent chunking with configurable size and overlap
- **High-Performance Embeddings**: GPU-accelerated embedding generation using SentenceTransformer models
- **Efficient Vector Store**: FAISS-based vector storage for lightning-fast similarity search
- **LM Studio Integration**: Seamlessly connect to LM Studio for RAG capabilities
- **User-Friendly GUI**: Modern tkinter interface with multiple specialized tabs
- **Multi-Processing**: Parallel document processing to maximize throughput
- **Comprehensive Configuration**: Easily configurable via GUI or .env file
- **Robust Error Handling**: Graceful handling of processing errors with detailed logging

## Installation

### Prerequisites

- Python 3.9+ 
- CUDA-compatible GPU (optional, for acceleration)
- CUDA Toolkit (if using GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Sarakael78/DarthVector.git
cd DarthVector
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file for configuration (or use the GUI to save one):
```
MODEL_NAME=all-mpnet-base-v2
MAX_WORKERS=4
CHUNK_SIZE=500
CHUNK_OVERLAP=50
DEFAULT_OUTPUT_DIR=output
DEFAULT_INDEX_NAME=vector_index.faiss
DEFAULT_METADATA_NAME=metadata.pkl
LMSTUDIO_API_URL=http://localhost:1234/v1
LMSTUDIO_MODEL_NAME=default
LMSTUDIO_MAX_TOKENS=2000
```

## Usage

### GUI Application

Launch the application with:

```bash
python gui.py
```

The GUI consists of multiple tabs:

1. **File Processing Tab**: Process documents and create/update vector stores
2. **LM Studio Integration Tab**: Connect to LM Studio and query your documents

### Command Line Interface

For batch processing or automation, use the command line interface:

```bash
python main.py --input_dir /path/to/documents --output_index output/vector_index.faiss --output_metadata output/metadata.pkl
```

Optional arguments:
- `--model_name`: SentenceTransformer model to use (default: all-mpnet-base-v2)
- `--max_workers`: Number of worker processes (default: 4)
- `--chunk_size`: Maximum words per chunk (default: 500)
- `--chunk_overlap`: Word overlap between chunks (default: 50)
- `--disable_multiprocessing`: Use single-threaded processing

## Configuration

DarthVector can be configured via:

1. The `.env` file (see Installation section)
2. Command line arguments (for CLI usage)
3. The GUI's configuration options, which can save to the `.env` file

Key configuration options:

| Option | Description | Default |
|--------|-------------|---------|
| MODEL_NAME | SentenceTransformer model name | all-mpnet-base-v2 |
| MAX_WORKERS | Number of parallel workers | 4 |
| CHUNK_SIZE | Maximum words per text chunk | 500 |
| CHUNK_OVERLAP | Word overlap between chunks | 50 |
| MAX_FILE_SIZE_MB | Maximum document file size | 100 |
| LMSTUDIO_API_URL | LM Studio API endpoint | http://localhost:1234/v1 |
| LMSTUDIO_MAX_TOKENS | Maximum tokens for LM Studio response | 2000 |

## Architecture

DarthVector consists of several key components:

- **Document Processing Pipeline**: Extracts text from documents and generates vector embeddings
- **Text Preprocessing**: Cleans, normalizes, and chunks text for optimal embeddings
- **Embedding Generation**: Creates vector representations using SentenceTransformer models
- **Vector Store Management**: Maintains and queries FAISS vector stores
- **LM Studio Integration**: Connects to LM Studio for AI-powered document queries
- **GUI Interface**: Provides user-friendly access to all functionality

The application uses a modular design with clear separation of concerns, allowing for easy extension and customization.

## Dependencies

Major dependencies include:

- **faiss-gpu**: High-performance similarity search library
- **sentence-transformers**: Neural text embedding models
- **pdfplumber**: PDF text extraction
- **python-docx**: DOCX file processing
- **striprtf**: RTF file processing
- **tkinter**: GUI framework
- **pydantic**: Data validation and configuration management
- **python-dotenv**: Environment variable management

## Troubleshooting

### CUDA Tensor Conversion Issues

If you encounter CUDA tensor errors during embedding generation, ensure that:

1. Your GPU drivers are up to date
2. You have CUDA toolkit installed that matches your PyTorch version
3. You have sufficient GPU memory for the model

### Memory Issues

For processing large document collections:

1. Reduce batch size in `embeddingGenerator.py`
2. Decrease the number of workers with `--max_workers`
3. Process documents in smaller batches

## License

This project is licensed under the MIT License - see the LICENSE file for details.
