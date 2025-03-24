# DarthVector/config.py
from pydantic import BaseModel, Field, ConfigDict, model_validator
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Load environment variables from .env file
load_dotenv()

class Configuration(BaseModel):
    """Configuration settings for the application."""
    maxFileSizeMB: int = Field(default=100, ge=1, le=1024, description="Maximum file size in MB (1-1024)")
    model_config = ConfigDict(arbitrary_types_allowed=True)
    defaultOutputDir: Path = Path(os.getenv("DEFAULT_OUTPUT_DIR", "output"))
    defaultIndexName: str = os.getenv("DEFAULT_INDEX_NAME", "vector_index.faiss")
    defaultMetadataName: str = os.getenv("DEFAULT_METADATA_NAME", "metadata.pkl")
    defaultIndexPath: Path = defaultOutputDir / defaultIndexName
    defaultMetadataPath: Path = defaultOutputDir / defaultMetadataName
    defaultModelName: str = os.getenv("MODEL_NAME", "all-mpnet-base-v2")
    defaultMaxWorkers: int = int(os.getenv("MAX_WORKERS", "4"))
    defaultChunkSize: int = int(os.getenv("CHUNK_SIZE", "500"))
    defaultChunkOverlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    searchResultCount: int = int(os.getenv("SEARCH_RESULT_COUNT", "5"))
    loggingLevel: str = os.getenv("LOGGING_LEVEL", "INFO").upper()
    loggingFormat: str = os.getenv("LOGGING_FORMAT", "%(asctime)s - %(levelname)s - %(message)s")
    lmStudioApiUrl: str = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
    lmStudioModelName: str = os.getenv("LMSTUDIO_MODEL_NAME", "default")
    defaultInputDir: Path = Path(os.getenv("DEFAULT_INPUT_DIR", ""))
    disableMultiprocessing: bool = Field(default=False, description="Disable multiprocessing", env="DISABLE_MULTIPROCESSING")
    defaultIngestedFilesPath: Path = defaultOutputDir / "ingested_files.json"
    supportedExtensions: List[str] = Field(default=['.rtf', '.pdf', '.docx'], description="Supported file extensions")
    lmStudioMaxTokens: int = Field(default=2000, description="Maximum tokens for LM Studio response", env="LMSTUDIO_MAX_TOKENS")
    rtfEncodings: List[str] = Field(default=['utf-8', 'latin-1', 'cp1252', 'utf-16', 'utf-16-le'], description="Supported RTF encodings", env="RTF_ENCODINGS")

    @model_validator(mode='after')
    def validate_fields(self) -> 'Configuration':
        """Validate configuration fields."""
        # Normalize logging level
        self.loggingLevel = self.loggingLevel.strip().upper()
        valid_logging_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.loggingLevel not in valid_logging_levels:
            raise ValueError(f"Invalid logging level: {self.loggingLevel}. Valid options: {valid_logging_levels}")
        if self.defaultMaxWorkers <= 0:
            raise ValueError("MAX_WORKERS must be positive")
        if self.defaultChunkSize <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.defaultChunkOverlap < 0 or self.defaultChunkOverlap >= self.defaultChunkSize:
            raise ValueError("CHUNK_OVERLAP must be non-negative and less than CHUNK_SIZE")
        if self.maxFileSizeMB <= 0:
            raise ValueError("maxFileSizeMB must be positive")
        if self.searchResultCount <= 0:
            raise ValueError("SEARCH_RESULT_COUNT must be positive")
        if self.lmStudioMaxTokens <= 0:
            raise ValueError("LMSTUDIO_MAX_TOKENS must be positive")
        if not self.rtfEncodings:
            raise ValueError("RTF_ENCODINGS must not be empty")
        if self.maxFileSizeMB > 1024:
            raise ValueError("maxFileSizeMB must be <= 1024")
        return self
    @classmethod
    def ensureOutputDir(cls) -> None:
        """Ensure the default output directory exists."""
        instance = cls()
        instance.defaultOutputDir.mkdir(exist_ok=True, parents=True)

# Instantiate the configuration
config = Configuration()