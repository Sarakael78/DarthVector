"""
Common exceptions for the RTF Processing Pipeline.
Centralizes exception definitions to avoid duplication.
"""

class FileProcessingError(Exception):
    """Base exception for file processing errors."""
    pass

class TextExtractionError(FileProcessingError):
    """Exception for text extraction errors."""
    pass

class FileSizeExceededError(FileProcessingError):
    """Exception when file size exceeds the limit."""
    pass

class UnsupportedFileTypeError(FileProcessingError):
    """Exception for unsupported file types."""
    pass

class RTFEncodingError(FileProcessingError):
    """Exception raised for RTF encoding issues."""
    pass