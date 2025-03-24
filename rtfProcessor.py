# ---------------------------
# File: rtfProcessor.py
# ---------------------------
import logging
from pathlib import Path
from striprtf.striprtf import rtf_to_text
from typing import Union
from exceptions import FileProcessingError, TextExtractionError, FileSizeExceededError, UnsupportedFileTypeError, RTFEncodingError
from config import config

class RtfProcessor:
    """Processes RTF files to extract plain text."""

    def __init__(self):
        """Initialise the RtfProcessor."""
        self.logger = logging.getLogger(__name__)
        self.encodings = getattr(config, 'rtfEncodings', ['utf-8', 'latin-1', 'cp1252', 'utf-16', 'utf-16-le'])

    def extractTextFromRtf(self, filePath: Union[str, Path]) -> str:
        """
        Extract plain text from an RTF file.

        Args:
            filePath: Path to the RTF document (str or Path).

        Returns:
            str: Extracted plain text.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            FileSizeExceededError: If file size exceeds the limit.
            RTFEncodingError: If encoding cannot be determined.
            FileProcessingError: For other processing errors.
        """
        try:
            path = Path(filePath).resolve()  # Prevent path traversal
            self.logger.debug(f"Processing file: {path}")
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            fileSizeMb = path.stat().st_size / (1024 * 1024)
            self.logger.debug(f"File size: {fileSizeMb:.2f}MB for {path}")
            if fileSizeMb > config.maxFileSizeMB:
                raise FileSizeExceededError(
                f"File size ({fileSizeMb:.2f}MB) exceeds limit ({config.maxFileSizeMB}MB)"
                )

            self.logger.debug(f"Reading RTF file: {path}")
            rtfContent = None
            decode_errors = []
        
            for encoding in self.encodings:
                try:
                    rtfContent = path.read_text(encoding=encoding)
                    self.logger.debug(f"Successfully decoded using {encoding} encoding")
                    break
                except UnicodeDecodeError as e:
                    decode_errors.append(f"{encoding}: {str(e)}")
                    self.logger.debug(f"Failed to decode RTF with {encoding}: {e}")
                    continue
                    
            if rtfContent is None:
                    error_msg = f"Failed to decode RTF file {path} with encodings: {', '.join(self.encodings)}"
                    self.logger.error(error_msg)
                    self.logger.debug(f"Decode errors: {decode_errors}")
                    raise RTFEncodingError(error_msg)

            plainText = rtf_to_text(rtfContent)
            if not plainText.strip():
                self.logger.warning(f"RTF content from {path} resulted in empty text.")
                self.logger.info(f"Extracted {len(plainText)} characters from {path}")
                return plainText

        except FileNotFoundError as e:
            self.logger.error(f"Error processing {filePath}: {e}")
            raise
        except FileSizeExceededError as e:
            self.logger.error(f"Error processing {filePath}: {e}")
            raise
        except RTFEncodingError as e:
            self.logger.error(str(e))
            raise
        except Exception as e:
            self.logger.error(f"Failed to process {path}: {e}")
            raise FileProcessingError(f"Error processing {path}: {str(e)}")