# textExtractor.py
import logging
from pathlib import Path
import pdfplumber
from docx import Document
from striprtf.striprtf import rtf_to_text
from config import config
from exceptions import FileProcessingError, TextExtractionError, FileSizeExceededError, UnsupportedFileTypeError, RTFEncodingError


def extractTextFromFile(filePath: Path) -> str:
    """
    Extract text from a file based on its extension.

    Args:
        filePath: Path to the file.

    Returns:
        str: Extracted text.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        FileSizeExceededError: If the file exceeds the size limit.
        UnsupportedFileTypeError: If the file type is not supported.
        TextExtractionError: If text extraction fails.
    """
    path = filePath.resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    fileSizeMb = path.stat().st_size / (1024 * 1024)
    if fileSizeMb > config.maxFileSizeMB:
        raise FileSizeExceededError(f"File size ({fileSizeMb:.2f}MB) exceeds limit ({config.maxFileSizeMB}MB)")

    ext = path.suffix.lower()
    if ext == '.rtf':
        return extractTextFromRtf(path)
    elif ext == '.pdf':
        return extractTextFromPdf(path)
    elif ext == '.docx':
        return extractTextFromDocx(path)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {ext}")

def extractTextFromRtf(filePath: Path) -> str:
    """
    Extract text from an RTF file.
    
    Args:
        filePath: Path to the RTF file
        
    Returns:
        str: Extracted text
        
    Raises:
        TextExtractionError: If text extraction fails
    """
    try:
        # Try multiple encodings
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "windows-1252"]
        rtfContent = None
        decode_errors = []
        
        for encoding in encodings:
            try:
                rtfContent = filePath.read_text(encoding=encoding)
                logging.debug(f"Successfully decoded {filePath} using {encoding} encoding")
                break
            except UnicodeDecodeError as e:
                decode_errors.append(f"{encoding}: {str(e)}")
                continue
                
        if rtfContent is None:
            error_msg = f"Failed to decode RTF file {filePath} with encodings: {', '.join(encodings)}"
            logging.error(error_msg)
            logging.debug(f"Decode errors: {decode_errors}")
            raise TextExtractionError(error_msg)
        
        text = rtf_to_text(rtfContent)
        logging.info(f"Extracted {len(text)} characters from {filePath}")
        return text
    except UnicodeDecodeError:
        logging.error(f"Encoding error in RTF file {filePath}")
        raise RTFEncodingError(f"Unable to decode RTF file {filePath} with UTF-8")
    except Exception as e:
        logging.error(f"Failed to extract text from RTF {filePath}: {e}")
        raise TextExtractionError(f"Error extracting text from RTF {filePath}: {e}")

def extractTextFromPdf(filePath: Path) -> str:
    """
    Extract text from a PDF file using pdfplumber with enhanced error handling.
    
    Args:
        filePath: Path to the PDF file
        
    Returns:
        str: Extracted text
        
    Raises:
        TextExtractionError: If text extraction fails
    """
    try:
        with pdfplumber.open(filePath) as pdf:
            text_parts = []
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                    else:
                        logging.warning(f"No text extracted from page {page.page_number} in {filePath}. Possibly an image-only page.")
                except Exception as e:
                    logging.error(f"Error extracting text from page {page.page_number} in {filePath}: {e}")
                    text_parts.append(f"[Error extracting page {page.page_number}]")
            
            if not text_parts:
                logging.warning(f"{filePath} appears to be an image-based PDF with no extractable text.")
                return f"[Image-based PDF: No text could be extracted from {filePath.name}]"
                
            text = "\n".join(text_parts)
            logging.info(f"Extracted {len(text)} characters from {filePath}")
            return text
    except pdfplumber.PDFSyntaxError as e:
        logging.error(f"PDF syntax error in {filePath}: {e}")
        raise TextExtractionError(f"PDF syntax error in {filePath}: {e}")
    except Exception as e:
        logging.error(f"Failed to extract text from PDF {filePath}: {e}")
        raise TextExtractionError(f"Error extracting text from PDF {filePath}: {e}")

def extractTextFromDocx(filePath: Path) -> str:
    """Extract text from a DOCX file using python-docx."""
    try:
        doc = Document(filePath)
        text = "\n".join(para.text for para in doc.paragraphs)
        logging.info(f"Extracted {len(text)} characters from {filePath}")
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from DOCX {filePath}: {e}")
        raise TextExtractionError(f"Error extracting text from DOCX {filePath}: {e}")