# DarthVector/textPreprocessor.py
# ---------------------------------
# File: textPreprocessor.py
# ---------------------------------
import re
import logging
from typing import List

class TextPreprocessor:
    """
    Class for preprocessing text by cleaning, tokenizing, chunking, and splitting into sentences.
    """

    def __init__(self, chunkSize: int = 500, chunkOverlap: int = 50) -> None:
        """
        Initialise the TextPreprocessor with chunking parameters.

        Args:
            chunkSize: Maximum number of words per text chunk (default: 500).
            chunkOverlap: Number of overlapping words between consecutive chunks (default: 50).
            
        Raises:
            ValueError: If chunk parameters are invalid.
        """
        self.logger = logging.getLogger(__name__)
        
        if chunkSize <= 0:
            raise ValueError("chunkSize must be positive")
        if chunkOverlap < 0:
            raise ValueError("chunkOverlap must be non-negative")
        if chunkOverlap >= chunkSize:
            raise ValueError("chunkOverlap must be less than chunkSize")
            
        self.chunkSize = chunkSize
        self.chunkOverlap = chunkOverlap

    def cleanText(self, text: str, preservePunctuation: bool = False) -> str:
        """
        Clean the input text by removing extra whitespace and normalizing.

        Args:
            text: The input string to be cleaned.
            preservePunctuation: If True, preserve punctuation marks (default: False).

        Returns:
            A cleaned version of the input string.
        """
        try:
            text = re.sub(r'\s+', ' ', text.strip())
            if not preservePunctuation:
                text = re.sub(r'[^\w\s\-.,;:!?\'"\(\)]', '', text)
            return text
        except Exception as e:
            self.logger.error(f"Error cleaning text: {e}")
            raise

    def tokenizeText(self, text: str) -> List[str]:
        """
        Tokenize the text into words.

        Args:
            text: The input string to tokenize.

        Returns:
            A list of word tokens.
        """
        try:
            tokens = re.findall(r"\b[\w']+(?:-[\w']+)*\b", text.lower())
            return tokens
        except Exception as e:
            self.logger.error(f"Error tokenizing text: {e}")
            raise

    def segmentText(self, text: str) -> List[str]:
        """
        Segment text into chunks, treating hyphenated words and apostrophes correctly.

        Args:
            text: The input string to be segmented.

        Returns:
            A list of text chunks.
        """
        try:
            clean_text = self.cleanText(text, preservePunctuation=True)
            tokens = self.tokenizeText(clean_text)
        
            if not tokens:
                return []
                
            chunks = []
            stride = max(1, self.chunkSize - self.chunkOverlap)
            
            for i in range(0, len(tokens), stride):
                chunk = tokens[i:i + self.chunkSize]
                if chunk:
                    chunks.append(" ".join(chunk))
                    
            return chunks
        except Exception as e:
            self.logger.error(f"Error segmenting text: {e}")
            raise

    def splitSentences(self, text: str) -> List[str]:
        """
        Split text into sentences, handling common abbreviations.

        Args:
            text: The input text to split.

        Returns:
            A list of sentences.
        """
        try:
            clean_text = self.cleanText(text, preservePunctuation=True)
            pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.!?])\s'
            sentences = re.split(pattern, clean_text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            self.logger.error(f"Error splitting sentences: {e}")
            raise

    def preprocess(self, text: str) -> str:
        """
        Preprocess the text by cleaning it (basic entry point).

        Args:
            text: The input string to preprocess.

        Returns:
            A cleaned version of the input string.
        """
        try:
            return self.cleanText(text)
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            raise