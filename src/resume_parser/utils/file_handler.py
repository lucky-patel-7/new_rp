"""
File handling utilities for resume processing.
"""

import magic
import fitz  # PyMuPDF
import sys
from pathlib import Path
from typing import Tuple, Optional
from docx import Document

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))
from config.settings import settings
from .logging import get_logger

logger = get_logger(__name__)


class FileHandler:
    """Handles file operations for resume processing."""

    @staticmethod
    def validate_file(file_path: Path, file_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file.

        Args:
            file_path: Path to the file
            file_size: Size of the file in bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if file_size > settings.max_file_size_bytes:
            return False, f"File size exceeds {settings.app.max_file_size_mb}MB limit"

        # Check file type
        try:
            file_type = magic.from_file(str(file_path), mime=True)
            logger.debug(f"Detected file type: {file_type}")

            # Map MIME types to extensions
            allowed_types = {
                'application/pdf': 'pdf',
                'application/msword': 'doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                'text/plain': 'txt'
            }

            if file_type not in allowed_types:
                return False, f"File type {file_type} not supported"

            detected_extension = allowed_types[file_type]
            if detected_extension not in settings.app.allowed_file_types:
                return False, f"File type {detected_extension} not allowed"

            return True, None

        except Exception as e:
            logger.error(f"Error validating file: {e}")
            return False, f"Error validating file: {str(e)}"

    @staticmethod
    def extract_text_from_file(file_path: Path) -> Tuple[str, str]:
        """
        Extract text from uploaded file.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (extracted_text, file_type)
        """
        try:
            file_type = magic.from_file(str(file_path), mime=True)
            logger.info(f"Extracting text from {file_type} file: {file_path.name}")

            if file_type == 'application/pdf':
                return FileHandler._extract_from_pdf(file_path), 'pdf'
            elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                return FileHandler._extract_from_docx(file_path), 'docx'
            elif file_type == 'application/msword':
                return FileHandler._extract_from_doc(file_path), 'doc'
            elif file_type == 'text/plain':
                return FileHandler._extract_from_txt(file_path), 'txt'
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

        except Exception as e:
            logger.error(f"Error extracting text from file: {e}")
            raise

    @staticmethod
    def _extract_from_pdf(file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            doc = fitz.open(str(file_path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            logger.debug(f"Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    @staticmethod
    def _extract_from_docx(file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(str(file_path))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            logger.debug(f"Extracted {len(text)} characters from DOCX")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            raise

    @staticmethod
    def _extract_from_doc(file_path: Path) -> str:
        """Extract text from DOC file (legacy Word format)."""
        # For DOC files, we might need additional libraries like python-docx2txt
        # For now, raise an error as it requires additional setup
        raise NotImplementedError("DOC file support requires additional libraries")

    @staticmethod
    def _extract_from_txt(file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.debug(f"Extracted {len(text)} characters from TXT")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from TXT: {e}")
            raise

    @staticmethod
    def cleanup_file(file_path: Path) -> bool:
        """
        Clean up temporary file.

        Args:
            file_path: Path to the file to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleaned up file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {e}")
            return False