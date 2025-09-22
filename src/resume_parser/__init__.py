"""
Resume Parser Package

A comprehensive resume parsing system using hybrid LLM + regex extraction
with Azure OpenAI integration and Qdrant vector storage.
"""

__version__ = "1.0.0"
__author__ = "Resume Parser Team"

from .core.parser import ResumeParser
from .core.models import ResumeData, ExtractedData

__all__ = ["ResumeParser", "ResumeData", "ExtractedData"]