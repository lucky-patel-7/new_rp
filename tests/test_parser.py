"""
Tests for the resume parser functionality.
"""

import pytest
from pathlib import Path
import tempfile

from src.resume_parser.core.parser import ResumeParser
from src.resume_parser.core.models import ResumeData


class TestResumeParser:
    """Test cases for the ResumeParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ResumeParser()

    @pytest.mark.asyncio
    async def test_parse_resume_text_basic(self):
        """Test basic resume text parsing."""
        sample_text = """
        John Doe
        john.doe@email.com
        (555) 123-4567
        Software Engineer

        SKILLS
        Python, JavaScript, React

        EXPERIENCE
        Senior Developer at Tech Corp (2020-2023)
        - Developed web applications
        - Led team of 5 developers
        """

        result = await self.parser.parse_resume_text(
            raw_text=sample_text,
            user_id="test-user-123"
        )

        assert isinstance(result, ResumeData)
        assert result.user_id == "test-user-123"
        assert result.name != "Unknown"
        assert "@email.com" in result.email

    def test_create_embedding_text(self):
        """Test embedding text creation."""
        sample_data = {
            "name": "John Doe",
            "current_position": "Software Engineer",
            "skills": ["Python", "JavaScript"],
            "work_history": [{
                "title": "Developer",
                "company": "Tech Corp",
                "responsibilities": ["Coding", "Testing"]
            }]
        }

        embedding_text = self.parser._create_embedding_text(sample_data)

        assert "John Doe" in embedding_text
        assert "Software Engineer" in embedding_text
        assert "Python" in embedding_text
        assert "Tech Corp" in embedding_text

    @pytest.mark.asyncio
    async def test_process_resume_file_invalid_size(self):
        """Test file processing with invalid size."""
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            temp_path = Path(temp_file.name)

            # Test with file size exceeding limit
            result = await self.parser.process_resume_file(
                file_path=temp_path,
                user_id="test-user",
                file_size=50 * 1024 * 1024  # 50MB
            )

            assert not result.success
            assert "File too large" in result.error_message or "size" in result.error_message