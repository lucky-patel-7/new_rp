"""
Main resume parser class that coordinates all parsing operations.
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..clients.azure_openai import azure_client
from ..extractors.hybrid_extractor import HybridExtractor
from ..parsers.section_parser import identify_resume_sections, calculate_total_experience
from ..utils.logging import get_logger
from ..utils.file_handler import FileHandler
from .models import ResumeData, ExtractedData, ProcessingResult, ExtractionStatistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))
from config.settings import settings


logger = get_logger(__name__)


def clean_json_response(raw_text: str) -> str:
    """Clean JSON response from LLM output."""
    cleaned = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", raw_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"`{3,}", "", cleaned)
    return cleaned.strip()


class ResumeParser:
    """
    Production-ready resume parser with Azure OpenAI integration.

    Coordinates file processing, text extraction, AI analysis, and data structuring.
    """

    def __init__(self):
        """Initialize the resume parser."""
        self.file_handler = FileHandler()
        self.hybrid_extractor = HybridExtractor()

        # Try to initialize Azure OpenAI client
        try:
            self.azure_client = azure_client.get_sync_client()
            self.chat_deployment = azure_client.get_chat_deployment()
            self.embedding_deployment = azure_client.get_embedding_deployment()
            logger.info("✅ Azure OpenAI client initialized successfully")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to initialize Azure OpenAI: {e}")
            logger.info("[INFO] Will use fallback parsing only")
            self.azure_client = None

    async def process_resume_file(
        self,
        file_path: Path,
        user_id: str,
        file_size: int
    ) -> ProcessingResult:
        """
        Process a resume file end-to-end.

        Args:
            file_path: Path to the resume file
            user_id: Unique identifier for the user
            file_size: Size of the file in bytes

        Returns:
            ProcessingResult with success status and data
        """
        start_time = datetime.utcnow()

        try:
            # Validate file
            is_valid, error_msg = self.file_handler.validate_file(file_path, file_size)
            if not is_valid:
                return ProcessingResult(# type: ignore
                    success=False,
                    error_message=error_msg,
                    processing_time=0.0
                )

            # Extract text from file
            raw_text, file_type = self.file_handler.extract_text_from_file(file_path)

            if not raw_text or len(raw_text.strip()) < 50:
                return ProcessingResult(# type: ignore
                    success=False,
                    error_message="Resume content is too short or empty",
                    processing_time=0.0
                )

            # Create extracted data object
            extracted_data = ExtractedData(
                raw_text=raw_text,
                file_type=file_type,
                file_size_bytes=file_size,
                extraction_method="file_handler"
            )

            # Parse the resume
            resume_data = await self.parse_resume_text(raw_text, user_id, file_path.name)

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            resume_data.extraction_statistics.processing_time_seconds = processing_time# type: ignore

            return ProcessingResult(# type: ignore
                success=True,
                resume_data=resume_data,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error processing resume file: {e}")
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            return ProcessingResult(# type: ignore
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )

    async def parse_resume_text(
        self,
        raw_text: str,
        user_id: str,
        filename: Optional[str] = None
    ) -> ResumeData:
        """
        Parse resume text and return structured data.

        Args:
            raw_text: Raw text extracted from resume
            user_id: Unique identifier for the user
            filename: Original filename (optional)

        Returns:
            ResumeData object with structured information
        """
        logger.info(f"Starting resume parsing for user: {user_id}")

        # Generate content hash for deduplication
        content_hash = hashlib.sha256(raw_text.encode()).hexdigest()

        try:
            # Use Azure OpenAI for initial structuring if available
            if self.azure_client and self.chat_deployment:
                logger.debug("Using Azure OpenAI for resume structuring")
                structured_data = await self._ai_structure_resume(raw_text)
                logger.debug(f"AI structuring completed, projects: {len(structured_data.get('projects', []))}")
            else:
                logger.debug("Using fallback parsing method")
                structured_data = self._fallback_parse(raw_text)

            # Enhance with hybrid extraction for critical fields
            logger.debug("Starting hybrid extraction enhancement")
            structured_data = await self._enhance_with_hybrid_extraction(
                structured_data, raw_text
            )
            logger.debug(f"Hybrid extraction completed, projects: {len(structured_data.get('projects', []))}")

            # Create extraction statistics
            logger.debug("Creating extraction statistics")
            stats = self._create_extraction_statistics(structured_data)

            # Build final ResumeData object
            logger.debug("Building ResumeData object")
            try:
                resume_data = ResumeData(
                    user_id=user_id,
                    name=structured_data.get('name', 'Unknown'),
                    email=structured_data.get('email', ''),
                    phone=structured_data.get('phone'),
                    location=structured_data.get('location', ''),
                    linkedin_url=structured_data.get('linkedin_url'),
                    current_position=structured_data.get('current_position', ''),
                    skills=structured_data.get('skills', []),
                    summary=structured_data.get('summary'),
                    total_experience=structured_data.get('total_experience', '0 years'),
                    work_history=structured_data.get('work_history', []),
                    current_employment=structured_data.get('current_employment'),
                    projects=structured_data.get('projects', []),
                    education=structured_data.get('education', []),
                    role_classification=structured_data.get('role_classification', {}),
                    best_role=structured_data.get('best_role', ''),
                    recommended_roles=structured_data.get('recommended_roles', []),
                    original_filename=filename,
                    extraction_statistics=stats
                )
                logger.debug("ResumeData object created successfully")
            except Exception as e:
                logger.error(f"Error creating ResumeData object: {e}")
                logger.debug(f"Structured data keys: {list(structured_data.keys())}")
                logger.debug(f"Projects data: {structured_data.get('projects', [])}")
                raise

            logger.info(f"✅ Resume parsing completed for user: {user_id}")
            return resume_data

        except Exception as e:
            logger.error(f"Error parsing resume text: {e}")
            raise

    async def _ai_structure_resume(self, raw_text: str) -> Dict[str, Any]:
        """Use Azure OpenAI to structure resume data."""
        try:
            # Prepare prompt for comprehensive extraction
            prompt = self._create_extraction_prompt()

            response = self.azure_client.chat.completions.create(# type: ignore
                model=self.chat_deployment,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": raw_text}
                ],
                temperature=0.1,
                max_tokens=4000
            )

            response_text = response.choices[0].message.content
            cleaned_response = clean_json_response(response_text)# type: ignore

            # Parse JSON response
            structured_data = json.loads(cleaned_response)
            logger.debug("✅ AI structuring completed successfully")
            return structured_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"Raw response: {response_text[:500]}...")# type: ignore
            return self._fallback_parse(raw_text)
        except Exception as e:
            logger.error(f"Error in AI structuring: {e}")
            return self._fallback_parse(raw_text)

    def _create_extraction_prompt(self) -> str:
        """Create comprehensive extraction prompt for AI."""
        return """You are an expert resume parser. Extract comprehensive information from the resume text and return ONLY a valid JSON object with this exact structure:

{
    "name": "Full name",
    "email": "email@example.com",
    "phone": "phone number",
    "location": "city, state/country",
    "linkedin_url": "LinkedIn profile URL if available",
    "current_position": "current job title",
    "summary": "professional summary/objective",
    "skills": ["skill1", "skill2", "skill3"],
    "total_experience": "X years Y months",
    "work_history": [
        {
            "title": "Job Title",
            "company": "Company Name",
            "duration": "MM/YYYY - MM/YYYY or Present",
            "start_date": "MM/YYYY",
            "end_date": "MM/YYYY or Present",
            "location": "City, State",
            "responsibilities": ["responsibility 1", "responsibility 2"],
            "technologies": ["tech1", "tech2"]
        }
    ],
    "current_employment": {
        "company": "Current Company",
        "position": "Current Position",
        "start_date": "MM/YYYY",
        "is_current": true
    },
    "projects": [
        {
            "name": "Project Name",
            "description": "Project description",
            "technologies": ["tech1", "tech2"],
            "duration": "duration if available",
            "role": "your role in project"
        }
    ],
    "education": [
        {
            "degree": "Degree Name",
            "field": "Field of Study",
            "institution": "University/School Name",
            "graduation_year": "YYYY",
            "gpa": "GPA if available",
            "location": "City, State"
        }
    ],
    "role_classification": {
        "primary_category": "Software Engineer/Data Scientist/etc",
        "secondary_categories": ["category1", "category2"],
        "seniority": "Junior/Mid/Senior/Lead",
        "confidence_score": 0.95
    },
    "best_role": "Most suitable role based on experience",
    "recommended_roles": ["role1", "role2", "role3"]
}

Extract all available information. Use "Unknown" or empty arrays for missing data. Be thorough and accurate."""

    def _fallback_parse(self, raw_text: str) -> Dict[str, Any]:
        """Fallback parsing when AI is not available."""
        logger.info("Using fallback parsing method")

        # Use hybrid extractor for basic extraction
        basic_data = self.hybrid_extractor.extract_basic_info(raw_text)

        # Add minimal structure required
        return {
            "name": basic_data.get("name", "Unknown"),
            "email": basic_data.get("email", ""),
            "phone": basic_data.get("phone", ""),
            "location": basic_data.get("location", ""),
            "current_position": basic_data.get("current_position", ""),
            "skills": basic_data.get("skills", []),
            "total_experience": "0 years",
            "work_history": [],
            "projects": [],
            "education": [],
            "role_classification": {
                "primary_category": "Unknown",
                "seniority": "Unknown",
                "confidence_score": 0.5
            },
            "best_role": "Unknown",
            "recommended_roles": []
        }

    async def _enhance_with_hybrid_extraction(
        self,
        structured_data: Dict[str, Any],
        raw_text: str
    ) -> Dict[str, Any]:
        """Enhance AI results with hybrid extraction."""
        logger.debug("Enhancing with hybrid extraction")

        try:
            # Get sections for detailed extraction
            lines = raw_text.split('\n')
            sections = identify_resume_sections(lines)
            logger.debug("Sections identified successfully")

            # Enhance contact information
            try:
                contact_info = self.hybrid_extractor.extract_contact_info(raw_text)
                if contact_info.get('phone') and not structured_data.get('phone'):
                    structured_data['phone'] = contact_info['phone']
                if contact_info.get('linkedin') and not structured_data.get('linkedin_url'):
                    structured_data['linkedin_url'] = contact_info['linkedin']
                logger.debug("Contact info enhancement completed")
            except Exception as e:
                logger.error(f"Error in contact info extraction: {e}")

            # Enhance work history if missing or incomplete
            try:
                if not structured_data.get('work_history'):
                    work_history = self.hybrid_extractor.extract_work_history_hybrid(
                        raw_text, sections.get('experience', [])
                    )
                    structured_data['work_history'] = work_history
                logger.debug("Work history enhancement completed")
            except Exception as e:
                logger.error(f"Error in work history extraction: {e}")

            # Enhance projects if missing
            try:
                if not structured_data.get('projects'):
                    projects = self.hybrid_extractor.extract_projects_hybrid(
                        raw_text, sections.get('projects', [])
                    )
                    structured_data['projects'] = projects
                logger.debug("Projects enhancement completed")
            except Exception as e:
                logger.error(f"Error in projects extraction: {e}")

            # Enhance education if missing
            try:
                if not structured_data.get('education'):
                    education = self.hybrid_extractor.extract_education_hybrid(
                        raw_text, sections.get('education', [])
                    )
                    structured_data['education'] = education
                logger.debug("Education enhancement completed")
            except Exception as e:
                logger.error(f"Error in education extraction: {e}")

            # Calculate total experience
            try:
                experience_string = calculate_total_experience(
                    structured_data.get('work_history', [])
                )
                structured_data['total_experience'] = experience_string
                logger.debug("Total experience calculation completed")
            except Exception as e:
                logger.error(f"Error in experience calculation: {e}")
                structured_data['total_experience'] = '0 years'

            return structured_data

        except Exception as e:
            logger.error(f"Error in hybrid extraction enhancement: {e}")
            return structured_data

    def _create_extraction_statistics(self, structured_data: Dict[str, Any]) -> ExtractionStatistics:
        """Create extraction statistics from parsed data."""
        # Parse total experience
        total_exp = structured_data.get('total_experience', '0 years 0 months')
        years = months = 0

        # Extract years and months from string
        year_match = re.search(r'(\d+)\s*years?', total_exp, re.IGNORECASE)
        month_match = re.search(r'(\d+)\s*months?', total_exp, re.IGNORECASE)

        if year_match:
            years = float(year_match.group(1))
        if month_match:
            months = int(month_match.group(1))

        return ExtractionStatistics(# type: ignore
            total_experience_years=years,
            total_experience_months=months,
            work_history_count=len(structured_data.get('work_history', [])),
            projects_count=len(structured_data.get('projects', [])),
            education_count=len(structured_data.get('education', [])),
            skills_count=len(structured_data.get('skills', [])),
            extraction_method="hybrid_ai" if self.azure_client else "hybrid_fallback"
        )

    async def create_embedding(self, resume_data: ResumeData) -> List[float]:
        """Create embedding for resume data."""
        if not self.azure_client:
            logger.warning("Azure OpenAI client not available for embeddings")
            return []

        try:
            # Create comprehensive text for embedding
            embedding_text = self._create_embedding_text(resume_data.dict())

            async_client = azure_client.get_async_client()
            response = await async_client.embeddings.create(
                model=self.embedding_deployment,
                input=embedding_text
            )

            embedding_vector = response.data[0].embedding
            logger.debug(f"✅ Created embedding vector of size {len(embedding_vector)}")
            return embedding_vector

        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return []

    def _create_embedding_text(self, resume_data: Dict[str, Any]) -> str:
        """Create comprehensive embedding text from ALL extracted data."""
        text_parts = []

        # Basic information
        if resume_data.get('name'):
            text_parts.append(f"Candidate: {resume_data['name']}")

        if resume_data.get('current_position'):
            text_parts.append(f"Current Position: {resume_data['current_position']}")

        # Skills
        skills = resume_data.get('skills', [])
        if skills:
            text_parts.append(f"Skills: {', '.join(skills[:10])}")  # Top 10 skills

        # Work history with comprehensive details
        work_history = resume_data.get('work_history', [])
        if work_history:
            for job in work_history:
                # Handle both dict and Pydantic model formats
                if hasattr(job, 'model_dump'):
                    job_dict = job.model_dump()
                elif hasattr(job, 'dict'):
                    job_dict = job.dict()
                elif isinstance(job, dict):
                    job_dict = job
                else:
                    job_dict = {}

                job_text = f"Work Experience: {job_dict.get('title', '')} at {job_dict.get('company', '')}"
                if job_dict.get('duration'):
                    job_text += f" ({job_dict['duration']})"
                text_parts.append(job_text)

                # Add responsibilities
                responsibilities = job_dict.get('responsibilities', [])
                if responsibilities:
                    resp_text = f"Responsibilities: {'. '.join(responsibilities[:3])}"
                    text_parts.append(resp_text)

                # Add technologies
                technologies = job_dict.get('technologies', [])
                if technologies:
                    tech_text = f"Technologies: {', '.join(technologies[:5])}"
                    text_parts.append(tech_text)

        # Projects with technologies
        projects = resume_data.get('projects', [])
        if projects:
            for project in projects:
                try:
                    # Handle both dict and Pydantic model formats
                    if hasattr(project, 'model_dump'):
                        project_dict = project.model_dump()
                    elif hasattr(project, 'dict'):
                        project_dict = project.dict()
                    elif isinstance(project, dict):
                        project_dict = project
                    else:
                        # Skip non-dict, non-model objects
                        logger.warning(f"Unexpected project type: {type(project)}")
                        continue

                    project_text = f"Project: {project_dict.get('name', '')}"
                    # Safely handle missing 'description' key
                    description = project_dict.get('description', '')
                    if description:
                        project_text += f" - {description[:100]}"
                    if project_dict.get('technologies'):
                        technologies = project_dict['technologies']
                        if isinstance(technologies, list):
                            project_text += f" using {', '.join(technologies[:5])}"
                    text_parts.append(project_text)
                except Exception as e:
                    logger.error(f"Error processing project: {e}, project: {project}")
                    continue

        # Education with complete details
        education = resume_data.get('education', [])
        if education:
            for edu in education:
                # Handle both dict and Pydantic model formats
                if hasattr(edu, 'model_dump'):
                    edu_dict = edu.model_dump()
                elif hasattr(edu, 'dict'):
                    edu_dict = edu.dict()
                elif isinstance(edu, dict):
                    edu_dict = edu
                else:
                    edu_dict = {}

                edu_text = f"Education: {edu_dict.get('degree', '')} in {edu_dict.get('field', '')}"
                if edu_dict.get('institution'):
                    edu_text += f" from {edu_dict['institution']}"
                text_parts.append(edu_text)

        # Professional summary
        if resume_data.get('summary'):
            text_parts.append(f"Summary: {resume_data['summary'][:200]}")

        # Role classification
        role_class = resume_data.get('role_classification', {})
        if role_class.get('primary_category'):
            text_parts.append(f"Role Category: {role_class['primary_category']}")
        if role_class.get('seniority'):
            text_parts.append(f"Seniority: {role_class['seniority']}")

        # Experience level
        if resume_data.get('total_experience'):
            text_parts.append(f"Total Experience: {resume_data['total_experience']}")

        return '. '.join(text_parts)