"""
Core data models for the resume parser.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid


class WorkExperience(BaseModel):
    """Work experience entry model."""

    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    duration: Optional[str] = Field(None, description="Duration of employment")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    location: Optional[str] = Field(None, description="Work location")
    responsibilities: List[str] = Field(default=[], description="Job responsibilities")
    technologies: List[str] = Field(default=[], description="Technologies used")


class Project(BaseModel):
    """Project entry model."""

    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    technologies: List[str] = Field(default=[], description="Technologies used")
    duration: Optional[str] = Field(None, description="Project duration")
    role: Optional[str] = Field(None, description="Role in project")
    url: Optional[str] = Field(None, description="Project URL")


class Education(BaseModel):
    """Education entry model."""

    degree: str = Field(..., description="Degree name")
    field: Optional[str] = Field(None, description="Field of study")
    institution: str = Field(..., description="Educational institution")
    graduation_year: Optional[str] = Field(None, description="Graduation year")
    gpa: Optional[str] = Field(None, description="GPA if available")
    location: Optional[str] = Field(None, description="Institution location")


class RoleClassification(BaseModel):
    """Role classification model."""

    primary_category: str = Field(..., description="Primary role category")
    secondary_categories: List[str] = Field(default=[], description="Secondary categories")
    seniority: str = Field(..., description="Seniority level")
    confidence_score: Optional[float] = Field(None, description="Classification confidence")


class CurrentEmployment(BaseModel):
    """Current employment information."""

    company: str = Field(..., description="Current company")
    position: str = Field(..., description="Current position")
    start_date: Optional[str] = Field(None, description="Employment start date")
    is_current: bool = Field(default=True, description="Currently employed")


class ExtractionStatistics(BaseModel):
    """Statistics about the extraction process."""

    total_experience_years: float = Field(0.0, description="Total years of experience")
    total_experience_months: int = Field(0, description="Additional months of experience")
    work_history_count: int = Field(0, description="Number of work entries")
    projects_count: int = Field(0, description="Number of projects")
    education_count: int = Field(0, description="Number of education entries")
    skills_count: int = Field(0, description="Number of skills identified")
    extraction_method: str = Field("hybrid", description="Extraction method used")
    processing_time_seconds: Optional[float] = Field(None, description="Processing time")


class ResumeData(BaseModel):
    """Complete resume data model."""

    # Personal Information
    user_id: str = Field(..., description="Unique user identifier")
    name: str = Field(..., description="Candidate name")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: str = Field(..., description="Location")
    linkedin_url: Optional[str] = Field(None, description="LinkedIn profile URL")

    # Professional Information
    current_position: str = Field(..., description="Current job title")
    skills: List[str] = Field(default=[], description="Skills list")
    summary: Optional[str] = Field(None, description="Professional summary")
    total_experience: str = Field(..., description="Total experience description")

    # Work and Education
    work_history: List[WorkExperience] = Field(default=[], description="Work experience")
    current_employment: Optional[CurrentEmployment] = Field(None, description="Current job")
    projects: List[Project] = Field(default=[], description="Projects")
    education: List[Education] = Field(default=[], description="Education history")

    # Classifications and Recommendations
    role_classification: RoleClassification = Field(..., description="Role classification")
    best_role: str = Field(..., description="Best matching role")
    recommended_roles: List[str] = Field(default=[], description="Recommended roles")

    # Metadata
    original_filename: Optional[str] = Field(None, description="Original file name")
    extraction_statistics: Optional[ExtractionStatistics] = Field(None, description="Extraction stats")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")


class ExtractedData(BaseModel):
    """Raw extracted data before processing."""

    raw_text: str = Field(..., description="Raw extracted text")
    sections: Dict[str, List[str]] = Field(default={}, description="Identified sections")
    file_type: str = Field(..., description="Source file type")
    file_size_bytes: int = Field(..., description="File size in bytes")
    extraction_method: str = Field(..., description="Extraction method used")


class EmbeddingData(BaseModel):
    """Embedding data for vector storage."""

    text: str = Field(..., description="Text for embedding")
    vector: List[float] = Field(..., description="Embedding vector")
    metadata: Dict[str, Any] = Field(default={}, description="Additional metadata")


class ProcessingResult(BaseModel):
    """Result of resume processing."""

    success: bool = Field(..., description="Processing success status")
    resume_data: Optional[ResumeData] = Field(None, description="Processed resume data")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    embedding_id: Optional[str] = Field(None, description="Qdrant vector ID")



# --- Models for New Features ---

class ShortlistUpdate(BaseModel):
    """Model for updating the shortlist status of a resume."""
    is_shortlisted: bool


class InterviewQuestionBase(BaseModel):
    """Base model for an interview question."""
    user_id: str = Field(..., description="The ID of the user creating the question.")
    question_text: str = Field(..., max_length=1000, description="The text of the interview question.")
    category: Optional[str] = Field(None, max_length=100, description="A category for the question (e.g., 'Technical', 'Behavioral').")


class InterviewQuestionCreate(InterviewQuestionBase):
    """Model for creating a new interview question."""
    pass


class InterviewQuestionUpdate(BaseModel):
    """Model for updating an existing interview question. All fields are optional."""
    question_text: Optional[str] = Field(None, max_length=1000, description="The updated text of the interview question.")
    category: Optional[str] = Field(None, max_length=100, description="The updated category for the question.")


class InterviewQuestionInDB(InterviewQuestionBase):
    """Model representing an interview question as stored in the database."""
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CallInitiationRequest(BaseModel):
    """Model for a request to initiate an interview call."""
    resume_id: uuid.UUID = Field(..., description="The ID of the resume/candidate to call.")
    user_id: str = Field(..., description="The ID of the user initiating the call.")
    notes: Optional[str] = Field(None, description="Optional notes for the call.")


class CallRecord(BaseModel):
    """Model representing a logged interview call as stored in the database."""
    id: uuid.UUID
    resume_id: uuid.UUID
    user_id: str
    call_status: str
    initiated_at: datetime
    notes: Optional[str] = None

    class Config:
        from_attributes = True