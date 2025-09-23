"""
Main FastAPI application for Resume Parser.

A modern, well-organized resume parsing API with comprehensive extraction capabilities.
"""

import os
import sys
import uuid
import tempfile
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from difflib import SequenceMatcher

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio

# Import organized modules
from src.resume_parser.core.parser import ResumeParser
from src.resume_parser.core.models import ProcessingResult, ResumeData
from src.resume_parser.database.qdrant_client import qdrant_client
from src.resume_parser.utils.logging import setup_logging, get_logger
from src.resume_parser.utils.file_handler import FileHandler
from config.settings import settings

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app.app_name,
    description="A comprehensive resume parsing API with Azure OpenAI integration",
    version="1.0.0",
    debug=settings.app.debug
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global resume parser instance
resume_parser = ResumeParser()

# Ensure upload directory exists
upload_dir = Path(settings.app.upload_dir)
upload_dir.mkdir(exist_ok=True)


async def _generate_embedding(text: str) -> List[float]:
    """Create an embedding vector for arbitrary search text."""
    if not text or not text.strip():
        return []

    if not resume_parser.azure_client:
        raise HTTPException(
            status_code=503,
            detail="Search functionality requires Azure OpenAI configuration"
        )

    try:
        from src.resume_parser.clients.azure_openai import azure_client

        async_client = azure_client.get_async_client()
        response = await async_client.embeddings.create(
            model=azure_client.get_embedding_deployment(),
            input=text
        )
        embedding_vector = response.data[0].embedding
        logger.debug(
            "🔮 Generated embedding of length %s for text snippet: '%s'",
            len(embedding_vector),
            text[:60].replace("\n", " ")
        )
        return embedding_vector
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to generate embedding: {exc}")
        raise HTTPException(status_code=503, detail="Embedding service unavailable")


def _extract_experience_years(payload: Dict[str, Any]) -> Optional[float]:
    """Extract total experience in years from stored payload."""
    stats = payload.get('extraction_statistics') or {}

    try:
        years = stats.get('total_experience_years')
        months = stats.get('total_experience_months')
        if years is not None:
            years_total = float(years)
            if months:
                years_total += float(months) / 12.0
            return years_total
    except (TypeError, ValueError):
        pass

    total_experience = payload.get('total_experience')
    if not isinstance(total_experience, str):
        return None

    years_match = re.search(r'(\d+(?:\.\d+)?)\s*years?', total_experience, re.IGNORECASE)
    months_match = re.search(r'(\d+(?:\.\d+)?)\s*months?', total_experience, re.IGNORECASE)

    years_total = 0.0
    found_value = False

    if years_match:
        try:
            years_total += float(years_match.group(1))
            found_value = True
        except ValueError:
            pass

    if months_match:
        try:
            years_total += float(months_match.group(1)) / 12.0
            found_value = True
        except ValueError:
            pass

    return years_total if found_value else None


def _candidate_meets_experience_requirement(
    payload: Dict[str, Any],
    minimum_years: Optional[int],
    maximum_years: Optional[int]
) -> bool:
    """Determine if a candidate satisfies minimum and/or maximum experience requirements."""
    experience_years = _extract_experience_years(payload)
    if experience_years is None:
        return False

    try:
        if minimum_years is not None and experience_years < float(minimum_years):
            return False
        if maximum_years is not None and experience_years > float(maximum_years):
            return False
        return True
    except (TypeError, ValueError):
        return False


GENERIC_COMPANY_TOKENS: Set[str] = {
    'inc', 'inc.', 'llc', 'llp', 'ltd', 'ltd.', 'pvt', 'pvt.', 'private', 'limited',
    'technologies', 'technology', 'tech', 'solutions', 'solution', 'systems', 'system',
    'software', 'services', 'service', 'corp', 'corp.', 'corporation', 'company',
    'co', 'co.', 'group', 'india', 'global', 'international', 'it', 'p', 'plc'
}


def _normalize_company_text(name: str) -> str:
    if not name or not isinstance(name, str):
        return ''
    # Replace punctuation with spaces and collapse whitespace
    cleaned = re.sub(r'[^a-z0-9]+', ' ', name.lower())
    return re.sub(r'\s+', ' ', cleaned).strip()


def _company_token_set(name: str) -> Set[str]:
    normalized = _normalize_company_text(name)
    tokens = set(re.split(r'\s+', normalized))
    return {token for token in tokens if token and token not in GENERIC_COMPANY_TOKENS}


def _company_names_match(query_company: str, candidate_company: str) -> bool:
    normalized_query = _normalize_company_text(query_company)
    normalized_candidate = _normalize_company_text(candidate_company)

    if not normalized_query or not normalized_candidate:
        return False

    if normalized_query == normalized_candidate:
        return True

    core_query_tokens = _company_token_set(query_company)
    core_candidate_tokens = _company_token_set(candidate_company)

    if core_query_tokens and core_candidate_tokens:
        if core_query_tokens.issubset(core_candidate_tokens) or core_candidate_tokens.issubset(core_query_tokens):
            return True

        token_overlap = core_query_tokens & core_candidate_tokens
        if any(len(token) >= 4 for token in token_overlap):
            return True

    # Check for strong substring match on normalized names while avoiding generic-only matches
    if core_query_tokens and core_candidate_tokens:
        if normalized_query in normalized_candidate or normalized_candidate in normalized_query:
            if core_query_tokens <= core_candidate_tokens or core_candidate_tokens <= core_query_tokens:
                return True

    similarity = SequenceMatcher(None, normalized_query, normalized_candidate).ratio()
    return similarity >= 0.78


def _extract_candidate_companies(payload: Dict[str, Any]) -> List[str]:
    companies: List[str] = []

    current_employment = payload.get('current_employment')
    if isinstance(current_employment, dict):
        company = current_employment.get('company')
        if company:
            companies.append(company)

    if payload.get('company'):
        companies.append(payload.get('company'))

    work_history = payload.get('work_history', [])
    for job in work_history:
        if isinstance(job, dict):
            company = job.get('company')
            if company:
                companies.append(company)

    # Remove duplicates while preserving order
    seen: Set[str] = set()
    unique_companies: List[str] = []
    for company in companies:
        normalized = _normalize_company_text(company)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_companies.append(company)

    return unique_companies


def _find_company_match(
    payload: Dict[str, Any],
    query_companies: List[str]
) -> Tuple[bool, Optional[str], Optional[str]]:
    if not query_companies:
        return False, None, None

    candidate_companies = _extract_candidate_companies(payload)
    for candidate_company in candidate_companies:
        for query_company in query_companies:
            if _company_names_match(query_company, candidate_company):
                return True, candidate_company, query_company

    return False, None, None


def _candidate_matches_company(payload: Dict[str, Any], query_companies: List[str]) -> bool:
    match_found, _, _ = _find_company_match(payload, query_companies)
    return match_found


def _candidate_matches_forced_keywords(payload: Dict[str, Any], forced_keywords: List[str]) -> bool:
    if not forced_keywords:
        return True

    from src.resume_parser.utils.search_intelligence import search_processor

    try:
        text = search_processor.extract_comprehensive_search_text(payload).lower()
    except Exception:
        text = ' '

    for keyword in forced_keywords:
        if keyword.lower() not in text:
            return False
    return True


def _candidate_matches_education(
    payload: Dict[str, Any],
    required_degrees: List[str],
    required_institutions: List[str]
) -> bool:
    if not required_degrees and not required_institutions:
        return True

    education = payload.get('education', [])
    if not isinstance(education, list):
        education = []

    degrees_matched = not required_degrees
    institutions_matched = not required_institutions

    for edu in education:
        if not isinstance(edu, dict):
            continue
        degree_val = (edu.get('degree') or '').lower()
        field_val = (edu.get('field') or '').lower()
        institution_val = (edu.get('institution') or '').lower()

        if not degrees_matched and degree_val:
            for required_degree in required_degrees:
                req_lower = required_degree.lower()
                if req_lower in degree_val or req_lower in field_val:
                    degrees_matched = True
                    break

        if not institutions_matched and institution_val:
            for required_institution in required_institutions:
                req_inst_lower = required_institution.lower()
                if req_inst_lower in institution_val:
                    institutions_matched = True
                    break

        if degrees_matched and institutions_matched:
            break

    return degrees_matched and institutions_matched


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info(f"🚀 Starting {settings.app.app_name}")
    logger.info(f"📁 Upload directory: {upload_dir}")
    logger.info(f"🔧 Debug mode: {settings.app.debug}")

    # Test Qdrant connection
    try:
        collection_info = qdrant_client.get_collection_info()
        logger.info(f"✅ Qdrant connected: {collection_info}")
    except Exception as e:
        logger.warning(f"⚠️ Qdrant connection issue: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown."""
    logger.info(f"🛑 Shutting down {settings.app.app_name}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": f"Welcome to {settings.app.app_name}",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "upload": "/upload-resume",
            "health": "/health",
            "search": "/search-resumes"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": str(asyncio.get_event_loop().time()),
        "services": {}
    }

    # Check Azure OpenAI
    try:
        if resume_parser.azure_client:
            health_status["services"]["azure_openai"] = "connected"
        else:
            health_status["services"]["azure_openai"] = "not_configured"
    except Exception as e:
        health_status["services"]["azure_openai"] = f"error: {str(e)}"

    # Check Qdrant
    try:
        collection_info = qdrant_client.get_collection_info()
        health_status["services"]["qdrant"] = "connected"
        health_status["services"]["qdrant_info"] = collection_info
    except Exception as e:
        health_status["services"]["qdrant"] = f"error: {str(e)}"

    return health_status


@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload and parse a resume file.

    Supports PDF, DOC, DOCX, and TXT files.
    Returns structured resume data and stores embeddings in Qdrant.
    """
    logger.info(f"📄 Processing resume upload: {file.filename}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Generate unique user ID
    user_id = str(uuid.uuid4())

    # Create temporary file
    temp_file = None
    temp_file_path = None  # Ensure temp_file_path is always defined
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        logger.info(f"📊 File size: {file_size} bytes")

        # Validate file size
        if file_size > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.app.max_file_size_mb}MB"
            )

        # Create temporary file
        suffix = f".{file.filename.split('.')[-1].lower()}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_content)
            temp_file_path = Path(temp_file.name)

        # Process the resume
        result: ProcessingResult = await resume_parser.process_resume_file(
            file_path=temp_file_path,
            user_id=user_id,
            file_size=file_size
        )

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        resume_data = result.resume_data

        # Defensive: ensure resume_data is not None and is a ResumeData instance
        if not resume_data:
            raise HTTPException(status_code=500, detail="Resume parsing failed: No resume data returned")

        # Ensure all projects have a 'description' key
        safe_projects = []
        for project in getattr(resume_data, 'projects', []):
            proj_dict = project.dict() if hasattr(project, 'dict') else dict(project)
            if 'description' not in proj_dict:
                proj_dict['description'] = ''
            safe_projects.append(proj_dict)

        # Defensive: handle other fields that may be None and .dict()
        safe_work_history = [job.dict() if hasattr(job, 'dict') else dict(job) for job in getattr(resume_data, 'work_history', [])]
        safe_education = [edu.dict() if hasattr(edu, 'dict') else dict(edu) for edu in getattr(resume_data, 'education', [])]
        safe_role_classification = resume_data.role_classification.dict() if getattr(resume_data, 'role_classification', None) and hasattr(resume_data.role_classification, 'dict') else getattr(resume_data, 'role_classification', {})
        safe_extraction_statistics = resume_data.extraction_statistics.dict() if getattr(resume_data, 'extraction_statistics', None) and hasattr(resume_data.extraction_statistics, 'dict') else getattr(resume_data, 'extraction_statistics', {})
        safe_current_employment = resume_data.current_employment.dict() if getattr(resume_data, 'current_employment', None) and hasattr(resume_data.current_employment, 'dict') else getattr(resume_data, 'current_employment', None)
        safe_created_at = getattr(resume_data, 'created_at', None)
        upload_timestamp = safe_created_at.isoformat() if safe_created_at else None

        payload = {
            "user_id": getattr(resume_data, 'user_id', None),
            "name": getattr(resume_data, 'name', None),
            "email": getattr(resume_data, 'email', None),
            "phone": getattr(resume_data, 'phone', ''),
            "location": getattr(resume_data, 'location', None),
            "linkedin_url": getattr(resume_data, 'linkedin_url', ''),
            "current_position": getattr(resume_data, 'current_position', None),
            "skills": getattr(resume_data, 'skills', []),
            "total_experience": getattr(resume_data, 'total_experience', None),
            "role_category": getattr(resume_data.role_classification, 'primary_category', None) if getattr(resume_data, 'role_classification', None) else None,
            "seniority": getattr(resume_data.role_classification, 'seniority', None) if getattr(resume_data, 'role_classification', None) else None,
            "best_role": getattr(resume_data, 'best_role', None),
            "summary": getattr(resume_data, 'summary', ''),
            "recommended_roles": getattr(resume_data, 'recommended_roles', []),
            "work_history": safe_work_history,
            "current_employment": safe_current_employment,
            "projects": safe_projects,
            "education": safe_education,
            "role_classification": safe_role_classification,
            "original_filename": file.filename,
            "extraction_statistics": safe_extraction_statistics,
            "upload_timestamp": upload_timestamp
        }

        # Create embedding and store in Qdrant
        embedding_vector = await resume_parser.create_embedding(resume_data)

        if embedding_vector:
            # Store in Qdrant
            try:
                point_id = await qdrant_client.store_embedding(
                    user_id=user_id,
                    embedding_vector=embedding_vector,
                    payload=payload
                )
                logger.info(f"✅ Stored in Qdrant with ID: {point_id}")
            except Exception as e:
                logger.error(f"❌ Failed to store in Qdrant: {e}")
                # Continue without failing the request

        # Prepare response
        response_data = {
            "success": True,
            "user_id": user_id,
            "processing_time": result.processing_time,
            "resume_data": resume_data.model_dump(mode='json') if hasattr(resume_data, 'model_dump') else (resume_data.dict() if hasattr(resume_data, 'dict') else {}),
            "message": "Resume processed successfully"
        }

        logger.info(f"✅ Resume processing completed for user: {user_id}")
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing resume: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # Clean up temporary file
        try:
            if temp_file_path and temp_file_path.exists():
                FileHandler.cleanup_file(temp_file_path)
        except Exception:
            pass


@app.post("/bulk-upload-resumes")
async def bulk_upload_resumes(files: List[UploadFile] = File(...)):
    """
    Upload and parse multiple resume files (up to 5).

    Supports PDF, DOC, DOCX, and TXT files.
    Returns structured resume data for all files and stores embeddings in Qdrant.

    Args:
        files: List of resume files (maximum 5 files)

    Returns:
        Dict containing results for each file with success/failure status
    """
    logger.info(f"📁 Processing bulk upload: {len(files)} files")

    # Validate file count
    if len(files) > 5:
        raise HTTPException(
            status_code=400,
            detail="Maximum 5 files allowed per bulk upload"
        )

    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )

    results = {
        "total_files": len(files),
        "successful_uploads": 0,
        "failed_uploads": 0,
        "results": []
    }

    # Process each file using the same logic as single upload
    for i, file in enumerate(files):
        file_result = {
            "file_index": i,
            "filename": file.filename,
            "status": "processing",
            "user_id": None,
            "error": None,
            "resume_data": None
        }

        try:
            logger.info(f"📄 Processing file {i+1}/{len(files)}: {file.filename}")

            if not file.filename:
                raise ValueError("No filename provided")

            # Call the same upload logic as the single file upload
            # This is essentially what the upload_resume function does
            user_id = str(uuid.uuid4())
            file_result["user_id"] = user_id

            # Create temporary file
            temp_file = None
            temp_file_path = None

            try:
                # Get file content and size
                content = await file.read()
                file_size = len(content)
                logger.info(f"📊 File size: {file_size} bytes")

                # Create temporary file with proper suffix
                file_suffix = Path(file.filename).suffix.lower()
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=file_suffix
                )
                temp_file_path = Path(temp_file.name)

                # Write content to temporary file
                temp_file.write(content)
                temp_file.close()

                # Process the resume using the same logic as single upload
                result = await resume_parser.process_resume_file(
                    file_path=temp_file_path,
                    user_id=user_id,
                    file_size=file_size
                )

                if not result.success:
                    raise ValueError(f"Failed to parse resume: {result.error_message}")

                resume_data = result.resume_data

                if not resume_data:
                    raise ValueError("Resume parsing failed: No resume data returned")

                # Use the same Qdrant storage logic as the original upload function
                # (Copy the exact same logic from upload_resume function)
                safe_projects = []
                for project in getattr(resume_data, 'projects', []):
                    proj_dict = project.dict() if hasattr(project, 'dict') else dict(project)
                    if 'description' not in proj_dict:
                        proj_dict['description'] = ''
                    safe_projects.append(proj_dict)

                safe_work_history = [job.dict() if hasattr(job, 'dict') else dict(job) for job in getattr(resume_data, 'work_history', [])]
                safe_education = [edu.dict() if hasattr(edu, 'dict') else dict(edu) for edu in getattr(resume_data, 'education', [])]
                safe_role_classification = resume_data.role_classification.dict() if getattr(resume_data, 'role_classification', None) and hasattr(resume_data.role_classification, 'dict') else getattr(resume_data, 'role_classification', {})
                safe_extraction_statistics = resume_data.extraction_statistics.dict() if getattr(resume_data, 'extraction_statistics', None) and hasattr(resume_data.extraction_statistics, 'dict') else getattr(resume_data, 'extraction_statistics', {})
                safe_current_employment = resume_data.current_employment.dict() if getattr(resume_data, 'current_employment', None) and hasattr(resume_data.current_employment, 'dict') else getattr(resume_data, 'current_employment', None)
                safe_created_at = getattr(resume_data, 'created_at', None)
                upload_timestamp = safe_created_at.isoformat() if safe_created_at else None

                payload = {
                    "user_id": getattr(resume_data, 'user_id', None),
                    "name": getattr(resume_data, 'name', None),
                    "email": getattr(resume_data, 'email', None),
                    "phone": getattr(resume_data, 'phone', None),
                    "location": getattr(resume_data, 'location', None),
                    "linkedin_url": getattr(resume_data, 'linkedin_url', None),
                    "current_position": getattr(resume_data, 'current_position', None),
                    "skills": getattr(resume_data, 'skills', []),
                    "total_experience": getattr(resume_data, 'total_experience', None),
                    "role_category": getattr(resume_data.role_classification, 'primary_category', None) if getattr(resume_data, 'role_classification', None) else None,
                    "seniority": getattr(resume_data.role_classification, 'seniority', None) if getattr(resume_data, 'role_classification', None) else None,
                    "best_role": getattr(resume_data, 'best_role', None),
                    "summary": getattr(resume_data, 'summary', None),
                    "recommended_roles": getattr(resume_data, 'recommended_roles', []),
                    "work_history": safe_work_history,
                    "current_employment": safe_current_employment,
                    "projects": safe_projects,
                    "education": safe_education,
                    "role_classification": safe_role_classification,
                    "original_filename": file.filename,
                    "extraction_statistics": safe_extraction_statistics,
                    "upload_timestamp": upload_timestamp
                }

                # Create embedding and store in Qdrant (same as single upload)
                embedding_vector = await resume_parser.create_embedding(resume_data)

                if embedding_vector:
                    # Store in Qdrant
                    try:
                        point_id = await qdrant_client.store_embedding(
                            user_id=user_id,
                            embedding_vector=embedding_vector,
                            payload=payload
                        )
                        logger.info(f"✅ Stored resume {i+1} in Qdrant with ID: {point_id}")
                    except Exception as e:
                        logger.error(f"❌ Failed to store resume {i+1} in Qdrant: {e}")
                        # Continue without failing the request

                # Success - prepare simplified response data
                file_result["status"] = "success"
                file_result["resume_data"] = {
                    "name": getattr(resume_data, 'name', None),
                    "email": getattr(resume_data, 'email', None),
                    "current_position": getattr(resume_data, 'current_position', None),
                    "total_experience": getattr(resume_data, 'total_experience', None),
                    "skills_count": len(getattr(resume_data, 'skills', [])),
                    "best_role": getattr(resume_data, 'best_role', None)
                }
                results["successful_uploads"] += 1

                logger.info(f"✅ Successfully processed file {i+1}: {file.filename}")

            finally:
                # Cleanup temporary file
                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        os.unlink(temp_file_path)
                    except Exception:
                        pass

        except Exception as e:
            # Handle individual file errors
            error_msg = str(e)
            logger.error(f"❌ Error processing file {i+1} ({file.filename}): {error_msg}")

            file_result["status"] = "failed"
            file_result["error"] = error_msg
            results["failed_uploads"] += 1

        # Add file result to results
        results["results"].append(file_result)

    # Log summary
    logger.info(f"📋 Bulk upload completed: {results['successful_uploads']} successful, {results['failed_uploads']} failed")

    return results


@app.post("/search-resumes")
async def search_resumes(
    query: str = Form(...),
    limit: int = Form(10),
    role_filter: Optional[str] = Form(None),
    seniority_filter: Optional[str] = Form(None)
):
    """
    Search for similar resumes using intelligent semantic search.

    Args:
        query: Natural language query (e.g., "I need an HR manager with 5+ years experience")
        limit: Maximum number of results
        role_filter: Optional manual role filter override
        seniority_filter: Optional manual seniority filter override
    """
    try:
        # Import intelligent search processor
        from src.resume_parser.utils.search_intelligence import search_processor

        logger.info(f"🔍 Processing search query: {query}")

        # Parse the query intelligently
        parsed_query = search_processor.parse_query(query)

        # Clamp limit to prevent overly large responses or negative numbers
        limit = max(1, min(limit, 100))
        logger.info(f"📏 Result limit normalized to: {limit}")

        effective_skills = parsed_query.effective_skills()
        explicit_skill_count = len(parsed_query.skills)
        effective_skill_count = len(effective_skills)
        forced_keywords = parsed_query.forced_keywords
        required_degrees = parsed_query.required_degrees
        required_institutions = parsed_query.required_institutions

        logger.info(f"🔧 Explicit skills detected: {parsed_query.skills[:5]}")
        if parsed_query.role_inferred_skills:
            logger.info(f"🔧 Role-inferred skills sample: {parsed_query.role_inferred_skills[:5]}")
        if forced_keywords:
            logger.info(f"🔑 Forced keywords: {forced_keywords[:5]}")
        if required_degrees:
            logger.info(f"🎓 Required degrees: {required_degrees[:5]}")
        if required_institutions:
            logger.info(f"🏛️ Required institutions: {required_institutions[:5]}")

        # Create comprehensive search text including all relevant terms
        search_components: List[str] = []
        seen_components: Set[str] = set()

        def add_component(text: Optional[str]):
            if not text or not isinstance(text, str):
                return
            cleaned = text.strip()
            key = cleaned.lower()
            if cleaned and key not in seen_components:
                search_components.append(cleaned)
                seen_components.add(key)

        add_component(query)

        # Add detected role synonyms
        for role in parsed_query.job_roles:
            role_titles = search_processor.job_db.get_role_titles(role)
            for title in role_titles[:5]:
                add_component(title)

        # Add explicit skills mentioned in the query
        for skill in parsed_query.skills:
            add_component(skill)

        # Add inferred skills from configuration to enrich semantic search
        for skill in parsed_query.role_inferred_skills[:40]:
            add_component(skill)

        for keyword in forced_keywords:
            add_component(keyword)

        for degree in required_degrees:
            add_component(degree)
        for institution in required_institutions:
            add_component(institution)

        # Use expanded query for better semantic search
        search_text = ' '.join(search_components) if search_components else query

        # Log detailed embedding information
        logger.info(f"🔮 Creating embedding for search text: '{search_text}'")
        logger.info(f"🔮 Search components count: {len(search_components)} | sample: {search_components[:10]}")

        query_vector = await _generate_embedding(search_text)

        # Log embedding vector details
        logger.info(f"🔮 Generated embedding vector with {len(query_vector)} dimensions")
        logger.info(f"🔮 First 10 embedding values: {query_vector[:10]}")
        logger.info(f"🔮 Embedding magnitude: {sum(x*x for x in query_vector)**0.5:.4f}")

        # Create intelligent filters with proper role detection
        filter_conditions: Dict[str, Any] = {}

        # Improved role filtering logic
        role_filters = search_processor.create_search_filters(parsed_query)
        experience_min_filter_value = role_filters.pop('min_experience_years', None)
        experience_max_filter_value = role_filters.pop('max_experience_years', None)

        # Always apply location/seniority filters even when no role detected
        location_filter_value = role_filters.pop('location', None)
        if location_filter_value:
            filter_conditions['location'] = location_filter_value

        seniority_filter_value = role_filters.pop('seniority', None)
        if seniority_filter_value and 'seniority' not in filter_conditions:
            filter_conditions['seniority'] = seniority_filter_value

        # Apply role filters if we have clear role detection
        if parsed_query.job_roles:
            skill_count_for_filter = explicit_skill_count if explicit_skill_count else effective_skill_count
            # For role-focused queries (like "HR manager"), always apply role filters
            if skill_count_for_filter == 0 or len(parsed_query.job_roles) >= skill_count_for_filter:
                filter_conditions.update(role_filters)
                logger.info(f"🎯 Applying role filters for role-focused query: {role_filters}")
            # For skill-focused queries, be more flexible with role filters
            elif skill_count_for_filter > len(parsed_query.job_roles):
                # Only apply role filters if we have very specific role detection
                if len(parsed_query.job_roles) == 1:
                    filter_conditions.update(role_filters)
                    logger.info(f"🎯 Applying role filters for skill+role query: {role_filters}")

        # Apply manual overrides if provided (only if they're valid filters, not placeholder strings)
        if role_filter and role_filter.lower() not in ['none', 'null', '', 'string']:
            filter_conditions["role_category"] = role_filter
        if seniority_filter and seniority_filter.lower() not in ['none', 'null', '', 'string']:
            filter_conditions["seniority"] = seniority_filter

        reported_filters = dict(filter_conditions)
        if experience_min_filter_value is not None:
            reported_filters['min_experience_years'] = experience_min_filter_value
        if experience_max_filter_value is not None:
            reported_filters['max_experience_years'] = experience_max_filter_value

        logger.info(
            "🎯 Search filters (Qdrant): %s | Python experience filter: min=%s, max=%s",
            filter_conditions,
            experience_min_filter_value,
            experience_max_filter_value
        )
        logger.info(f"📊 Detected roles: {parsed_query.job_roles}")
        logger.info(f"🔧 Effective skills used for search: {effective_skills[:10]}")
        logger.info(f"🗺️ Detected location: {parsed_query.location}")
        skill_focused = effective_skill_count > len(parsed_query.job_roles)
        logger.info(f"💡 Search strategy: {'skill-focused' if skill_focused else 'role-focused'}")

        # Search strategy: For technical queries, cast a wider net
        search_limit_multiplier = 4 if effective_skill_count > 0 else 3

        # Check for unavailable roles using configuration system
        if parsed_query.unavailable_role_info:
            unavailable_info = parsed_query.unavailable_role_info
            return {
                "query": query,
                "parsed_query": {
                    "detected_roles": [],
                    "detected_skills": parsed_query.skills,
                    "role_inferred_skills": parsed_query.role_inferred_skills,
                    "forced_keywords": forced_keywords,
                    "required_degrees": required_degrees,
                    "required_institutions": required_institutions,
                    "experience_years": parsed_query.experience_years,
                    "min_experience_years": parsed_query.min_experience_years,
                    "max_experience_years": parsed_query.max_experience_years,
                    "seniority_level": parsed_query.seniority_level,
                    "location": parsed_query.location,
                    "intent": parsed_query.intent
                },
                "search_strategy": {
                    "type": "unavailable_role",
                    "role": unavailable_info.get("role"),
                    "message": unavailable_info.get("message"),
                    "suggestions": unavailable_info.get("suggestions", []),
                    "available_roles": search_processor.job_db.get_database_roles()
                },
                "total_results": 0,
                "results": []
            }

        # Check if this is a generic candidate search - allow these to proceed for general browsing
        generic_keywords = ['candidate', 'candidates', 'people', 'person', 'resume', 'resumes', 'talent', 'profile', 'profiles']
        has_generic_keywords = any(word in query.lower() for word in generic_keywords)
        has_only_generic_keywords = (
            parsed_query.keywords and
            all(keyword.lower() in generic_keywords for keyword in parsed_query.keywords)
        )

        is_generic_search = (
            not parsed_query.job_roles and
            not parsed_query.skills and
            not parsed_query.location and
            not parsed_query.companies and
            not forced_keywords and
            not required_degrees and
            not required_institutions and
            (not parsed_query.keywords or has_only_generic_keywords) and
            parsed_query.intent in ['find', 'search', 'hire', 'general'] and
            has_generic_keywords
        )


        # Only search if we have proper criteria OR it's a generic candidate search
        if (not parsed_query.job_roles and not parsed_query.skills and not parsed_query.location and not parsed_query.companies and
                not forced_keywords and not required_degrees and not required_institutions and
                (not parsed_query.keywords or has_only_generic_keywords) and not is_generic_search):
            logger.info("🚫 No roles, skills, location, companies, or keywords detected - returning empty results")
            return {
                "query": query,
                "parsed_query": {
                    "detected_roles": [],
                    "detected_skills": [],
                    "role_inferred_skills": parsed_query.role_inferred_skills,
                    "forced_keywords": forced_keywords,
                    "required_degrees": required_degrees,
                    "required_institutions": required_institutions,
                    "experience_years": parsed_query.experience_years,
                    "min_experience_years": parsed_query.min_experience_years,
                    "max_experience_years": parsed_query.max_experience_years,
                    "seniority_level": parsed_query.seniority_level,
                    "location": parsed_query.location,
                    "intent": parsed_query.intent
                },
                "search_strategy": {
                    "type": "no_match",
                    "message": "No specific roles or skills detected in query",
                    "available_roles": search_processor.job_db.get_database_roles()[:20],  # Show first 20 roles as examples
                    "suggestion": "Please specify a role (e.g., 'Software Engineer', 'Marketing Manager') or skills (e.g., 'Python developer')"
                },
                "total_results": 0,
                "results": []
            }

        # Hybrid approach: First try with strict role filters, then fall back to semantic search with re-ranking
        results = []

        # Step 1: Try strict role filtering if we have role detection
        if filter_conditions:
            logger.info(f"🎯 Step 1: Searching with strict role/skill filters: {filter_conditions}")
            results = await qdrant_client.search_similar(
                query_vector=query_vector,
                limit=1000,  # Get ALL candidates for comprehensive matching
                filter_conditions=filter_conditions
            )
            logger.info(f"🔍 Found {len(results)} results with strict filtering")

        # Step 2: If no results with strict filters, try semantic search without role filters but with re-ranking
        if len(results) == 0 and parsed_query.job_roles:
            logger.info("🔄 No results with strict filters, trying semantic search with re-ranking")
            # Get semantic results without strict role filters
            semantic_results = await qdrant_client.search_similar(
                query_vector=query_vector,
                limit=1000,  # Get ALL candidates for comprehensive re-ranking
                filter_conditions=None  # No filters - pure semantic matching
            )

            if len(semantic_results) > 0:
                # Re-rank based on comprehensive relevance and role matching
                logger.info(f"🎯 Re-ranking {len(semantic_results)} semantic results for role relevance")
                has_location_filter = 'location' in filter_conditions
                results = _rerank_with_role_matching(semantic_results, parsed_query, search_processor, has_location_filter)

        # Step 2.5: If no results and only location was provided, do location-only search
        if len(results) == 0 and parsed_query.location and not parsed_query.job_roles and not parsed_query.skills:
            logger.info(f"🗺️ Doing location-only search for: {parsed_query.location}")
            # Get all candidates and filter by location only
            location_results = await qdrant_client.search_similar(
                query_vector=query_vector,
                limit=1000,  # Get ALL candidates for location filtering
                filter_conditions=None  # No role/skill filters, just semantic matching
            )

            # STRICT location filtering for location-only searches
            location_query = parsed_query.location.lower()
            query_city = location_query.split(',')[0].strip()

            strictly_filtered_results = []
            for result in location_results:
                payload = result.get('payload', {})
                candidate_location = payload.get('location', '').lower()

                if candidate_location:
                    candidate_city = candidate_location.split(',')[0].strip()

                    # Strict matching: exact city match or contains the city name
                    location_match = (
                        location_query in candidate_location or
                        candidate_location in location_query or
                        query_city == candidate_city or
                        (len(query_city) > 3 and query_city in candidate_city) or
                        (len(candidate_city) > 3 and candidate_city in query_city)
                    )

                    if location_match:
                        strictly_filtered_results.append(result)
                        logger.info(f"✅ Location match: '{candidate_location}' matches query '{location_query}'")
                    else:
                        logger.info(f"❌ Location rejected: '{candidate_location}' doesn't match query '{location_query}'")

            results = strictly_filtered_results
            logger.info(f"🗺️ After strict location filtering: {len(results)} candidates from {parsed_query.location}")

        # Step 3: Word-based fallback search using keywords/companies
        if len(results) == 0 and (parsed_query.companies or parsed_query.keywords):
            logger.info(f"🔍 No specific intent found - performing word-based search using companies: {parsed_query.companies} and keywords: {parsed_query.keywords}")

            # Create search terms from companies and keywords
            search_terms = []
            if parsed_query.companies:
                search_terms.extend(parsed_query.companies)
            if parsed_query.keywords:
                search_terms.extend(parsed_query.keywords)
            if forced_keywords:
                search_terms.extend(forced_keywords)
            if required_degrees:
                search_terms.extend(required_degrees)
            if required_institutions:
                search_terms.extend(required_institutions)

            # Use these terms for semantic search
            word_search_query = " ".join(search_terms)
            logger.info(f"🔍 Word-based search query: '{word_search_query}'")

            word_query_vector = await _generate_embedding(word_search_query)
            if word_query_vector:
                word_results = await qdrant_client.search_similar(
                    query_vector=word_query_vector,
                    limit=1000,  # Get ALL candidates for word-based matching
                    filter_conditions=None
                )

                # Filter results based on word matches in work history and companies
                matched_results = []
                for result in word_results:
                    payload = result.get('payload', {})
                    candidate_name = payload.get('name', 'Unknown')

                    company_match, matched_company, matched_query_company = _find_company_match(
                        payload,
                        parsed_query.companies
                    )
                    if company_match and matched_company and matched_query_company:
                        logger.info(
                            "🎯 Company match found: %s worked at '%s' (matches '%s')",
                            candidate_name,
                            matched_company,
                            matched_query_company
                        )

                    # Check for keyword matches in summary, skills, current position
                    summary = payload.get('summary', '').lower()
                    current_position = payload.get('current_position', '').lower()
                    skills = [s.lower() for s in payload.get('skills', [])]
                    keyword_match = False

                    for keyword in parsed_query.keywords:
                        keyword_lower = keyword.lower()
                        if (keyword_lower in summary or
                            keyword_lower in current_position or
                            any(keyword_lower in skill for skill in skills)):
                            keyword_match = True
                            logger.info(f"🎯 Keyword match found: {candidate_name} matches keyword '{keyword}'")
                            break

                    education_match = _candidate_matches_education(
                        payload,
                        required_degrees,
                        required_institutions
                    )

                    forced_match = _candidate_matches_forced_keywords(payload, forced_keywords)

                    include_candidate = False
                    if parsed_query.companies:
                        if company_match and forced_match and education_match:
                            include_candidate = True
                    else:
                        if (keyword_match or company_match) and forced_match and education_match:
                            include_candidate = True

                    if include_candidate:
                        matched_results.append(result)

                results = matched_results
                logger.info(f"🔍 Word-based search found {len(results)} matching candidates")

        # Step 3.5: Generic candidate search for broad queries like "i need candidates"
        if len(results) == 0 and is_generic_search:
            logger.info(f"👥 Performing generic candidate search for query: '{query}'")

            # For generic searches, return a diverse set of candidates ranked by relevance
            generic_results = await qdrant_client.search_similar(
                query_vector=query_vector,
                limit=1000,  # Get ALL candidates for generic browsing
                filter_conditions=None  # No filters - pure semantic matching
            )

            if generic_results:
                # Sort by similarity score and diversify by role categories
                seen_roles = set()
                diverse_results = []

                for result in generic_results:
                    payload = result.get('payload', {})
                    role_category = payload.get('role_category', 'Unknown')

                    # Limit results per role category to ensure diversity
                    role_count = sum(1 for r in diverse_results if r.get('payload', {}).get('role_category') == role_category)
                    if role_count < 3:  # Max 3 candidates per role category
                        diverse_results.append(result)
                        seen_roles.add(role_category)

                    # Stop once we have enough diverse candidates
                    if len(diverse_results) >= 30:  # Return top 30 diverse candidates
                        break

                results = diverse_results
                logger.info(f"👥 Generic search found {len(results)} diverse candidates across {len(seen_roles)} role categories")

        # Step 4: If still no results, return informative message
        if len(results) == 0:
            logger.info("🚫 No relevant candidates found - returning empty results")
            detected_role_str = ", ".join(parsed_query.job_roles) if parsed_query.job_roles else "Unknown"

            response_payload = {
                "query": query,
                "parsed_query": {
                    "detected_roles": parsed_query.job_roles,
                    "detected_skills": parsed_query.skills,
                    "role_inferred_skills": parsed_query.role_inferred_skills,
                    "forced_keywords": forced_keywords,
                    "required_degrees": required_degrees,
                    "required_institutions": required_institutions,
                    "experience_years": parsed_query.experience_years,
                    "min_experience_years": parsed_query.min_experience_years,
                    "max_experience_years": parsed_query.max_experience_years,
                    "seniority_level": parsed_query.seniority_level,
                    "location": parsed_query.location,
                    "companies": parsed_query.companies,
                    "keywords": parsed_query.keywords,
                    "intent": parsed_query.intent
                },
                "search_strategy": {
                    "type": "no_candidates_found",
                    "message": f"No {detected_role_str} candidates found in database",
                    "filters_attempted": reported_filters if reported_filters else {},
                    "semantic_search_attempted": True,
                    "available_roles": search_processor.job_db.get_database_roles()[:10],
                    "forced_keyword_sample": forced_keywords[:10] if forced_keywords else [],
                    "required_degree_sample": required_degrees[:5] if required_degrees else [],
                    "required_institution_sample": required_institutions[:5] if required_institutions else [],
                    "effective_skill_sample": effective_skills[:10] if effective_skills else []
                },
                "total_results": 0,
                "results": []
            }

            if experience_min_filter_value is not None or experience_max_filter_value is not None:
                experience_requirement = {"messages": []}
                if experience_min_filter_value is not None:
                    experience_requirement["minimum_years"] = experience_min_filter_value
                    experience_requirement["messages"].append(
                        f"Filtered out candidates with less than {experience_min_filter_value} years experience"
                    )
                if experience_max_filter_value is not None:
                    experience_requirement["maximum_years"] = experience_max_filter_value
                    experience_requirement["messages"].append(
                        f"Filtered out candidates with more than {experience_max_filter_value} years experience"
                    )
                response_payload["search_strategy"]["experience_requirement"] = experience_requirement

            return response_payload

        logger.info(f"✅ Found {len(results)} total results for final ranking")

        experience_filtered_out = 0

        # Apply Python-based location filtering if location filter was requested
        filtered_results = results
        if 'location' in filter_conditions:
            location_query = filter_conditions['location'].lower()
            logger.info(f"🗺️ Applying Python location filter for: {location_query}")

            # Check if this is a location-only search (no roles or skills specified)
            is_location_only = not parsed_query.job_roles and not parsed_query.skills
            logger.info(f"🗺️ Location-only search: {is_location_only}")

            filtered_results = []
            for result in results:
                payload = result.get('payload', {})
                candidate_name = payload.get('name', 'NO_NAME')
                candidate_location = payload.get('location', '').lower()

                logger.info(f"🗺️ Checking {candidate_name}: query='{location_query}' vs candidate='{candidate_location}'")

                # Check if location query matches candidate location
                # Extract city names for comparison
                query_city = location_query.split(',')[0].strip().lower()
                candidate_city = candidate_location.split(',')[0].strip().lower()

                # For location-only searches, be STRICT - only show candidates from that location
                if is_location_only:
                    # STRICT matching for location-only searches
                    location_match = (
                        query_city == candidate_city or
                        (len(query_city) > 3 and query_city in candidate_city) or
                        (len(candidate_city) > 3 and candidate_city in query_city)
                    )
                else:
                    # More flexible matching for role+location searches
                    location_match = (
                        location_query in candidate_location or
                        candidate_location in location_query or
                        query_city == candidate_city or
                        (len(query_city) > 3 and query_city in candidate_city) or
                        (len(candidate_city) > 3 and candidate_city in query_city)
                    )

                if location_match:
                    filtered_results.append(result)
                    logger.info(f"🗺️ ✅ Location match: '{location_query}' matches '{candidate_location}'")
                else:
                    logger.info(f"🗺️ ❌ Location rejected: '{location_query}' vs '{candidate_location}'")

            logger.info(f"🗺️ Location filtering: {len(results)} -> {len(filtered_results)} results")

        if parsed_query.companies:
            pre_company_count = len(filtered_results)
            filtered_results = [
                result for result in filtered_results
                if _candidate_matches_company(result.get('payload', {}), parsed_query.companies)
            ]
            logger.info(
                "🏢 Company filtering: %s -> %s results for companies %s",
                pre_company_count,
                len(filtered_results),
                parsed_query.companies
            )

        # Apply experience filtering in Python to support legacy payloads without indexed fields
        if experience_min_filter_value is not None or experience_max_filter_value is not None:
            pre_filter_count = len(filtered_results)
            filtered_results = [
                result for result in filtered_results
                if _candidate_meets_experience_requirement(
                    result.get('payload', {}),
                    experience_min_filter_value,
                    experience_max_filter_value
                )
            ]
            experience_filtered_out = pre_filter_count - len(filtered_results)
            logger.info(
                "⏳ Experience filtering: %s -> %s candidates with requirements min=%s, max=%s",
                pre_filter_count,
                len(filtered_results),
                experience_min_filter_value,
                experience_max_filter_value
            )

        if forced_keywords:
            pre_forced_count = len(filtered_results)
            filtered_results = [
                result for result in filtered_results
                if _candidate_matches_forced_keywords(result.get('payload', {}), forced_keywords)
            ]
            logger.info(
                "🔑 Forced keyword filtering: %s -> %s candidates for terms %s",
                pre_forced_count,
                len(filtered_results),
                forced_keywords
            )

        if required_degrees or required_institutions:
            pre_edu_count = len(filtered_results)
            filtered_results = [
                result for result in filtered_results
                if _candidate_matches_education(result.get('payload', {}), required_degrees, required_institutions)
            ]
            logger.info(
                "🎓 Education filtering: %s -> %s candidates for degrees %s institutions %s",
                pre_edu_count,
                len(filtered_results),
                required_degrees,
                required_institutions
            )

        # Remove duplicates based on name+email combination (keep the best score for each person)
        unique_results = {}
        for result in filtered_results:
            payload = result.get('payload', {})
            # Create unique key from name and email
            unique_key = f"{payload.get('name', 'Unknown')}_{payload.get('email', 'Unknown')}"
            if unique_key not in unique_results or result.get('score', 0) > unique_results[unique_key].get('score', 0):
                unique_results[unique_key] = result

        deduplicated_results = list(unique_results.values())

        # Post-process results with comprehensive ranking using ALL available data
        ranked_results = _rank_search_results(deduplicated_results, parsed_query)

        # Limit to requested number
        limited_results = ranked_results[:limit]

        # Format results with comprehensive information and detailed selection reasons
        formatted_results = _format_search_results(limited_results, parsed_query)

        # CRITICAL: Filter out candidates with 0.0 comprehensive scores (they don't meet requirements)
        # This is essential for AI/ML queries where candidates must have specific skills
        qualified_results = []
        for result in formatted_results:
            comprehensive_score = result.get('comprehensive_score', 0.0)
            candidate_name = result.get('name', 'Unknown')

            # Only include candidates with meaningful comprehensive scores
            if comprehensive_score > 0.0:
                qualified_results.append(result)
                logger.info(f"✅ Including qualified candidate: {candidate_name} (comprehensive_score: {comprehensive_score})")
            else:
                logger.info(f"❌ Filtering out unqualified candidate: {candidate_name} (comprehensive_score: {comprehensive_score})")

        # Use qualified results instead of all formatted results
        formatted_results = qualified_results

        response_payload = {
            "query": query,
            "parsed_query": {
                "detected_roles": parsed_query.job_roles,
                "detected_skills": parsed_query.skills,
                "role_inferred_skills": parsed_query.role_inferred_skills,
                "forced_keywords": forced_keywords,
                "required_degrees": required_degrees,
                "required_institutions": required_institutions,
                "experience_years": parsed_query.experience_years,
                "min_experience_years": parsed_query.min_experience_years,
                "max_experience_years": parsed_query.max_experience_years,
                "seniority_level": parsed_query.seniority_level,
                "location": parsed_query.location,
                "companies": parsed_query.companies,
                "keywords": parsed_query.keywords,
                "intent": parsed_query.intent
            },
            "search_strategy": {
                "type": "generic-candidate-search" if is_generic_search else ("skill-focused" if skill_focused else "role-focused"),
                "filters_applied": reported_filters,
                "total_candidates_analyzed": len(results),
                "final_results_returned": len(formatted_results),
                "effective_skill_sample": effective_skills[:10] if effective_skills else [],
                "forced_keyword_sample": forced_keywords[:10] if forced_keywords else [],
                "required_degree_sample": required_degrees[:5] if required_degrees else [],
                "required_institution_sample": required_institutions[:5] if required_institutions else [],
                "search_approach": "Generic candidate browsing with role diversity" if is_generic_search else None
            },
            "total_results": len(formatted_results),
            "results": formatted_results
        }

        if experience_min_filter_value is not None or experience_max_filter_value is not None:
            experience_requirement = {
                "filtered_out_candidates": max(experience_filtered_out, 0),
                "messages": []
            }
            if experience_min_filter_value is not None:
                experience_requirement["minimum_years"] = experience_min_filter_value
                experience_requirement["messages"].append(
                    f"Applied minimum experience requirement of {experience_min_filter_value} years"
                )
            if experience_max_filter_value is not None:
                experience_requirement["maximum_years"] = experience_max_filter_value
                experience_requirement["messages"].append(
                    f"Applied maximum experience requirement of {experience_max_filter_value} years"
                )
            if not experience_requirement["messages"]:
                experience_requirement.pop("messages")
            response_payload["search_strategy"]["experience_requirement"] = experience_requirement

        return response_payload

    except Exception as e:
        logger.error(f"Error in resume search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _rerank_with_role_matching(results: List[Dict], parsed_query, search_processor, has_location_filter: bool = False) -> List[Dict]:
    """Re-rank semantic search results based on role relevance and comprehensive scoring."""
    if not results:
        return results

    scored_results = []

    for result in results:
        payload = result.get('payload', {})
        base_semantic_score = result.get('score', 0.0)

        # Calculate role relevance score
        role_relevance_score = 0.0
        current_position = payload.get('current_position', '').lower()
        role_category = payload.get('role_category', '').lower()

        # Check if candidate's role matches any of the query roles
        for query_role in parsed_query.job_roles:
            query_role_lower = query_role.lower()

            # Direct role category match (highest score)
            if query_role_lower in role_category:
                role_relevance_score = max(role_relevance_score, 1.0)
            # Current position match (high score)
            elif query_role_lower in current_position:
                role_relevance_score = max(role_relevance_score, 0.8)
            # Partial match in position or category (medium score)
            elif any(word in current_position or word in role_category for word in query_role_lower.split()):
                role_relevance_score = max(role_relevance_score, 0.6)

        # Calculate comprehensive relevance score
        comprehensive_score = search_processor.calculate_comprehensive_relevance_score(payload, parsed_query)

        # Combined scoring: 40% semantic + 30% role relevance + 30% comprehensive
        final_score = (base_semantic_score * 0.4) + (role_relevance_score * 0.3) + (comprehensive_score * 0.3)

        # STRICT COMPANY FILTERING: If companies are specified, ONLY include candidates from those companies
        if parsed_query.companies:
            company_match_found, _, _ = _find_company_match(payload, parsed_query.companies)
            if not company_match_found:
                continue  # Skip this candidate entirely

        # Filter thresholds: be more inclusive when location filtering is involved
        if has_location_filter:
            # When location filtering, include more diverse roles since location is the primary filter
            if role_relevance_score > 0.2 or base_semantic_score > 0.3:
                result['rerank_score'] = final_score
                result['role_relevance'] = role_relevance_score
                scored_results.append(result)
        else:
            # Standard filtering: only keep results with some role relevance (> 0.5) or very high semantic similarity (> 0.7)
            if role_relevance_score > 0.5 or base_semantic_score > 0.7:
                result['rerank_score'] = final_score
                result['role_relevance'] = role_relevance_score
                scored_results.append(result)

    # Sort by combined score
    scored_results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

    return scored_results


def _rank_search_results(results: List[Dict], parsed_query) -> List[Dict]:
    """Rank search results based on comprehensive query relevance using ALL available data."""
    if not results:
        return results

    from src.resume_parser.utils.search_intelligence import search_processor

    for result in results:
        base_score = result.get('score', 0.0)
        payload = result.get('payload', {})

        # Calculate comprehensive relevance score using ALL data
        comprehensive_score = search_processor.calculate_comprehensive_relevance_score(payload, parsed_query)

        # Combine semantic similarity (base_score) with comprehensive relevance
        # 60% comprehensive relevance + 40% semantic similarity
        final_score = (comprehensive_score * 0.6) + (base_score * 0.4)

        # Additional specific boosts for key matches
        additional_boost = 0.0

        # Boost for exact role matches in ANY field
        role_category = payload.get('role_category', '').lower()
        current_position = payload.get('current_position', '').lower()
        best_role = payload.get('best_role', '').lower()

        for detected_role in parsed_query.job_roles:
            role_term = detected_role.replace('_', ' ')
            if (role_term in role_category or role_term in current_position or role_term in best_role):
                additional_boost += 0.1
                break

        # Boost for work history title matches
        work_history = payload.get('work_history', [])
        for job in work_history:
            if isinstance(job, dict):
                job_title = job.get('title', '').lower()
                for detected_role in parsed_query.job_roles:
                    if detected_role.replace('_', ' ') in job_title:
                        additional_boost += 0.05
                        break

        # Boost for skill density (more skills = better match)
        candidate_skills = [s.lower() for s in payload.get('skills', [])]
        query_skills = parsed_query.skills if parsed_query.skills else parsed_query.role_inferred_skills
        matching_skills = len([s for s in query_skills if s.lower() in candidate_skills])
        if matching_skills > 0 and query_skills:
            skill_density = matching_skills / len(query_skills)
            additional_boost += skill_density * 0.2

        # Boost for technology overlap in work history and projects
        tech_matches = 0
        total_tech = 0

        # Check work history technologies
        for job in work_history:
            if isinstance(job, dict):
                technologies = job.get('technologies', [])
                total_tech += len(technologies)
                for tech in technologies:
                    if any(skill.lower() in tech.lower() for skill in query_skills):
                        tech_matches += 1

        # Check project technologies
        projects = payload.get('projects', [])
        for project in projects:
            if isinstance(project, dict):
                technologies = project.get('technologies', [])
                total_tech += len(technologies)
                for tech in technologies:
                    if any(skill.lower() in tech.lower() for skill in query_skills):
                        tech_matches += 1

        if total_tech > 0:
            tech_overlap = tech_matches / total_tech
            additional_boost += tech_overlap * 0.15

        # Boost for seniority match
        if parsed_query.seniority_level:
            candidate_seniority = payload.get('seniority', '').lower()
            if parsed_query.seniority_level in candidate_seniority:
                additional_boost += 0.1

        # Boost for education relevance
        education = payload.get('education', [])
        for edu in education:
            if isinstance(edu, dict):
                field = edu.get('field', '').lower()
                degree = edu.get('degree', '').lower()
                # Check if education field is relevant to query roles
                for role in parsed_query.job_roles:
                    if role.replace('_', ' ') in field or role.replace('_', ' ') in degree:
                        additional_boost += 0.05
                        break

        # MAJOR boost for location matches (especially for location-only searches)
        if parsed_query.location:
            candidate_location = payload.get('location', '').lower()
            location_query = parsed_query.location.lower()

            # Check for location matches with flexible matching
            query_city = location_query.split(',')[0].strip()
            candidate_city = candidate_location.split(',')[0].strip()

            location_match = False
            location_boost_value = 0.0

            # Exact location match
            if location_query in candidate_location or candidate_location in location_query:
                location_match = True
                location_boost_value = 0.3  # Strong boost for exact match
            # City-level match
            elif query_city == candidate_city:
                location_match = True
                location_boost_value = 0.25  # Good boost for city match
            # Partial city match (flexible matching)
            elif (len(query_city) > 3 and query_city in candidate_city) or (len(candidate_city) > 3 and candidate_city in query_city):
                location_match = True
                location_boost_value = 0.2  # Moderate boost for partial match

            if location_match:
                # For location-only searches, give even higher boost
                if not parsed_query.job_roles and not parsed_query.skills:
                    location_boost_value += 0.2  # Extra boost for location-only searches
                additional_boost += location_boost_value
                logger.info(f"🎯 Location match found for {payload.get('name', 'Unknown')}: Query '{location_query}' matches '{candidate_location}' (boost: +{location_boost_value})")

        # MAJOR boost for company matches
        if parsed_query.companies:
            company_match_found, matched_company, matched_query_company = _find_company_match(
                payload,
                parsed_query.companies
            )
            if company_match_found:
                company_boost = 0.5  # HUGE boost for company match
                additional_boost += company_boost
                logger.info(
                    "🎯 COMPANY MATCH FOUND for %s: Worked at '%s' (matches '%s') (boost: +%s)",
                    payload.get('name', 'Unknown'),
                    matched_company,
                    matched_query_company,
                    company_boost
                )
                match_details = result.setdefault('_match_details', {})
                match_details['company'] = {
                    'candidate_company': matched_company,
                    'query_company': matched_query_company
                }

        # Apply final score calculation
        result['adjusted_score'] = final_score + additional_boost
        result['comprehensive_score'] = comprehensive_score
        result['semantic_score'] = base_score
        result['boost_applied'] = additional_boost

    # Sort by adjusted score
    return sorted(results, key=lambda x: x.get('adjusted_score', x.get('score', 0)), reverse=True)


def _format_search_results(results: List[Dict], parsed_query) -> List[Dict]:
    """Format search results with comprehensive information and detailed selection reasons."""
    formatted_results = []

    for result in results:
        payload = result.get('payload', {})

        # Extract comprehensive information
        formatted_result = {
            "name": payload.get('name', 'Unknown'),
            "email": payload.get('email', 'Unknown'),
            "phone": payload.get('phone', 'Unknown'),
            "current_position": payload.get('current_position', 'Unknown'),
            "location": payload.get('location', 'Unknown'),
            "total_experience": payload.get('total_experience', 'Unknown'),
            "match_score": round(result.get('adjusted_score', result.get('score', 0)), 3),
            "semantic_score": round(result.get('semantic_score', 0), 3),
            "comprehensive_score": round(result.get('comprehensive_score', 0), 3)
        }

        # Generate comprehensive and detailed selection reasons
        reasons = []
        detailed_explanations = []

        # Import search processor for detailed analysis
        from src.resume_parser.utils.search_intelligence import search_processor

        match_details = result.setdefault('_match_details', {})
        effective_query_skills = parsed_query.effective_skills()

        # Location match reason
        if parsed_query.location:
            candidate_location = payload.get('location', '').lower()
            query_location = parsed_query.location.lower()
            if query_location in candidate_location or candidate_location in query_location:
                detailed_explanations.append(f"📍 Location match: Candidate is in {payload.get('location', 'Unknown')}, matches query requirement for {parsed_query.location}")

        # Company match reason - PRIORITY
        if parsed_query.companies:
            if match_details.get('company'):
                matched_company = match_details['company'].get('candidate_company')
                matched_query_company = match_details['company'].get('query_company')
            else:
                company_match_found, matched_company, matched_query_company = _find_company_match(
                    payload,
                    parsed_query.companies
                )
                if company_match_found:
                    match_details.setdefault('company', {})
                    match_details['company']['candidate_company'] = matched_company
                    match_details['company']['query_company'] = matched_query_company
            matched_company = match_details.get('company', {}).get('candidate_company')
            matched_query_company = match_details.get('company', {}).get('query_company')

            if matched_company and matched_query_company:
                current_employment = payload.get('current_employment')
                current_title = None
                if isinstance(current_employment, dict) and _company_names_match(
                    matched_query_company,
                    current_employment.get('company', '')
                ):
                    current_title = current_employment.get('position') or payload.get('current_position', 'Current Role')

                if not current_title:
                    for job in payload.get('work_history', []):
                        if isinstance(job, dict) and _company_names_match(matched_query_company, job.get('company', '')):
                            current_title = job.get('title', 'Role')
                            break

                role_phrase = f" as {current_title}" if current_title else ""
                detailed_explanations.insert(
                    0,
                    f"🏢 Company Match: Experience{role_phrase} at {matched_company} aligns with your request for {matched_query_company}"
                )

        # Detailed role matching explanation
        role_explanations = []
        role_category = payload.get('role_category', '')
        current_position = payload.get('current_position', '')

        for detected_role in parsed_query.job_roles:
            # Get the sector/category for this role from JSON config
            from src.resume_parser.utils.roles_config import roles_config
            role_sector = roles_config.get_role_database_category(detected_role)
            role_variations = roles_config.get_role_search_terms(detected_role)

            # Detailed role matching logic
            if role_sector and role_sector.lower() in role_category.lower():
                role_explanations.append(f"🎯 Perfect sector match: Query '{detected_role}' belongs to '{role_sector}' sector, candidate's role category is '{role_category}'")
            elif detected_role.lower() in current_position.lower():
                role_explanations.append(f"🎯 Direct position match: Query role '{detected_role}' found in candidate's current position '{current_position}'")
            elif any(variation.lower() in current_position.lower() for variation in role_variations if variation):
                matching_variation = next(var for var in role_variations if var and var.lower() in current_position.lower())
                role_explanations.append(f"🎯 Role variation match: '{matching_variation}' (variation of {detected_role}) found in position '{current_position}'")
            elif role_category and role_category != 'Unknown':
                # Semantic/related role match
                role_explanations.append(f"🔄 Related field match: Query '{detected_role}' is similar to candidate's '{role_category}' category")

        # Check work history for detailed experience matching
        work_history = payload.get('work_history', [])
        projects = payload.get('projects', [])  # Define projects variable here
        experience_explanations = []

        for i, job in enumerate(work_history[:3]):  # Check top 3 jobs
            if isinstance(job, dict):
                job_title = job.get('title', '').lower()
                company = job.get('company', '')

                for detected_role in parsed_query.job_roles:
                    from src.resume_parser.utils.roles_config import roles_config
                    role_variations = roles_config.get_role_search_terms(detected_role)

                    # Check for direct role match in job title
                    if detected_role.lower() in job_title:
                        experience_explanations.append(f"💼 Direct role experience: Previous role '{job.get('title', '')}' at {company} directly matches query '{detected_role}'")
                        break
                    # Check for role variations in job title
                    elif any(var.lower() in job_title for var in role_variations if var and len(var) > 3):
                        matching_var = next(var for var in role_variations if var and len(var) > 3 and var.lower() in job_title)
                        experience_explanations.append(f"💼 Related role experience: Previous role '{job.get('title', '')}' at {company} contains '{matching_var}', related to query '{detected_role}'")
                        break

        # Combine role explanations
        if role_explanations:
            detailed_explanations.extend(role_explanations[:2])  # Top 2 role matches
        if experience_explanations:
            detailed_explanations.extend(experience_explanations[:1])  # Top 1 experience match

        # Detailed skills analysis
        if effective_query_skills:
            skill_explanations = []
            candidate_skills = [s.lower() for s in payload.get('skills', [])]
            direct_skill_matches = []
            work_skill_matches = []
            project_skill_matches = []

            # Check direct skills matches
            for query_skill in effective_query_skills:
                if query_skill.lower() in candidate_skills:
                    # Find the exact match in original case
                    original_skill = next((s for s in payload.get('skills', []) if s.lower() == query_skill.lower()), query_skill)
                    direct_skill_matches.append(original_skill)

            # Check skills in work history technologies/responsibilities
            for job in work_history[:3]:  # Check top 3 jobs
                if isinstance(job, dict):
                    technologies = job.get('technologies', [])
                    responsibilities = job.get('responsibilities', [])
                    job_title = job.get('title', '')
                    company = job.get('company', '')

                    for query_skill in effective_query_skills:
                        # Check in technologies
                        tech_matches = [tech for tech in technologies if query_skill.lower() in tech.lower()]
                        if tech_matches:
                            work_skill_matches.append(f"{query_skill} used in {job_title} at {company} ({', '.join(tech_matches[:2])})")

                        # Check in responsibilities
                        resp_matches = [resp for resp in responsibilities if query_skill.lower() in resp.lower()]
                        if resp_matches and not tech_matches:  # Avoid duplicate
                            work_skill_matches.append(f"{query_skill} mentioned in {job_title} responsibilities at {company}")

            # Check skills in project technologies
            for project in projects[:3]:  # Check top 3 projects (projects already defined above)
                if isinstance(project, dict):
                    technologies = project.get('technologies', [])
                    project_name = project.get('name', '')

                    for query_skill in effective_query_skills:
                        tech_matches = [tech for tech in technologies if query_skill.lower() in tech.lower()]
                        if tech_matches:
                            project_skill_matches.append(f"{query_skill} used in project '{project_name}' ({', '.join(tech_matches[:2])})")

            # Create detailed skill explanations
            if direct_skill_matches:
                skill_explanations.append(f"🔧 Direct skills match: {', '.join(direct_skill_matches)} listed in candidate's skills")

            if work_skill_matches:
                skill_explanations.append(f"💻 Work experience skills: {work_skill_matches[0]}")  # Show top work skill match

            if project_skill_matches:
                skill_explanations.append(f"🚀 Project skills: {project_skill_matches[0]}")  # Show top project skill match

            if skill_explanations:
                detailed_explanations.extend(skill_explanations[:2])  # Show top 2 skill explanations

        # Experience reason with more detail
        total_experience = payload.get('total_experience', '')
        if total_experience and total_experience != '0 years':
            reasons.append(f"Experience: {total_experience}")

        # Work history highlights
        if work_history and len(work_history) > 0:
            recent_companies = []
            for job in work_history[:3]:  # Top 3 recent jobs
                if isinstance(job, dict):
                    company = job.get('company', '')
                    title = job.get('title', '')
                    if company and title:
                        recent_companies.append(f"{title} at {company}")

            if recent_companies:
                formatted_result["recent_experience"] = recent_companies[:2]  # Show top 2

        # Project highlights
        if projects and len(projects) > 0:
            project_highlights = []
            for project in projects[:2]:  # Top 2 projects
                if isinstance(project, dict):
                    name = project.get('name', '')
                    technologies = project.get('technologies', [])
                    if name:
                        proj_summary = name
                        if technologies:
                            proj_summary += f" ({', '.join(technologies[:3])})"
                        project_highlights.append(proj_summary)

            if project_highlights:
                formatted_result["key_projects"] = project_highlights

        # Education highlights
        education = payload.get('education', [])
        if education and len(education) > 0:
            edu_highlights = []
            for edu in education[:2]:  # Top 2 education entries
                if isinstance(edu, dict):
                    degree = edu.get('degree', '')
                    field = edu.get('field', '')
                    institution = edu.get('institution', '')
                    edu_summary = f"{degree}"
                    if field:
                        edu_summary += f" in {field}"
                    if institution:
                        edu_summary += f" from {institution}"
                    edu_highlights.append(edu_summary)

            if edu_highlights:
                formatted_result["education"] = edu_highlights
                # Add education relevance to reasons
                for edu in education:
                    if isinstance(edu, dict):
                        field = edu.get('field', '').lower()
                        degree = edu.get('degree', '').lower()
                        for role in parsed_query.job_roles:
                            if role.replace('_', ' ') in field or role.replace('_', ' ') in degree:
                                reasons.append(f"Relevant education: {edu.get('degree', '')} in {edu.get('field', '')}")
                                break
                        break

        # Seniority match reason
        if parsed_query.seniority_level:
            candidate_seniority = payload.get('seniority', '')
            if candidate_seniority and parsed_query.seniority_level.lower() in candidate_seniority.lower():
                reasons.append(f"Seniority: {candidate_seniority}")

        # Add comprehensive summary
        if payload.get('summary'):
            formatted_result["summary"] = payload.get('summary', '')[:200] + "..." if len(payload.get('summary', '')) > 200 else payload.get('summary', '')

        # Generate specific reasons based on actual candidate data - NO BULLSHIT GENERIC MESSAGES
        if not reasons:
            specific_reasons = []

            # Check role/position alignment
            candidate_role = payload.get('role_category', '')
            current_position = payload.get('current_position', '')

            if candidate_role:
                specific_reasons.append(f"Role category: {candidate_role}")

            if current_position and current_position != 'Unknown':
                specific_reasons.append(f"Current role: {current_position}")

            # Check semantic similarity
            semantic_score = result.get('semantic_score', 0)
            if semantic_score > 0.3:
                specific_reasons.append(f"High semantic similarity ({semantic_score:.2f})")
            elif semantic_score > 0.1:
                specific_reasons.append(f"Moderate semantic similarity ({semantic_score:.2f})")

            # Check experience level
            total_exp = payload.get('total_experience', '')
            if total_exp and total_exp != '0 years' and total_exp != 'Unknown':
                specific_reasons.append(f"Has {total_exp} experience")

            # Check education relevance
            education = payload.get('education', [])
            if education:
                for edu in education[:1]:  # Just first education
                    if isinstance(edu, dict):
                        degree = edu.get('degree', '')
                        field = edu.get('field', '')
                        if degree:
                            edu_text = degree
                            if field and field != degree:
                                edu_text += f" in {field}"
                            specific_reasons.append(f"Education: {edu_text}")
                            break

            # Check location if relevant
            if parsed_query.location:
                candidate_location = payload.get('location', '')
                if candidate_location and candidate_location != 'Unknown':
                    specific_reasons.append(f"Located in {candidate_location}")

            # Check skills count
            skills = payload.get('skills', [])
            if skills and len(skills) > 0:
                specific_reasons.append(f"Has {len(skills)} listed skills")

            # Use specific reasons or explain why included
            if specific_reasons:
                reasons.extend(specific_reasons[:3])  # Max 3 specific reasons
            else:
                # Last resort - explain search methodology
                reasons.append(f"Included due to vector similarity (score: {result.get('semantic_score', 0):.3f})")

        # Use detailed explanations if available, otherwise fall back to old reasons
        if detailed_explanations:
            # Use detailed explanations with emojis and specific matching logic
            formatted_result["selection_reason"] = " | ".join(detailed_explanations[:4])
        elif reasons:
            formatted_result["selection_reason"] = " | ".join(reasons)
        else:
            # Fallback with honest semantic score explanation
            semantic_score = result.get('semantic_score', 0)
            if semantic_score > 0.3:
                formatted_result["selection_reason"] = f"🧠 Semantic similarity match ({semantic_score:.3f}) - Resume content aligns with search query"
            else:
                formatted_result["selection_reason"] = f"📋 Low confidence match based on vector similarity (score: {semantic_score:.3f})"
        formatted_results.append(formatted_result)

    return formatted_results


@app.post("/search-resumes-advanced")
async def search_resumes_advanced(
    query: str,
    limit: int = 10,
    must_have_roles: Optional[List[str]] = None,
    preferred_skills: Optional[List[str]] = None,
    min_experience_years: Optional[int] = None,
    location: Optional[str] = None,
    include_analysis: bool = True
):
    """
    Advanced resume search with explicit filters and detailed analysis.

    Args:
        query: Natural language search query
        limit: Maximum number of results
        must_have_roles: Required role categories
        preferred_skills: Skills to prioritize in ranking
        min_experience_years: Minimum years of experience
        location: Location requirement
        include_analysis: Include detailed search analysis
    """
    try:
        from src.resume_parser.utils.search_intelligence import search_processor

        logger.info(f"🔍 Advanced search query: {query}")

        # Parse the query
        parsed_query = search_processor.parse_query(query)

        # Override with explicit filters
        if must_have_roles:
            parsed_query.job_roles = must_have_roles
            try:
                role_skills = search_processor._aggregate_role_skills(parsed_query.job_roles)
            except AttributeError:
                role_skills = []
            parsed_query.role_inferred_skills = role_skills
        if preferred_skills:
            existing_skills = {skill.lower() for skill in parsed_query.skills}
            for skill in preferred_skills:
                if skill and isinstance(skill, str):
                    skill_clean = skill.strip()
                    if skill_clean and skill_clean.lower() not in existing_skills:
                        parsed_query.skills.append(skill_clean)
                        existing_skills.add(skill_clean.lower())
        if min_experience_years:
            parsed_query.experience_years = min_experience_years
            parsed_query.min_experience_years = min_experience_years
        if location:
            parsed_query.location = location

        effective_skills = parsed_query.effective_skills()
        forced_keywords = parsed_query.forced_keywords

        if forced_keywords:
            logger.info(f"🔑 Forced keywords (advanced search): {forced_keywords[:5]}")

        # Generate enhanced search text
        search_components: List[str] = []
        seen_components: Set[str] = set()

        def add_component(text: Optional[str]):
            if not text or not isinstance(text, str):
                return
            cleaned = text.strip()
            key = cleaned.lower()
            if cleaned and key not in seen_components:
                search_components.append(cleaned)
                seen_components.add(key)

        add_component(query)
        for role in parsed_query.job_roles:
            role_titles = search_processor.job_db.get_role_titles(role)
            for title in role_titles[:5]:
                add_component(title)

        for skill in parsed_query.skills:
            add_component(skill)

        for skill in parsed_query.role_inferred_skills[:40]:
            add_component(skill)

        for keyword in forced_keywords:
            add_component(keyword)

        search_text = ' '.join(search_components) if search_components else query
        logger.info(f"🔧 Advanced search effective skills sample: {effective_skills[:10]}")
        logger.info(f"🔮 Advanced search components count: {len(search_components)} | sample: {search_components[:10]}")

        # Create embedding
        from src.resume_parser.clients.azure_openai import azure_client
        async_client = azure_client.get_async_client()
        response = await async_client.embeddings.create(
            model=azure_client.get_embedding_deployment(),
            input=search_text
        )
        query_vector = response.data[0].embedding

        # Create advanced filters
        filter_conditions = {}
        if must_have_roles:
            # Map role names to categories stored in Qdrant
            role_mappings = {
                'hr': ['HR Manager', 'Human Resources', 'People Manager', 'Talent Acquisition'],
                'software': ['Software Engineer', 'Developer', 'Programmer', 'Full Stack'],
                'data': ['Data Scientist', 'Data Analyst', 'ML Engineer', 'Analytics'],
                'product': ['Product Manager', 'Product Owner', 'Product Marketing'],
                'marketing': ['Marketing Manager', 'Digital Marketing', 'Brand Manager'],
                'sales': ['Sales Manager', 'Account Manager', 'Business Development']
            }

            categories = []
            for role in must_have_roles:
                role_lower = role.lower()
                for key, values in role_mappings.items():
                    if key in role_lower or any(v.lower() in role_lower for v in values):
                        categories.extend(values)

            if categories:
                filter_conditions['role_category'] = categories

        # Search with filters
        results = await qdrant_client.search_similar(
            query_vector=query_vector,
            limit=limit * 3,  # Get more for better filtering
            filter_conditions=filter_conditions if filter_conditions else None
        )

        # Advanced ranking
        ranked_results = _advanced_rank_results(results, parsed_query, preferred_skills)

        if forced_keywords:
            pre_forced_count = len(ranked_results)
            ranked_results = [
                result for result in ranked_results
                if _candidate_matches_forced_keywords(result.get('payload', {}), forced_keywords)
            ]
            logger.info(
                "🔑 Advanced forced keyword filtering: %s -> %s results for terms %s",
                pre_forced_count,
                len(ranked_results),
                forced_keywords
            )

        final_results = ranked_results[:limit]

        response_data = {
            "query": query,
            "total_results": len(final_results),
            "results": final_results
        }

        if include_analysis:
            response_data["search_analysis"] = {
                "parsed_query": {
                    "detected_roles": parsed_query.job_roles,
                    "detected_skills": parsed_query.skills,
                    "role_inferred_skills": parsed_query.role_inferred_skills,
                    "forced_keywords": forced_keywords,
                    "required_degrees": parsed_query.required_degrees,
                    "required_institutions": parsed_query.required_institutions,
                    "experience_years": parsed_query.experience_years,
                    "seniority_level": parsed_query.seniority_level,
                    "location": parsed_query.location,
                    "companies": parsed_query.companies,
                    "keywords": parsed_query.keywords,
                    "intent": parsed_query.intent
                },
                "applied_filters": filter_conditions,
                "ranking_factors": {
                    "role_match_boost": 0.1,
                    "skill_match_boost": 0.05,
                    "experience_match_boost": 0.15,
                    "seniority_match_boost": 0.15
                },
                "effective_skill_sample": effective_skills[:10] if effective_skills else [],
                "forced_keyword_sample": forced_keywords[:10] if forced_keywords else []
            }

        return response_data

    except Exception as e:
        logger.error(f"Error in advanced resume search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _advanced_rank_results(results: List[Dict], parsed_query, preferred_skills: Optional[List[str]] = None) -> List[Dict]:
    """Advanced ranking with additional factors."""
    if not results:
        return results

    for result in results:
        score = result.get('score', 0.0)
        payload = result.get('payload', {})

        # Role matching (higher weight)
        role_category = payload.get('role_category', '').lower()
        for detected_role in parsed_query.job_roles:
            if detected_role.replace('_', ' ') in role_category:
                score += 0.2  # Higher boost for role match

        # Skill matching
        candidate_skills = [s.lower() for s in payload.get('skills', [])]
        effective_query_skills = parsed_query.skills if parsed_query.skills else parsed_query.role_inferred_skills

        # Regular skill matches
        matching_skills = len([s for s in effective_query_skills if s.lower() in candidate_skills])
        score += matching_skills * 0.05

        # Preferred skill matches (higher weight)
        if preferred_skills:
            preferred_matches = len([s for s in preferred_skills if s.lower() in candidate_skills])
            score += preferred_matches * 0.1

        # Experience matching
        if parsed_query.experience_years:
            candidate_experience = payload.get('total_experience', '0 years')
            # Extract years from string like "5 years 3 months"
            exp_match = re.search(r'(\d+)', candidate_experience)
            if exp_match:
                candidate_years = int(exp_match.group(1))
                if candidate_years >= parsed_query.experience_years:
                    score += 0.15
                elif candidate_years >= parsed_query.experience_years * 0.8:  # Close match
                    score += 0.1

        # Seniority matching
        if parsed_query.seniority_level:
            candidate_seniority = payload.get('seniority', '').lower()
            if parsed_query.seniority_level in candidate_seniority:
                score += 0.15

        result['final_score'] = score

    return sorted(results, key=lambda x: x.get('final_score', x.get('score', 0)), reverse=True)


@app.get("/resume/{user_id}")
async def get_resume(user_id: str):
    """Get resume data by user ID."""
    try:
        resume_data = await qdrant_client.get_resume_by_id(user_id)
        if not resume_data:
            raise HTTPException(status_code=404, detail="Resume not found")

        return {
            "user_id": user_id,
            "resume_data": resume_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving resume {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app.debug,
        log_level=settings.app.log_level.lower()
    )
