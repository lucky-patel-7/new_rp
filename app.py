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
from typing import Dict, Any, Optional, List

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

        # Create embedding for the expanded query (better semantic matching)
        if not resume_parser.azure_client:
            raise HTTPException(
                status_code=503,
                detail="Search functionality requires Azure OpenAI configuration"
            )

        # Use expanded query for better semantic search
        search_text = parsed_query.expanded_query if parsed_query.expanded_query else query

        from src.resume_parser.clients.azure_openai import azure_client
        async_client = azure_client.get_async_client()
        response = await async_client.embeddings.create(
            model=azure_client.get_embedding_deployment(),
            input=search_text
        )
        query_vector = response.data[0].embedding

        # Create intelligent filters
        filter_conditions = search_processor.create_search_filters(parsed_query)

        # Apply manual overrides if provided (only if they're valid filters, not placeholder strings)
        if role_filter and role_filter.lower() not in ['none', 'null', '', 'string']:
            filter_conditions["role_category"] = role_filter
        if seniority_filter and seniority_filter.lower() not in ['none', 'null', '', 'string']:
            filter_conditions["seniority"] = seniority_filter

        logger.info(f"🎯 Search filters: {filter_conditions}")
        logger.info(f"📊 Detected roles: {parsed_query.job_roles}")
        logger.info(f"🔧 Detected skills: {parsed_query.skills[:5]}")
        logger.info(f"🗺️ Detected location: {parsed_query.location}")

        # Search in Qdrant with intelligent filtering
        results = await qdrant_client.search_similar(
            query_vector=query_vector,
            limit=limit * 2,  # Get more results for better filtering
            filter_conditions=filter_conditions if filter_conditions else None
        )

        # Apply Python-based location filtering if location filter was requested
        filtered_results = results
        if 'location' in filter_conditions:
            location_query = filter_conditions['location'].lower()
            logger.info(f"🗺️ Applying Python location filter for: {location_query}")

            filtered_results = []
            for result in results:
                payload = result.get('payload', {})
                candidate_location = payload.get('location', '').lower()

                # Check if location query is contained in candidate location
                if location_query in candidate_location or candidate_location in location_query:
                    filtered_results.append(result)
                    logger.info(f"🗺️ Location match: '{location_query}' in '{candidate_location}'")

            logger.info(f"🗺️ Location filtering: {len(results)} -> {len(filtered_results)} results")

        # Remove duplicates based on name+email combination (keep the best score for each person)
        unique_results = {}
        for result in filtered_results:
            payload = result.get('payload', {})
            # Create unique key from name and email
            unique_key = f"{payload.get('name', 'Unknown')}_{payload.get('email', 'Unknown')}"
            if unique_key not in unique_results or result.get('score', 0) > unique_results[unique_key].get('score', 0):
                unique_results[unique_key] = result

        deduplicated_results = list(unique_results.values())

        # Post-process results for better ranking
        ranked_results = _rank_search_results(deduplicated_results, parsed_query)

        # Limit to requested number
        limited_results = ranked_results[:limit]

        # Format results to include only essential information and selection reasons
        formatted_results = _format_search_results(limited_results, parsed_query)

        return {
            "query": query,
            "parsed_query": {
                "detected_roles": parsed_query.job_roles,
                "detected_skills": parsed_query.skills,
                "experience_years": parsed_query.experience_years,
                "seniority_level": parsed_query.seniority_level,
                "intent": parsed_query.intent
            },
            "total_results": len(formatted_results),
            "filters_applied": filter_conditions,
            "results": formatted_results
        }

    except Exception as e:
        logger.error(f"Error in resume search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _rank_search_results(results: List[Dict], parsed_query) -> List[Dict]:
    """Rank search results based on query relevance."""
    if not results:
        return results

    for result in results:
        score = result.get('score', 0.0)
        payload = result.get('payload', {})

        # Boost score for exact role matches
        role_category = payload.get('role_category', '').lower()
        for detected_role in parsed_query.job_roles:
            if detected_role.replace('_', ' ') in role_category:
                score += 0.1

        # Boost score for skill matches
        candidate_skills = [s.lower() for s in payload.get('skills', [])]
        matching_skills = len([s for s in parsed_query.skills if s.lower() in candidate_skills])
        if matching_skills > 0:
            score += matching_skills * 0.05

        # Boost score for seniority match
        if parsed_query.seniority_level:
            candidate_seniority = payload.get('seniority', '').lower()
            if parsed_query.seniority_level in candidate_seniority:
                score += 0.15

        result['adjusted_score'] = score

    # Sort by adjusted score
    return sorted(results, key=lambda x: x.get('adjusted_score', x.get('score', 0)), reverse=True)


def _format_search_results(results: List[Dict], parsed_query) -> List[Dict]:
    """Format search results to include only essential information and selection reason."""
    formatted_results = []

    for result in results:
        payload = result.get('payload', {})

        # Extract essential information
        formatted_result = {
            "name": payload.get('name', 'Unknown'),
            "email": payload.get('email', 'Unknown'),
            "phone": payload.get('phone', 'Unknown'),
            "current_position": payload.get('current_position', 'Unknown'),
            "match_score": round(result.get('adjusted_score', result.get('score', 0)), 3)
        }

        # Generate selection reason
        reasons = []

        # Location match reason (check first for location-based queries)
        if parsed_query.location:
            candidate_location = payload.get('location', '').lower()
            query_location = parsed_query.location.lower()
            if query_location in candidate_location or candidate_location in query_location:
                reasons.append(f"Location: {payload.get('location', 'Unknown')}")

        # Role match reason
        role_category = payload.get('role_category', '').lower()
        for detected_role in parsed_query.job_roles:
            if detected_role.replace('_', ' ') in role_category:
                reasons.append(f"Role matches: {role_category}")
                break

        # Skills match reason
        candidate_skills = [s.lower() for s in payload.get('skills', [])]
        matching_skills = [s for s in parsed_query.skills if s.lower() in candidate_skills]
        if matching_skills:
            if len(matching_skills) <= 3:
                reasons.append(f"Skills match: {', '.join(matching_skills)}")
            else:
                reasons.append(f"Skills match: {', '.join(matching_skills[:3])} and {len(matching_skills)-3} more")

        # Experience reason
        total_experience = payload.get('total_experience', '')
        if total_experience and total_experience != '0 years':
            reasons.append(f"Experience: {total_experience}")

        # Seniority match reason
        if parsed_query.seniority_level:
            candidate_seniority = payload.get('seniority', '')
            if candidate_seniority and parsed_query.seniority_level.lower() in candidate_seniority.lower():
                reasons.append(f"Seniority level: {candidate_seniority}")

        # Default reason if no specific matches
        if not reasons:
            reasons.append("Profile matches search criteria")

        formatted_result["selection_reason"] = " | ".join(reasons)
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
        if preferred_skills:
            parsed_query.skills.extend(preferred_skills)
        if min_experience_years:
            parsed_query.experience_years = min_experience_years
        if location:
            parsed_query.location = location

        # Generate enhanced search text
        search_components = [query]
        if must_have_roles:
            search_components.extend(must_have_roles)
        if preferred_skills:
            search_components.extend(preferred_skills)

        search_text = ' '.join(search_components)

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
                    "experience_years": parsed_query.experience_years,
                    "seniority_level": parsed_query.seniority_level,
                    "location": parsed_query.location,
                    "intent": parsed_query.intent
                },
                "applied_filters": filter_conditions,
                "ranking_factors": {
                    "role_match_boost": 0.1,
                    "skill_match_boost": 0.05,
                    "experience_match_boost": 0.15,
                    "seniority_match_boost": 0.15
                }
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

        # Regular skill matches
        matching_skills = len([s for s in parsed_query.skills if s.lower() in candidate_skills])
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