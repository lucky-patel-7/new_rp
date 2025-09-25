from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from config.settings import settings
from src.resume_parser.core.models import ProcessingResult
from src.resume_parser.core.parser import ResumeParser
from src.resume_parser.database.postgres_client import pg_client
from src.resume_parser.database.qdrant_client import qdrant_client
from src.resume_parser.utils.file_handler import FileHandler
from src.resume_parser.utils.logging import get_logger
from src.resume_parser.clients.azure_openai import azure_client


router = APIRouter(tags=["uploads"])
logger = get_logger(__name__)
resume_parser = ResumeParser()


@router.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), user_id: Optional[str] = Form(None)):
    """Upload and parse a resume file; store in Qdrant and mirror to Postgres."""
    logger.info(f"📄 Processing resume upload: {file.filename}")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    if user_id:
        can_upload, limit_info = await pg_client.check_user_resume_limit(user_id, 1)
        if not can_upload:
            error_detail = (
                f"Resume upload limit exceeded. You have uploaded {limit_info['current_resumes']}/"
                f"{limit_info['resume_limit']} resumes. Available slots: {limit_info['available_slots']}"
            )
            raise HTTPException(status_code=429, detail=error_detail)

    resume_id = str(uuid.uuid4())

    temp_file_path: Optional[Path] = None
    try:
        file_content = await file.read()
        file_size = len(file_content)
        if file_size > settings.max_file_size_bytes:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {settings.app.max_file_size_mb}MB")

        suffix = f".{file.filename.split('.')[-1].lower()}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_content)
            temp_file_path = Path(temp_file.name)

        result: ProcessingResult = await resume_parser.process_resume_file(
            file_path=temp_file_path,
            user_id=resume_id,
            file_size=file_size,
        )
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error_message)

        resume_data = result.resume_data
        if not resume_data:
            raise HTTPException(status_code=500, detail="Resume parsing failed: No resume data returned")

        safe_projects = []
        for project in getattr(resume_data, 'projects', []):
            proj_dict = project.dict() if hasattr(project, 'dict') else dict(project)
            if 'description' not in proj_dict:
                proj_dict['description'] = ''
            safe_projects.append(proj_dict)

        safe_work_history = [job.dict() if hasattr(job, 'dict') else dict(job) for job in getattr(resume_data, 'work_history', [])]
        safe_education = [edu.dict() if hasattr(edu, 'dict') else dict(edu) for edu in getattr(resume_data, 'education', [])]
        safe_role_classification = resume_data.role_classification.dict() if getattr(resume_data, 'role_classification', None) and hasattr(resume_data.role_classification, 'dict') else getattr(resume_data, 'role_classification', {})
        safe_extraction_statistics = resume_data.extraction_statistics.dict() if getattr(resume_data, 'extraction_statistics', None) and hasattr(resume_data.extraction_statistics, 'dict') else getattr(resume_data, 'extraction_statistics', {})  # type: ignore
        safe_current_employment = resume_data.current_employment.dict() if getattr(resume_data, 'current_employment', None) and hasattr(resume_data.current_employment, 'dict') else getattr(resume_data, 'current_employment', None)  # type: ignore

        payload: Dict[str, Any] = {
            "name": getattr(resume_data, 'name', None),
            "email": getattr(resume_data, 'email', None),
            "phone": getattr(resume_data, 'phone', None),
            "location": getattr(resume_data, 'location', None),
            "linkedin_url": getattr(resume_data, 'linkedin_url', None),
            "skills": getattr(resume_data, 'skills', []),
            "projects": safe_projects,
            "work_history": safe_work_history,
            "education": safe_education,
            "summary": getattr(resume_data, 'summary', None),
            "total_experience": getattr(resume_data, 'total_experience', None),
            "recommended_roles": getattr(resume_data, 'recommended_roles', []),
            "role_classification": safe_role_classification,
            "best_role": getattr(resume_data, 'best_role', None),
            "current_position": getattr(resume_data, 'current_employment', {}).get('title', None) if isinstance(getattr(resume_data, 'current_employment', {}), dict) else getattr(safe_current_employment, 'title', None),
            "original_filename": file.filename,
            "upload_timestamp": getattr(resume_data, 'created_at', None),
        }

        if resume_parser.azure_client and getattr(result, 'embedding_vector', None) is not None:
            try:
                await qdrant_client.store_embedding(user_id=resume_id, embedding_vector=result.embedding_vector, payload=payload)
            except Exception as e:
                logger.error(f"❌ Failed to store in Qdrant: {e}")

        response_data = {
            "success": True,
            "user_id": resume_id,
            "processing_time": result.processing_time,
            "resume_data": resume_data.model_dump(mode='json') if hasattr(resume_data, 'model_dump') else (resume_data.dict() if hasattr(resume_data, 'dict') else {}),
            "message": "Resume processed successfully",
        }

        if user_id:
            tokens_used = getattr(result, 'tokens_used', 0) if result else 0
            await pg_client.increment_user_resume_count(user_id, 1, tokens_used)

        return JSONResponse(content=response_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing resume: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        try:
            if temp_file_path and temp_file_path.exists():
                FileHandler.cleanup_file(temp_file_path)
        except Exception:
            pass


@router.post("/bulk-upload-resumes")
async def bulk_upload_resumes(files: List[UploadFile] = File(...), user_id: Optional[str] = Form(None)):
    """Bulk upload multiple resumes; processes each file and stores results."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if user_id:
        can_upload, limit_info = await pg_client.check_user_resume_limit(user_id, len(files))
        if not can_upload:
            error_detail = (
                f"Resume upload limit exceeded. You have uploaded {limit_info['current_resumes']}/"
                f"{limit_info['resume_limit']} resumes. Available slots: {limit_info['available_slots']}"
            )
            raise HTTPException(status_code=429, detail=error_detail)

    results_summary: Dict[str, Any] = {"processed": 0, "skipped": 0, "results": []}
    for file in files:
        resume_id = str(uuid.uuid4())
        r: Dict[str, Any] = {"filename": file.filename, "id": resume_id}
        temp_path: Optional[Path] = None
        try:
            content = await file.read()
            suffix = f".{file.filename.split('.')[-1].lower()}"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tf.write(content)
                temp_path = Path(tf.name)

            result: ProcessingResult = await resume_parser.process_resume_file(
                file_path=temp_path,
                user_id=resume_id,
                file_size=len(content),
            )
            if not result.success:
                r["status"] = "failed"
                r["error"] = result.error_message
                results_summary["skipped"] += 1
                results_summary["results"].append(r)
                continue

            resume_data = result.resume_data
            if not resume_data:
                r["status"] = "failed"
                r["error"] = "No resume data returned"
                results_summary["skipped"] += 1
                results_summary["results"].append(r)
                continue

            payload = {
                "name": getattr(resume_data, 'name', None),
                "email": getattr(resume_data, 'email', None),
                "phone": getattr(resume_data, 'phone', None),
                "location": getattr(resume_data, 'location', None),
                "linkedin_url": getattr(resume_data, 'linkedin_url', None),
                "skills": getattr(resume_data, 'skills', []),
                "projects": [p.dict() if hasattr(p, 'dict') else p for p in getattr(resume_data, 'projects', [])],
                "work_history": [w.dict() if hasattr(w, 'dict') else w for w in getattr(resume_data, 'work_history', [])],
                "education": [e.dict() if hasattr(e, 'dict') else e for e in getattr(resume_data, 'education', [])],
                "summary": getattr(resume_data, 'summary', None),
                "total_experience": getattr(resume_data, 'total_experience', None),
                "recommended_roles": getattr(resume_data, 'recommended_roles', []),
                "role_classification": getattr(resume_data, 'role_classification', {}),
                "best_role": getattr(resume_data, 'best_role', None),
                "current_position": getattr(resume_data, 'current_employment', {}).get('title', None) if isinstance(getattr(resume_data, 'current_employment', {}), dict) else None,
                "original_filename": file.filename,
                "upload_timestamp": getattr(resume_data, 'created_at', None),
            }

            if resume_parser.azure_client and getattr(result, 'embedding_vector', None) is not None:
                try:
                    await qdrant_client.store_embedding(user_id=resume_id, embedding_vector=result.embedding_vector, payload=payload)
                except Exception as e:
                    logger.error(f"❌ Failed to store in Qdrant: {e}")

            r["status"] = "success"
            r["resume_data"] = {
                "name": getattr(resume_data, 'name', None),
                "email": getattr(resume_data, 'email', None),
                "current_position": getattr(resume_data, 'current_position', None),
                "total_experience": getattr(resume_data, 'total_experience', None),
                "skills_count": len(getattr(resume_data, 'skills', [])),
                "best_role": getattr(resume_data, 'best_role', None),
            }

            if user_id:
                tokens_used = getattr(result, 'tokens_used', 0) if result else 0
                await pg_client.increment_user_resume_count(user_id, 1, tokens_used)

            results_summary["processed"] += 1
        except Exception as e:
            r["status"] = "failed"
            r["error"] = str(e)
            results_summary["skipped"] += 1
        finally:
            try:
                if temp_path and temp_path.exists():
                    FileHandler.cleanup_file(temp_path)
            except Exception:
                pass
            results_summary["results"].append(r)

    logger.info(
        f"📋 Bulk upload complete: processed={results_summary['processed']} skipped={results_summary['skipped']} files={len(files)}"
    )
    return results_summary


@router.post("/bulk-upload-folder")
async def bulk_upload_folder(folder: str = Form(...), user_id: Optional[str] = Form(None)):
    """Upload all files from a local folder path (server-side)."""
    if not folder or not str(folder).strip():
        raise HTTPException(status_code=400, detail="folder is required")
    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid folder path")

    file_paths = [p for p in folder_path.iterdir() if p.is_file()]
    if not file_paths:
        raise HTTPException(status_code=400, detail="No files found in folder")

    if user_id:
        can_upload, limit_info = await pg_client.check_user_resume_limit(user_id, len(file_paths))
        if not can_upload:
            error_detail = (
                f"Resume upload limit exceeded. You have uploaded {limit_info['current_resumes']}/"
                f"{limit_info['resume_limit']} resumes. Available slots: {limit_info['available_slots']}"
            )
            raise HTTPException(status_code=429, detail=error_detail)

    results_summary: Dict[str, Any] = {"processed": 0, "skipped": 0, "results": []}
    for fpath in file_paths:
        resume_id = str(uuid.uuid4())
        r: Dict[str, Any] = {"filename": fpath.name, "id": resume_id}
        try:
            result: ProcessingResult = await resume_parser.process_resume_file(
                file_path=fpath,
                user_id=resume_id,
                file_size=fpath.stat().st_size,
            )
            if not result.success:
                r["status"] = "failed"
                r["error"] = result.error_message
                results_summary["skipped"] += 1
                results_summary["results"].append(r)
                continue

            resume_data = result.resume_data
            if not resume_data:
                r["status"] = "failed"
                r["error"] = "No resume data returned"
                results_summary["skipped"] += 1
                results_summary["results"].append(r)
                continue

            payload = {
                "name": getattr(resume_data, 'name', None),
                "email": getattr(resume_data, 'email', None),
                "phone": getattr(resume_data, 'phone', None),
                "location": getattr(resume_data, 'location', None),
                "linkedin_url": getattr(resume_data, 'linkedin_url', None),
                "skills": getattr(resume_data, 'skills', []),
                "projects": [p.dict() if hasattr(p, 'dict') else p for p in getattr(resume_data, 'projects', [])],
                "work_history": [w.dict() if hasattr(w, 'dict') else w for w in getattr(resume_data, 'work_history', [])],
                "education": [e.dict() if hasattr(e, 'dict') else e for e in getattr(resume_data, 'education', [])],
                "summary": getattr(resume_data, 'summary', None),
                "total_experience": getattr(resume_data, 'total_experience', None),
                "recommended_roles": getattr(resume_data, 'recommended_roles', []),
                "role_classification": getattr(resume_data, 'role_classification', {}),
                "best_role": getattr(resume_data, 'best_role', None),
                "current_position": getattr(resume_data, 'current_employment', {}).get('title', None) if isinstance(getattr(resume_data, 'current_employment', {}), dict) else None,
                "original_filename": fpath.name,
                "upload_timestamp": getattr(resume_data, 'created_at', None),
            }

            if resume_parser.azure_client and getattr(result, 'embedding_vector', None) is not None:
                try:
                    await qdrant_client.store_embedding(user_id=resume_id, embedding_vector=result.embedding_vector, payload=payload)
                except Exception as e:
                    logger.error(f"❌ Failed to store in Qdrant: {e}")

            r["status"] = "success"
            r["resume_data"] = {
                "name": getattr(resume_data, 'name', None),
                "email": getattr(resume_data, 'email', None),
                "current_position": getattr(resume_data, 'current_position', None),
                "total_experience": getattr(resume_data, 'total_experience', None),
                "skills_count": len(getattr(resume_data, 'skills', [])),
                "best_role": getattr(resume_data, 'best_role', None),
            }

            if user_id:
                tokens_used = getattr(result, 'tokens_used', 0) if result else 0
                await pg_client.increment_user_resume_count(user_id, 1, tokens_used)

            results_summary["processed"] += 1
        except Exception as e:
            r["status"] = "failed"
            r["error"] = str(e)
            results_summary["skipped"] += 1
        results_summary["results"].append(r)

    logger.info(
        f"📋 Folder upload complete: processed={results_summary['processed']} skipped={results_summary['skipped']} folder={folder}"
    )
    return results_summary

