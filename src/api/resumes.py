from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from src.resume_parser.database.postgres_client import pg_client
from src.resume_parser.database.qdrant_client import qdrant_client


router = APIRouter(tags=["resumes"])


@router.get("/resumes")
async def get_all_resumes(
    page: int = 1,
    page_size: int = 20,
    limit: Optional[int] = Query(None, description="Alias for page_size for compatibility"),
    search: Optional[str] = None,
    name: Optional[str] = None,
    email: Optional[str] = None,
    location: Optional[str] = None,
    job_title: Optional[str] = None,
    role_category: Optional[str] = None,
    company: Optional[str] = None,
    user_id: Optional[str] = Query(None, description="Return resumes owned by this user_id only"),
    admin_view: bool = Query(False, description="If true, ignore user_id and return all resumes"),
    order_by: str = "-created_at",
):
    """Return paginated list of resumes from PostgreSQL mirror (qdrant_resumes).

    - If admin_view=true, lists across all users and ignores user_id
    - Otherwise, requires user_id and filters by owner_user_id
    """
    page = max(1, int(page))
    if isinstance(limit, int) and limit > 0:
        page_size = limit
    page_size = max(1, min(100, int(page_size)))
    offset = (page - 1) * page_size

    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    owner_user_id: Optional[str]
    if admin_view:
        owner_user_id = None
    else:
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required unless admin_view=true")
        owner_user_id = user_id

    total, items = await pg_client.list_resumes(
        offset=offset,
        limit=page_size,
        search=search,
        name=name,
        email=email,
        location=location,
        job_title=job_title,
        role_category=role_category,
        company=company,
        owner_user_id=owner_user_id,
        order_by=order_by,
    )

    return {
        "success": True,
        "page": page,
        "page_size": page_size,
        "total": total,
        "items": items,
        "resumes": items,
    }


@router.get("/resume/{user_id}")
async def get_resume(user_id: str):
    try:
        resume_data = await qdrant_client.get_resume_by_id(user_id)
        if not resume_data:
            raise HTTPException(status_code=404, detail="Resume not found")
        return {"user_id": user_id, "resume_data": resume_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resumes/{resume_id}")
async def get_resume_detail(resume_id: str, fallback_to_qdrant: bool = True):
    ok = await pg_client.connect()
    if ok:
        row = await pg_client.get_resume(resume_id)
        if row:
            return {"success": True, "source": "postgres", "item": row}
    if fallback_to_qdrant:
        try:
            payload = await qdrant_client.get_resume_by_id(resume_id)
            if payload:
                return {"success": True, "source": "qdrant", "item": payload}
        except Exception:
            pass
    raise HTTPException(status_code=404, detail="Resume not found")
