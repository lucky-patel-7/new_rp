from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from src.resume_parser.database.postgres_client import pg_client


router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/metrics")
async def dashboard_metrics(user_id: str = Query(..., description="Owner user_id to scope metrics")):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    data = await pg_client.get_dashboard_metrics(user_id)
    return {"success": True, "user_id": user_id, "metrics": data}


@router.get("/recent-activity")
async def dashboard_recent_activity(
    user_id: str = Query(..., description="Owner user_id to scope activity"),
    limit: int = Query(10, ge=1, le=50)
):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    items = await pg_client.get_recent_activity(user_id, limit)
    return {"success": True, "user_id": user_id, "items": items}


@router.get("/top-candidates")
async def top_candidates(
    user_id: str,
    query: Optional[str] = None,
    limit: int = 3,
):
    # Import here to avoid circular import at module load time
    from app import dashboard_top_candidates as _impl  # type: ignore
    return await _impl(user_id=user_id, query=query, limit=limit)
