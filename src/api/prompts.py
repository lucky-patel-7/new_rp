from typing import Any, Dict, Optional
from fastapi import APIRouter, Body, HTTPException, Request, Form

from src.resume_parser.database.postgres_client import pg_client


router = APIRouter(tags=["prompts"])


@router.post("/user-search-prompts")
async def create_user_search_prompt(
    request: Request,
    user_id: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    route: Optional[str] = Form(None),
    liked: Optional[bool] = Form(None),
    asked_at: Optional[str] = Form(None),
    response_meta: Optional[str] = Form(None),
    json_body: Optional[Dict[str, Any]] = Body(None),
):
    if not prompt or not str(prompt).strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    payload: Optional[Dict[str, Any]] = None
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            payload = await request.json()
    except Exception:
        payload = None
    if payload and isinstance(payload, dict):
        user_id = payload.get("user_id", user_id)
        prompt = payload.get("prompt", prompt)
        route = payload.get("route", route)
        liked = payload.get("liked", liked)
        asked_at = payload.get("asked_at", asked_at)
        response_meta = payload.get("response_meta", response_meta)
    elif json_body and isinstance(json_body, dict):
        user_id = json_body.get("user_id", user_id)
        prompt = json_body.get("prompt", prompt)
        route = json_body.get("route", route)
        liked = json_body.get("liked", liked)
        asked_at = json_body.get("asked_at", asked_at)
        response_meta = json_body.get("response_meta", response_meta)

    from datetime import datetime
    asked_dt = None
    if asked_at:
        try:
            asked_dt = datetime.fromisoformat(str(asked_at))
        except Exception:
            asked_dt = None

    rm: Optional[Dict[str, Any]] = None
    if isinstance(response_meta, str):
        try:
            import json as _json
            rm = _json.loads(response_meta)
        except Exception:
            rm = None
    elif isinstance(response_meta, dict):
        rm = response_meta

    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    new_id = await pg_client.insert_user_search_prompt(
        user_id=user_id or "anonymous",
        prompt=str(prompt),
        route=str((route or "search-resumes-intent-based")),
        liked=liked if isinstance(liked, bool) else None,
        asked_at=asked_dt,
        response_meta=rm,
    )
    if not new_id:
        raise HTTPException(status_code=500, detail="Failed to store prompt")
    return {"success": True, "id": new_id}


@router.patch("/user-search-prompts/{prompt_id}/feedback")
async def update_user_search_prompt_feedback(
    prompt_id: str,
    request: Request,
    liked: Optional[bool] = Form(None),
    json_body: Optional[Dict[str, Any]] = Body(None),
):
    payload: Optional[Dict[str, Any]] = None
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            payload = await request.json()
    except Exception:
        payload = None

    if isinstance(payload, dict) and "liked" in payload:
        liked = payload.get("liked")
    elif isinstance(json_body, dict) and "liked" in json_body:
        liked = json_body.get("liked")

    def _normalize_liked(val: Any) -> Optional[bool]:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int,)) and val in (0, 1):
            return bool(val)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("true", "1", "yes", "y"): return True
            if v in ("false", "0", "no", "n"): return False
        return None

    liked_norm = _normalize_liked(liked)
    if liked_norm is None:
        raise HTTPException(status_code=400, detail="liked is required")

    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    done = await pg_client.update_user_search_prompt_feedback(prompt_id, liked_norm)
    if not done:
        raise HTTPException(status_code=404, detail="Prompt not found or update failed")

    return {"success": True, "id": prompt_id, "liked": liked_norm}

