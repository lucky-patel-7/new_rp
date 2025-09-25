from typing import Optional
from fastapi import APIRouter, Form


router = APIRouter(tags=["search"])


@router.post("/analyze-query-intent")
async def analyze_query_intent_proxy(query: str = Form(...), user_id: Optional[str] = Form(None)):
    # Defer import to avoid circular dependency at import time
    from app import analyze_query_intent as _analyze
    return await _analyze(query, user_id)


@router.post("/search-resumes-intent-based")
async def search_resumes_intent_based_proxy(
    query: str = Form(...),
    limit: int = Form(10),
    strict_matching: bool = Form(False),
    user_id: Optional[str] = Form(None),
):
    from app import search_resumes_intent_based as _search
    return await _search(query=query, limit=limit, strict_matching=strict_matching, user_id=user_id)

