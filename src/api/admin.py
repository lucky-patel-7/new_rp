from typing import Any, Dict, List, Optional
import os
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Header

from src.resume_parser.database.postgres_client import pg_client
from src.resume_parser.database.qdrant_client import qdrant_client
from fastapi import Request
from typing import Optional

_ADMIN_API_KEY = os.getenv("AUTH_API_KEY") or os.getenv("ADMIN_API_KEY")

async def require_admin(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    if _ADMIN_API_KEY:
        if not x_api_key or x_api_key != _ADMIN_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing admin API key")
    return None


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users")
async def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    _auth: None = Depends(require_admin),
):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    assert pg_client._pool is not None  # type: ignore[attr-defined]
    offset = (page - 1) * limit
    total = 0
    items: List[Dict[str, Any]] = []
    async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
        total = await conn.fetchval("SELECT COUNT(*) FROM public.users")
        rows = await conn.fetch(
            """
            SELECT u.id, u.email, u.name, u.image, u.created_at, u.updated_at,
                   COALESCE(l.total_resumes_uploaded, 0) AS total_resumes_uploaded,
                   COALESCE(l.resume_limit, 10) AS resume_limit,
                   COALESCE(l.tokens_used, 0) AS tokens_used,
                   COALESCE(l.token_limit, 1000000) AS token_limit,
                   l.last_resume_uploaded_at
            FROM public.users u
            LEFT JOIN public.user_resume_limits l ON l.user_id = u.id
            ORDER BY u.created_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset,
        )
        for r in rows:
            items.append({
                "id": r["id"],
                "email": r["email"],
                "name": r["name"],
                "image": r["image"],
                "created_at": (r["created_at"].isoformat() if r["created_at"] else None),
                "updated_at": (r["updated_at"].isoformat() if r["updated_at"] else None),
                "total_resumes_uploaded": int(r["total_resumes_uploaded"] or 0),
                "resume_limit": int(r["resume_limit"] or 0),
                "tokens_used": int(r["tokens_used"] or 0),
                "token_limit": int(r["token_limit"] or 0),
                "last_resume_uploaded_at": (r["last_resume_uploaded_at"].isoformat() if r["last_resume_uploaded_at"] else None),
            })

    return {"success": True, "page": page, "limit": limit, "total": total, "items": items, "users": items}


@router.get("/stats/resumes")
async def stats_resumes(_auth: None = Depends(require_admin)):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    assert pg_client._pool is not None  # type: ignore[attr-defined]
    async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
        total = await conn.fetchval(f"SELECT COUNT(*) FROM {pg_client._table}")  # type: ignore[attr-defined]
        week = await conn.fetchval(
            f"SELECT COUNT(*) FROM {pg_client._table} WHERE created_at >= date_trunc('week', now())"  # type: ignore[attr-defined]
        )
        month = await conn.fetchval(
            f"SELECT COUNT(*) FROM {pg_client._table} WHERE created_at >= date_trunc('month', now())"  # type: ignore[attr-defined]
        )
    return {"success": True, "total": int(total or 0), "this_week": int(week or 0), "this_month": int(month or 0)}


@router.get("/stats/users")
async def stats_users(_auth: None = Depends(require_admin)):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    assert pg_client._pool is not None  # type: ignore[attr-defined]
    async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
        total = await conn.fetchval("SELECT COUNT(*) FROM public.users")
        new_30d = await conn.fetchval("SELECT COUNT(*) FROM public.users WHERE created_at >= now() - interval '30 days'")
        active_30d = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT u.id) FROM public.users u
            LEFT JOIN public.user_resume_limits l ON l.user_id = u.id
            LEFT JOIN public.user_search_prompts p ON p.user_id = u.id
            WHERE (l.last_resume_uploaded_at >= now() - interval '30 days')
               OR (p.asked_at >= now() - interval '30 days')
            """
        )
    return {"success": True, "total": int(total or 0), "active_30d": int(active_30d or 0), "new_30d": int(new_30d or 0)}


@router.get("/search-logs")
async def search_logs(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    _auth: None = Depends(require_admin),
):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    assert pg_client._pool is not None  # type: ignore[attr-defined]
    offset = (page - 1) * limit
    items: List[Dict[str, Any]] = []
    async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
        try:
            total = await conn.fetchval("SELECT COUNT(*) FROM public.user_search_prompts")
        except Exception:
            return {"success": True, "page": page, "limit": limit, "total": 0, "items": items, "logs": items}
        rows = await conn.fetch(
            """
            SELECT id, user_id, prompt, route, liked, asked_at, response_meta
            FROM public.user_search_prompts
            ORDER BY asked_at DESC
            LIMIT $1 OFFSET $2
            """,
            limit, offset,
        )
        for r in rows:
            items.append({
                "id": str(r["id"]),
                "user_id": r["user_id"],
                "prompt": r["prompt"],
                "route": r["route"],
                "liked": r["liked"],
                "asked_at": (r["asked_at"].isoformat() if r["asked_at"] else None),
                "response_meta": r["response_meta"],
            })
    return {"success": True, "page": page, "limit": limit, "total": total, "items": items, "logs": items}


@router.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: str, _auth: None = Depends(require_admin)):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    assert pg_client._pool is not None  # type: ignore[attr-defined]
    owner_user_id: Optional[str] = None
    deleted_pg = False
    async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
        row = await conn.fetchrow(f"SELECT owner_user_id FROM {pg_client._table} WHERE id = $1", resume_id)  # type: ignore[attr-defined]
        if row and row.get("owner_user_id"):
            owner_user_id = str(row["owner_user_id"]) if row["owner_user_id"] else None
        res = await conn.execute(f"DELETE FROM {pg_client._table} WHERE id = $1", resume_id)  # type: ignore[attr-defined]
        deleted_pg = res.upper().startswith("DELETE")

    deleted_q = await qdrant_client.delete_resume(resume_id)
    decremented = False
    if owner_user_id:
        decremented = await pg_client.decrement_user_resume_count(owner_user_id, 1, 0)

    return {
        "success": True,
        "resume_id": resume_id,
        "deleted_postgres": deleted_pg,
        "deleted_qdrant": deleted_q,
        "owner_user_id": owner_user_id,
        "decremented_user_count": decremented,
    }


@router.delete("/users/{user_id}")
async def delete_user(user_id: str, _auth: None = Depends(require_admin)):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    assert pg_client._pool is not None  # type: ignore[attr-defined]
    deleted_counts: Dict[str, int] = {"postgres_resumes": 0, "qdrant_points": 0, "prompts": 0, "limits_row": 0, "user_row": 0}
    async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
        r = await conn.execute(f"DELETE FROM {pg_client._table} WHERE owner_user_id = $1", user_id)  # type: ignore[attr-defined]
        deleted_counts["postgres_resumes"] = int(r.split()[-1]) if r.upper().startswith("DELETE") else 0
        r = await conn.execute("DELETE FROM public.user_search_prompts WHERE user_id = $1", user_id)
        deleted_counts["prompts"] = int(r.split()[-1]) if r.upper().startswith("DELETE") else 0
        r = await conn.execute("DELETE FROM public.user_resume_limits WHERE user_id = $1", user_id)
        deleted_counts["limits_row"] = int(r.split()[-1]) if r.upper().startswith("DELETE") else 0
        r = await conn.execute("DELETE FROM public.users WHERE id = $1", user_id)
        deleted_counts["user_row"] = int(r.split()[-1]) if r.upper().startswith("DELETE") else 0

    client = qdrant_client.client
    if client is not None:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        f = Filter(must=[FieldCondition(key='owner_user_id', match=MatchValue(value=user_id))])
        client.delete(collection_name=qdrant_client.collection_name, filter=f)

    return {"success": True, "user_id": user_id, "deleted": deleted_counts}


@router.get("/users/{user_id}/limits")
async def get_user_limits_admin(user_id: str, _auth: None = Depends(require_admin)):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    data = await pg_client.get_user_resume_limits(user_id)
    if not data:
        await pg_client.init_user_resume_limits(user_id)
        data = await pg_client.get_user_resume_limits(user_id)
    return {"success": True, "user_id": user_id, "limits": data}


@router.patch("/users/{user_id}/limits")
async def update_user_limits_admin(
    user_id: str,
    resume_limit: Optional[int] = Body(None),
    token_limit: Optional[int] = Body(None),
    reset_counts: bool = Body(False),
    new_count: Optional[int] = Body(None),
    new_tokens: Optional[int] = Body(None),
    _auth: None = Depends(require_admin),
):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    assert pg_client._pool is not None  # type: ignore[attr-defined]
    await pg_client.init_user_resume_limits(user_id)

    updated = False
    async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
        if resume_limit is not None or token_limit is not None:
            sets = []
            args: List[Any] = []
            if resume_limit is not None:
                sets.append(f"resume_limit = ${len(args)+1}")
                args.append(int(resume_limit))
            if token_limit is not None:
                sets.append(f"token_limit = ${len(args)+1}")
                args.append(int(token_limit))
            args.append(user_id)
            sql = f"UPDATE public.user_resume_limits SET {', '.join(sets)}, updated_at = NOW() WHERE user_id = ${len(args)}"
            r = await conn.execute(sql, *args)
            updated = r.upper().startswith("UPDATE")

    counts_changed = False
    if reset_counts:
        counts_changed = await pg_client.reset_user_resume_count(user_id, new_count or 0, new_tokens or 0)
    elif new_count is not None or new_tokens is not None:
        cur = await pg_client.get_user_resume_limits(user_id)
        if cur:
            nc = int(new_count if new_count is not None else cur.get("total_resumes_uploaded", 0))
            nt = int(new_tokens if new_tokens is not None else cur.get("tokens_used", 0))
            counts_changed = await pg_client.reset_user_resume_count(user_id, nc, nt)

    data = await pg_client.get_user_resume_limits(user_id)
    return {"success": True, "user_id": user_id, "updated_limits": updated, "counts_changed": bool(counts_changed), "limits": data}


@router.post("/dump-qdrant-to-postgres")
async def dump_qdrant_to_postgres(
    limit: int = 0,
    batch_size: int = 256,
    dry_run: bool = False,
    _auth: None = Depends(require_admin),
):
    try:
        collection_info = qdrant_client.get_collection_info()
        if collection_info.get("error"):
            raise RuntimeError(collection_info.get("error"))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant not available: {e}")

    if not dry_run:
        ok = await pg_client.connect()
        if not ok:
            raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    client = qdrant_client.client
    collection = qdrant_client.collection_name
    scanned = mirrored = failed = 0
    offset = None

    while True:
        this_limit = batch_size
        if limit and limit > 0:
            remaining = limit - scanned
            if remaining <= 0:
                break
            this_limit = min(this_limit, remaining)

        points, offset = client.scroll(
            collection_name=collection, with_payload=True, with_vectors=False, limit=this_limit, offset=offset
        )
        if not points:
            break
        for p in points:
            scanned += 1
            payload = p.payload or {}
            resume_id = str(p.id)
            if dry_run:
                mirrored += 1
                continue
            try:
                await pg_client.upsert_parsed_resume(
                    resume_id=resume_id,
                    payload=payload,
                    embedding_model="text-embedding-3-large",
                    vector_id=resume_id,
                )
                mirrored += 1
            except Exception:
                failed += 1
        if offset is None:
            break

    return {
        "success": True,
        "collection": collection,
        "limit": limit,
        "batch_size": batch_size,
        "dry_run": dry_run,
        "scanned": scanned,
        "mirrored": mirrored,
        "failed": failed,
    }


@router.post("/assign-owner-all")
async def assign_owner_all(
    request: Request,
    owner_user_id: Optional[str] = None,
    batch_size: int = 512,
    _auth: None = Depends(require_admin),
):
    try:
        payload = await request.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        owner_user_id = payload.get("owner_user_id", payload.get("user_id", owner_user_id))
        try:
            batch_size = int(payload.get("batch_size", batch_size))
        except Exception:
            pass
    if not owner_user_id or not str(owner_user_id).strip():
        raise HTTPException(status_code=400, detail="owner_user_id is required (or provide user_id)")

    owner_user_id = str(owner_user_id).strip()
    pg_updated = 0
    if await pg_client.connect():
        assert pg_client._pool is not None  # type: ignore[attr-defined]
        sql = f"UPDATE {pg_client._table} SET owner_user_id = $1 WHERE owner_user_id IS DISTINCT FROM $1"  # type: ignore[attr-defined]
        async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
            status = await conn.execute(sql, owner_user_id)
            try:
                pg_updated = int(status.split()[-1])
            except Exception:
                pg_updated = 0

    q_updated = 0
    try:
        client = qdrant_client.client
        collection = qdrant_client.collection_name
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection, with_payload=False, with_vectors=False, limit=batch_size, offset=offset
            )
            if not points:
                break
            ids = [p.id for p in points]
            if ids:
                client.set_payload(collection_name=collection, payload={"owner_user_id": owner_user_id}, points=ids)
                q_updated += len(ids)
            if offset is None:
                break
    except Exception:
        pass

    return {
        "success": True,
        "owner_user_id": owner_user_id,
        "postgres_updated_rows": pg_updated,
        "qdrant_updated_points": q_updated,
        "batch_size": batch_size,
    }

