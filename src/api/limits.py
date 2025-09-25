from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException

from src.resume_parser.database.postgres_client import pg_client


router = APIRouter(prefix="/user-limits", tags=["limits"])


@router.get("/{user_id}")
async def get_user_limits(user_id: str):
    try:
        limits = await pg_client.get_user_resume_limits(user_id)
        if not limits:
            await pg_client.init_user_resume_limits(user_id)
            limits = await pg_client.get_user_resume_limits(user_id)
        return {"success": True, "data": limits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user limits: {str(e)}")


@router.post("/{user_id}/decrement")
async def decrement_user_resume_count(user_id: str, count: int = 1, tokens_used: int = 0):
    try:
        success = await pg_client.decrement_user_resume_count(user_id, count, tokens_used)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to decrement resume count")
        limits = await pg_client.get_user_resume_limits(user_id)
        return {"success": True, "message": f"Decremented resume count by {count} and tokens by {tokens_used}", "data": limits}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decrement resume count: {str(e)}")


@router.post("/{user_id}/reset")
async def reset_user_resume_count(user_id: str, new_count: int = 0, new_tokens: int = 0):
    try:
        success = await pg_client.reset_user_resume_count(user_id, new_count, new_tokens)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to reset resume count")
        limits = await pg_client.get_user_resume_limits(user_id)
        return {"success": True, "message": f"Reset resume count to {new_count} and tokens to {new_tokens}", "data": limits}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset resume count: {str(e)}")


@router.post("/{user_id}/increment")
async def manual_increment_user_resume_count(user_id: str, count: int = 1, tokens_used: int = 0):
    try:
        success = await pg_client.increment_user_resume_count(user_id, count, tokens_used)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to increment resume count")
        limits = await pg_client.get_user_resume_limits(user_id)
        return {"success": True, "message": f"Incremented resume count by {count} and tokens by {tokens_used}", "data": limits}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to increment resume count: {str(e)}")

