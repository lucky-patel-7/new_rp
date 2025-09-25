from typing import Any, Dict
import asyncio
from fastapi import APIRouter

from config.settings import settings
from src.resume_parser.database.qdrant_client import qdrant_client
from src.resume_parser.core.parser import ResumeParser


router = APIRouter(tags=["system"])
parser = ResumeParser()


@router.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.app.app_name}",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {"upload": "/upload-resume", "health": "/health", "search": "/search-resumes"},
    }


@router.get("/health")
async def health_check():
    health_status: Dict[str, Any] = {
        "status": "healthy",
        "timestamp": str(asyncio.get_event_loop().time()),
        "services": {},
    }
    try:
        if parser.azure_client:
            health_status["services"]["azure_openai"] = "connected"
        else:
            health_status["services"]["azure_openai"] = "not_configured"
    except Exception as e:
        health_status["services"]["azure_openai"] = f"error: {str(e)}"

    try:
        collection_info = qdrant_client.get_collection_info()
        health_status["services"]["qdrant"] = "connected"
        health_status["services"]["qdrant_info"] = collection_info
    except Exception as e:
        health_status["services"]["qdrant"] = f"error: {str(e)}"

    return health_status

