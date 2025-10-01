"""
Main FastAPI application for Resume Parser.

A modern, well-organized resume parsing API with comprehensive extraction capabilities.
"""

import os
import sys
import uuid
import tempfile
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from difflib import SequenceMatcher
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, Body, Request, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
from pydantic import BaseModel, Field

# Import organized modules
from src.resume_parser.core.parser import ResumeParser
from src.resume_parser.core.models import ProcessingResult, ResumeData
from src.resume_parser.database.qdrant_client import qdrant_client
from src.resume_parser.database.postgres_client import pg_client
from src.resume_parser.utils.logging import setup_logging, get_logger
from src.resume_parser.utils.file_handler import FileHandler
from src.resume_parser.clients.azure_openai import azure_client
from config.settings import settings

# Setup logging
setup_logging()
logger = get_logger(__name__)

# --- Pydantic Models for New Features ---

class ShortlistUpdate(BaseModel):
    is_shortlisted: bool

class InterviewQuestionBase(BaseModel):
    user_id: str = Field(..., description="The ID of the user creating the question.")
    question_text: str = Field(..., max_length=1000, description="The text of the interview question.")
    category: Optional[str] = Field(None, max_length=100, description="A category for the question (e.g., 'Technical', 'Behavioral').")

class InterviewQuestionCreate(InterviewQuestionBase):
    pass

class InterviewQuestionUpdate(BaseModel):
    question_text: Optional[str] = Field(None, max_length=1000, description="The updated text of the interview question.")
    category: Optional[str] = Field(None, max_length=100, description="The updated category for the question.")

class InterviewQuestionInDB(InterviewQuestionBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class CallInitiationRequest(BaseModel):
    resume_id: uuid.UUID = Field(..., description="The ID of the resume/candidate to call.")
    user_id: str = Field(..., description="The ID of the user initiating the call.")
    notes: Optional[str] = Field(None, description="Optional notes for the call.")

class CallRecord(BaseModel):
    id: uuid.UUID
    resume_id: uuid.UUID
    user_id: str
    call_status: str
    initiated_at: datetime
    notes: Optional[str] = None

    class Config:
        from_attributes = True

# --- New Interview Session Models ---

class InterviewSessionCreate(BaseModel):
    user_id: str = Field(..., description="The ID of the user creating the interview session.")
    session_type: str = Field(..., description="Type of session: 'test' or 'live'")
    question_ids: List[uuid.UUID] = Field(..., description="List of question IDs to include in the interview.")
    candidate_ids: Optional[List[uuid.UUID]] = Field(None, description="List of candidate IDs (for live interviews).")

class InterviewSessionInDB(BaseModel):
    id: uuid.UUID
    user_id: str
    session_type: str
    question_ids: List[uuid.UUID]
    candidate_ids: Optional[List[uuid.UUID]]
    current_question_index: int
    status: str  # 'active', 'completed', 'paused'
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class InterviewResponse(BaseModel):
    question_id: uuid.UUID
    answer_text: str
    audio_duration: Optional[float] = None
    response_time_seconds: Optional[float] = None

class InterviewSetupResponse(BaseModel):
    available_questions: List[InterviewQuestionInDB]
    available_candidates: List[Dict[str, Any]]
    can_create_session: bool
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events."""
    # Startup
    logger.info(f"[STARTUP] Starting {settings.app.app_name}")
    logger.info(f"[CONFIG] Upload directory: {upload_dir}")
    logger.info(f"[CONFIG] Debug mode: {settings.app.debug}")

    # Test Qdrant connection
    try:
        collection_info = qdrant_client.get_collection_info()
        logger.info(f"[SUCCESS] Qdrant connected: {collection_info}")
    except Exception as e:
        logger.warning(f"[WARNING] Qdrant connection issue: {e}")

    yield

    # Shutdown
    logger.info(f"[SHUTDOWN] Shutting down {settings.app.app_name}")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app.app_name,
    description="A comprehensive resume parsing API with Azure OpenAI integration",
    version="1.0.0",
    debug=settings.app.debug,
    lifespan=lifespan
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

_ADMIN_API_KEY = os.getenv("AUTH_API_KEY") or os.getenv("ADMIN_API_KEY")

async def require_admin(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> None:
    if _ADMIN_API_KEY:
        if not x_api_key or x_api_key != _ADMIN_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing admin API key")
    return None


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


def _extract_identifier_filters_from_query(query: str) -> Dict[str, List[str]]:
    """Extract name, email, and phone filters directly from the raw query text."""
    identifiers: Dict[str, List[str]] = {}

    if not query:
        return identifiers

    # Email addresses
    email_pattern = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    email_matches = email_pattern.findall(query)
    if email_matches:
        normalized_emails: List[str] = []
        for raw_email in email_matches:
            cleaned_email = raw_email.strip().lower()
            if cleaned_email and cleaned_email not in normalized_emails:
                normalized_emails.append(cleaned_email)
        if normalized_emails:
            identifiers["email"] = normalized_emails

    # Phone numbers (store both digit-only and lightly formatted variants)
    phone_pattern = re.compile(r"(\+?\d[\d\s\-()]{6,}\d)")
    phone_matches = phone_pattern.findall(query)
    if phone_matches:
        phone_values: List[str] = []
        for raw_phone in phone_matches:
            candidate = raw_phone.strip()
            if not candidate:
                continue
            condensed = re.sub(r"\s+", " ", candidate)
            digits_only = re.sub(r"\D", "", candidate)
            for variant in (condensed, digits_only):
                if variant and variant not in phone_values:
                    phone_values.append(variant)
        if phone_values:
            identifiers["phone"] = phone_values

    # Candidate names mentioned explicitly (common prompt patterns)
    # Examples: "give me lucky patel's resume", "i need dhruv rana profile", "resume of John Doe", "profile of Jane D."
    name_patterns = [
        r"(?:named|name is|called)\s+([A-Za-z][A-Za-z.'\- ]{1,60})",
        r"(?:candidate|applicant)\s+named\s+([A-Za-z][A-Za-z.'\- ]{1,60})",
        r"\b([A-Za-z][A-Za-z.'\- ]{1,60})'?s\s+(?:resume|cv|profile)\b",
        r"\b(?:resume|cv|profile)\s+of\s+([A-Za-z][A-Za-z.'\- ]{1,60})\b",
        r"\b(?:give me|show me|get|find|need|i need|looking for)\s+([A-Za-z][A-Za-z.'\- ]{1,60})\s+(?:resume|cv|profile)\b"
    ]
    name_candidates: List[str] = []
    seen_names: Set[str] = set()
    for pattern in name_patterns:
        matches = re.findall(pattern, query, flags=re.IGNORECASE)
        for raw_match in matches:
            base_name = raw_match.strip().strip('"').strip("'")

            # Clean possessives and trailing qualifiers
            if base_name.lower().endswith("'s"):
                base_name = base_name[:-2].strip()

            def _add_name_variant(value: str) -> None:
                normalized = value.strip() if value else ''
                if not normalized:
                    return
                key = normalized.lower()
                if key not in seen_names:
                    name_candidates.append(normalized)
                    seen_names.add(key)

            if not base_name:
                continue

            _add_name_variant(base_name)

            truncated = base_name
            for separator in ["'s", ',', ' with ', ' who ', ' that ', ' which ', ' and ', ' or ', ' for ', ' from ', '?', '.', '!', ' resume', ' cv', ' profile']:
                idx = truncated.lower().find(separator)
                if idx != -1:
                    truncated = truncated[:idx].strip()
                    break
            truncated = truncated.strip('"').strip("'")
            _add_name_variant(truncated)

    if name_candidates:
        # Post-filter and normalize names: remove directives, possessives, artifacts
        def _strip_directives(text: str) -> str:
            tl = text.strip()
            directives = [
                'give me ', 'show me ', 'get ', 'find ', 'need ', 'i need ', 'looking for ',
                'please ', 'kindly '
            ]
            tl_lower = tl.lower()
            for d in directives:
                if tl_lower.startswith(d):
                    tl = tl[len(d):].strip()
                    tl_lower = tl.lower()
                    break
            return tl

        seen_out: Set[str] = set()
        cleaned: List[str] = []
        BAD = {'resume', 'cv', 'profile', 'candidate', 'applicant', 'email', 'phone', 'number'}
        ROLE_WORDS = {
            'engineer','developer','manager','analyst','designer','consultant','architect','scientist','tester','qa',
            'hr','recruiter','marketing','sales','executive','lead','principal','senior','junior','intern','assistant',
            'devops','full','stack','frontend','backend','data','machine','learning','ml','ai','nlp','doctor','nurse',
            'teacher','accountant','supervisor','director','officer','administrator','operator','specialist'
        }
        def _looks_like_role(text: str) -> bool:
            tl = text.lower()
            # exact multi-word role indicators
            if 'full stack' in tl or 'human resources' in tl:
                return True
            tokens = [t for t in re.split(r"\s+", tl) if t]
            return any(t in ROLE_WORDS for t in tokens)
        for raw in name_candidates:
            name = _strip_directives(raw)
            # Remove trailing artifacts
            name = re.sub(r"\b(?:resume|cv|profile)\b$", "", name, flags=re.IGNORECASE).strip()
            # Remove trailing possessive if any leftover
            name = re.sub(r"'s\b$", "", name, flags=re.IGNORECASE).strip()
            # Trim extra punctuation
            name = name.strip(' .,!?:;"\'')

            tokens = [t for t in name.split() if t]
            if not tokens:
                continue
            if any(t.lower() in BAD for t in tokens):
                continue
            # Keep 1-4 token names and skip obvious role phrases
            if 1 <= len(tokens) <= 4 and not _looks_like_role(name):
                key = name.lower()
                if key not in seen_out:
                    cleaned.append(name)
                    seen_out.add(key)

        if cleaned:
            identifiers["name"] = cleaned

    return identifiers


def _extract_candidate_companies(payload: Dict[str, Any]) -> List[str]:
    companies: List[str] = []

    current_employment = payload.get('current_employment')
    if isinstance(current_employment, dict):
        company = current_employment.get('company')
        if company:
            companies.append(company)

    if payload.get('company'):
        companies.append(payload.get('company')) # type: ignore

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


def _extract_education_info(payload: Dict[str, Any]) -> Dict[str, str]:
    """Extract education information from the education field."""
    education_data = payload.get('education', [])

    # Initialize default values
    education_info = {
        'education_level': 'N/A',
        'education_field': 'N/A',
        'university': 'N/A'
    }

    # If education is a list, extract from the most recent/relevant entry
    if isinstance(education_data, list) and education_data:
        # Get the first education entry (usually the most recent/highest)
        education_entry = education_data[0] if education_data else {}

        if isinstance(education_entry, dict):
            # Extract degree/level
            degree = education_entry.get('degree') or education_entry.get('level') or education_entry.get('qualification')
            if degree:
                education_info['education_level'] = str(degree)

            # Extract field of study
            field = (education_entry.get('field_of_study') or
                    education_entry.get('major') or
                    education_entry.get('subject') or
                    education_entry.get('specialization'))
            if field:
                education_info['education_field'] = str(field)

            # Extract institution/university
            institution = (education_entry.get('institution') or
                          education_entry.get('university') or
                          education_entry.get('school') or
                          education_entry.get('college'))
            if institution:
                education_info['university'] = str(institution)

    elif isinstance(education_data, dict):
        # If education is a dict directly
        degree = education_data.get('degree') or education_data.get('level') or education_data.get('qualification')
        if degree:
            education_info['education_level'] = str(degree)

        field = (education_data.get('field_of_study') or
                education_data.get('major') or
                education_data.get('subject') or
                education_data.get('specialization'))
        if field:
            education_info['education_field'] = str(field)

        institution = (education_data.get('institution') or
                      education_data.get('university') or
                      education_data.get('school') or
                      education_data.get('college'))
        if institution:
            education_info['university'] = str(institution)

    return education_info


async def _llm_semantic_similarity(query_term: str, candidate_term: str) -> bool:
    """
    Use LLM to determine semantic similarity between query and candidate terms.
    Returns True if terms are semantically similar, False otherwise.
    """
    if not query_term or not candidate_term:
        return False

    # Quick check for exact matches first
    if query_term.lower().strip() == candidate_term.lower().strip():
        return True

    try:
        from src.resume_parser.clients.azure_openai import azure_client

        # Create a prompt for semantic similarity comparison
        prompt = f"""
        Compare these two terms and determine if they are semantically similar or related in a professional/job context:

        Term 1: "{query_term}"
        Term 2: "{candidate_term}"

        Consider these as similar if they:
        - Refer to the same or related job roles/positions
        - Are in the same professional category or field
        - Have overlapping responsibilities or skills
        - Are different names for similar roles (e.g., "Software Engineer" vs "Software Developer")

        Answer only "YES" if they are similar/related, or "NO" if they are not.
        """

        client = azure_client.get_async_client()
        response = await client.chat.completions.create(
            model=azure_client.get_chat_deployment(),
            messages=[
                {"role": "system", "content": "You are a professional career matching expert. Answer only YES or NO."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5,
            temperature=0.1
        )

        ai_response = response.choices[0].message.content
        if ai_response:
            return ai_response.strip().upper() == "YES"
        return False

    except Exception as e:
        # Fallback to basic string matching
        return query_term.lower().strip() == candidate_term.lower().strip()


def _llm_refine_final_requirements(query: str, final_requirements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Use LLM to refine and normalize final requirements (filters + keywords).

    The model should:
    - Normalize job titles (e.g., 'full stack developer' -> 'Full Stack Developer')
    - Confirm person names (exclude role phrases like 'software engineer')
    - Keep emails lowercased; provide digits-only phone alongside formatted if available
    - Avoid adding requirements not clearly present
    - Return JSON with keys: {"qdrant_filters": {...}, "search_keywords": [...], "primary_intent": "..."}
    """
    try:
        from src.resume_parser.clients.azure_openai import azure_client
        import json as _json

        client = azure_client.get_sync_client()
        chat_deployment = azure_client.get_chat_deployment()

        fr_json = _json.dumps(final_requirements, ensure_ascii=False)
        system = (
            "You refine search intents for a resume search engine. "
            "Return precise, minimal filters with normalized values. Do not invent data."
        )
        user = f"""
Query: {query}

Current final_requirements JSON:
{fr_json}

Refine and normalize the filters and keywords for vector + metadata search. Rules:
- Normalize job titles to Title Case; include common synonym if obvious (avoid overgeneralization)
- If a 'name' is present, ensure it is a person name (not a role phrase); remove directive words (e.g., 'give me')
- Keep emails lowercased; phones may include formatted form, and provide a digits-only variant if evident
- Do not add filters that are not clearly requested
- Prefer compact lists (max 3 items per key)
- Provide primary_intent among: role | person_name | contact_information | company | location | hybrid

Return ONLY JSON with keys: qdrant_filters, search_keywords, primary_intent.
"""

        response = client.chat.completions.create(
            model=chat_deployment,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=600,
            temperature=0.05
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            return None

        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        refined = _json.loads(content)

        # Basic validation
        if not isinstance(refined, dict) or 'qdrant_filters' not in refined:
            return None

        # Ensure lists and strings are well-typed
        qf = refined.get('qdrant_filters') or {}
        if not isinstance(qf, dict):
            return None
        for k, v in list(qf.items()):
            if v is None:
                del qf[k]
                continue
            if isinstance(v, (str, int)):
                qf[k] = [str(v)]
            elif isinstance(v, list):
                qf[k] = [str(x).strip() for x in v if str(x).strip()]
            else:
                # Remove unexpected types
                del qf[k]

        refined['qdrant_filters'] = qf

        # Sanitize keywords
        kw = refined.get('search_keywords') or []
        if isinstance(kw, str):
            kw = [kw]
        if isinstance(kw, list):
            refined['search_keywords'] = [str(x).strip().lower() for x in kw if str(x).strip()]
        else:
            refined['search_keywords'] = []

        # primary_intent optional normalization
        pi = refined.get('primary_intent')
        if pi and not isinstance(pi, str):
            refined['primary_intent'] = str(pi)

        return refined
    except Exception as _e:
        logger.info(f"[INTENT_LLM] Refinement skipped due to error: {_e}")
        return None


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

    candidate_name = payload.get('name', 'UNKNOWN')
    # Only log a small snippet to avoid noisy logs
    text_snippet = text[:160].replace('\n', ' ')

    for keyword in forced_keywords:
        kw = keyword.lower()
        if kw not in text:
            logger.info(
                f"🔎 Forced keyword MISS for '{candidate_name}': '{keyword}' not found in text snippet: '{text_snippet}...'"
            )
            return False

    logger.info(
        f"🔎 Forced keyword HIT for '{candidate_name}': all terms matched: {forced_keywords}"
    )
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
async def upload_resume(file: UploadFile = File(...), user_id: Optional[str] = Form(None)):
    """
    Upload and parse a resume file.

    Supports PDF, DOC, DOCX, and TXT files.
    Returns structured resume data and stores embeddings in Qdrant.
    """
    logger.info(f"Processing resume upload: {file.filename}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Check user resume upload limit before processing
    if user_id:
        can_upload, limit_info = await pg_client.check_user_resume_limit(user_id, 1)
        if not can_upload:
            error_detail = f"Resume upload limit exceeded. You have uploaded {limit_info['current_resumes']}/{limit_info['resume_limit']} resumes. Available slots: {limit_info['available_slots']}"
            raise HTTPException(status_code=429, detail=error_detail)

    # Generate unique resume ID (point id)
    resume_id = str(uuid.uuid4())

    # Create temporary file
    temp_file = None
    temp_file_path = None  # Ensure temp_file_path is always defined
    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        file_hash = hashlib.sha256(file_content).hexdigest()
        duplicate_resume_id = None
        if user_id and file_hash:
            try:
                duplicate_resume_id = await pg_client.find_duplicate_resume(user_id, file_hash, None)
            except Exception as exc:
                logger.warning(f"[UPLOAD] Duplicate check skipped for user {user_id}: {exc}")
            if duplicate_resume_id:
                raise HTTPException(status_code=409, detail="Duplicate resume detected for this user.")

        logger.info(f"[INFO] File size: {file_size} bytes")

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
            user_id=resume_id,
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
        safe_extraction_statistics = resume_data.extraction_statistics.dict() if getattr(resume_data, 'extraction_statistics', None) and hasattr(resume_data.extraction_statistics, 'dict') else getattr(resume_data, 'extraction_statistics', {}) # type: ignore
        safe_current_employment = resume_data.current_employment.dict() if getattr(resume_data, 'current_employment', None) and hasattr(resume_data.current_employment, 'dict') else getattr(resume_data, 'current_employment', None) # type: ignore
        safe_created_at = getattr(resume_data, 'created_at', None)
        upload_timestamp = safe_created_at.isoformat() if safe_created_at else None

        payload = {
            "user_id": getattr(resume_data, 'user_id', None),
            "name": getattr(resume_data, 'name', None),
            "name_lc": (getattr(resume_data, 'name', '') or '').strip().lower(),
            "email": getattr(resume_data, 'email', None),
            "email_lc": (getattr(resume_data, 'email', '') or '').strip().lower(),
            "phone": getattr(resume_data, 'phone', ''),
            "phone_digits": re.sub(r"\D", "", str(getattr(resume_data, 'phone', ''))) if getattr(resume_data, 'phone', None) else "",
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
            "content_hash": file_hash,
            "extraction_statistics": safe_extraction_statistics,
            "upload_timestamp": upload_timestamp,
            "owner_user_id": user_id,
            "is_shortlisted": False  # Initialize as not shortlisted
        }

        metadata_hash = pg_client.compute_metadata_hash(payload)
        if metadata_hash:
            payload["metadata_hash"] = metadata_hash
        else:
            payload["metadata_hash"] = None

        if user_id:
            try:
                duplicate_resume_id = await pg_client.find_duplicate_resume(user_id, file_hash, metadata_hash)
            except Exception as exc:
                logger.warning(f"[UPLOAD] Duplicate check (post-parse) skipped for user {user_id}: {exc}")
            else:
                if duplicate_resume_id:
                    raise HTTPException(status_code=409, detail="Duplicate resume detected for this user.")

        # Create embedding and store in Qdrant
        embedding_vector = await resume_parser.create_embedding(resume_data)

        if embedding_vector:
            # Store in Qdrant
            try:
                point_id = await qdrant_client.store_embedding(
                    user_id=resume_id,
                    embedding_vector=embedding_vector,
                    payload=payload
                )
                logger.info(f"[SUCCESS] Stored in Qdrant with ID: {point_id}")
                # Mirror to PostgreSQL (best-effort)
                try:
                    await pg_client.upsert_parsed_resume(
                        resume_id=resume_id,
                        payload=payload,
                        embedding_model=azure_client.get_embedding_deployment(),
                        vector_id=str(point_id)
                    )
                except Exception as e:
                    logger.info(f"[PG] Skipped mirroring to Postgres: {e}")
            except Exception as e:
                logger.error(f"❌ Failed to store in Qdrant: {e}")
                # Continue without failing the request

        # Prepare response
        response_data = {
            "success": True,
            "user_id": resume_id,
            "processing_time": result.processing_time,
            "resume_data": resume_data.model_dump(mode='json') if hasattr(resume_data, 'model_dump') else (resume_data.dict() if hasattr(resume_data, 'dict') else {}),
            "message": "Resume processed successfully"
        }

        # Increment user's resume count after successful processing
        if user_id:
            tokens_used = getattr(result, 'tokens_used', 0) if result else 0
            await pg_client.increment_user_resume_count(user_id, 1, tokens_used)

        logger.info(f"[SUCCESS] Resume processing completed for user: {user_id}")
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

@app.get("/user-limits/{user_id}")
async def get_user_limits(user_id: str):
    """
    Get user's current resume upload limits and usage.

    Args:
        user_id: User identifier

    Returns:
        Dict containing current usage, limits, and available slots
    """
    try:
        limits = await pg_client.get_user_resume_limits(user_id)

        if not limits:
            # Initialize limits for new user
            await pg_client.init_user_resume_limits(user_id)
            limits = await pg_client.get_user_resume_limits(user_id)

        return {
            "success": True,
            "data": limits
        }
    except Exception as e:
        logger.error(f"❌ Error fetching user limits for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch user limits: {str(e)}")


@app.post("/user-limits/{user_id}/decrement")
async def decrement_user_resume_count(
    user_id: str,
    count: int = 1,
    tokens_used: int = 0
):
    """
    Decrease user's resume count (e.g., when resumes are deleted).

    Args:
        user_id: User identifier
        count: Number of resumes to subtract (default: 1)
        tokens_used: Tokens to subtract from usage (default: 0)

    Returns:
        Dict with success status and updated limits
    """
    try:
        success = await pg_client.decrement_user_resume_count(user_id, count, tokens_used)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to decrement resume count")

        # Get updated limits
        limits = await pg_client.get_user_resume_limits(user_id)

        return {
            "success": True,
            "message": f"Decremented resume count by {count} and tokens by {tokens_used}",
            "data": limits
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error decrementing user resume count for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to decrement resume count: {str(e)}")


@app.post("/user-limits/{user_id}/reset")
async def reset_user_resume_count(
    user_id: str,
    new_count: int = 0,
    new_tokens: int = 0
):
    """
    Reset user's resume count and token usage to specific values.

    Args:
        user_id: User identifier
        new_count: New resume count (default: 0)
        new_tokens: New token count (default: 0)

    Returns:
        Dict with success status and updated limits
    """
    try:
        success = await pg_client.reset_user_resume_count(user_id, new_count, new_tokens)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to reset resume count")

        # Get updated limits
        limits = await pg_client.get_user_resume_limits(user_id)

        return {
            "success": True,
            "message": f"Reset resume count to {new_count} and tokens to {new_tokens}",
            "data": limits
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error resetting user resume count for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset resume count: {str(e)}")


@app.post("/user-limits/{user_id}/increment")
async def manual_increment_user_resume_count(
    user_id: str,
    count: int = 1,
    tokens_used: int = 0
):
    """
    Manually increment user's resume count (for admin purposes).

    Args:
        user_id: User identifier
        count: Number of resumes to add (default: 1)
        tokens_used: Tokens to add to usage (default: 0)

    Returns:
        Dict with success status and updated limits
    """
    try:
        success = await pg_client.increment_user_resume_count(user_id, count, tokens_used)

        if not success:
            raise HTTPException(status_code=400, detail="Failed to increment resume count")

        # Get updated limits
        limits = await pg_client.get_user_resume_limits(user_id)

        return {
            "success": True,
            "message": f"Incremented resume count by {count} and tokens by {tokens_used}",
            "data": limits
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error incrementing user resume count for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to increment resume count: {str(e)}")


@app.post("/analyze-query-intent")
async def analyze_query_intent(
    query: str = Form(...),
    user_id: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Enhanced query intent analyzer that prepares filters and keywords for downstream search."""
    import json
    import re

    started = time.perf_counter()
    normalized_query = (query or "").strip()
    if not normalized_query:
        return {
            "success": False,
            "error": "Query text is required for intent analysis.",
            "query": query,
            "user_id": user_id,
        }

    ai_response: Optional[str] = None
    intent_data: Dict[str, Any] = {}
    used_llm = False

    client = None
    chat_deployment = None
    try:
        from src.resume_parser.clients.azure_openai import azure_client  # local import to avoid circular load
        client = azure_client.get_sync_client()
        chat_deployment = azure_client.get_chat_deployment()
    except Exception as exc:
        logger.info(f"[INTENT] Azure OpenAI client unavailable: {exc}")
        client = None
        chat_deployment = None

    if client and chat_deployment:
        try:
            enhanced_intent_prompt = f"""
You are an advanced query intent analyzer for a resume search system. Analyze this job search query and extract ALL components with high precision. Return ONLY a JSON object.

Query: "{normalized_query}"

Extract and classify ALL components from this query. Be thorough and precise:

{{
    "query_metadata": {{
        "complexity_level": "simple|moderate|complex|very_complex",
        "query_type": "single_criteria|multi_criteria|comprehensive",
        "primary_intent": "education|role|skills|company|location|experience|hybrid",
        "secondary_intents": ["education", "role", "skills", "company", "location"],
        "confidence_score": 0.95,
        "intent_explanation": "Detailed explanation of the query structure and why this classification was chosen"
    }},

    "extracted_components": {{
        "education_requirements": {{
            "has_requirement": true/false,
            "degree_levels": ["PhD", "Master's", "Bachelor's", "Associate"],
            "specific_degrees": ["PhD in Computer Science", "Master of Science"],
            "fields_of_study": ["Computer Science", "Engineering", "Data Science"],
            "institutions": ["Stanford University", "MIT", "Carnegie Mellon University"],
            "institution_tiers": ["top_tier", "ivy_league", "technical_schools"],
            "education_keywords": ["from top universities", "prestigious"]
        }},

        "role_requirements": {{
            "has_requirement": true/false,
            "job_titles": ["Principal Software Architect", "Senior Engineer"],
            "role_levels": ["Principal", "Senior", "Lead", "Staff"],
            "role_categories": ["Engineering", "Management", "Technical"],
            "role_keywords": ["architect", "lead", "principal"]
        }},

        "skill_requirements": {{
            "has_requirement": true/false,
            "technical_skills": ["distributed systems", "Kubernetes", "Java", "Go"],
            "frameworks": ["React", "Angular", "Spring"],
            "technologies": ["AWS", "Docker", "Microservices"],
            "domains": ["machine learning", "data science", "cybersecurity"],
            "skill_categories": ["programming", "architecture", "devops", "cloud"],
            "proficiency_indicators": ["expert", "advanced", "proficient"]
        }},

        "company_requirements": {{
            "has_requirement": true/false,
            "specific_companies": ["Google", "Facebook", "Apple", "Amazon", "Netflix"],
            "company_groups": ["FAANG", "Big Tech", "Fortune 500"],
            "company_types": ["startup", "enterprise", "public", "private"],
            "company_sizes": ["large", "medium", "small"],
            "industry_sectors": ["technology", "finance", "healthcare"]
        }},

        "experience_requirements": {{
            "has_requirement": true/false,
            "min_years": 10,
            "max_years": null,
            "specific_experience": ["10+ years", "5-8 years"],
            "experience_types": ["industry", "relevant", "total"],
            "seniority_levels": ["junior", "mid", "senior", "principal", "staff"]
        }},

        "location_requirements": {{
            "has_requirement": true/false,
            "current_locations": ["Silicon Valley", "San Francisco", "New York"],
            "preferred_locations": ["San Francisco", "Bay Area"],
            "relocation_indicators": ["willing to relocate", "open to relocation"],
            "location_flexibility": "strict|flexible|remote_ok",
            "geographic_regions": ["West Coast", "East Coast", "US", "Global"]
        }},

        "additional_criteria": {{
            "certifications": ["AWS Certified Solutions Architect"],
            "languages": ["English", "Spanish"],
            "work_authorization": ["US Citizen", "Green Card"],
            "work_preferences": ["remote", "onsite", "hybrid"],
            "soft_skills": ["communication", "leadership"],
            "culture_fit": ["startup mindset", "enterprise ready"]
        }}
    }},

    "search_strategy": {{
        "recommended_approach": "semantic_search|hybrid_search|metadata_first",
        "search_complexity": "single_pass|multi_pass",
        "post_filters": ["education", "experience", "skills"]
    }},

    "query_ambiguities": ["Any unclear parts where human confirmation might be needed"]
}}
""".strip()
            response = client.chat.completions.create(
                model=chat_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an advanced query intent analyzer that returns comprehensive JSON analysis of job search queries. Focus on extracting all components with high precision.",
                    },
                    {"role": "user", "content": enhanced_intent_prompt},
                ],
                max_tokens=1500,
                temperature=0.05,
            )
            if response and response.choices:
                ai_response = (response.choices[0].message.content or "").strip()
                used_llm = bool(ai_response)
        except Exception as exc:
            logger.error(f"[INTENT] Enhanced intent analysis failed: {exc}")

    if ai_response:
        cleaned_response = ai_response
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        if cleaned_response:
            try:
                intent_data = json.loads(cleaned_response)
            except json.JSONDecodeError as exc:
                logger.error(f"[INTENT] Failed to parse AI response: {exc}; payload={cleaned_response[:400]}")
                intent_data = {}
    elif client and not ai_response:
        logger.info("[INTENT] Azure OpenAI intent analyzer returned no content.")

    if not isinstance(intent_data, dict):
        intent_data = {}

    components = intent_data.get("extracted_components")
    if not isinstance(components, dict):
        components = {}
    intent_data["extracted_components"] = components

    identifier_filters = _extract_identifier_filters_from_query(normalized_query)
    qdrant_filters: Dict[str, Any] = {}
    search_keywords: List[str] = []

    def _normalize_filter_value(value: Any) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return value
        text = str(value).strip()
        return text or None

    def _merge_filter(field: str, values: Any) -> None:
        if values is None:
            return
        if isinstance(values, (list, tuple, set)):
            values_iter = list(values)
        else:
            values_iter = [values]
        existing = qdrant_filters.get(field)
        if isinstance(existing, list):
            collected = existing.copy()
        elif existing is None:
            collected = []
        else:
            collected = [existing]
        seen = {repr(item).lower() if isinstance(item, str) else repr(item) for item in collected}
        for raw in values_iter:
            normalized = _normalize_filter_value(raw)
            if normalized is None:
                continue
            key = normalized.lower() if isinstance(normalized, str) else repr(normalized)
            if key in seen:
                continue
            collected.append(normalized)
            seen.add(key)
        if collected:
            qdrant_filters[field] = collected

    def _extend_keywords(values: Any) -> None:
        if values is None:
            return
        if isinstance(values, (list, tuple, set)):
            values_iter = values
        else:
            values_iter = [values]
        for raw in values_iter:
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                search_keywords.append(text)

    for field, values in identifier_filters.items():
        _merge_filter(field, values)
        _extend_keywords(values)

    edu_req = components.get("education_requirements")
    if isinstance(edu_req, dict) and edu_req.get("has_requirement"):
        _merge_filter("degree_level", edu_req.get("degree_levels"))
        _merge_filter("field_of_study", edu_req.get("fields_of_study"))
        _merge_filter("institution", edu_req.get("institutions"))
        _extend_keywords(edu_req.get("degree_levels"))
        _extend_keywords(edu_req.get("specific_degrees"))
        _extend_keywords(edu_req.get("fields_of_study"))
        _extend_keywords(edu_req.get("institutions"))
        _extend_keywords(edu_req.get("education_keywords"))

    role_req = components.get("role_requirements")
    if isinstance(role_req, dict) and role_req.get("has_requirement"):
        _merge_filter("job_title", role_req.get("job_titles"))
        _merge_filter("seniority_level", role_req.get("role_levels"))
        _merge_filter("role_category", role_req.get("role_categories"))
        _extend_keywords(role_req.get("job_titles"))
        _extend_keywords(role_req.get("role_levels"))
        _extend_keywords(role_req.get("role_keywords"))

    skill_req = components.get("skill_requirements")
    if isinstance(skill_req, dict) and skill_req.get("has_requirement"):
        all_skills: List[str] = []
        for key in ("technical_skills", "frameworks", "technologies", "domains"):
            values = skill_req.get(key)
            if isinstance(values, list):
                all_skills.extend(values)
            elif isinstance(values, str):
                all_skills.append(values)
        if all_skills:
            _merge_filter("skills", all_skills)
            _extend_keywords(all_skills)
        if skill_req.get("skill_categories"):
            _merge_filter("skill_category", skill_req.get("skill_categories"))
            _extend_keywords(skill_req.get("skill_categories"))
        if skill_req.get("proficiency_indicators"):
            _extend_keywords(skill_req.get("proficiency_indicators"))

    comp_req = components.get("company_requirements")
    if isinstance(comp_req, dict) and comp_req.get("has_requirement"):
        _merge_filter("company", comp_req.get("specific_companies"))
        _merge_filter("company_type", comp_req.get("company_groups"))
        _merge_filter("industry", comp_req.get("industry_sectors"))
        _extend_keywords(comp_req.get("specific_companies"))
        _extend_keywords(comp_req.get("company_groups"))
        _extend_keywords(comp_req.get("company_types"))
        _extend_keywords(comp_req.get("industry_sectors"))

    exp_req = components.get("experience_requirements")
    if isinstance(exp_req, dict) and exp_req.get("has_requirement"):
        if exp_req.get("min_years") is not None:
            _merge_filter("min_experience", [exp_req.get("min_years")])
        if exp_req.get("max_years") is not None:
            _merge_filter("max_experience", [exp_req.get("max_years")])
        _merge_filter("seniority_level", exp_req.get("seniority_levels"))
        _extend_keywords(exp_req.get("specific_experience"))
        _extend_keywords(exp_req.get("experience_types"))

    loc_req = components.get("location_requirements")
    if isinstance(loc_req, dict) and loc_req.get("has_requirement"):
        all_locations: List[str] = []
        for key in ("current_locations", "preferred_locations"):
            values = loc_req.get(key)
            if isinstance(values, list):
                all_locations.extend(values)
            elif isinstance(values, str):
                all_locations.append(values)
        unique_locations: List[str] = []
        seen_locations = set()
        for item in all_locations:
            text = str(item).strip()
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen_locations:
                continue
            seen_locations.add(lowered)
            unique_locations.append(text)
        if unique_locations:
            qdrant_filters["location"] = unique_locations
            search_keywords.extend(unique_locations)
        if loc_req.get("geographic_regions"):
            _merge_filter("region", loc_req.get("geographic_regions"))
            _extend_keywords(loc_req.get("geographic_regions"))
        if loc_req.get("relocation_indicators"):
            _extend_keywords(loc_req.get("relocation_indicators"))

    add_req = components.get("additional_criteria")
    if isinstance(add_req, dict):
        if add_req.get("certifications"):
            _merge_filter("certifications", add_req.get("certifications"))
            _extend_keywords(add_req.get("certifications"))
        if add_req.get("languages"):
            _merge_filter("languages", add_req.get("languages"))
            _extend_keywords(add_req.get("languages"))
        if add_req.get("work_authorization"):
            _merge_filter("work_authorization", add_req.get("work_authorization"))
            _extend_keywords(add_req.get("work_authorization"))
        if add_req.get("work_preferences"):
            _merge_filter("work_preference", add_req.get("work_preferences"))
            _extend_keywords(add_req.get("work_preferences"))
        if add_req.get("soft_skills"):
            _extend_keywords(add_req.get("soft_skills"))
        if add_req.get("culture_fit"):
            _extend_keywords(add_req.get("culture_fit"))

    dedup_keywords: List[str] = []
    seen_kw = set()
    for kw in search_keywords:
        text = str(kw).strip()
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen_kw:
            continue
        seen_kw.add(normalized)
        dedup_keywords.append(text)

    if not dedup_keywords:
        tokens = re.findall(r"[A-Za-z]{3,}", normalized_query.lower())
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "from",
            "that",
            "this",
            "have",
            "need",
            "looking",
            "search",
            "find",
            "resume",
            "candidate",
            "profile",
            "give",
            "show",
            "please",
            "kindly",
            "about",
            "into",
            "over",
            "want",
            "require",
            "seeking",
            "someone",
            "who",
            "skill",
            "skills",
            "years",
            "year",
            "experience",
        }
        for token in tokens:
            if token in stopwords or token in seen_kw:
                continue
            seen_kw.add(token)
            dedup_keywords.append(token)
            if len(dedup_keywords) >= 25:
                break

    query_lower = normalized_query.lower()
    pattern_groups = {
        "degree_patterns": [
            r"(phd|ph\.d\.?|doctorate|doctoral)",
            r"(master['’]s|ms|m\.s\.?|mba|m\.b\.a\.?)",
            r"(bachelor['’]s|bs|b\.s\.?|ba|b\.a\.?)",
        ],
        "university_patterns": [
            r"(stanford|mit|harvard|berkeley|caltech|carnegie mellon|princeton|yale)",
            r"(top universities|prestigious|ivy league|tier[- ]?1)",
        ],
        "company_patterns": [
            r"(google|facebook|apple|amazon|netflix|microsoft|meta)",
            r"(faang|big tech|fortune 500)",
        ],
        "experience_patterns": [
            r"\d+\+?\s*years?",
            r"(senior|principal|staff|lead|director|vp|c-level)",
        ],
        "location_patterns": [
            r"(silicon valley|san francisco|nyc|new york|seattle|austin|remote)",
            r"(relocate|relocation|willing to move|open to)",
        ],
        "skill_patterns": [
            r"(python|java|javascript|typescript|go|rust|c\+\+)",
            r"(kubernetes|docker|aws|azure|gcp|terraform)",
            r"(machine learning|ml|ai|distributed systems|microservices|data science)",
        ],
    }
    pattern_matches: Dict[str, int] = {}
    total_pattern_hits = 0
    for key, regex_list in pattern_groups.items():
        count = 0
        for pattern in regex_list:
            count += len(re.findall(pattern, query_lower))
        pattern_matches[key] = count
        total_pattern_hits += count

    words = re.findall(r"\w+", normalized_query)
    sentences = [segment for segment in re.split(r"[.!?]+", normalized_query) if segment.strip()]
    comma_segments = [segment for segment in normalized_query.split(",") if segment.strip()]
    parenthetical_segments = re.findall(r"\([^)]+\)", normalized_query)
    complexity_score = min(100, total_pattern_hits * 15 + len(words) * 2)

    def _score_to_level(score: int) -> str:
        if score >= 75:
            return "very_complex"
        if score >= 55:
            return "complex"
        if score >= 35:
            return "moderate"
        return "simple"

    complexity_level = _score_to_level(complexity_score)

    override_primary_intent: Optional[str] = None
    if qdrant_filters.get("email") or qdrant_filters.get("phone"):
        override_primary_intent = "contact_information"
    elif qdrant_filters.get("name"):
        override_primary_intent = "person_name"
    elif qdrant_filters.get("company"):
        override_primary_intent = "company"
    elif qdrant_filters.get("location"):
        override_primary_intent = "location"
    elif qdrant_filters.get("job_title") or qdrant_filters.get("skills"):
        override_primary_intent = "role"

    query_metadata = intent_data.get("query_metadata")
    if not isinstance(query_metadata, dict):
        query_metadata = {}

    existing_primary = str(query_metadata.get("primary_intent") or "").strip()
    if override_primary_intent:
        if existing_primary.lower() in ("", "unknown", "hybrid"):
            query_metadata["primary_intent"] = override_primary_intent
        else:
            query_metadata["primary_intent"] = override_primary_intent
    query_metadata.setdefault("primary_intent", "hybrid")
    query_metadata.setdefault(
        "complexity_level",
        complexity_level,
    )
    query_metadata.setdefault(
        "query_type",
        "multi_criteria" if len(qdrant_filters) > 1 or len(dedup_keywords) > 5 else "single_criteria",
    )
    secondary_intents: List[str] = []
    if qdrant_filters.get("job_title") or qdrant_filters.get("seniority_level"):
        secondary_intents.append("role")
    if qdrant_filters.get("skills") or qdrant_filters.get("skill_category"):
        secondary_intents.append("skills")
    if qdrant_filters.get("company") or qdrant_filters.get("company_type"):
        secondary_intents.append("company")
    if qdrant_filters.get("location") or qdrant_filters.get("region"):
        secondary_intents.append("location")
    if qdrant_filters.get("degree_level") or qdrant_filters.get("institution"):
        secondary_intents.append("education")
    if qdrant_filters.get("min_experience") or qdrant_filters.get("max_experience"):
        secondary_intents.append("experience")
    query_metadata["secondary_intents"] = sorted({value for value in secondary_intents})
    if "confidence_score" not in query_metadata:
        query_metadata["confidence_score"] = 0.85 if used_llm else 0.55
    if "intent_explanation" not in query_metadata:
        query_metadata["intent_explanation"] = "LLM-based intent analysis" if used_llm else "Heuristic fallback intent analysis"
    intent_data["query_metadata"] = query_metadata

    advanced_analysis = intent_data.get("advanced_analysis")
    if not isinstance(advanced_analysis, dict):
        advanced_analysis = {}
    advanced_analysis["query_statistics"] = {
        "query_length": len(normalized_query),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "comma_separated_criteria": len(comma_segments),
        "parenthetical_info": len(parenthetical_segments),
    }
    advanced_analysis["pattern_analysis"] = pattern_matches
    advanced_analysis["complexity_indicators"] = {
        "multiple_criteria": len(comma_segments) > 0 or total_pattern_hits >= 3,
        "specific_institutions": pattern_matches.get("degree_patterns", 0) > 0,
        "company_requirements": pattern_matches.get("company_patterns", 0) > 0,
        "technical_depth": pattern_matches.get("skill_patterns", 0) >= 2,
        "experience_specific": pattern_matches.get("experience_patterns", 0) > 0,
        "location_constraints": pattern_matches.get("location_patterns", 0) > 0,
    }
    advanced_analysis["query_structure"] = {
        "has_conjunctions": any(token in query_lower for token in (" and ", " with ", " who ", " that ")),
        "has_qualifiers": any(token in query_lower for token in (" prefer", " ideal", " nice to have", " bonus")),
        "has_requirements": any(token in query_lower for token in (" must", " required", " need ", " should ")),
        "has_alternatives": any(token in query_lower for token in (" or ", " alternatively", " either ")),
    }
    semantic_signals = {
        "urgency_indicators": any(token in query_lower for token in (" asap", " urgent", " immediately", " quickly")),
        "flexibility_indicators": any(token in query_lower for token in (" flexible", " open to", " willing to", " consider")),
        "exclusivity_indicators": any(token in query_lower for token in (" only", " exclusively", " strictly", " must have")),
    }
    advanced_analysis["semantic_signals"] = semantic_signals
    advanced_analysis["overall_complexity_score"] = complexity_score
    advanced_analysis["complexity_level"] = complexity_level
    intent_data["advanced_analysis"] = advanced_analysis

    intent_data["processing_recommendations"] = {
        "use_multi_stage_filtering": complexity_score > 60 or len(qdrant_filters) >= 3,
        "require_fuzzy_matching": pattern_matches.get("skill_patterns", 0) > 3,
        "prioritize_exact_matches": semantic_signals["exclusivity_indicators"],
        "enable_fallback_search": True,
        "suggested_result_limit": 50 if complexity_score > 70 else 100,
    }

    search_strategy = intent_data.get("search_strategy")
    if not isinstance(search_strategy, dict):
        search_strategy = {}
    search_strategy.setdefault("primary_intent", query_metadata.get("primary_intent", "hybrid"))
    search_strategy.setdefault("recommended_approach", "semantic_search")
    search_strategy.setdefault(
        "search_complexity",
        "multi_pass" if complexity_level in {"complex", "very_complex"} else "single_pass",
    )
    search_strategy["complexity"] = search_strategy.get("search_complexity")
    intent_data["search_strategy"] = search_strategy

    final_requirements = intent_data.get("final_requirements")
    if not isinstance(final_requirements, dict):
        final_requirements = {}
    final_requirements["qdrant_filters"] = qdrant_filters
    final_requirements["search_keywords"] = dedup_keywords
    final_requirements["filter_count"] = len(qdrant_filters)
    final_requirements["has_strict_requirements"] = bool(qdrant_filters)
    final_requirements["search_strategy"] = {
        "primary_intent": search_strategy.get("primary_intent"),
        "recommended_approach": search_strategy.get("recommended_approach"),
        "complexity": search_strategy.get("search_complexity"),
    }

    refined_final = None
    try:
        refined_final = _llm_refine_final_requirements(normalized_query, final_requirements)
    except Exception as exc:
        logger.info(f"[INTENT] Refinement skipped: {exc}")
        refined_final = None

    if refined_final:
        refined_filters = refined_final.get("qdrant_filters")
        if isinstance(refined_filters, dict):
            final_requirements["qdrant_filters"] = refined_filters
            final_requirements["filter_count"] = len(refined_filters)
            final_requirements["has_strict_requirements"] = bool(refined_filters)
        refined_keywords = refined_final.get("search_keywords")
        if isinstance(refined_keywords, list):
            final_requirements["search_keywords"] = refined_keywords
        refined_primary = refined_final.get("primary_intent")
        if isinstance(refined_primary, str) and refined_primary.strip():
            refined_primary_clean = refined_primary.strip()
            query_metadata["primary_intent"] = refined_primary_clean
            search_strategy["primary_intent"] = refined_primary_clean
            final_requirements["search_strategy"]["primary_intent"] = refined_primary_clean

    intent_data["final_requirements"] = final_requirements
    intent_data["query_metadata"] = query_metadata
    intent_data["search_strategy"] = search_strategy

    logger.info(f"[INTENT] Final requirements prepared: {final_requirements}")

    duration_ms = (time.perf_counter() - started) * 1000.0

    return {
        "success": True,
        "query": normalized_query,
        "user_id": user_id,
        "intent_analysis": intent_data,
        "processing_time_ms": round(duration_ms, 2),
        "used_llm": used_llm,
    }

@app.post("/bulk-upload-resumes")
async def bulk_upload_resumes(files: List[UploadFile] = File(...), user_id: Optional[str] = Form(None)):
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

    # Check user resume upload limit before processing bulk upload
    if user_id:
        can_upload, limit_info = await pg_client.check_user_resume_limit(user_id, len(files))
        if not can_upload:
            error_detail = f"Bulk upload rejected. You want to upload {len(files)} resumes but only have {limit_info['available_slots']} slots remaining. Current usage: {limit_info['current_resumes']}/{limit_info['resume_limit']}"
            raise HTTPException(status_code=429, detail=error_detail)

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
            logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}")

            if not file.filename:
                raise ValueError("No filename provided")

            # Call the same upload logic as the single file upload
            # This is essentially what the upload_resume function does
            resume_id = str(uuid.uuid4())
            file_result["user_id"] = resume_id

            # Create temporary file
            temp_file = None
            temp_file_path = None

            try:
                # Get file content and size
                content = await file.read()
                file_size = len(content)
                logger.info(f"[INFO] File size: {file_size} bytes")

                file_hash = hashlib.sha256(content).hexdigest()
                duplicate_resume_id = None
                if user_id and file_hash:
                    try:
                        duplicate_resume_id = await pg_client.find_duplicate_resume(user_id, file_hash, None)
                    except Exception as exc:
                        logger.warning(f"[BULK_UPLOAD] Duplicate check skipped for user {user_id}: {exc}")
                    if duplicate_resume_id:
                        file_result["status"] = "duplicate"
                        file_result["error"] = "Duplicate resume detected for this user"
                        results["failed_uploads"] += 1
                        results["results"].append(file_result)
                        continue


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
                    user_id=resume_id,
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
                safe_extraction_statistics = resume_data.extraction_statistics.dict() if getattr(resume_data, 'extraction_statistics', None) and hasattr(resume_data.extraction_statistics, 'dict') else getattr(resume_data, 'extraction_statistics', {}) # type: ignore
                safe_current_employment = resume_data.current_employment.dict() if getattr(resume_data, 'current_employment', None) and hasattr(resume_data.current_employment, 'dict') else getattr(resume_data, 'current_employment', None) # type: ignore
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
                    "summary": getattr(resume_data, 'summary', ''),
                    "recommended_roles": getattr(resume_data, 'recommended_roles', []),
                    "work_history": safe_work_history,
                    "current_employment": safe_current_employment,
                    "projects": safe_projects,
                    "education": safe_education,
                    "role_classification": safe_role_classification,
                    "original_filename": file.filename,
                    "content_hash": file_hash,
                    "extraction_statistics": safe_extraction_statistics,
                    "upload_timestamp": upload_timestamp,
                    "owner_user_id": owner_user_id,
                    "is_shortlisted": False  # Initialize as not shortlisted
                }

                metadata_hash = pg_client.compute_metadata_hash(payload)
                if metadata_hash:
                    payload["metadata_hash"] = metadata_hash
                else:
                    payload["metadata_hash"] = None

                if user_id:
                    try:
                        duplicate_resume_id = await pg_client.find_duplicate_resume(user_id, file_hash, metadata_hash)
                    except Exception as exc:
                        logger.warning(f"[BULK_UPLOAD] Duplicate check (post-parse) skipped for user {user_id}: {exc}")
                    else:
                        if duplicate_resume_id:
                            file_result["status"] = "duplicate"
                            file_result["error"] = "Duplicate resume detected for this user"
                            results["failed_uploads"] += 1
                            results["results"].append(file_result)
                            continue

                # Create embedding and store in Qdrant (same as single upload)
                embedding_vector = await resume_parser.create_embedding(resume_data)

                if embedding_vector:
                    # Store in Qdrant
                    try:
                        point_id = await qdrant_client.store_embedding(
                            user_id=resume_id,
                            embedding_vector=embedding_vector,
                            payload=payload
                        )
                        logger.info(f"[SUCCESS] Stored resume {i+1} in Qdrant with ID: {point_id}")
                        # Mirror to PostgreSQL (best-effort)
                        try:
                            await pg_client.upsert_parsed_resume(
                                resume_id=resume_id,
                                payload=payload,
                                embedding_model=azure_client.get_embedding_deployment(),
                                vector_id=str(point_id)
                            )
                        except Exception as e:
                            logger.info(f"[PG] Skipped mirroring to Postgres (bulk): {e}")
                    except Exception as e:
                        logger.error(f"❌ Failed to store in Qdrant: {e}")
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

                # Increment user's resume count for each successful upload
                if user_id:
                    tokens_used = getattr(result, 'tokens_used', 0) if result else 0
                    await pg_client.increment_user_resume_count(user_id, 1, tokens_used)

                logger.info(f"[SUCCESS] Successfully processed file {i+1}: {file.filename}")

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


@app.post("/bulk-upload-folder")
async def bulk_upload_folder(
    folder_path: Optional[str] = Form(None),
    recursive: bool = Form(False),
    limit: int = Form(0),
    user_id: Optional[str] = Form(None),
    json_body: Optional[Dict[str, Any]] = Body(None)
):
    """
    Process and upload all resumes from a local folder path.

    Accepts either form-encoded fields or JSON body with keys: folder_path, recursive, limit.

    Args:
        folder_path: Absolute or relative path to a folder on the server machine
        recursive: Whether to recurse into subdirectories
        limit: Optional max number of files to process (0 = no limit)
        json_body: Optional JSON body with the same keys

    Returns:
        Summary with counts and per-file results
    """
    # Allow JSON body as an alternative to form fields
    if not folder_path and json_body:
        folder_path = json_body.get("folder_path")
        recursive = bool(json_body.get("recursive", recursive))
        limit = int(json_body.get("limit", limit))
        user_id = json_body.get("user_id", user_id)

    if not folder_path or not str(folder_path).strip():
        raise HTTPException(status_code=400, detail="folder_path is required")

    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found or not a directory: {folder}")

    # Allowed file types from settings
    allowed = set((settings.app.allowed_file_types or []))
    allowed = {str(ext).lower().lstrip('.') for ext in allowed}
    if not allowed:
        allowed = {"pdf", "doc", "docx", "txt"}

    # Build file list
    pattern = "**/*" if recursive else "*"
    all_paths = [p for p in folder.glob(pattern) if p.is_file()]
    file_paths = [p for p in all_paths if p.suffix.lower().lstrip('.') in allowed]

    if not file_paths:
        return {
            "success": True,
            "folder": str(folder),
            "total_found": 0,
            "processed": 0,
            "skipped": 0,
            "results": []
        }

    if limit and limit > 0:
        file_paths = file_paths[:limit]

    # Check user resume upload limit before processing folder upload
    if user_id:
        can_upload, limit_info = await pg_client.check_user_resume_limit(user_id, len(file_paths))
        if not can_upload:
            error_detail = f"Folder upload rejected. Found {len(file_paths)} resume files but only have {limit_info['available_slots']} slots remaining. Current usage: {limit_info['current_resumes']}/{limit_info['resume_limit']}"
            raise HTTPException(status_code=429, detail=error_detail)

    logger.info(f"📁 Bulk folder upload: folder={folder} files={len(file_paths)} recursive={recursive} limit={limit}")

    results_summary: Dict[str, Any] = {
        "success": True,
        "folder": str(folder),
        "total_found": len(file_paths),
        "processed": 0,
        "skipped": 0,
        "results": []
    }

    for idx, path in enumerate(file_paths, 1):
        r: Dict[str, Any] = {
            "index": idx,
            "file": str(path),
            "status": "processing",
            "user_id": None,
            "error": None
        }
        try:
            if not path.exists() or not path.is_file():
                raise ValueError("File not found or not a regular file")

            resume_id = str(uuid.uuid4())
            r["resume_id"] = resume_id
            r["user_id"] = resume_id
            r["owner_user_id"] = owner_user_id

            file_size = path.stat().st_size
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

            duplicate_resume_id = None
            if owner_user_id and file_hash:
                try:
                    duplicate_resume_id = await pg_client.find_duplicate_resume(owner_user_id, file_hash, None)
                except Exception as exc:
                    logger.warning(f"[FOLDER_UPLOAD] Duplicate check skipped for user {owner_user_id}: {exc}")
                if duplicate_resume_id:
                    r["status"] = "duplicate"
                    r["error"] = "Duplicate resume detected for this user"
                    results_summary["skipped"] += 1
                    results_summary["results"].append(r)
                    continue

            # Process the resume directly from disk
            result: ProcessingResult = await resume_parser.process_resume_file(
                file_path=path,
                user_id=user_id,
                file_size=file_size
            )

            if not result.success:
                raise ValueError(result.error_message or "Resume parsing failed")

            resume_data = result.resume_data
            if not resume_data:
                raise ValueError("No resume data returned")

            # Prepare payload (aligned with /upload-resume)
            safe_projects: List[Dict[str, Any]] = []
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
                "summary": getattr(resume_data, 'summary', ''),
                "recommended_roles": getattr(resume_data, 'recommended_roles', []),
                "work_history": safe_work_history,
                "current_employment": safe_current_employment,
                "projects": safe_projects,
                "education": safe_education,
                "role_classification": safe_role_classification,
                "original_filename": path.name,
                "content_hash": file_hash,
                "extraction_statistics": safe_extraction_statistics,
                "upload_timestamp": upload_timestamp,
                "owner_user_id": owner_user_id,
                "is_shortlisted": False  # Initialize as not shortlisted
            }

            metadata_hash = pg_client.compute_metadata_hash(payload)
            if metadata_hash:
                payload["metadata_hash"] = metadata_hash
            else:
                payload["metadata_hash"] = None

            if owner_user_id:
                try:
                    duplicate_resume_id = await pg_client.find_duplicate_resume(owner_user_id, file_hash, metadata_hash)
                except Exception as exc:
                    logger.warning(f"[FOLDER_UPLOAD] Duplicate check (post-parse) skipped for user {owner_user_id}: {exc}")
                else:
                    if duplicate_resume_id:
                        r["status"] = "duplicate"
                        r["error"] = "Duplicate resume detected for this user"
                        results_summary["skipped"] += 1
                        results_summary["results"].append(r)
                        continue

            # Create embedding and store in Qdrant
            embedding_vector = await resume_parser.create_embedding(resume_data)
            if embedding_vector:
                try:
                    point_id = await qdrant_client.store_embedding(
                        user_id=resume_id,
                        embedding_vector=embedding_vector,
                        payload=payload
                    )
                    try:
                        await pg_client.upsert_parsed_resume(
                            resume_id=resume_id,
                            payload=payload,
                            embedding_model=azure_client.get_embedding_deployment(),
                            vector_id=str(point_id)
                        )
                    except Exception as e:
                        logger.info(f"[PG] Skipped mirroring to Postgres (folder): {e}")
                except Exception as e:
                    logger.error(f"❌ Failed to store in Qdrant for file {path.name}: {e}")

            r["status"] = "success"
            r["resume_data"] = {
                "name": getattr(resume_data, 'name', None),
                "email": getattr(resume_data, 'email', None),
                "current_position": getattr(resume_data, 'current_position', None),
                "total_experience": getattr(resume_data, 'total_experience', None),
                "skills_count": len(getattr(resume_data, 'skills', [])),
                "best_role": getattr(resume_data, 'best_role', None)
            }
            # Increment user's resume count for each successful folder upload
            if owner_user_id:
                tokens_used = getattr(result, 'tokens_used', 0) if result else 0
                await pg_client.increment_user_resume_count(owner_user_id, 1, tokens_used)

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


@app.post("/admin/dump-qdrant-to-postgres")
async def dump_qdrant_to_postgres(
    limit: int = Form(0),
    batch_size: int = Form(256),
    dry_run: bool = Form(False),
    json_body: Optional[Dict[str, Any]] = Body(None)
):
    """
    Mirror all Qdrant points into PostgreSQL table qdrant_resumes.

    Accepts form fields or JSON body with keys: limit, batch_size, dry_run.
    - limit: 0 for all points, or a positive number to cap processing
    - batch_size: Qdrant scroll batch size (default 256)
    - dry_run: if true, do not write to Postgres; only count
    """
    # Allow JSON override
    if json_body:
        try:
            limit = int(json_body.get("limit", limit))
        except Exception:
            pass
        try:
            batch_size = int(json_body.get("batch_size", batch_size))
        except Exception:
            pass
        dry_run = bool(json_body.get("dry_run", dry_run))

    # Validate Qdrant connection
    try:
        collection_info = qdrant_client.get_collection_info()
        if collection_info.get("error"):
            raise RuntimeError(collection_info.get("error"))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant not available: {e}")

    # Validate Postgres availability (best-effort; we still can dry-run)
    if not dry_run:
        ok = await pg_client.connect()
        if not ok:
            raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    client = qdrant_client.client  # underlying qdrant_client.QdrantClient
    collection = qdrant_client.collection_name

    scanned = 0
    mirrored = 0
    failed = 0
    offset = None

    logger.info(f"[DUMP] Starting dump from Qdrant '{collection}' to Postgres (limit={limit}, batch={batch_size}, dry_run={dry_run})")

    # Iterate via scroll
    while True:
        this_limit = batch_size
        if limit and limit > 0:
            remaining = limit - scanned
            if remaining <= 0:
                break
            this_limit = min(this_limit, remaining)

        points, offset = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=this_limit,
            offset=offset,
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
                    embedding_model=azure_client.get_embedding_deployment(),
                    vector_id=resume_id,
                )
                mirrored += 1
            except Exception as e:
                failed += 1
                logger.info(f"[DUMP] Failed to mirror {resume_id}: {e}")

        if offset is None:
            break

    logger.info(f"[DUMP] Completed: scanned={scanned}, mirrored={mirrored}, failed={failed}")

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


@app.post("/admin/assign-owner-all")
async def assign_owner_all(
    request: Request,
    owner_user_id: Optional[str] = Form(None),
    batch_size: int = Form(512),
    json_body: Optional[Dict[str, Any]] = Body(None)
):
    """
    Assign the same owner_user_id to all resumes in both Postgres and Qdrant.

    Body (form or JSON):
    - owner_user_id (required)
    - batch_size (optional, for Qdrant scrolling)
    """
    # Merge JSON if provided (root-level or nested json_body)
    payload: Optional[Dict[str, Any]] = None
    try:
        if request.headers.get("content-type", "").startswith("application/json"):
            payload = await request.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        # Accept both owner_user_id and user_id (alias)
        owner_user_id = payload.get("owner_user_id", payload.get("user_id", owner_user_id))
        if "batch_size" in payload:
            try:
                batch_size = int(payload.get("batch_size", batch_size))
            except Exception:
                pass
    elif json_body and isinstance(json_body, dict):
        owner_user_id = json_body.get("owner_user_id", json_body.get("user_id", owner_user_id))
        try:
            batch_size = int(json_body.get("batch_size", batch_size))
        except Exception:
            pass

    if not owner_user_id or not str(owner_user_id).strip():
        raise HTTPException(status_code=400, detail="owner_user_id is required (or provide user_id)")

    owner_user_id = str(owner_user_id).strip()

    # Update Postgres
    pg_updated = 0
    if await pg_client.connect():
        assert pg_client._pool is not None  # type: ignore[attr-defined]
        sql = f"UPDATE {pg_client._table} SET owner_user_id = $1 WHERE owner_user_id IS DISTINCT FROM $1"  # type: ignore[attr-defined]
        try:
            async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
                status = await conn.execute(sql, owner_user_id)
                # status like 'UPDATE 42'
                try:
                    pg_updated = int(status.split()[-1])
                except Exception:
                    pg_updated = 0
        except Exception as e:
            logger.error(f"[ADMIN] Failed to update Postgres owner_user_id: {e}")

    # Update Qdrant payloads
    q_updated = 0
    try:
        client = qdrant_client.client
        collection = qdrant_client.collection_name
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                with_payload=False,
                with_vectors=False,
                limit=batch_size,
                offset=offset,
            )
            if not points:
                break
            ids = [p.id for p in points]
            if ids:
                client.set_payload(
                    collection_name=collection,
                    payload={"owner_user_id": owner_user_id},
                    points=ids,
                )
                q_updated += len(ids)
            if offset is None:
                break
    except Exception as e:
        logger.error(f"[ADMIN] Failed to update Qdrant owner_user_id: {e}")

    return {
        "success": True,
        "owner_user_id": owner_user_id,
        "postgres_updated_rows": pg_updated,
        "qdrant_updated_points": q_updated,
        "batch_size": batch_size,
    }



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
            candidate_location = payload.get('location', '')
            query_location = parsed_query.location
            if candidate_location and query_location:
                if candidate_location in query_location or query_location in candidate_location:
                    reasons.append(f"Location: {candidate_location} matches {query_location}")

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
                    f"Company Match: Experience{role_phrase} at {matched_company} aligns with your request for {matched_query_company}"
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
                role_explanations.append(f"Perfect sector match: Query '{detected_role}' belongs to '{role_sector}' sector, candidate's role category is '{role_category}'")
            elif detected_role.lower() in current_position.lower():
                role_explanations.append(f"Direct position match: Query role '{detected_role}' found in candidate's current position '{current_position}'")
            elif any(variation.lower() in current_position.lower() for variation in role_variations if variation):
                matching_variation = next(var for var in role_variations if var and var.lower() in current_position.lower())
                role_explanations.append(f"Role variation match: '{matching_variation}' (variation of {detected_role}) found in position '{current_position}'")
            elif role_category and role_category != 'Unknown':
                # Semantic/related role match
                role_explanations.append(f"Related field match: Query '{detected_role}' is similar to candidate's '{role_category}' category")

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
                        experience_explanations.append(f"Direct role experience: Previous role '{job.get('title', '')}' at {company} directly matches query '{detected_role}'")
                        break
                    # Check for role variations in job title
                    elif any(var.lower() in job_title for var in role_variations if var and len(var) > 3):
                        matching_var = next(var for var in role_variations if var and len(var) > 3 and var.lower() in job_title)
                        experience_explanations.append(f"Related role experience: Previous role '{job.get('title', '')}' at {company} contains '{matching_var}', related to query '{detected_role}'")
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
                skill_explanations.append(f"Direct skills match: {', '.join(direct_skill_matches)} listed in candidate's skills")

            if work_skill_matches:
                skill_explanations.append(f"Work experience skills: {work_skill_matches[0]}")  # Show top work skill match

            if project_skill_matches:
                skill_explanations.append(f"Project skills: {project_skill_matches[0]}")  # Show top project skill match

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




@app.post("/search-resumes-intent-based")
async def search_resumes_intent_based(
    query: str = Form(...),
    limit: int = Form(10),
    strict_matching: bool = Form(False),
    user_id: Optional[str] = Form(None)
):
    """
    Enhanced resume search using analyze-query-intent to find exact candidates based on final requirements.

    This route uses the analyze-query-intent function to understand exactly what the user wants,
    then performs targeted Qdrant search with precise filters for more accurate results.

    Args:
        query: Natural language query describing the ideal candidate
        limit: Maximum number of results to return
        strict_matching: If True, candidates must match ALL specified criteria exactly.
                        If False, candidates can match partially (more lenient matching).
                        Default: False

    Returns:
        Dict containing search results with detailed matching analysis
    """
    try:
        logger.info(f"🎯 Intent-based search for: {query}")

        # Step 1: Analyze the query to get final requirements
        intent_result = await analyze_query_intent(query, user_id)

        if not intent_result.get("success"):
            raise HTTPException(
                status_code=400,
                detail=f"Failed to analyze query intent: {intent_result.get('error', 'Unknown error')}"
            )

        # Extract final requirements
        intent_data = intent_result.get("intent_analysis", {})
        final_requirements = intent_data.get("final_requirements", {})

        if not final_requirements:
            raise HTTPException(
                status_code=400,
                detail="No final requirements extracted from query"
            )

        logger.info(f"📋 Final requirements extracted: {final_requirements}")

        search_strategy = final_requirements.get("search_strategy", {})
        if (
            not final_requirements.get("qdrant_filters")
            and not final_requirements.get("search_keywords")
            and final_requirements.get("filter_count", 0) == 0
            and search_strategy.get("primary_intent") == "unknown"
        ):
            logger.info(
                "Intent analysis returned unknown intent with no actionable filters; skipping semantic search."
            )
            return {
                "success": True,
                "query": query,
                "requested_by": user_id,
                "total_results": 0,
                "intent_analysis": {
                    "final_requirements": final_requirements,
                    "filters_applied": {},
                    "search_strategy": search_strategy,
                },
                "results": [],
                "processing_summary": {
                    "query_analyzed": True,
                    "filters_used": False,
                    "post_retrieval_filtering_applied": False,
                    "semantic_fallback": True,
                    "strict_matching_enabled": strict_matching,
                    "matching_mode": "strict" if strict_matching else "partial",
                    "candidates_before_filtering": 0,
                    "candidates_after_filtering": 0,
                    "filter_types_applied": [],
                    "search_approach": "semantic_only",
                    "margin": {"enabled": False, "ratio": None, "top_semantic_score": 0.0},
                    "notes": "No actionable criteria were extracted from the query; nothing to search.",
                },
            }

        # Step 2: Generate embedding for the query
        try:
            from src.resume_parser.clients.azure_openai import azure_client
            client = azure_client.get_sync_client()

            search_keywords_for_embedding = final_requirements.get("search_keywords", [])
            if search_keywords_for_embedding:
                keyword_snippet = ' '.join(search_keywords_for_embedding[:20])
                embedding_input = f"{query}\n\nKeywords: {keyword_snippet}"
                logger.info(f"dYZ_ Embedding enriched with intent keywords: {search_keywords_for_embedding[:5]}")
            else:
                embedding_input = query

            response = client.embeddings.create(
                input=embedding_input,
                model=azure_client.get_embedding_deployment()
            )

            query_vector = response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate query embedding"
            )

        # Step 3: Prepare Qdrant filters (post-retrieval focus)
        qdrant_filters = final_requirements.get("qdrant_filters", {}) or {}

        # For hiring workflows we prefer a full semantic retrieval first, then apply filters manually
        pre_filter_fields = {"name", "email", "phone"}
        filter_conditions = {}

        def _expand_name_variants(values: List[str]) -> List[str]:
            variants = set()
            for v in values:
                if not isinstance(v, str):
                    continue
                s = ' '.join(v.strip().split())  # collapse whitespace
                if not s:
                    continue
                # Common case variants
                variants.add(s)
                variants.add(s.lower())
                variants.add(s.title())
                # Preserve original token casing per word (capitalize)
                variants.add(' '.join(w.capitalize() for w in s.split(' ')))
            return list(variants)

        def _expand_phone_variants(values: List[str]) -> List[str]:
            """Generate elastic variants for phone numbers across country codes and formats.

            Strategy:
            - Preserve raw, digits-only, and +digits
            - Generate last-10-digit form (NSN) and common groupings
            - Prepend common country codes for NSN: +<cc><nsn>, +<cc> <nsn>, <cc><nsn>, 00<cc><nsn>, 0<nsn>
            Note: This is best-effort without libphonenumber; we keep variants bounded.
            """
            import re as _re
            out: set[str] = set()
            COMMON_DIAL_CODES = [
                '1','20','27','30','31','32','33','34','36','39','40','41','43','44','45','46','47','48','49',
                '52','55','56','57','58','60','61','62','63','64','65','66','81','82','84','86','90','91','92','93','94','95','98',
                '971','972','973','974','975','976','977','966','968','970','380','351','353','354','358'
            ]
            MAX_VARIANTS = 50

            for v in values:
                if not isinstance(v, str):
                    continue
                raw = v.strip()
                if not raw:
                    continue
                digits = ''.join(_re.findall(r'\d+', raw))
                if not digits:
                    continue

                # Base forms
                out.add(raw)
                out.add(digits)
                out.add('+' + digits)

                # National significant number (last 10 as a common heuristic)
                nsn = digits[-10:] if len(digits) >= 10 else digits
                if len(nsn) >= 7:
                    out.add(nsn)
                    # Common groupings
                    if len(nsn) == 10:
                        out.add(nsn[:5] + ' ' + nsn[5:])  # 5-5 split
                        out.add(nsn[:4] + ' ' + nsn[4:7] + ' ' + nsn[7:])  # 4-3-3 split
                        out.add(nsn[:3] + ' ' + nsn[3:6] + ' ' + nsn[6:])  # 3-3-4 split
                        out.add(nsn[:2] + ' ' + nsn[2:6] + ' ' + nsn[6:])  # 2-4-4 split

                    # Local trunk prefix form
                    out.add('0' + nsn)

                    # Add variants with common dial codes
                    for cc in COMMON_DIAL_CODES:
                        out.add(cc + nsn)
                        out.add('+' + cc + nsn)
                        out.add('+' + cc + ' ' + nsn)
                        out.add('00' + cc + nsn)
                        if len(out) >= MAX_VARIANTS:
                            break

                # If looks like already includes a country code (length >= 11)
                if len(digits) >= 11:
                    out.add(digits)
                    out.add('+' + digits)

            # Bound the output deterministically
            variants = sorted(out)
            return variants[:MAX_VARIANTS]

        for field in pre_filter_fields:
            field_value = qdrant_filters.get(field)
            if field_value:
                if isinstance(field_value, list):
                    normalized_values = [str(v).strip() for v in field_value if str(v).strip()]
                else:
                    normalized_values = [str(field_value).strip()]
                if normalized_values:
                    # For names, add case/whitespace variants to handle exact-match behavior in Qdrant
                    if field == 'name':
                        expanded = _expand_name_variants(normalized_values)
                        filter_conditions[field] = expanded
                        logger.info(f"dYt Name filter expanded to variants: {expanded}")
                    elif field == 'phone':
                        expanded_p = _expand_phone_variants(normalized_values)
                        filter_conditions[field] = expanded_p
                        logger.info(f"dYt Phone filter expanded to variants: {expanded_p}")
                    else:
                        filter_conditions[field] = normalized_values

        # Always restrict to this user's resumes when user_id is provided
        if user_id:
            filter_conditions['owner_user_id'] = str(user_id).strip()

        post_retrieval_fields = set(qdrant_filters.keys())

        logger.info(f"dY<? Qdrant filters (post-processing): {qdrant_filters}")
        if filter_conditions:
            logger.info(f"dY<? Qdrant pre-filter conditions: {filter_conditions}")
        else:
            logger.info("dYZ_ Running full semantic search without pre-applied Qdrant filters")

        # Step 4: Search Qdrant using full semantic retrieval
        from src.resume_parser.database.qdrant_client import qdrant_client

        has_post_retrieval_filters = bool(post_retrieval_fields)

        base_multiplier = 10 if has_post_retrieval_filters else 4
        if strict_matching:
            base_multiplier = max(base_multiplier, 6)

        search_limit = max(limit * base_multiplier, limit)

        results = await qdrant_client.search_similar(
            query_vector=query_vector,
            limit=search_limit,
            filter_conditions=filter_conditions if filter_conditions else None  # Apply strict filters only when provided
        )
        logger.info(f"dYZ_ Retrieved {len(results)} candidates from full semantic search (limit={search_limit})")

        # Fallback: if strict identifier pre-filters yielded no results, rerun without pre-filter
        if not results and any(key in filter_conditions for key in ('name', 'phone', 'email')):
            logger.info("dY! No results with pre-applied identifier filter(s). Retrying semantic search without pre-filter for post-filtering.")
            # Keep owner filter even in fallback
            owner_only_filter = {'owner_user_id': str(user_id).strip()} if user_id else None
            results = await qdrant_client.search_similar(
                query_vector=query_vector,
                limit=max(search_limit, 200),
                filter_conditions=owner_only_filter
            )
            logger.info(f"dY! Unfiltered semantic search returned {len(results)} candidates; will apply identifier filters in post-processing")

        # Step 4.5: Apply post-retrieval filtering for fields not available in Qdrant
        if results:
            filtered_results = []

            critical_criteria = {"experience", "location", "job_title", "role_category", "company", "education", "name", "email", "phone"}
            soft_criteria = {"job_title", "role_category", "skills"}
            top_semantic_score = max((r.get('score', 0.0) for r in results), default=0.0)
            margin_ratio = 0.90

            # Helper function to check strict vs non-strict matching
            def should_exclude_candidate(criteria_matches: list, strict_mode: bool = False, cand_sem_score: float = 0.0) -> bool:
                """
                Determine if candidate should be excluded based on matching criteria.

                Args:
                    criteria_matches: List of tuples [(criteria_name, has_match, is_required), ...]
                    strict_mode: If True, ALL required criteria must match. If False, partial matches OK.

                Returns:
                    True if candidate should be excluded, False otherwise.
                """
                if not criteria_matches:
                    return False

                required_criteria = [match for match in criteria_matches if match[2]]  # is_required = True

                if not required_criteria:
                    return False

                if strict_mode:
                    # In strict mode, ALL required criteria must match
                    failed_criteria = [match for match in required_criteria if not match[1]]  # has_match = False
                    return len(failed_criteria) > 0
                else:
                    # In non-strict mode, enforce critical requirements but allow partial matches otherwise
                    allow_soft = cand_sem_score >= (top_semantic_score * margin_ratio)
                    soft_fail_budget = 1 if allow_soft else 0
                    soft_fail_used = 0

                    for criterion_name, has_match, _ in required_criteria:
                        if not has_match and (criterion_name in critical_criteria):
                            if criterion_name in soft_criteria and soft_fail_used < soft_fail_budget:
                                soft_fail_used += 1
                                continue
                            return True

                    skill_matches = [match for match in required_criteria if match[0].startswith("skill_")]
                    if skill_matches and not any(match[1] for match in skill_matches):
                        return True

                    non_skill_required = [match for match in required_criteria if not match[0].startswith("skill_")]
                    matched_non_skill = [match for match in non_skill_required if match[1]]

                    if non_skill_required:
                        match_ratio = len(matched_non_skill) / len(non_skill_required)
                    else:
                        matched_total = len([match for match in required_criteria if match[1]])
                        match_ratio = matched_total / len(required_criteria) if required_criteria else 1.0

                    return match_ratio < 0.5  # Exclude if less than 50% of non-skill criteria match

            # Helper: digits-only and elastic equality for phone numbers
            def _digits_only(val: str) -> str:
                return re.sub(r"\D", "", str(val or ""))

            def _phone_digits_match(cand_digits: str, req_digits: str) -> bool:
                if not cand_digits or not req_digits:
                    return False
                if cand_digits == req_digits:
                    return True
                # Accept suffix/prefix matches to account for country/trunk codes
                MIN_LEN = 7
                if len(req_digits) >= MIN_LEN and cand_digits.endswith(req_digits):
                    return True
                if len(cand_digits) >= MIN_LEN and req_digits.endswith(cand_digits):
                    return True
                return False

            for result in results:
                payload = result.get('payload', {})
                should_include = True
                criteria_matches = []

                # Enforce owner filter in post-processing as a safety net
                if user_id:
                    owner = str(payload.get('owner_user_id', '')).strip()
                    if owner != str(user_id).strip():
                        # Skip candidates not owned by the requesting user
                        continue

                def _normalize_filter_values(value):
                    if isinstance(value, list):
                        return [str(v).lower().strip() for v in value if str(v).strip()]
                    value_str = str(value).lower().strip()
                    return [value_str] if value_str else []

                # Experience filtering
                min_experience = qdrant_filters.get('min_experience')
                max_experience = qdrant_filters.get('max_experience')

                if min_experience is not None or max_experience is not None:
                    candidate_exp_str = payload.get('total_experience', '0')

                    # Parse experience from strings like "2 years 8 months"
                    years_match = re.search(r'(\d+)\s*years?', candidate_exp_str)
                    months_match = re.search(r'(\d+)\s*months?', candidate_exp_str)

                    total_years = 0.0
                    if years_match:
                        total_years += float(years_match.group(1))
                    if months_match:
                        total_years += float(months_match.group(1)) / 12.0

                    # Check experience requirements
                    experience_match = True
                    if min_experience is not None and total_years < min_experience:
                        experience_match = False
                    if max_experience is not None and total_years > max_experience:
                        experience_match = False

                    criteria_matches.append(("experience", experience_match, True))

                    if not experience_match:
                        logger.info(f"🚫 Experience mismatch for {payload.get('name', 'Unknown')} - {total_years:.1f} years (required: {min_experience}-{max_experience} years)")

                # Location filtering (flexible matching)
                location_requirements = qdrant_filters.get('location')
                if location_requirements:
                    candidate_location = payload.get('location', '').lower()
                    location_match = False

                    if isinstance(location_requirements, list):
                        for req_location in location_requirements:
                            if req_location.lower() in candidate_location or candidate_location in req_location.lower():
                                location_match = True
                                break
                    else:
                        req_location = str(location_requirements).lower()
                        location_match = req_location in candidate_location or candidate_location in req_location

                    criteria_matches.append(("location", location_match, True))

                    if not location_match:
                        logger.info(f"🚫 Location mismatch for {payload.get('name', 'Unknown')} - location '{candidate_location}' doesn't match '{location_requirements}'")

                # Job title filtering (flexible matching)
                job_title_requirements = qdrant_filters.get('job_title')
                if job_title_requirements:
                    candidate_position = payload.get('current_position', '').lower()
                    title_match = False

                    if isinstance(job_title_requirements, list):
                        for req_title in job_title_requirements:
                            if req_title.lower() in candidate_position or any(word in candidate_position for word in req_title.lower().split()):
                                title_match = True
                                break
                    else:
                        req_title = str(job_title_requirements).lower()
                        if req_title in candidate_position or any(word in candidate_position for word in req_title.split()):
                            title_match = True

                    criteria_matches.append(("job_title", title_match, True))

                    if not title_match:
                        logger.info(f"🚫 Job title mismatch for {payload.get('name', 'Unknown')} - '{candidate_position}' doesn't match '{job_title_requirements}'")

                # Name filtering (exact or substring match)
                name_requirements = qdrant_filters.get('name')
                if name_requirements:
                    candidate_name = str(payload.get('name', '')).lower().strip()
                    name_match = False

                    for req_name in _normalize_filter_values(name_requirements):
                        if req_name and req_name in candidate_name:
                            name_match = True
                            break

                    criteria_matches.append(("name", name_match, True))

                    if not name_match:
                        logger.info(f"dYs_ Name mismatch for {payload.get('name', 'Unknown')} - does not contain required name '{name_requirements}'")

                # Email filtering (exact match)
                email_requirements = qdrant_filters.get('email')
                if email_requirements:
                    candidate_email = str(payload.get('email', '')).lower().strip()
                    email_match = False

                    for req_email in _normalize_filter_values(email_requirements):
                        if req_email and req_email == candidate_email:
                            email_match = True
                            break

                    criteria_matches.append(("email", email_match, True))

                    if not email_match:
                        logger.info(f"dYs_ Email mismatch for {payload.get('name', 'Unknown')} - email '{candidate_email}' doesn't match '{email_requirements}'")

                # Phone filtering (elastic digits-only match)
                phone_requirements = qdrant_filters.get('phone')
                if phone_requirements:
                    candidate_phone = str(payload.get('phone', '')).strip()
                    candidate_phone_digits = _digits_only(candidate_phone)
                    phone_match = False

                    for req_phone in _normalize_filter_values(phone_requirements):
                        req_digits = _digits_only(req_phone)
                        if _phone_digits_match(candidate_phone_digits, req_digits):
                            phone_match = True
                            break

                    criteria_matches.append(("phone", phone_match, True))

                    if not phone_match:
                        logger.info(f"dYs_ Phone mismatch for {payload.get('name', 'Unknown')} - phone '{candidate_phone}' doesn't match '{phone_requirements}'")

                # Role category filtering (flexible matching with LLM semantic similarity)
                role_category_requirements = qdrant_filters.get('role_category')
                if role_category_requirements:
                    candidate_role_category = payload.get('role_category', '').lower()
                    category_match = False

                    if isinstance(role_category_requirements, list):
                        for req_category in role_category_requirements:
                            # First try exact matching
                            if req_category.lower() in candidate_role_category or candidate_role_category in req_category.lower():
                                category_match = True
                                break
                            # Then try LLM semantic matching for better accuracy
                            try:
                                if await _llm_semantic_similarity(req_category, candidate_role_category):
                                    category_match = True
                                    break
                            except:
                                # Fallback to basic matching if LLM fails
                                pass
                    else:
                        req_category = str(role_category_requirements).lower()
                        # First try exact matching
                        if req_category in candidate_role_category or candidate_role_category in req_category:
                            category_match = True
                        # Then try LLM semantic matching
                        else:
                            try:
                                if await _llm_semantic_similarity(req_category, candidate_role_category):
                                    category_match = True
                            except:
                                # Fallback if LLM fails
                                pass

                    # Treat role category as optional if candidate metadata is missing to avoid false negatives.
                    is_role_category_required = bool(candidate_role_category.strip())
                    criteria_matches.append(("role_category", category_match, is_role_category_required))

                    if not category_match:
                        logger.info(f"🚫 Role category mismatch for {payload.get('name', 'Unknown')} - '{candidate_role_category}' doesn't match '{role_category_requirements}'")

                # Skills filtering (flexible matching)
                skills_requirements = qdrant_filters.get('skills')
                if skills_requirements:
                    candidate_skills = [s.lower() for s in payload.get('skills', [])]

                    if isinstance(skills_requirements, list):
                        required_skills = [skill.lower() for skill in skills_requirements]
                        matched_skills = []

                        # Check each required skill
                        for req_skill in required_skills:
                            skill_found = any(req_skill in cand_skill for cand_skill in candidate_skills)
                            matched_skills.append(skill_found)

                        # For skills, we track individual skill matches
                        for i, req_skill in enumerate(required_skills):
                            criteria_matches.append((f"skill_{req_skill}", matched_skills[i], True))

                        # Overall skills match logic
                        total_matched = sum(matched_skills)
                        skills_match = total_matched == len(required_skills)  # Default: ALL skills required

                        if not skills_match:
                            missing_skills = [req_skill for i, req_skill in enumerate(required_skills) if not matched_skills[i]]
                            logger.info(f"🚫 Skills mismatch for {payload.get('name', 'Unknown')} - missing: {', '.join(missing_skills)}")

                    else:
                        req_skill = str(skills_requirements).lower()
                        skills_match = any(req_skill in cand_skill for cand_skill in candidate_skills)
                        criteria_matches.append((f"skill_{req_skill}", skills_match, True))

                        if not skills_match:
                            logger.info(f"🚫 Skills mismatch for {payload.get('name', 'Unknown')} - missing: {req_skill}")

                # Company filtering (flexible matching)
                company_requirements = qdrant_filters.get('company')
                if company_requirements:
                    candidate_companies = [c.lower() for c in _extract_candidate_companies(payload)]
                    company_match = False

                    if isinstance(company_requirements, list):
                        for req_company in company_requirements:
                            if any(req_company.lower() in cand_company or cand_company in req_company.lower()
                                  for cand_company in candidate_companies):
                                company_match = True
                                break
                    else:
                        req_company = str(company_requirements).lower()
                        company_match = any(req_company in cand_company or cand_company in req_company
                                          for cand_company in candidate_companies)

                    criteria_matches.append(("company", company_match, True))

                    if not company_match:
                        logger.info(f"🚫 Company mismatch for {payload.get('name', 'Unknown')} - companies: {candidate_companies}, required: {company_requirements}")

                # Education filtering (flexible matching)
                education_requirements = qdrant_filters.get('education_level') or qdrant_filters.get('field_of_study') or qdrant_filters.get('institution')
                if education_requirements:
                    education_info = _extract_education_info(payload)
                    education_match = False

                    # Check education level
                    education_level_req = qdrant_filters.get('education_level')
                    if education_level_req:
                        candidate_edu_level = education_info['education_level'].lower()
                        if isinstance(education_level_req, list):
                            education_match = any(req.lower() in candidate_edu_level for req in education_level_req)
                        else:
                            education_match = str(education_level_req).lower() in candidate_edu_level

                    # Check field of study
                    field_of_study_req = qdrant_filters.get('field_of_study')
                    if field_of_study_req and not education_match:
                        candidate_field = education_info['education_field'].lower()
                        if isinstance(field_of_study_req, list):
                            education_match = any(req.lower() in candidate_field for req in field_of_study_req)
                        else:
                            education_match = str(field_of_study_req).lower() in candidate_field

                    # Check institution
                    institution_req = qdrant_filters.get('institution')
                    if institution_req and not education_match:
                        candidate_institution = education_info['university'].lower()
                        if isinstance(institution_req, list):
                            education_match = any(req.lower() in candidate_institution for req in institution_req)
                        else:
                            education_match = str(institution_req).lower() in candidate_institution

                    criteria_matches.append(("education", education_match, True))

                    if not education_match:
                        logger.info(f"🚫 Education mismatch for {payload.get('name', 'Unknown')} - education: {education_info}, required: {education_requirements}")

                # Final decision: Apply strict vs non-strict matching logic
                should_exclude = should_exclude_candidate(criteria_matches, strict_matching, result.get('score', 0.0))

                if not should_exclude:
                    filtered_results.append(result)
                else:
                    # Log exclusion reason
                    failed_criteria = [match[0] for match in criteria_matches if not match[1]]
                    matching_mode = "strict" if strict_matching else "non-strict"
                    logger.info(f"🚫 Excluding {payload.get('name', 'Unknown')} ({matching_mode} mode) - failed criteria: {', '.join(failed_criteria)}")

            logger.info(f"🎯 Post-retrieval filtering: {len(results)} → {len(filtered_results)} candidates")
            results = filtered_results

        # Helper function to generate selection reasons
        async def generate_selection_reason(payload: Dict[str, Any], match_details: Dict[str, bool], qdrant_filters: Dict[str, Any], search_keywords: List[str]) -> str:
            """Generate a detailed explanation of why this candidate was selected."""
            candidate_name = payload.get('name', 'Unknown')
            candidate_role = payload.get('current_position', 'Unknown')
            candidate_location = payload.get('location', 'Unknown')

            logger.info(f"[REASON_DEBUG] =========================")
            logger.info(f"[REASON_DEBUG] Generating selection reason for: {candidate_name}")
            logger.info(f"[REASON_DEBUG] Candidate role: {candidate_role}")
            logger.info(f"[REASON_DEBUG] Candidate location: {candidate_location}")
            logger.info(f"[REASON_DEBUG] Search keywords provided: {search_keywords}")
            logger.info(f"[REASON_DEBUG] Qdrant filters applied: {list(qdrant_filters.keys()) if qdrant_filters else 'None'}")
            logger.info(f"[REASON_DEBUG] Match details flags: {match_details}")
            logger.info(f"[REASON_DEBUG] =========================")

            # Validate inputs
            if not search_keywords:
                logger.warning(f"[REASON_DEBUG] WARNING: No search keywords provided for reason generation!")
            if not qdrant_filters:
                logger.warning(f"[REASON_DEBUG] WARNING: No qdrant filters provided for reason generation!")
            if not any(match_details.values()):
                logger.warning(f"[REASON_DEBUG] WARNING: No match details flags are True - this may indicate semantic-only matching!")

            reasons = []

            # Identifier matches first for clarity
            email_req = qdrant_filters.get('email')
            if email_req and match_details.get('email_match'):
                candidate_email = payload.get('email', 'Unknown')
                if candidate_email:
                    reasons.append(f"📧 Exact email match: {candidate_email}")

            phone_req = qdrant_filters.get('phone')
            if phone_req and match_details.get('phone_match'):
                candidate_phone = payload.get('phone', 'Unknown')
                if candidate_phone:
                    reasons.append(f"📞 Exact phone match: {candidate_phone}")

            name_req = qdrant_filters.get('name')
            if name_req and match_details.get('name_match'):
                candidate_full_name = payload.get('name', 'Unknown')
                req_name_display = name_req[0] if isinstance(name_req, list) and name_req else str(name_req)
                reasons.append(f"🧑 Name match: '{candidate_full_name}' matches requested '{req_name_display}'")

            # Location match
            if match_details.get('location_match'):
                candidate_location = payload.get('location', 'Unknown')
                req_locations = qdrant_filters.get('location', [])
                if isinstance(req_locations, list):
                    location_text = ', '.join(req_locations)
                else:
                    location_text = str(req_locations)
                reasons.append(f"🗺️ Location match: Located in '{candidate_location}' (matches requirement: {location_text})")

            # Experience match
            if match_details.get('experience_match'):
                candidate_exp = payload.get('total_experience', 'Unknown')
                min_exp = qdrant_filters.get('min_experience')
                max_exp = qdrant_filters.get('max_experience')
                if min_exp is not None:
                    reasons.append(f"📅 Experience match: {candidate_exp} experience (meets minimum {min_exp} years requirement)")
                elif max_exp is not None:
                    reasons.append(f"📅 Experience match: {candidate_exp} experience (within maximum {max_exp} years limit)")

            # Role/Position match
            if match_details.get('role_match'):
                candidate_role = payload.get('current_position', 'Unknown')
                job_titles = qdrant_filters.get('job_title', [])
                role_categories = qdrant_filters.get('role_category', [])
                if job_titles:
                    job_title_text = ', '.join(job_titles) if isinstance(job_titles, list) else str(job_titles)
                    reasons.append(f"💼 Role match: Current position '{candidate_role}' aligns with requirement: {job_title_text}")
                if role_categories:
                    category_text = ', '.join(role_categories) if isinstance(role_categories, list) else str(role_categories)
                    role_category = payload.get('role_category', 'Unknown')
                    reasons.append(f"🎯 Category match: Role category '{role_category}' matches: {category_text}")

            # Skills match
            if match_details.get('skills_match'):
                candidate_skills = payload.get('skills', [])
                required_skills = qdrant_filters.get('skills', [])
                if isinstance(required_skills, list):
                    matched_skills = []
                    candidate_name = str(payload.get('name', '')).lower().strip()
                    name_match = False

                    for req_name in _normalize_filter_values(name_requirements):
                        if req_name and req_name in candidate_name:
                            name_match = True
                            break

                    criteria_matches.append(("name", name_match, True))

                    if not name_match:
                        logger.info(f"dYs_ Name mismatch for {payload.get('name', 'Unknown')} - does not contain required name '{name_requirements}'")

                # Email filtering (exact match)
                email_requirements = qdrant_filters.get('email')
                if email_requirements:
                    candidate_email = str(payload.get('email', '')).lower().strip()
                    email_match = False

                    for req_email in _normalize_filter_values(email_requirements):
                        if req_email and req_email == candidate_email:
                            email_match = True
                            break

                    criteria_matches.append(("email", email_match, True))

                    if not email_match:
                        logger.info(f"dYs_ Email mismatch for {payload.get('name', 'Unknown')} - email '{candidate_email}' doesn't match '{email_requirements}'")

                # Phone filtering (elastic digits-only match)
                phone_requirements = qdrant_filters.get('phone')
                if phone_requirements:
                    candidate_phone = str(payload.get('phone', '')).strip()
                    candidate_phone_digits = _digits_only(candidate_phone)
                    phone_match = False

                    for req_phone in _normalize_filter_values(phone_requirements):
                        req_digits = _digits_only(req_phone)
                        if _phone_digits_match(candidate_phone_digits, req_digits):
                            phone_match = True
                            break

                    criteria_matches.append(("phone", phone_match, True))

                    if not phone_match:
                        logger.info(f"dYs_ Phone mismatch for {payload.get('name', 'Unknown')} - phone '{candidate_phone}' doesn't match '{phone_requirements}'")

                # Role category filtering (flexible matching with LLM semantic similarity)
                role_category_requirements = qdrant_filters.get('role_category')
                if role_category_requirements:
                    candidate_role_category = payload.get('role_category', '').lower()
                    category_match = False

                    if isinstance(role_category_requirements, list):
                        for req_category in role_category_requirements:
                            # First try exact matching
                            if req_category.lower() in candidate_role_category or candidate_role_category in req_category.lower():
                                category_match = True
                                break
                            # Then try LLM semantic matching for better accuracy
                            try:
                                if await _llm_semantic_similarity(req_category, candidate_role_category):
                                    category_match = True
                                    break
                            except:
                                # Fallback to basic matching if LLM fails
                                pass
                    else:
                        req_category = str(role_category_requirements).lower()
                        # First try exact matching
                        if req_category in candidate_role_category or candidate_role_category in req_category:
                            category_match = True
                        # Then try LLM semantic matching
                        else:
                            try:
                                if await _llm_semantic_similarity(req_category, candidate_role_category):
                                    category_match = True
                            except:
                                # Fallback if LLM fails
                                pass

                    # Treat role category as optional if candidate metadata is missing to avoid false negatives.
                    is_role_category_required = bool(candidate_role_category.strip())
                    criteria_matches.append(("role_category", category_match, is_role_category_required))

                    if not category_match:
                        logger.info(f"🚫 Role category mismatch for {payload.get('name', 'Unknown')} - '{candidate_role_category}' doesn't match '{role_category_requirements}'")

                # Skills filtering (flexible matching)
                skills_requirements = qdrant_filters.get('skills')
                if skills_requirements:
                    candidate_skills = [s.lower() for s in payload.get('skills', [])]

                    if isinstance(skills_requirements, list):
                        required_skills = [skill.lower() for skill in skills_requirements]
                        matched_skills = []

                        # Check each required skill
                        for req_skill in required_skills:
                            skill_found = any(req_skill in cand_skill for cand_skill in candidate_skills)
                            matched_skills.append(skill_found)

                        # For skills, we track individual skill matches
                        for i, req_skill in enumerate(required_skills):
                            criteria_matches.append((f"skill_{req_skill}", matched_skills[i], True))

                        # Overall skills match logic
                        total_matched = sum(matched_skills)
                        skills_match = total_matched == len(required_skills)  # Default: ALL skills required

                        if not skills_match:
                            missing_skills = [req_skill for i, req_skill in enumerate(required_skills) if not matched_skills[i]]
                            logger.info(f"🚫 Skills mismatch for {payload.get('name', 'Unknown')} - missing: {', '.join(missing_skills)}")

                    else:
                        req_skill = str(skills_requirements).lower()
                        skills_match = any(req_skill in cand_skill for cand_skill in candidate_skills)
                        criteria_matches.append((f"skill_{req_skill}", skills_match, True))

                        if not skills_match:
                            logger.info(f"🚫 Skills mismatch for {payload.get('name', 'Unknown')} - missing: {req_skill}")

                # Company filtering (flexible matching)
                company_requirements = qdrant_filters.get('company')
                if company_requirements:
                    candidate_companies = [c.lower() for c in _extract_candidate_companies(payload)]
                    company_match = False

                    if isinstance(company_requirements, list):
                        for req_company in company_requirements:
                            if any(req_company.lower() in cand_company or cand_company in req_company.lower()
                                  for cand_company in candidate_companies):
                                company_match = True
                                break
                    else:
                        req_company = str(company_requirements).lower()
                        company_match = any(req_company in cand_company or cand_company in req_company
                                          for cand_company in candidate_companies)

                    criteria_matches.append(("company", company_match, True))

                    if not company_match:
                        logger.info(f"🚫 Company mismatch for {payload.get('name', 'Unknown')} - companies: {candidate_companies}, required: {company_requirements}")

                # Education filtering (flexible matching)
                education_requirements = qdrant_filters.get('education_level') or qdrant_filters.get('field_of_study') or qdrant_filters.get('institution')
                if education_requirements:
                    education_info = _extract_education_info(payload)
                    education_match = False

                    # Check education level
                    education_level_req = qdrant_filters.get('education_level')
                    if education_level_req:
                        candidate_edu_level = education_info['education_level'].lower()
                        if isinstance(education_level_req, list):
                            education_match = any(req.lower() in candidate_edu_level for req in education_level_req)
                        else:
                            education_match = str(education_level_req).lower() in candidate_edu_level

                    # Check field of study
                    field_of_study_req = qdrant_filters.get('field_of_study')
                    if field_of_study_req and not education_match:
                        candidate_field = education_info['education_field'].lower()
                        if isinstance(field_of_study_req, list):
                            education_match = any(req.lower() in candidate_field for req in field_of_study_req)
                        else:
                            education_match = str(field_of_study_req).lower() in candidate_field

                    # Check institution
                    institution_req = qdrant_filters.get('institution')
                    if institution_req and not education_match:
                        candidate_institution = education_info['university'].lower()
                        if isinstance(institution_req, list):
                            education_match = any(req.lower() in candidate_institution for req in institution_req)
                        else:
                            education_match = str(institution_req).lower() in candidate_institution

                    criteria_matches.append(("education", education_match, True))

                    if not education_match:
                        logger.info(f"🚫 Education mismatch for {payload.get('name', 'Unknown')} - education: {education_info}, required: {education_requirements}")

                # Final decision: Apply strict vs non-strict matching logic
                should_exclude = should_exclude_candidate(criteria_matches, strict_matching, result.get('score', 0.0))

                if not should_exclude:
                    filtered_results.append(result)
                else:
                    # Log exclusion reason
                    failed_criteria = [match[0] for match in criteria_matches if not match[1]]
                    matching_mode = "strict" if strict_matching else "non-strict"
                    logger.info(f"🚫 Excluding {payload.get('name', 'Unknown')} ({matching_mode} mode) - failed criteria: {', '.join(failed_criteria)}")

            logger.info(f"🎯 Post-retrieval filtering: {len(results)} → {len(filtered_results)} candidates")
            results = filtered_results

        # Helper function to generate selection reasons
        async def generate_selection_reason(payload: Dict[str, Any], match_details: Dict[str, bool], qdrant_filters: Dict[str, Any], search_keywords: List[str]) -> str:
            """Generate a detailed explanation of why this candidate was selected."""
            candidate_name = payload.get('name', 'Unknown')
            candidate_role = payload.get('current_position', 'Unknown')
            candidate_location = payload.get('location', 'Unknown')

            logger.info(f"[REASON_DEBUG] =========================")
            logger.info(f"[REASON_DEBUG] Generating selection reason for: {candidate_name}")
            logger.info(f"[REASON_DEBUG] Candidate role: {candidate_role}")
            logger.info(f"[REASON_DEBUG] Candidate location: {candidate_location}")
            logger.info(f"[REASON_DEBUG] Search keywords provided: {search_keywords}")
            logger.info(f"[REASON_DEBUG] Qdrant filters applied: {list(qdrant_filters.keys()) if qdrant_filters else 'None'}")
            logger.info(f"[REASON_DEBUG] Match details flags: {match_details}")
            logger.info(f"[REASON_DEBUG] =========================")

            # Validate inputs
            if not search_keywords:
                logger.warning(f"[REASON_DEBUG] WARNING: No search keywords provided for reason generation!")
            if not qdrant_filters:
                logger.warning(f"[REASON_DEBUG] WARNING: No qdrant filters provided for reason generation!")
            if not any(match_details.values()):
                logger.warning(f"[REASON_DEBUG] WARNING: No match details flags are True - this may indicate semantic-only matching!")

            reasons = []

            # Identifier matches first for clarity
            email_req = qdrant_filters.get('email')
            if email_req and match_details.get('email_match'):
                candidate_email = payload.get('email', 'Unknown')
                if candidate_email:
                    reasons.append(f"📧 Exact email match: {candidate_email}")

            phone_req = qdrant_filters.get('phone')
            if phone_req and match_details.get('phone_match'):
                candidate_phone = payload.get('phone', 'Unknown')
                if candidate_phone:
                    reasons.append(f"📞 Exact phone match: {candidate_phone}")

            name_req = qdrant_filters.get('name')
            if name_req and match_details.get('name_match'):
                candidate_full_name = payload.get('name', 'Unknown')
                req_name_display = name_req[0] if isinstance(name_req, list) and name_req else str(name_req)
                reasons.append(f"🧑 Name match: '{candidate_full_name}' matches requested '{req_name_display}'")

            # Location match
            if match_details.get('location_match'):
                candidate_location = payload.get('location', 'Unknown')
                req_locations = qdrant_filters.get('location', [])
                if isinstance(req_locations, list):
                    location_text = ', '.join(req_locations)
                else:
                    location_text = str(req_locations)
                reasons.append(f"🗺️ Location match: Located in '{candidate_location}' (matches requirement: {location_text})")

            # Experience match
            if match_details.get('experience_match'):
                candidate_exp = payload.get('total_experience', 'Unknown')
                min_exp = qdrant_filters.get('min_experience')
                max_exp = qdrant_filters.get('max_experience')
                if min_exp is not None:
                    reasons.append(f"📅 Experience match: {candidate_exp} experience (meets minimum {min_exp} years requirement)")
                elif max_exp is not None:
                    reasons.append(f"📅 Experience match: {candidate_exp} experience (within maximum {max_exp} years limit)")

            # Role/Position match
            if match_details.get('role_match'):
                candidate_role = payload.get('current_position', 'Unknown')
                job_titles = qdrant_filters.get('job_title', [])
                role_categories = qdrant_filters.get('role_category', [])
                if job_titles:
                    job_title_text = ', '.join(job_titles) if isinstance(job_titles, list) else str(job_titles)
                    reasons.append(f"💼 Role match: Current position '{candidate_role}' aligns with requirement: {job_title_text}")
                if role_categories:
                    category_text = ', '.join(role_categories) if isinstance(role_categories, list) else str(role_categories)
                    role_category = payload.get('role_category', 'Unknown')
                    reasons.append(f"🎯 Category match: Role category '{role_category}' matches: {category_text}")

            # Skills match
            if match_details.get('skills_match'):
                candidate_skills = payload.get('skills', [])
                required_skills = qdrant_filters.get('skills', [])
                if isinstance(required_skills, list):
                    matched_skills = []
                    for req_skill in required_skills:
                        for cand_skill in candidate_skills:
                            if req_skill.lower() in cand_skill.lower():
                                matched_skills.append(cand_skill)
                                break
                    if matched_skills:
                        skills_text = ', '.join(matched_skills[:3])  # Show first 3 matched skills
                        if len(matched_skills) > 3:
                            skills_text += f" and {len(matched_skills) - 3} more"
                        reasons.append(f"🛠️ Skills match: Has required skills including {skills_text}")
                else:
                    for cand_skill in candidate_skills:
                        if str(required_skills).lower() in cand_skill.lower():
                            reasons.append(f"🛠️ Skills match: Has required skill '{cand_skill}'")
                            break

            # Company match
            if match_details.get('company_match'):
                candidate_companies = _extract_candidate_companies(payload)
                required_companies = qdrant_filters.get('company', [])
                if candidate_companies:
                    company_text = ', '.join(candidate_companies[:2])  # Show first 2 companies
                    if len(candidate_companies) > 2:
                        company_text += f" and {len(candidate_companies) - 2} more"
                    reasons.append(f"🏢 Company match: Experience at {company_text}")

            # Education match
            if match_details.get('education_match'):
                education_info = _extract_education_info(payload)
                edu_level = qdrant_filters.get('education_level')
                field_of_study = qdrant_filters.get('field_of_study')
                institution = qdrant_filters.get('institution')

                if edu_level and education_info['education_level'] != 'N/A':
                    reasons.append(f"🎓 Education level match: {education_info['education_level']} (meets requirement)")
                if field_of_study and education_info['education_field'] != 'N/A':
                    reasons.append(f"📚 Field of study match: {education_info['education_field']}")
                if institution and education_info['university'] != 'N/A':
                    reasons.append(f"🏛️ Institution match: {education_info['university']}")

            # Check for search keyword matches
            keyword_matches = []
            if search_keywords:
                logger.info(f"[REASON_DEBUG] Starting keyword matching for candidate: {payload.get('name', 'Unknown')}")
                logger.info(f"[REASON_DEBUG] Search keywords to match: {search_keywords}")

                candidate_text_fields = {
                    'role': payload.get('current_position', '').lower(),
                    'name': payload.get('name', '').lower(),
                    'location': payload.get('location', '').lower(),
                    'summary': payload.get('summary', '').lower(),
                    'skills': ' '.join(payload.get('skills', [])).lower(),
                    'companies': ' '.join([job.get('company', '') for job in payload.get('work_history', [])]).lower()
                }

                logger.info(f"[REASON_DEBUG] Candidate text fields prepared: {list(candidate_text_fields.keys())}")

                for keyword in search_keywords:
                    keyword_lower = keyword.lower().strip()
                    logger.info(f"[REASON_DEBUG] Processing keyword: '{keyword_lower}'")

                    matched_fields = []
                    field_match_details = {}

                    for field_name, field_text in candidate_text_fields.items():
                        if keyword_lower in field_text:
                            matched_fields.append(field_name)
                            # Extract context around the match
                            match_index = field_text.find(keyword_lower)
                            context_start = max(0, match_index - 20)
                            context_end = min(len(field_text), match_index + len(keyword_lower) + 20)
                            context = field_text[context_start:context_end]
                            field_match_details[field_name] = context.strip()
                            logger.info(f"[REASON_DEBUG] ✓ Keyword '{keyword_lower}' found in {field_name}: '...{context}...'")

                    if matched_fields:
                        keyword_matches.append({
                            'keyword': keyword,
                            'fields': matched_fields,
                            'context': field_match_details
                        })
                        logger.info(f"[REASON_DEBUG] ✓ Keyword '{keyword_lower}' matched in fields: {matched_fields}")
                    else:
                        logger.info(f"[REASON_DEBUG] ✗ Keyword '{keyword_lower}' not found in any field")

                # Generate detailed keyword match reasons
                if keyword_matches:
                    logger.info(f"[REASON_DEBUG] Total keyword matches found: {len(keyword_matches)}")
                    logger.info(f"[REASON_DEBUG] Matched keywords summary: {[m['keyword'] for m in keyword_matches]}")

                    # Check for multi-keyword matches across different categories
                    role_keywords = []
                    location_keywords = []
                    skill_keywords = []

                    for match in keyword_matches:
                        keyword = match['keyword']
                        fields = match['fields']

                        if 'role' in fields or 'summary' in fields:
                            role_keywords.append(keyword)
                        if 'location' in fields:
                            location_keywords.append(keyword)
                        if 'skills' in fields:
                            skill_keywords.append(keyword)

                    logger.info(f"[REASON_DEBUG] Keyword categorization - Role: {role_keywords}, Location: {location_keywords}, Skills: {skill_keywords}")

                    # Check for the specific case you mentioned (role + location combination)
                    if role_keywords and location_keywords:
                        role_keyword = role_keywords[0]
                        location_keyword = location_keywords[0]
                        reasons.append(f"🎯 Excellent match: Both '{role_keyword}' (role requirement) and '{location_keyword}' (location requirement) found in candidate profile")
                        logger.info(f"[REASON_DEBUG] ★ PERFECT COMBO: Role keyword '{role_keyword}' + Location keyword '{location_keyword}' both matched!")

                    # Create more specific reasons for keyword matches
                    for match in keyword_matches[:3]:  # Show details for first 3 matches
                        keyword = match['keyword']
                        fields = match['fields']
                        context = match.get('context', {})

                        # Skip if already handled in combo above
                        if role_keywords and location_keywords and (keyword in role_keywords[:1] or keyword in location_keywords[:1]):
                            continue

                        if 'role' in fields and 'location' in fields:
                            # Both role and location match - most relevant case
                            reasons.append(f"🎯 Perfect match: '{keyword}' found in both role ({payload.get('current_position', '')}) and location ({payload.get('location', '')})")
                        elif 'role' in fields:
                            reasons.append(f"💼 Role match: '{keyword}' matches current position '{payload.get('current_position', '')}'")
                        elif 'location' in fields:
                            reasons.append(f"🗺️ Location match: '{keyword}' matches candidate location '{payload.get('location', '')}'")
                        elif 'skills' in fields:
                            reasons.append(f"🛠️ Skills match: '{keyword}' found in technical skills")
                        elif 'companies' in fields:
                            reasons.append(f"🏢 Company match: '{keyword}' matches work experience")
                        elif 'summary' in fields:
                            reasons.append(f"📝 Profile match: '{keyword}' found in professional summary")
                        else:
                            # Fallback for other fields
                            field_names = ', '.join(fields)
                            reasons.append(f"🔍 Keyword match: '{keyword}' found in {field_names}")

                    # Add summary if there are more matches
                    if len(keyword_matches) > 3:
                        additional_keywords = [m['keyword'] for m in keyword_matches[3:]]
                        reasons.append(f"➕ Additional matches: {len(additional_keywords)} more keywords ({', '.join(additional_keywords[:2])}{'...' if len(additional_keywords) > 2 else ''})")
                else:
                    logger.info(f"[REASON_DEBUG] No keyword matches found for any search keywords")

            # If no specific matches but candidate was selected (semantic relevance)
            if not any(match_details.values()) and len(keyword_matches) == 0 and len(reasons) == 0:
                candidate_role = payload.get('current_position', 'Unknown')
                candidate_skills = payload.get('skills', [])
                if candidate_skills:
                    skills_sample = ', '.join(candidate_skills[:3])
                    if len(candidate_skills) > 3:
                        skills_sample += f" and {len(candidate_skills) - 3} more"
                    reasons.append(f"🔍 Semantic relevance: Profile matches query context - {candidate_role} with skills in {skills_sample}")
                else:
                    reasons.append(f"🔍 Semantic relevance: Profile '{candidate_role}' matches query context")

            final_reason = " | ".join(reasons) if reasons else "Selected based on overall profile relevance"

            # Final logging summary
            logger.info(f"[REASON_DEBUG] =========================")
            logger.info(f"[REASON_DEBUG] FINAL REASON for {candidate_name}:")
            logger.info(f"[REASON_DEBUG] Generated {len(reasons)} reason components:")
            for i, reason in enumerate(reasons, 1):
                logger.info(f"[REASON_DEBUG]   {i}. {reason}")
            logger.info(f"[REASON_DEBUG] Combined reason: {final_reason}")
            if keyword_matches:
                logger.info(f"[REASON_DEBUG] Keywords that contributed to selection: {[m['keyword'] for m in keyword_matches]}")
            else:
                logger.info(f"[REASON_DEBUG] No search keywords matched - reason based on other criteria")
            logger.info(f"[REASON_DEBUG] =========================")

            return final_reason

        # Step 5: Enhanced ranking based on final requirements
        def calculate_intent_score(result: Dict[str, Any]) -> float:
            """Calculate enhanced score based on final requirements matching."""
            base_score = result.get('score', 0.0)
            payload = result.get('payload', {})
            bonus_score = 0.0

            # Track what criteria were matched for transparency
            match_details = {
                'education_match': False,
                'role_match': False,
                'skills_match': False,
                'company_match': False,
                'experience_match': False,
                'location_match': False,
                'email_match': False,
                'phone_match': False,
                'name_match': False
            }

            # Identifier matching bonuses (email/phone/name)
            email_req = qdrant_filters.get('email')
            if email_req:
                candidate_email = (payload.get('email') or '').strip().lower()
                req_emails = email_req if isinstance(email_req, list) else [email_req]
                req_emails_norm = [str(e).strip().lower() for e in req_emails if str(e).strip()]
                if candidate_email and candidate_email in req_emails_norm:
                    bonus_score += 1.2
                    match_details['email_match'] = True

            phone_req = qdrant_filters.get('phone')
            if phone_req:
                cand_digits = re.sub(r"\D", "", str(payload.get('phone') or ''))
                req_phones = phone_req if isinstance(phone_req, list) else [phone_req]
                req_digits_list = [re.sub(r"\D", "", str(p)) for p in req_phones]
                for rd in req_digits_list:
                    if cand_digits and rd and (cand_digits == rd or cand_digits.endswith(rd) or rd.endswith(cand_digits)):
                        bonus_score += 1.2
                        match_details['phone_match'] = True
                        break

            name_req = qdrant_filters.get('name')
            if name_req:
                cand_name = (payload.get('name') or '').strip().lower()
                req_names = name_req if isinstance(name_req, list) else [name_req]
                req_names_norm = [str(n).strip().lower() for n in req_names if str(n).strip()]
                if cand_name and req_names_norm:
                    if any(cand_name == rn for rn in req_names_norm):
                        bonus_score += 1.0
                        match_details['name_match'] = True
                    elif any(rn in cand_name for rn in req_names_norm):
                        bonus_score += 0.6
                        match_details['name_match'] = True

            # Education matching bonus
            edu_reqs = intent_data.get("extracted_components", {}).get("education_requirements", {})
            if edu_reqs.get("has_requirement", False):
                candidate_edu = payload.get('education_level', '').lower()
                candidate_field = payload.get('education_field', '').lower()
                candidate_uni = payload.get('university', '').lower()

                # Degree level match
                if edu_reqs.get("degree_levels"):
                    for degree in edu_reqs["degree_levels"]:
                        if degree.lower() in candidate_edu:
                            bonus_score += 0.3
                            match_details['education_match'] = True
                            break

                # Field of study match
                if edu_reqs.get("fields_of_study"):
                    for field in edu_reqs["fields_of_study"]:
                        if field.lower() in candidate_field:
                            bonus_score += 0.25
                            match_details['education_match'] = True
                            break

                # Institution match
                if edu_reqs.get("institutions"):
                    for inst in edu_reqs["institutions"]:
                        if inst.lower() in candidate_uni:
                            bonus_score += 0.4  # High bonus for specific institution match
                            match_details['education_match'] = True
                            break

            # Role matching bonus
            role_reqs = intent_data.get("extracted_components", {}).get("role_requirements", {})
            if role_reqs.get("has_requirement", False) or qdrant_filters.get('role_category') or qdrant_filters.get('job_title'):
                # Prefer 'current_position' stored in payload; fallback to 'current_role'
                candidate_role = (payload.get('current_position') or payload.get('current_role') or '').lower()
                candidate_seniority = (payload.get('seniority') or '').lower()

                # Job title match (exact phrase or token overlap)
                job_titles = role_reqs.get("job_titles") or qdrant_filters.get('job_title') or []
                if job_titles:
                    for title in job_titles:
                        title_l = str(title).lower()
                        if not title_l:
                            continue
                        if title_l in candidate_role:
                            bonus_score += 0.35
                            match_details['role_match'] = True
                            break
                        # Token overlap heuristic
                        title_tokens = [t for t in re.split(r"\W+", title_l) if len(t) > 1]
                        role_tokens = [t for t in re.split(r"\W+", candidate_role) if len(t) > 1]
                        if not role_tokens:
                            continue
                        overlap = set(title_tokens) & set(role_tokens)
                        if 'full' in title_tokens and 'stack' in title_tokens and {'full', 'stack'} <= set(role_tokens):
                            bonus_score += 0.25
                            match_details['role_match'] = True
                            break
                        if len(overlap) >= 2:
                            bonus_score += 0.2
                            match_details['role_match'] = True
                            break
                        if len(overlap) >= 1 and any(tok in overlap for tok in ['developer', 'engineer', 'architect']):
                            bonus_score += 0.1
                            match_details['role_match'] = True
                            # don't break; allow stronger matches to be found for higher bonus in other titles

                # Seniority match
                if role_reqs.get("role_levels"):
                    for level in role_reqs["role_levels"]:
                        if str(level).lower() in candidate_seniority:
                            bonus_score += 0.25
                            match_details['role_match'] = True
                            break

                # Role category match (from final qdrant filters)
                role_cat_reqs = qdrant_filters.get('role_category')
                if role_cat_reqs:
                    cand_role_cat = (payload.get('role_category') or '').lower()
                    for rc in role_cat_reqs:
                        if str(rc).lower() in cand_role_cat:
                            bonus_score += 0.2
                            match_details['role_match'] = True
                            break

            # Skills matching bonus
            skill_reqs = intent_data.get("extracted_components", {}).get("skill_requirements", {})
            if skill_reqs.get("has_requirement", False):
                candidate_skills = [s.lower() for s in payload.get('skills', [])]

                # Technical skills match
                all_required_skills = []
                all_required_skills.extend(skill_reqs.get("technical_skills", []))
                all_required_skills.extend(skill_reqs.get("frameworks", []))
                all_required_skills.extend(skill_reqs.get("technologies", []))

                matched_skills = 0
                for skill in all_required_skills:
                    if skill.lower() in candidate_skills:
                        matched_skills += 1

                if all_required_skills:
                    skill_match_ratio = matched_skills / len(all_required_skills)
                    bonus_score += skill_match_ratio * 0.4  # Up to 0.4 bonus for all skills matched
                    if skill_match_ratio > 0:
                        match_details['skills_match'] = True

            # Company matching bonus
            comp_reqs = intent_data.get("extracted_components", {}).get("company_requirements", {})
            if comp_reqs.get("has_requirement", False):
                candidate_companies = [c.lower() for c in payload.get('companies', [])]

                # Specific company match
                if comp_reqs.get("specific_companies"):
                    for company in comp_reqs["specific_companies"]:
                        if any(company.lower() in cand_comp for cand_comp in candidate_companies):
                            bonus_score += 0.35
                            match_details['company_match'] = True
                            break

                # Company group match (like FAANG)
                if comp_reqs.get("company_groups"):
                    faang_companies = ["google", "facebook", "meta", "apple", "amazon", "netflix"]
                    big_tech = ["microsoft", "uber", "airbnb", "tesla", "salesforce"]

                    for group in comp_reqs["company_groups"]:
                        if "faang" in group.lower():
                            if any(faang in cand_comp for faang in faang_companies for cand_comp in candidate_companies):
                                bonus_score += 0.4
                                match_details['company_match'] = True
                                break
                        elif "big tech" in group.lower():
                            all_big_tech = faang_companies + big_tech
                            if any(tech in cand_comp for tech in all_big_tech for cand_comp in candidate_companies):
                                bonus_score += 0.3
                                match_details['company_match'] = True
                                break

            # Experience matching bonus
            exp_reqs = intent_data.get("extracted_components", {}).get("experience_requirements", {})
            if exp_reqs.get("has_requirement", False):
                candidate_exp_str = payload.get('total_experience', '0')

                # Extract years from candidate experience
                exp_match = re.search(r'(\d+)', candidate_exp_str)
                if exp_match:
                    candidate_years = int(exp_match.group(1))
                    min_years = exp_reqs.get("min_years", 0)

                    if candidate_years >= min_years:
                        bonus_score += 0.2
                        match_details['experience_match'] = True
                    elif candidate_years >= min_years * 0.8:  # Close match
                        bonus_score += 0.1
                        match_details['experience_match'] = True

            # Location matching bonus
            location_reqs = qdrant_filters.get('location')
            if location_reqs:
                candidate_location = payload.get('location', '').lower()
                location_match = False

                if isinstance(location_reqs, list):
                    for req_location in location_reqs:
                        if req_location.lower() in candidate_location or candidate_location in req_location.lower():
                            location_match = True
                            break
                else:
                    req_location = str(location_reqs).lower()
                    location_match = req_location in candidate_location or candidate_location in req_location

                if location_match:
                    bonus_score += 0.25  # Significant bonus for location match
                    match_details['location_match'] = True

            # Store match details for debugging
            result['match_details'] = match_details

            return base_score + bonus_score

        # Apply enhanced scoring and ranking
        for result in results:
            result['intent_score'] = calculate_intent_score(result)

        # Filter out candidates that don't meet location requirements if location is specified
        if has_post_retrieval_filters:
            location_requirements = qdrant_filters.get('location')
            if location_requirements:
                logger.info(f"Applying location filtering for: {location_requirements}")
                filtered_results = []

                for result in results:
                    payload = result.get('payload', {})
                    candidate_location = payload.get('location', '').lower()
                    location_match = False

                    if isinstance(location_requirements, list):
                        for req_location in location_requirements:
                            if req_location.lower() in candidate_location or candidate_location in req_location.lower():
                                location_match = True
                                break
                    else:
                        req_location = str(location_requirements).lower()
                        location_match = req_location in candidate_location or candidate_location in req_location

                    if location_match:
                        filtered_results.append(result)
                    else:
                        logger.info(f"🚫 Excluding {payload.get('name', 'Unknown')} - location '{payload.get('location', '')}' doesn't match '{location_requirements}'")

                results = filtered_results
                logger.info(f"Location filtering: {len(results)} candidates remaining")

        # Sort by intent score
        results = sorted(results, key=lambda x: x.get('intent_score', 0), reverse=True)

        # Limit results
        results = results[:limit]

        # Step 6: Format response with detailed analysis
        # Extract search keywords from final requirements for reason generation
        search_keywords = final_requirements.get("search_keywords", [])
        logger.info(f"[REASON_DEBUG] Extracted search keywords for reason generation: {search_keywords}")

        # Get shortlist status for all results
        resume_ids = [str(result.get('id')) for result in results]
        shortlist_status_map = {}
        if resume_ids:
            try:
                # Query shortlist status for all resume IDs
                placeholders = ', '.join(f'${i+1}' for i in range(len(resume_ids)))
                sql = f"SELECT id, is_shortlisted FROM public.qdrant_resumes WHERE id IN ({placeholders})"
                async with pg_client._pool.acquire() as conn:
                    rows = await conn.fetch(sql, *resume_ids)
                    shortlist_status_map = {str(row['id']): row['is_shortlisted'] for row in rows}
            except Exception as e:
                logger.warning(f"Could not fetch shortlist status: {e}")

        formatted_results = []
        for result in results:
            payload = result.get('payload', {})
            education_info = _extract_education_info(payload)
            match_details = result.get('match_details', {})

            # Get shortlist status
            resume_id = str(result.get('id'))
            is_shortlisted = shortlist_status_map.get(resume_id, False)

            # Generate selection reason
            logger.info(f"[REASON_DEBUG] About to generate selection reason for candidate {result.get('id')} (Score: {result.get('score', 0):.3f})")
            logger.info(f"[REASON_DEBUG] Passing parameters - Keywords: {len(search_keywords)} items, Filters: {len(qdrant_filters)} types, Match flags: {sum(match_details.values())}/{len(match_details)} true")

            selection_reason = await generate_selection_reason(payload, match_details, qdrant_filters, search_keywords)

            formatted_result = {
                'id': result.get('id'),
                'score': result.get('score', 0),
                'intent_score': result.get('intent_score', 0),
                'is_shortlisted': is_shortlisted,
                'candidate': {
                    'name': payload.get('name', 'N/A'),
                    'email': payload.get('email', 'N/A'),
                    'phone': payload.get('phone', 'N/A'),
                    'current_role': payload.get('current_position', 'N/A'),
                    'seniority': payload.get('seniority', 'N/A'),
                    'total_experience': payload.get('total_experience', 'N/A'),
                    'education_level': education_info['education_level'],
                    'education_field': education_info['education_field'],
                    'university': education_info['university'],
                    'skills': payload.get('skills', []),
                    'companies': _extract_candidate_companies(payload),
                    'location': payload.get('location', 'N/A')
                },
                'match_analysis': {
                    'semantic_relevance': result.get('score', 0),
                    'requirements_match': result.get('intent_score', 0) - result.get('score', 0),
                    'total_score': result.get('intent_score', 0),
                    'match_details': match_details,
                    'matching_criteria': [
                        criterion for criterion, matched in match_details.items()
                        if matched
                    ]
                },
                'selection_reason': selection_reason
            }
            formatted_results.append(formatted_result)

        return {
            "success": True,
            "query": query,
            "requested_by": user_id,
            "total_results": len(formatted_results),
            "intent_analysis": {
                "final_requirements": final_requirements,
                "filters_applied": qdrant_filters,
                "search_strategy": final_requirements.get("search_strategy", {})
            },
            "results": formatted_results,
            "processing_summary": {
                "query_analyzed": True,
                "filters_used": bool(qdrant_filters),
                "post_retrieval_filtering_applied": has_post_retrieval_filters,
                "semantic_fallback": not bool(qdrant_filters),
                "strict_matching_enabled": strict_matching,
                "matching_mode": "strict" if strict_matching else "partial",
                "candidates_before_filtering": len(results) if 'results' in locals() else 0,
                "candidates_after_filtering": len(formatted_results),
                "filter_types_applied": list(qdrant_filters.keys()),
                "search_approach": "hybrid" if has_post_retrieval_filters else "semantic_only",
                "margin": {"enabled": True, "ratio": margin_ratio, "top_semantic_score": top_semantic_score}
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in intent-based resume search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


async def llm_candidate_matcher(query: str, candidates: List[Dict], limit: int = 10) -> List[Dict]:
    """
    Use LLM to intelligently match candidates to query requirements.
    Instead of strict filtering, uses AI to understand semantic similarity.
    """
    if not candidates:
        return []

    try:
        # Create a summary of each candidate for LLM analysis
        candidate_summaries = []
        for i, candidate in enumerate(candidates):
            payload = candidate.get('payload', {})
            summary = {
                'index': i,
                'name': payload.get('name', 'Unknown'),
                'role': payload.get('current_position', 'Unknown'),
                'location': payload.get('location', 'Unknown'),
                'skills': payload.get('skills', [])[:10],  # Top 10 skills
                'experience': payload.get('total_experience', 'Unknown'),
                'role_category': payload.get('role_category', 'Unknown'),
                'summary': payload.get('summary', '')[:200] + '...' if payload.get('summary') else ''
            }
            candidate_summaries.append(summary)

        # Create prompt for LLM to rank candidates
        prompt = f"""
You are an expert recruiter. Analyze these candidates and rank them based on how well they match this query: "{query}"

Query: {query}

Candidates:
{chr(10).join([f"{i+1}. {c['name']} - {c['role']} from {c['location']} - Skills: {', '.join(c['skills'][:5])} - Experience: {c['experience']}" for i, c in enumerate(candidate_summaries)])}

Instructions:
1. Consider semantic similarity, not just exact matches
2. For location: "Ahmedabad, Gujarat" matches "Ahmedabad", "Gujarat", etc.
3. For skills: "AI/ML" matches "Machine Learning", "Deep Learning", "NLP", "TensorFlow", etc.
4. For roles: "AI Developer" matches "AI Engineer", "ML Engineer", "Software Engineer (AI/ML)", etc.

Return ONLY a JSON array of candidate indices (0-based) ranked by relevance, limited to top {limit}:
[0, 2, 1]
"""

        # Get LLM response
        client = azure_client.get_sync_client()
        if not client:
            logger.warning("Azure OpenAI not available, falling back to vector similarity")
            return candidates[:limit]

        response = client.chat.completions.create(
            model=azure_client.get_chat_deployment(),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.1
        )

        # Parse LLM response
        content = response.choices[0].message.content
        llm_response = content.strip() if content else ""

        # Clean and parse JSON
        import re, json
        llm_response = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", llm_response, flags=re.IGNORECASE)
        llm_response = re.sub(r"`{3,}", "", llm_response)

        try:
            ranked_indices = json.loads(llm_response)
            if not isinstance(ranked_indices, list):
                raise ValueError("Response is not a list")

            # Return candidates in ranked order
            ranked_candidates = []
            for idx in ranked_indices:
                if 0 <= idx < len(candidates):
                    ranked_candidates.append(candidates[idx])

            logger.info(f"LLM ranked {len(ranked_candidates)} candidates for query: {query}")
            return ranked_candidates[:limit]

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse LLM ranking response: {e}. Response: {llm_response}")
            return candidates[:limit]

    except Exception as e:
        logger.error(f"LLM candidate matching failed: {e}")
        return candidates[:limit]


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


# --- Shortlisting Endpoints ---

@app.get("/resumes/shortlisted", summary="Get Shortlisted Candidates", tags=["Shortlist"])
async def get_shortlisted_resumes(user_id: str):
    """Retrieve all shortlisted resumes for a specific user."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required.")
    
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="Database connection failed.")
    
    try:
        shortlisted = await pg_client.get_shortlisted_resumes(user_id)
        return JSONResponse(content={"shortlisted_candidates": shortlisted}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to get shortlisted resumes for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve shortlisted candidates.")

@app.patch("/resumes/{resume_id}/shortlist", summary="Update Shortlist Status", tags=["Shortlist"])
async def update_shortlist_status(resume_id: uuid.UUID, payload: ShortlistUpdate):
    """Update the shortlist status of a single resume."""
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="Database connection failed.")
    
    try:
        success = await pg_client.update_shortlist_status(resume_id, payload.is_shortlisted)
        if not success:
            raise HTTPException(status_code=404, detail="Resume not found.")
        return JSONResponse(content={"message": "Shortlist status updated successfully.", "is_shortlisted": payload.is_shortlisted}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to update shortlist status for resume {resume_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not update shortlist status.")

# --- Interview Question CRUD Endpoints ---

@app.post("/users/{user_id}/questions", response_model=InterviewQuestionInDB, status_code=201, summary="Create Interview Question", tags=["Interviews"])
async def create_interview_question(user_id: str, question: InterviewQuestionCreate):
    """Create a new custom interview question for a user."""
    if user_id != question.user_id:
        raise HTTPException(status_code=403, detail="User ID mismatch.")
    
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="Database connection failed.")
    
    try:
        # Convert Pydantic model to dictionary
        question_data = question.model_dump()
        new_question = await pg_client.create_interview_question(question_data)
        if not new_question:
            raise HTTPException(status_code=500, detail="Failed to create question.")
        return new_question
    except Exception as e:
        logger.error(f"Failed to create interview question for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not create interview question.")

@app.get("/users/{user_id}/questions", response_model=List[InterviewQuestionInDB], summary="Get All Interview Questions", tags=["Interviews"])
async def get_interview_questions(user_id: str):
    """Retrieve all custom interview questions for a user."""
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="Database connection failed.")
    
    try:
        questions = await pg_client.get_interview_questions(user_id)
        return questions
    except Exception as e:
        logger.error(f"Failed to get interview questions for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not retrieve interview questions.")

@app.put("/questions/{question_id}", response_model=InterviewQuestionInDB, summary="Update Interview Question", tags=["Interviews"])
async def update_interview_question(question_id: uuid.UUID, question_update: InterviewQuestionUpdate):
    """Update an existing interview question."""
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Convert Pydantic model to dictionary
        update_data = question_update.model_dump(exclude_unset=True)
        updated_question = await pg_client.update_interview_question(question_id, update_data)
        if not updated_question:
            raise HTTPException(status_code=404, detail="Question not found.")
        return updated_question
    except Exception as e:
        logger.error(f"Failed to update interview question {question_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not update interview question.")

@app.delete("/questions/{question_id}", status_code=204, summary="Delete Interview Question", tags=["Interviews"])
async def delete_interview_question(question_id: uuid.UUID):
    """Delete an interview question."""
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        success = await pg_client.delete_interview_question(question_id)
        if not success:
            raise HTTPException(status_code=404, detail="Question not found.")
        return None  # No content response
    except Exception as e:
        logger.error(f"Failed to delete interview question {question_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not delete interview question.")

# --- Interview Response Endpoints ---

@app.post("/users/{user_id}/interview-responses", response_model=dict, status_code=201, summary="Save Interview Response", tags=["Interviews"])
async def save_interview_response(
    user_id: str,
    question_id: str = Form(...),
    answer_text: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    audio_duration: Optional[float] = Form(None),
    response_time_seconds: Optional[float] = Form(None)
):
    """Save an interview response (text and/or audio)."""
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    try:
        # Validate question exists and belongs to user
        question_uuid = uuid.UUID(question_id)
        question = await pg_client.get_interview_question(question_uuid)
        if not question or question['user_id'] != user_id:
            raise HTTPException(status_code=404, detail="Question not found or access denied.")

        audio_file_path = None
        if audio_file:
            # Save audio file
            audio_dir = Path(settings.app.upload_dir) / "interview_audio"
            audio_dir.mkdir(exist_ok=True)
            
            file_extension = Path(audio_file.filename).suffix.lower() if audio_file.filename else ".wav"
            audio_filename = f"{user_id}_{question_id}_{uuid.uuid4()}{file_extension}"
            audio_file_path = audio_dir / audio_filename
            
            with open(audio_file_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)

        # Save response to database
        response_id = await pg_client.save_interview_response(
            user_id=user_id,
            question_id=question_uuid,
            answer_text=answer_text,
            audio_file_path=str(audio_file_path) if audio_file_path else None,
            audio_duration=audio_duration,
            response_time_seconds=response_time_seconds
        )

        return {
            "success": True,
            "response_id": response_id,
            "message": "Interview response saved successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save interview response for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not save interview response.")

# --- Call Initiation Endpoint ---

@app.post("/interviews/call", response_model=CallRecord, status_code=201, summary="Initiate an Interview Call", tags=["Interviews"])
async def initiate_interview_call(call_request: CallInitiationRequest):
    """
    Initiates an interview call for a shortlisted candidate.
    This is a placeholder and will log the call attempt.
    """
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="Database connection failed.")

    # Check if the candidate is shortlisted before allowing a call
    is_shortlisted = await pg_client.get_shortlist_status(call_request.resume_id)
    if not is_shortlisted:
        raise HTTPException(status_code=403, detail="Cannot initiate a call for a candidate who is not shortlisted.")

    logger.info(f"Initiating call for resume_id: {call_request.resume_id} by user_id: {call_request.user_id}")
    
    # In a real-world scenario, you would integrate with a service like Twilio here.
    # For now, we will just log the call to our database.
    try:
        call_record = await pg_client.log_interview_call(
            resume_id=call_request.resume_id,
            user_id=call_request.user_id,
            status="initiated",
            notes=call_request.notes
        )
        if not call_record:
            raise HTTPException(status_code=500, detail="Failed to log the call initiation.")
        
        return call_record
    except Exception as e:
        logger.error(f"Error initiating call for resume {call_request.resume_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not initiate the call.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app.debug,
        log_level=settings.app.log_level.lower()
    )
@app.get("/resumes")
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
    admin_view: bool = Query(True, description="If true, ignore user_id and return all resumes"),  # Changed default to True for development
    order_by: str = "-created_at",
):
    """Return paginated list of resumes from PostgreSQL mirror (qdrant_resumes).

    Query params:
    - page (1-based), page_size (max 100)
    - search (applies to name, current_position, summary, location, email)
    - name, email, location, job_title, role_category (ILIKE filters)
    - user_id: Optional. Only resumes with owner_user_id == user_id are returned if provided
    - admin_view: Defaults to true for development - returns all resumes if true, filters by user_id if false
    - order_by: one of created_at|embedding_generated_at|upload_timestamp|name|current_position with optional '-' prefix for DESC
    """
    # Normalize pagination
    page = max(1, int(page))
    if isinstance(limit, int) and limit > 0:
        page_size = limit
    page_size = max(1, min(100, int(page_size)))
    offset = (page - 1) * page_size

    # Ensure DB connectivity
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    # Scope resumes: admin lists all, otherwise require user_id
    owner_user_id: Optional[str]
    if admin_view:
        owner_user_id = str(user_id).strip() if user_id and str(user_id).strip() else None
    else:
        if not user_id or not str(user_id).strip():
            raise HTTPException(status_code=400, detail="user_id is required when admin_view=false")
        owner_user_id = str(user_id).strip()

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


@app.get("/admin/users")
async def admin_list_users(
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
    try:
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
    except Exception as e:
        logger.error(f"[ADMIN] Failed to list users: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch users")

    return {"success": True, "page": page, "limit": limit, "total": total, "items": items, "users": items}


@app.get("/admin/stats/resumes")
async def admin_stats_resumes(_auth: None = Depends(require_admin)):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    assert pg_client._pool is not None  # type: ignore[attr-defined]
    try:
        async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
            total = await conn.fetchval(f"SELECT COUNT(*) FROM {pg_client._table}")  # type: ignore[attr-defined]
            week = await conn.fetchval(
                f"SELECT COUNT(*) FROM {pg_client._table} WHERE created_at >= date_trunc('week', now())"  # type: ignore[attr-defined]
            )
            month = await conn.fetchval(
                f"SELECT COUNT(*) FROM {pg_client._table} WHERE created_at >= date_trunc('month', now())"  # type: ignore[attr-defined]
            )
    except Exception as e:
        logger.error(f"[ADMIN] Failed to compute resume stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute resume stats")

    return {"success": True, "total": int(total or 0), "this_week": int(week or 0), "this_month": int(month or 0)}


@app.get("/admin/stats/users")
async def admin_stats_users(_auth: None = Depends(require_admin)):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    assert pg_client._pool is not None  # type: ignore[attr-defined]
    try:
        async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
            total = await conn.fetchval("SELECT COUNT(*) FROM public.users")
            new_30d = await conn.fetchval(
                "SELECT COUNT(*) FROM public.users WHERE created_at >= now() - interval '30 days'"
            )
            active_30d = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT u.id) FROM public.users u
                LEFT JOIN public.user_resume_limits l ON l.user_id = u.id
                LEFT JOIN public.user_search_prompts p ON p.user_id = u.id
                WHERE (l.last_resume_uploaded_at >= now() - interval '30 days')
                   OR (p.asked_at >= now() - interval '30 days')
                """
            )
    except Exception as e:
        logger.error(f"[ADMIN] Failed to compute user stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute user stats")

    return {
        "success": True,
        "total": int(total or 0),
        "active_30d": int(active_30d or 0),
        "new_30d": int(new_30d or 0),
    }


@app.get("/admin/search-logs")
async def admin_search_logs(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=200),
    _auth: None = Depends(require_admin),
):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    assert pg_client._pool is not None  # type: ignore[attr-defined]
    offset = (page - 1) * limit
    total = 0
    items: List[Dict[str, Any]] = []
    try:
        async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
            try:
                total = await conn.fetchval("SELECT COUNT(*) FROM public.user_search_prompts")
            except Exception:
                total = 0
                return {"success": True, "page": page, "limit": limit, "total": total, "items": items, "logs": items}

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
    except Exception as e:
        logger.error(f"[ADMIN] Failed to fetch search logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch search logs")

    return {"success": True, "page": page, "limit": limit, "total": total, "items": items, "logs": items}





@app.delete("/resumes/{resume_id}")
async def delete_user_resume(
    resume_id: str,
    user_id: str = Query(..., description="The owner user_id of the resume")
):
    """Allow a user to delete one of their own resumes."""
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    cleaned_user_id = (user_id or "").strip()
    if not cleaned_user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    try:
        resume_uuid = uuid.UUID(str(resume_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid resume_id format")

    assert pg_client._pool is not None  # type: ignore[attr-defined]

    owner_user_id: Optional[str] = None
    deleted_pg = False
    try:
        async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
            row = await conn.fetchrow(
                f"SELECT owner_user_id FROM {pg_client._table} WHERE id = $1",
                resume_uuid,
            )  # type: ignore[attr-defined]
            if not row:
                raise HTTPException(status_code=404, detail="Resume not found")

            owner_user_id = (row.get("owner_user_id") or "").strip()
            if owner_user_id != cleaned_user_id:
                raise HTTPException(status_code=403, detail="You do not have permission to delete this resume")

            res = await conn.execute(
                f"DELETE FROM {pg_client._table} WHERE id = $1",
                resume_uuid,
            )  # type: ignore[attr-defined]
            deleted_pg = res.upper().startswith("DELETE")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"Error deleting resume {resume_id} for user {cleaned_user_id}: {exc}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to delete resume")

    deleted_qdrant = False
    try:
        deleted_qdrant = await qdrant_client.delete_resume(str(resume_uuid))
    except Exception as exc:
        logger.warning(
            f"Failed to delete resume {resume_id} for user {cleaned_user_id} from Qdrant: {exc}"
        )

    if not deleted_pg:
        raise HTTPException(status_code=500, detail="Resume removal failed")

    return {
        "success": True,
        "resume_id": str(resume_uuid),
        "deleted_postgres": deleted_pg,
        "deleted_qdrant": deleted_qdrant,
        "decremented_user_count": False,
    }

@app.delete("/admin/resumes/{resume_id}")
async def admin_delete_resume(resume_id: str, _auth: None = Depends(require_admin)):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    assert pg_client._pool is not None  # type: ignore[attr-defined]
    owner_user_id: Optional[str] = None
    deleted_pg = False
    deleted_q = False
    decremented = False
    try:
        async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
            try:
                row = await conn.fetchrow(f"SELECT owner_user_id FROM {pg_client._table} WHERE id = $1", resume_id)  # type: ignore[attr-defined]
                if row and row.get("owner_user_id"):
                    owner_user_id = str(row["owner_user_id"]) if row["owner_user_id"] else None
            except Exception:
                owner_user_id = None
            try:
                res = await conn.execute(f"DELETE FROM {pg_client._table} WHERE id = $1", resume_id)  # type: ignore[attr-defined]
                deleted_pg = res.upper().startswith("DELETE")
            except Exception as e:
                logger.error(f"[ADMIN] Failed to delete resume {resume_id} from Postgres: {e}")
                deleted_pg = False
    except Exception as e:
        logger.error(f"[ADMIN] Postgres operation error deleting resume {resume_id}: {e}")

    try:
        deleted_q = await qdrant_client.delete_resume(resume_id)
    except Exception as e:
        logger.error(f"[ADMIN] Failed to delete resume {resume_id} from Qdrant: {e}")

    if owner_user_id:
        try:
            decremented = await pg_client.decrement_user_resume_count(owner_user_id, 1, 0)
        except Exception:
            decremented = False

    return {
        "success": True,
        "resume_id": resume_id,
        "deleted_postgres": deleted_pg,
        "deleted_qdrant": deleted_q,
        "owner_user_id": owner_user_id,
        "decremented_user_count": decremented,
    }


@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: str, _auth: None = Depends(require_admin)):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")

    assert pg_client._pool is not None  # type: ignore[attr-defined]
    deleted_counts: Dict[str, int] = {
        "postgres_resumes": 0,
        "qdrant_points": 0,
        "prompts": 0,
        "limits_row": 0,
        "user_row": 0,
    }

    try:
        async with pg_client._pool.acquire() as conn:  # type: ignore[attr-defined]
            res = await conn.execute(f"DELETE FROM {pg_client._table} WHERE owner_user_id = $1", user_id)  # type: ignore[attr-defined]
            try:
                deleted_counts["postgres_resumes"] = int(res.split()[-1]) if res.upper().startswith("DELETE") else 0
            except Exception:
                deleted_counts["postgres_resumes"] = 0
            try:
                r = await conn.execute("DELETE FROM public.user_search_prompts WHERE user_id = $1", user_id)
                deleted_counts["prompts"] = int(r.split()[-1]) if r.upper().startswith("DELETE") else 0
            except Exception:
                pass
            try:
                r = await conn.execute("DELETE FROM public.user_resume_limits WHERE user_id = $1", user_id)
                deleted_counts["limits_row"] = int(r.split()[-1]) if r.upper().startswith("DELETE") else 0
            except Exception:
                pass
            try:
                r = await conn.execute("DELETE FROM public.users WHERE id = $1", user_id)
                deleted_counts["user_row"] = int(r.split()[-1]) if r.upper().startswith("DELETE") else 0
            except Exception:
                pass
    except Exception as e:
        logger.error(f"[ADMIN] Failed Postgres deletes for user {user_id}: {e}")

    try:
        client = qdrant_client.client
        collection = qdrant_client.collection_name
        if client is not None:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            f = Filter(must=[FieldCondition(key='owner_user_id', match=MatchValue(value=user_id))])
            client.delete(collection_name=collection, filter=f)
            deleted_counts["qdrant_points"] = 0
    except Exception as e:
        logger.error(f"[ADMIN] Failed Qdrant deletes for user {user_id}: {e}")

    return {"success": True, "user_id": user_id, "deleted": deleted_counts}


@app.get("/admin/users/{user_id}/limits")
async def admin_get_user_limits(user_id: str, _auth: None = Depends(require_admin)):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    data = await pg_client.get_user_resume_limits(user_id)
    if not data:
        await pg_client.init_user_resume_limits(user_id)
        data = await pg_client.get_user_resume_limits(user_id)
    return {"success": True, "user_id": user_id, "limits": data}


@app.patch("/admin/users/{user_id}/limits")
async def admin_update_user_limits(
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
    try:
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
    except Exception as e:
        logger.error(f"[ADMIN] Failed updating limits for {user_id}: {e}")

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


@app.get("/dashboard/metrics")
async def dashboard_metrics(user_id: str = Query(..., description="Owner user_id to scope metrics")):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    data = await pg_client.get_dashboard_metrics(user_id)
    return {"success": True, "user_id": user_id, "metrics": data}

@app.get("/dashboard/recent-activity")
async def dashboard_recent_activity(
    user_id: str = Query(..., description="Owner user_id to scope activity"),
    limit: int = Query(10, ge=1, le=50)
):
    ok = await pg_client.connect()
    if not ok:
        raise HTTPException(status_code=503, detail="PostgreSQL not configured or unavailable")
    items = await pg_client.get_recent_activity(user_id, limit)
    return {"success": True, "user_id": user_id, "items": items}

@app.get("/dashboard/top-candidates")
async def dashboard_top_candidates(
    user_id: str = Query(..., description="Owner user_id to scope candidates"),
    query: Optional[str] = Query(None, description="Optional query; falls back to recent prompt or top role"),
    limit: int = Query(3, ge=1, le=10)
):
    """Compute top matching candidates for this user and query (this week preferred).
    We reuse the intent analysis + vector search, restrict by owner_user_id, then post-filter to this week.
    """
    # Fallbacks if query not provided: recent prompt -> top role category -> generic
    effective_query = (query or "").strip() if isinstance(query, str) else ""
    if not effective_query:
        try:
            last_prompt = await pg_client.get_recent_prompt(user_id, days=30)
            if last_prompt:
                effective_query = last_prompt
        except Exception:
            pass
    if not effective_query:
        try:
            top_role = await pg_client.get_top_role_category(user_id)
            if top_role:
                effective_query = top_role
        except Exception:
            pass
    if not effective_query:
        effective_query = "best candidates"

    # Analyze intent
    intent = await analyze_query_intent(effective_query, user_id)
    if not intent.get("success"):
        raise HTTPException(status_code=400, detail="Intent analysis failed")
    final_requirements = intent.get("intent_analysis", {}).get("final_requirements", {}) or {}

    # Embed query enriched with keywords
    try:
        from src.resume_parser.clients.azure_openai import azure_client
        client = azure_client.get_sync_client()
        kws = final_requirements.get("search_keywords", [])
        emb_in = f"{effective_query}\n\nKeywords: {' '.join(kws[:20])}" if kws else effective_query
        resp = client.embeddings.create(input=emb_in, model=azure_client.get_embedding_deployment())
        vec = resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # Qdrant search restricted to owner
    filter_conditions = {"owner_user_id": user_id}
    results = await qdrant_client.search_similar(query_vector=vec, limit=300, filter_conditions=filter_conditions)
    if not results:
        return {"success": True, "user_id": user_id, "items": []}

    # Prefer candidates uploaded this week
    from datetime import datetime, timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)

    def within_week(payload: Dict[str, Any]) -> bool:
        ts = payload.get('upload_timestamp')
        if not ts:
            return False
        try:
            dt = datetime.fromisoformat(ts)
            return dt >= week_ago
        except Exception:
            return False

    # Score normalization (relative to top)
    top_score = max(r.get('score', 0.0) for r in results)
    def pct(r):
        s = r.get('score', 0.0)
        return int(round(100 * (s / top_score), 0)) if top_score > 0 else 0

    weekly = [r for r in results if within_week(r.get('payload', {}))]
    pool = weekly if weekly else results
    pool.sort(key=lambda r: r.get('score', 0.0), reverse=True)
    items = []
    for r in pool[:limit]:
        p = r.get('payload', {})
        skills = p.get('skills', []) or []
        items.append({
            "name": p.get('name', 'Unknown'),
            "initials": ''.join([w[0].upper() for w in str(p.get('name','?')).split()[:2]]),
            "match_percent": pct(r),
            "current_role": p.get('current_position', 'Unknown'),
            "location": p.get('location', 'Unknown'),
            "skills": skills[:2],
        })

    return {"success": True, "user_id": user_id, "items": items}

@app.get("/resumes/{resume_id}")
async def get_resume_detail(resume_id: str, fallback_to_qdrant: bool = True):
    """Get a single resume by id from the Postgres mirror; optionally fall back to Qdrant payload."""
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


@app.post("/user-search-prompts")
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
    """
    Store a user search prompt for analytics/feedback.

    Accepts form or JSON. JSON example:
    {"user_id":"u123","prompt":"senior backend","route":"search-resumes-intent-based","liked":true,"asked_at":"2025-09-25T10:01:47+00:00","response_meta":{"total_results":10}}
    """
    # Merge JSON body if provided (root-level) or nested json_body
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

    if not prompt or not str(prompt).strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    # Parse asked_at to datetime
    from datetime import datetime
    asked_dt = None
    if asked_at:
        try:
            asked_dt = datetime.fromisoformat(str(asked_at))
        except Exception:
            asked_dt = None

    # response_meta can be JSON string or dict
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


@app.patch("/user-search-prompts/{prompt_id}/feedback")
async def update_user_search_prompt_feedback(
    prompt_id: str,
    request: Request,
    liked: Optional[bool] = Form(None),
    json_body: Optional[Dict[str, Any]] = Body(None),
):
    """Update user feedback (liked) for a stored prompt.
    Accepts form or JSON body {"liked": true/false}.
    """
    # Normalize 'liked' from JSON when Content-Type is application/json
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

    # Coerce liked from common string forms
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






