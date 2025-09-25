"""
Analyze disliked prompts and generate diagnostic reports.

This script:
 - Fetches prompts from public.user_search_prompts where liked = false
 - Runs intent analysis and Qdrant retrieval diagnostics
 - Emits a JSONL report summarizing likely causes and suggestions

Usage:
  python scripts/analyze_disliked_prompts.py --limit 50 --out logs/disliked_reports.jsonl

Environment:
  Requires Postgres (POSTGRES_DSN or POSTGRES_*), Qdrant, and Azure OpenAI configured (for best results).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config.settings import settings
from src.resume_parser.database.postgres_client import pg_client
from src.resume_parser.database.qdrant_client import qdrant_client
from src.resume_parser.clients.azure_openai import azure_client


@dataclass
class PromptRow:
    id: str
    user_id: str
    prompt: str
    route: str
    liked: Optional[bool]
    asked_at: Optional[str]


@dataclass
class PromptReport:
    prompt_id: str
    user_id: str
    prompt: str
    asked_at: Optional[str]
    route: str
    parsed_primary_intent: Optional[str]
    qdrant_filters: Dict[str, Any]
    search_keywords: List[str]
    vector_dim: Optional[int]
    unfiltered_candidates: int
    prefiltered_candidates: int
    issues: List[str]
    suggestions: List[str]
    timestamp: str


async def fetch_disliked_prompts(limit: int) -> List[PromptRow]:
    ok = await pg_client.connect()
    if not ok:
        print("[WARN] Postgres not configured; returning empty prompt list")
        return []
    assert pg_client._pool is not None  # noqa: SLF001
    sql = (
        "SELECT id, user_id, prompt, COALESCE(route, 'search-resumes-intent-based') as route,"
        " liked, asked_at FROM public.user_search_prompts"
        " WHERE liked = false ORDER BY asked_at DESC NULLS LAST LIMIT $1"
    )
    async with pg_client._pool.acquire() as conn:  # noqa: SLF001
        rows = await conn.fetch(sql, limit)
    prompts: List[PromptRow] = []
    for r in rows:
        prompts.append(
            PromptRow(
                id=str(r["id"]),
                user_id=str(r["user_id"]),
                prompt=str(r["prompt"]),
                route=str(r["route"]),
                liked=bool(r["liked"]) if r["liked"] is not None else None,
                asked_at=r["asked_at"].isoformat() if r["asked_at"] else None,
            )
        )
    return prompts


async def run_intent_analysis(query: str) -> Tuple[Dict[str, Any], List[str]]:
    """Call in-process intent analyzer and extract final requirements."""
    issues: List[str] = []
    try:
        from app import analyze_query_intent  # import the function, not HTTP route
        res = await analyze_query_intent(query=query)  # type: ignore[arg-type]
        if not isinstance(res, dict) or not res.get("success"):
            issues.append("intent_analysis_failed")
            return {}, issues
        intent = res.get("intent_analysis", {})
        final_reqs = intent.get("final_requirements", {}) or {}
        if not final_reqs:
            issues.append("final_requirements_missing")
        return final_reqs, issues
    except Exception as e:
        issues.append(f"intent_exception:{e}")
        return {}, issues


async def generate_embedding_for_intent(query: str, final_requirements: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    issues: List[str] = []
    try:
        client = azure_client.get_sync_client()
        snippet_terms: List[str] = final_requirements.get("search_keywords", [])[:20]
        if snippet_terms:
            embedding_input = f"{query}\n\nKeywords: {' '.join(snippet_terms)}"
        else:
            embedding_input = query
        resp = client.embeddings.create(input=embedding_input, model=azure_client.get_embedding_deployment())
        vec = resp.data[0].embedding
        return vec, issues
    except Exception as e:
        issues.append(f"embedding_error:{e}")
        return [], issues


def build_identifier_prefilters(final_requirements: Dict[str, Any]) -> Dict[str, Any]:
    qf = final_requirements.get("qdrant_filters", {}) or {}
    pre: Dict[str, Any] = {}
    # The intent route only pre-filters identifiers (name/email/phone)
    for field in ("name", "email", "phone"):
        v = qf.get(field)
        if v:
            pre[field] = v if isinstance(v, list) else [v]
    return pre


async def qdrant_diagnostics(vec: List[float], pre_filters: Dict[str, Any], search_limit: int = 200) -> Tuple[int, int, List[str]]:
    issues: List[str] = []
    unfiltered = 0
    filtered = 0
    try:
        unres = await qdrant_client.search_similar(query_vector=vec, limit=search_limit, filter_conditions=None)
        unfiltered = len(unres or [])
    except Exception as e:
        issues.append(f"qdrant_unfiltered_error:{e}")
    try:
        if pre_filters:
            fres = await qdrant_client.search_similar(query_vector=vec, limit=search_limit, filter_conditions=pre_filters)
            filtered = len(fres or [])
        else:
            filtered = unfiltered
    except Exception as e:
        issues.append(f"qdrant_filtered_error:{e}")
    return unfiltered, filtered, issues


def suggest_improvements(final_requirements: Dict[str, Any], unfiltered: int, filtered: int, issues: List[str]) -> List[str]:
    suggestions: List[str] = []
    qf = final_requirements.get("qdrant_filters", {}) or {}
    if not final_requirements:
        suggestions.append("Intent parser returned no final requirements; examine prompt parsing and enable LLM refinement.")
    if "embedding_error" in " ".join(issues):
        suggestions.append("Embedding failed; verify Azure OpenAI credentials and embedding deployment.")
    if unfiltered == 0:
        suggestions.append("No semantic candidates from Qdrant; check if collection has points and vector size is correct.")
    if qf and filtered == 0 and unfiltered > 0:
        suggestions.append("Identifier pre-filter eliminated all candidates; relax name/email/phone filters or retry with post-filtering only.")
    if final_requirements.get("search_keywords") and unfiltered < 5:
        suggestions.append("Consider expanding keywords/role synonyms in intent refinement for better recall.")
    if not suggestions:
        suggestions.append("Review ranking/filters; consider margin softening or re-ranking logic for near-miss results.")
    return suggestions


async def analyze_disliked(limit: int, out_path: Optional[str]) -> int:
    prompts = await fetch_disliked_prompts(limit)
    if not prompts:
        print("No disliked prompts found or Postgres unavailable.")
        return 0

    out_fp = None
    if out_path:
        out_fp = open(out_path, "w", encoding="utf-8")

    count = 0
    for row in prompts:
        final_reqs, intent_issues = await run_intent_analysis(row.prompt)
        vec, embed_issues = await generate_embedding_for_intent(row.prompt, final_reqs)
        pre = build_identifier_prefilters(final_reqs)
        unfiltered, filtered, qdrant_issues = await qdrant_diagnostics(vec, pre)

        primary_intent = (final_reqs.get("search_strategy", {}) or {}).get("primary_intent")
        report = PromptReport(
            prompt_id=row.id,
            user_id=row.user_id,
            prompt=row.prompt,
            asked_at=row.asked_at,
            route=row.route,
            parsed_primary_intent=primary_intent,
            qdrant_filters=final_reqs.get("qdrant_filters", {}) or {},
            search_keywords=final_reqs.get("search_keywords", []) or [],
            vector_dim=(len(vec) if isinstance(vec, list) else None),
            unfiltered_candidates=unfiltered,
            prefiltered_candidates=filtered,
            issues=[*intent_issues, *embed_issues, *qdrant_issues],
            suggestions=suggest_improvements(final_reqs, unfiltered, filtered, [*intent_issues, *embed_issues, *qdrant_issues]),
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        line = json.dumps(asdict(report), ensure_ascii=False)
        if out_fp:
            out_fp.write(line + "\n")
        else:
            print(line)
        count += 1

    if out_fp:
        out_fp.close()
        print(f"Wrote {count} reports to {out_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze disliked prompts and generate diagnostics")
    parser.add_argument("--limit", type=int, default=50, help="Max prompts to analyze")
    parser.add_argument("--out", type=str, default="", help="Output JSONL file (default: stdout)")
    args = parser.parse_args()

    out_path = args.out if args.out else None
    asyncio.run(analyze_disliked(args.limit, out_path))


if __name__ == "__main__":
    main()

