"""
Intent test runner: posts a suite of prompts to /analyze-query-intent
and prints refined qdrant_filters, search_keywords, and primary_intent.

Usage:
  INTENT_BASE_URL=http://localhost:8000 python scripts/intent_test_runner.py

If INTENT_BASE_URL is not set, defaults to http://localhost:8000.
"""

from __future__ import annotations

import os
import sys
import json
from typing import Dict, Any, List

import httpx


BASE_URL = os.getenv("INTENT_BASE_URL", "").rstrip("/") if os.getenv("INTENT_BASE_URL") else ""
ROUTE = "/analyze-query-intent"


PROMPTS: Dict[str, List[str]] = {
    "person_name": [
        "give me lucky patel’s resume",
        "i need dhruv rana profile",
        "resume of jane doe",
        "show me john d.’s cv",
        "candidate named akash shah",
    ],
    "contact": [
        "is there an email like rohampriya@gmail.com",
        "do we have a phone number like +91 9975717019",
        "find profile with phone 99757 17019",
        "candidate with email john.smith@acme.com",
    ],
    "role": [
        "give me full stack developer",
        "looking for backend engineer",
        "need senior software engineer",
        "show principal data engineer resumes",
        "find devops engineer profiles",
    ],
    "skills": [
        "need react and node.js developer",
        "python + django developer with celery",
        "go or rust backend developer",
        "java Spring Boot microservices engineer",
        "aws + kubernetes + terraform devops",
    ],
    "company": [
        "people working at google",
        "candidates from silvertouch technologies",
        "ex-amazon software engineers",
        "faang data scientists",
        "startup experience full stack developers",
    ],
    "location": [
        "frontend developer in mumbai",
        "react native developer from pune or remote",
        "hyderabad based data analyst",
        "willing to relocate to bangalore",
        "candidates near ahmedabad",
    ],
    "experience": [
        "3-5 years experience backend engineer",
        "at least 7 years as software architect",
        "up to 2 years experience data analyst",
        "fresher python developer",
    ],
    "education": [
        "b.tech in computer science required",
        "mba from top universities",
        "ph.d in machine learning",
        "graduates from iit or nit",
    ],
    "hybrid": [
        "senior backend engineer in bangalore with golang and kubernetes",
        "full stack developer react + node, 4+ years, pune",
        "data scientist from iit with pytorch and nlp, 3-6 years",
        "devops engineer aws terraform, remote ok, faang experience preferred",
        "ml engineer in hyderabad, masters in cs, computer vision",
    ],
    "very_complex": [
        "Senior Backend Engineer in Bangalore, 6–9 years, Go + Kubernetes + Kafka, AWS, product-based, FAANG or similar, notice ≤30 days, willing to relocate within India",
        "Principal Software Architect, 12+ years, microservices on Azure, .NET + C#, event-driven architectures, previously at Microsoft/Amazon, remote-first, US time overlap",
        "Data Scientist with PhD in ML/Stats from top-tier (IIT/IISc/MIT/CMU), 5–8 years, NLP + Transformers + PyTorch, MLOps (MLflow/Kubeflow), published papers preferred, Bengaluru or Hyderabad",
        "Full Stack Lead, 8–12 years, React + Node + GraphQL + Postgres, team lead experience, startup background, hybrid in Pune/Mumbai, joining in ≤45 days",
    ],
}


def detect_base_url() -> str:
    # 1) Explicit env var takes precedence
    if BASE_URL:
        return BASE_URL
    # 2) Try common defaults by pinging /health
    candidates = ["http://localhost:8000", "http://localhost:8000"]
    for base in candidates:
        try:
            with httpx.Client(timeout=3.0) as c:
                r = c.get(base + "/health")
                if r.status_code == 200:
                    return base
        except Exception:
            continue
    # 3) Fallback to 8000
    return "http://localhost:8000"


def run_suite() -> int:
    errors = 0
    base = detect_base_url()
    print(f"Posting to {base}{ROUTE}\n")
    with httpx.Client(timeout=30.0) as client:
        for section, prompts in PROMPTS.items():
            print(f"=== {section.upper()} ===")
            for q in prompts:
                try:
                    resp = client.post(f"{base}{ROUTE}", data={"query": q})
                    resp.raise_for_status()
                    data = resp.json()
                    intent = data.get("intent_analysis", {})
                    final_req = intent.get("final_requirements", {})
                    filters = final_req.get("qdrant_filters", {})
                    keywords = final_req.get("search_keywords", [])
                    strat = final_req.get("search_strategy", {})
                    primary = strat.get("primary_intent") or intent.get("query_metadata", {}).get("primary_intent")

                    print(f"- {q}")
                    print(f"  primary_intent: {primary}")
                    print(f"  qdrant_filters: {json.dumps(filters, ensure_ascii=False)}")
                    if keywords:
                        print(f"  search_keywords: {keywords[:8]}")
                except Exception as e:
                    errors += 1
                    print(f"- {q}")
                    print(f"  ERROR: {e}")
            print()
    print(f"Done. Errors: {errors}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(run_suite())
