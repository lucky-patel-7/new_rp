"""
Backfill script to normalize and add helper fields in Qdrant payloads:
- phone_digits: digits-only version of phone
- email_lc: lowercased email
- name_lc: lowercased name

Usage:
  python scripts/backfill_payloads.py

Optional env via .env (loaded through config.settings):
  QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME
"""

from typing import Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from config.settings import settings
import re


def normalize_phone_digits(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\D", "", str(value))


def normalize_lower(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def main(batch: int = 256) -> None:
    client = QdrantClient(host=settings.qdrant.host, port=settings.qdrant.port, timeout=30)
    collection = settings.qdrant.collection_name

    offset = None
    total = 0
    updated = 0

    print(f"Connecting to Qdrant {settings.qdrant.host}:{settings.qdrant.port}, collection={collection}")

    while True:
        points, offset = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=batch,
            offset=offset,
        )

        if not points:
            break

        total += len(points)

        for p in points:
            payload = p.payload or {}
            to_set = {}

            # Compute normalized fields
            phone_digits = normalize_phone_digits(payload.get('phone'))
            email_lc = normalize_lower(payload.get('email'))
            name_lc = normalize_lower(payload.get('name'))

            # Only set if different or missing
            if payload.get('phone_digits') != phone_digits:
                to_set['phone_digits'] = phone_digits
            if payload.get('email_lc') != email_lc:
                to_set['email_lc'] = email_lc
            if payload.get('name_lc') != name_lc:
                to_set['name_lc'] = name_lc

            if to_set:
                client.set_payload(collection_name=collection, payload=to_set, points=[p.id])
                updated += 1

        if offset is None:
            break

    print(f"Processed {total} points. Updated {updated} payloads.")


if __name__ == "__main__":
    main()

