import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure project root on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")

async def clear_postgres(connection_string: str) -> None:
    conn: Optional[asyncpg.Connection] = None
    try:
        conn = await asyncpg.connect(dsn=connection_string)
        await conn.execute("TRUNCATE TABLE public.qdrant_resumes RESTART IDENTITY CASCADE")
        logging.info("Cleared Postgres table public.qdrant_resumes")
    finally:
        if conn:
            await conn.close()


def clear_qdrant(host: str, port: int, collection: str, vector_size: int) -> None:
    client = QdrantClient(host=host, port=port, timeout=10)

    if client.collection_exists(collection):
        client.delete_collection(collection)
        logging.info(f"Deleted Qdrant collection '{collection}'")
    else:
        logging.info(f"Collection '{collection}' did not exist; nothing to delete")

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    logging.info(f"Created empty Qdrant collection '{collection}'")


async def main() -> None:
    conn_str = settings.postgres.connection_string
    if not conn_str:
        raise SystemExit("Postgres connection string not configured")

    clear_qdrant(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        collection=settings.qdrant.collection_name,
        vector_size=settings.qdrant.vector_size,
    )

    await clear_postgres(conn_str)


if __name__ == "__main__":
    asyncio.run(main())
