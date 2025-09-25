"""
Qdrant vector database client for resume embeddings.
"""

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
from config.settings import settings
from ..core.models import EmbeddingData

logger = logging.getLogger(__name__)


class QdrantVectorClient:
    """Client for managing Qdrant vector database operations."""

    def __init__(self):
        """Initialize Qdrant client."""
        self.client = None
        self.collection_name = settings.qdrant.collection_name
        self.vector_size = settings.qdrant.vector_size
        self._connected = False
        self._connect()

    def _connect(self):
        """Connect to Qdrant and ensure collection exists."""
        try:
            logger.info(f"Attempting to connect to Qdrant at {settings.qdrant.host}:{settings.qdrant.port}")

            # Add timeout to avoid hanging
            self.client = QdrantClient(
                host=settings.qdrant.host,
                port=settings.qdrant.port,
                timeout=10  # 10 second timeout
            )

            logger.info("Client created, checking collection...")
            self._ensure_collection_exists()
            self._connected = True
            logger.info("Successfully connected to Qdrant")
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant: {e}")
            logger.warning("Vector search functionality will be disabled")
            self._connected = False
            self.client = None

    def _ensure_collection_exists(self):
        """Ensure the collection exists, create if it doesn't."""
        try:
            collections = self.client.get_collections() # type: ignore
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection( # type: ignore
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    async def store_embedding(
        self,
        user_id: str,
        embedding_vector: List[float],
        payload: Dict[str, Any]
    ) -> str:
        """
        Store embedding in Qdrant with payload data.

        Args:
            user_id: Unique identifier for the resume
            embedding_vector: Vector representation
            payload: Additional data to store with the vector

        Returns:
            str: Point ID in Qdrant
        """
        if not self._connected:
            logger.warning("Qdrant not connected, skipping embedding storage")
            return user_id

        try:
            # Create point with user_id as the point ID
            point = PointStruct(
                id=user_id,
                vector=embedding_vector,
                payload=payload
            )

            # Store in Qdrant
            self.client.upsert( # type: ignore
                collection_name=self.collection_name,
                points=[point]
            )

            logger.info(f"Successfully stored embedding for user: {user_id}")
            return user_id

        except Exception as e:
            logger.error(f"Error storing embedding for user {user_id}: {e}")
            raise

    async def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar resume embeddings.

        Args:
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            filter_conditions: Optional filter conditions

        Returns:
            List of similar resumes with scores
        """
        if not self._connected:
            logger.warning("Qdrant not connected, returning empty search results")
            return []

        try:
            # High-level request log
            vec_len = len(query_vector) if isinstance(query_vector, list) else 0
            logger.info(
                f"🚀 Qdrant.search_similar | vector_len={vec_len} | limit={limit} | filters={filter_conditions}"
            )
            # Prepare filter if provided
            search_filter = None
            if filter_conditions:
                logger.info(f"🔍 Qdrant search called with filter_conditions: {filter_conditions}")
                logger.info(f"🔍 Filter conditions type: {type(filter_conditions)}")
                # Convert our filter format to Qdrant filter format
                qdrant_must = []
                qdrant_should = []

                for field, condition in filter_conditions.items():
                    # Defer location to Python post-processing for flexible matching
                    if field == 'location' and isinstance(condition, str):
                        logger.info(f"🗺️ Will apply location filter in post-processing: {condition}")
                        continue

                    # Normalize values from condition
                    if isinstance(condition, dict) and '$in' in condition:
                        values = list(condition['$in'])
                    elif isinstance(condition, list):
                        values = list(condition)
                    else:
                        values = [condition]

                    # Special case: phone → OR across phone and phone_digits
                    if field == 'phone':
                        from qdrant_client.models import FieldCondition, MatchAny
                        import re as _re
                        try:
                            qdrant_should.append(
                                FieldCondition(key='phone', match=MatchAny(any=values)) # type: ignore
                            )
                        except Exception:
                            pass

                        # Build digits-only set for phone_digits
                        digit_values = []
                        seen = set()
                        for v in values:
                            try:
                                d = ''.join(_re.findall(r'\d+', str(v)))
                                if d and d not in seen:
                                    digit_values.append(d)
                                    seen.add(d)
                            except Exception:
                                continue
                        if digit_values:
                            try:
                                qdrant_should.append(
                                    FieldCondition(key='phone_digits', match=MatchAny(any=digit_values))
                                )
                            except Exception:
                                pass
                        continue

                    # Special case: email → OR across email and email_lc
                    if field == 'email':
                        from qdrant_client.models import FieldCondition, MatchAny
                        try:
                            qdrant_should.append(FieldCondition(key='email', match=MatchAny(any=values)))# type: ignore
                        except Exception:
                            pass
                        try:
                            values_lc = [str(v).strip().lower() for v in values if str(v).strip()]
                            if values_lc:
                                qdrant_should.append(FieldCondition(key='email_lc', match=MatchAny(any=values_lc)))
                        except Exception:
                            pass
                        continue

                    # Special case: name → OR across name and name_lc
                    if field == 'name':
                        from qdrant_client.models import FieldCondition, MatchAny
                        try:
                            qdrant_should.append(FieldCondition(key='name', match=MatchAny(any=values)))# type: ignore
                        except Exception:
                            pass
                        try:
                            values_lc = [str(v).strip().lower() for v in values if str(v).strip()]
                            if values_lc:
                                qdrant_should.append(FieldCondition(key='name_lc', match=MatchAny(any=values_lc)))
                        except Exception:
                            pass
                        continue

                    # Default handling for other fields
                    if isinstance(condition, dict) and '$in' in condition:
                        from qdrant_client.models import FieldCondition, MatchAny
                        qdrant_must.append(
                            FieldCondition(
                                key=field,
                                match=MatchAny(any=condition['$in'])
                            )
                        )
                    elif isinstance(condition, list):
                        from qdrant_client.models import FieldCondition, MatchAny
                        qdrant_must.append(
                            FieldCondition(
                                key=field,
                                match=MatchAny(any=condition)
                            )
                        )
                    else:
                        if isinstance(condition, (str, int, bool)):
                            from qdrant_client.models import FieldCondition, MatchValue
                            qdrant_must.append(
                                FieldCondition(
                                    key=field,
                                    match=MatchValue(value=condition)
                                )
                            )
                        elif isinstance(condition, float):
                            from qdrant_client.models import FieldCondition, MatchValue
                            qdrant_must.append(
                                FieldCondition(
                                    key=field,
                                    match=MatchValue(value=int(condition))
                                )
                            )

                if qdrant_must or qdrant_should:
                    search_filter = Filter(
                        must=qdrant_must if qdrant_must else None,
                        should=qdrant_should if qdrant_should else None
                    )
                    # Summarize constructed filter conditions for debugging
                    try:
                        summaries = []
                        for cond in qdrant_must or []:
                            summaries.append({'where': 'must', 'key': getattr(cond, 'key', None), 'match': cond.match.__class__.__name__ if getattr(cond, 'match', None) else None})
                        for cond in qdrant_should or []:
                            summaries.append({'where': 'should', 'key': getattr(cond, 'key', None), 'match': cond.match.__class__.__name__ if getattr(cond, 'match', None) else None})
                        logger.info(f"🧩 Constructed Qdrant filter summary: {summaries}")
                    except Exception:
                        logger.info("🧩 Constructed Qdrant filter (summary unavailable)")

            # Search for similar vectors
            search_result = self.client.search(# type: ignore
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                with_payload=True
            )

            # Format results
            results = []
            for scored_point in search_result:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": scored_point.payload
                }
                results.append(result)

            # Log detailed Qdrant search results
            logger.info(f"🎯 Qdrant returned {len(results)} results from similarity search")
            if results:
                logger.info(f"📊 Detailed Qdrant results with similarity scores:")
                for i, result in enumerate(results):
                    payload = result.get('payload', {})
                    score = result.get('score', 0)
                    name = payload.get('name', 'NO_NAME')
                    location = payload.get('location', 'NO_LOCATION')
                    role_category = payload.get('role_category', 'NO_ROLE')
                    current_position = payload.get('current_position', 'NO_POSITION')

                    logger.info(f"📊 Result #{i+1}: {name} | Score: {score:.4f} | Location: '{location}' | Role: '{role_category}' | Position: '{current_position}'")

                    # Show work history companies for comprehensive matching debugging
                    work_history = payload.get('work_history', [])
                    companies = []
                    for job in work_history:
                        if isinstance(job, dict) and job.get('company'):
                            companies.append(job.get('company'))
                    if companies:
                        logger.info(f"📊   └─ Work History Companies: {companies[:3]}")  # Show first 3 companies

                    # Show available skills for debugging
                    skills = payload.get('skills', [])
                    if skills:
                        logger.info(f"📊   └─ Skills: {skills[:5]}")  # Show first 5 skills

                    # Show summary excerpt for keyword matching
                    summary = payload.get('summary', '')
                    if summary:
                        summary_excerpt = summary[:100] + "..." if len(summary) > 100 else summary
                        logger.info(f"📊   └─ Summary: {summary_excerpt}")

            # Log some sample location data for debugging
            if results:
                logger.info(f"📍 Sample location data from results:")
                for i, result in enumerate(results[:3]):  # Show first 3 results
                    payload = result.get('payload', {})
                    location = payload.get('location', 'NO_LOCATION')
                    name = payload.get('name', 'NO_NAME')
                    logger.info(f"📍 Result {i+1}: {name} -> location: '{location}'")

            logger.info(f"Found {len(results)} similar resumes")
            return results

        except Exception as e:
            logger.error(f"Error searching similar embeddings: {e}")
            raise

    async def get_resume_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve resume data by user ID.

        Args:
            user_id: User identifier

        Returns:
            Resume data if found, None otherwise
        """
        if not self._connected:
            logger.warning("Qdrant not connected, cannot retrieve resume")
            return None

        try:
            points = self.client.retrieve(# type: ignore
                collection_name=self.collection_name,
                ids=[user_id],
                with_payload=True,
                with_vectors=False
            )

            if points:
                return points[0].payload
            return None

        except Exception as e:
            logger.error(f"Error retrieving resume for user {user_id}: {e}")
            return None

    async def delete_resume(self, user_id: str) -> bool:
        """
        Delete resume from Qdrant.

        Args:
            user_id: User identifier

        Returns:
            True if successful, False otherwise
        """
        if not self._connected:
            logger.warning("Qdrant not connected, skipping resume deletion")
            return False

        try:
            self.client.delete(# type: ignore
                collection_name=self.collection_name,
                points_selector=[user_id]
            )
            logger.info(f"Successfully deleted resume for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting resume for user {user_id}: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        if not self._connected:
            return {"error": "Qdrant not connected"}

        try:
            info = self.client.get_collection(self.collection_name)# type: ignore
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}


# Global Qdrant client instance
qdrant_client = QdrantVectorClient()
