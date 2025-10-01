"""
Async PostgreSQL client to persist parsed resume payloads alongside Qdrant.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional, List, Tuple
import uuid


import asyncpg
from datetime import datetime, date

from config.settings import settings

logger = logging.getLogger(__name__)


class PostgresClient:
    def __init__(self) -> None:
        self._pool: Optional[asyncpg.Pool] = None
        self._connecting_lock = asyncio.Lock()
        self._table = "public.qdrant_resumes"
        self._schema_ensured = False
        self._prompts_table = "public.user_search_prompts"
        self._prompts_schema_ensured = False
        self._interview_schema_ensured = False

    

    async def connect(self) -> bool:
        if self._pool:
            return True
        async with self._connecting_lock:
            if self._pool:
                return True
            conn_str = settings.postgres.connection_string
            print(f"Postgres connection string: {conn_str}")
            if not conn_str:
                logger.info("PostgreSQL not configured (no credentials); DB layer disabled")
                return False
            try:
                self._pool = await asyncpg.create_pool(dsn=conn_str, min_size=1, max_size=4)
                logger.info("PostgreSQL connection pool created")
                # Ensure the mirror table exists with expected columns
                try:
                    async with self._pool.acquire() as conn:
                        await self._ensure_schema(conn)
                        self._schema_ensured = True
                        await self._ensure_prompts_schema(conn)
                        self._prompts_schema_ensured = True
                        await self._ensure_interview_schema(conn) # Ensure new tables
                        self._interview_schema_ensured = True
                except Exception as ee:
                    logger.info(f"[PG] Schema ensure skipped: {ee}")
                return True
            except Exception as e:
                logger.error(f"Failed to connect to PostgreSQL: {e}")
                self._pool = None
                return False

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def upsert_parsed_resume(
        self,
        resume_id: str,
        payload: Dict[str, Any],
        embedding_model: Optional[str] = None,
        vector_id: Optional[str] = None,
    ) -> bool:
        """Insert or update parsed_resume row with payload fields.

        Expects table public.parsed_resumes (see ddl.sql/updated_schema.sql).
        """
        if not await self.connect():
            return False

        assert self._pool is not None

        # Prepare JSONB fields
        def j(value: Any, empty: str) -> str:
            try:
                if value is None:
                    return empty
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return empty

        skills = j(payload.get("skills", []), "[]")
        education = j(payload.get("education", []), "[]")
        experience = j(payload.get("work_history", []), "[]")
        projects = j(payload.get("projects", []), "[]")
        recommended_roles = j(payload.get("recommended_roles", []), "[]")
        role_classification = j(payload.get("role_classification", {}), "{}")

        # Coerce upload_timestamp to datetime for timestamptz column
        def parse_timestamp(val: Any) -> Optional[datetime]:
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, date):
                return datetime(val.year, val.month, val.day)
            if isinstance(val, (int, float)):
                try:
                    return datetime.fromtimestamp(val)
                except Exception:
                    return None
            if isinstance(val, str):
                try:
                    # fromisoformat supports "YYYY-MM-DDTHH:MM:SS.sss+00:00"
                    return datetime.fromisoformat(val)
                except Exception:
                    return None
            return None

        sql = f"""
        INSERT INTO {self._table} (
            id, vector_id, owner_user_id,
            name, email, phone, location, linkedin_url,
            current_position, summary, total_experience,
            role_category, seniority, best_role,
            skills, work_history, projects, education,
            role_classification, recommended_roles,
            original_filename, upload_timestamp,
            embedding_model, embedding_generated_at, created_at, is_shortlisted
        ) VALUES (
            $1, $2, $3,
            $4, $5, $6, $7, $8,
            $9, $10, $11,
            $12, $13, $14,
            $15::jsonb, $16::jsonb, $17::jsonb, $18::jsonb,
            $19::jsonb, $20::jsonb,
            $21, $22,
            $23, NOW(), NOW(), FALSE
        )
        ON CONFLICT (id) DO UPDATE SET
            vector_id = EXCLUDED.vector_id,
            owner_user_id = EXCLUDED.owner_user_id,
            name = EXCLUDED.name,
            email = EXCLUDED.email,
            phone = EXCLUDED.phone,
            location = EXCLUDED.location,
            linkedin_url = EXCLUDED.linkedin_url,
            current_position = EXCLUDED.current_position,
            summary = EXCLUDED.summary,
            total_experience = EXCLUDED.total_experience,
            role_category = EXCLUDED.role_category,
            seniority = EXCLUDED.seniority,
            best_role = EXCLUDED.best_role,
            skills = EXCLUDED.skills,
            work_history = EXCLUDED.work_history,
            projects = EXCLUDED.projects,
            education = EXCLUDED.education,
            role_classification = EXCLUDED.role_classification,
            recommended_roles = EXCLUDED.recommended_roles,
            original_filename = EXCLUDED.original_filename,
            upload_timestamp = EXCLUDED.upload_timestamp,
            embedding_model = EXCLUDED.embedding_model,
            embedding_generated_at = NOW();
        """

        try:
            async with self._pool.acquire() as conn:
                if not self._schema_ensured:
                    try:
                        await self._ensure_schema(conn)
                        self._schema_ensured = True
                    except Exception as ee:
                        logger.info(f"[PG] Schema ensure skipped during upsert: {ee}")
                await conn.execute(
                    sql,
                    resume_id,
                    vector_id or resume_id,
                    payload.get("owner_user_id"),
                    payload.get("name"),
                    payload.get("email"),
                    payload.get("phone"),
                    payload.get("location"),
                    payload.get("linkedin_url"),
                    payload.get("current_position"),
                    payload.get("summary"),
                    payload.get("total_experience"),
                    payload.get("role_category"),
                    payload.get("seniority"),
                    payload.get("best_role"),
                    skills,
                    experience,
                    projects,
                    education,
                    role_classification,
                    recommended_roles,
                    payload.get("original_filename"),
                    parse_timestamp(payload.get("upload_timestamp")),
                    embedding_model or 'text-embedding-3-large',
                )
            logger.info(f"[PG] Upserted qdrant_resumes row for: {resume_id}")
            return True
        except Exception as e:
            logger.error(f"[PG] Failed to upsert qdrant_resumes {resume_id}: {e}")
            return False

    async def _ensure_schema(self, conn: asyncpg.Connection) -> None:
        """Best-effort ensure of qdrant_resumes table and columns.
        Works whether table exists or not; adds missing columns if needed."""
        table = self._table
        # Create table if not exists (no-op if exists)
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id uuid PRIMARY KEY,
            vector_id varchar(255),
            owner_user_id varchar(255),
            name text,
            email text,
            phone varchar(50),
            location text,
            linkedin_url text,
            current_position text,
            summary text,
            total_experience text,
            role_category text,
            seniority text,
            best_role text,
            skills jsonb DEFAULT '[]'::jsonb,
            work_history jsonb DEFAULT '[]'::jsonb,
            projects jsonb DEFAULT '[]'::jsonb,
            education jsonb DEFAULT '[]'::jsonb,
            role_classification jsonb DEFAULT '{{}}'::jsonb,
            recommended_roles jsonb DEFAULT '[]'::jsonb,
            original_filename text,
            upload_timestamp timestamptz,
            embedding_model varchar(100) DEFAULT 'text-embedding-3-large',
            embedding_generated_at timestamptz,
            created_at timestamptz DEFAULT now(),
            is_shortlisted BOOLEAN NOT NULL DEFAULT FALSE
        );
        """
        await conn.execute(create_sql)

        # Add missing columns defensively
        alter_statements = [
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS id uuid",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS vector_id varchar(255)",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS owner_user_id varchar(255)",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS name text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS email text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS phone varchar(50)",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS location text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS linkedin_url text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS current_position text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS summary text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS total_experience text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS role_category text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS seniority text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS best_role text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS skills jsonb DEFAULT '[]'::jsonb",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS work_history jsonb DEFAULT '[]'::jsonb",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS projects jsonb DEFAULT '[]'::jsonb",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS education jsonb DEFAULT '[]'::jsonb",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS role_classification jsonb DEFAULT '{{}}'::jsonb",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS recommended_roles jsonb DEFAULT '[]'::jsonb",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS original_filename text",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS upload_timestamp timestamptz",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_model varchar(100) DEFAULT 'text-embedding-3-large'",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS embedding_generated_at timestamptz",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now()",
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS is_shortlisted BOOLEAN NOT NULL DEFAULT FALSE",
        ]
        for stmt in alter_statements:
            try:
                await conn.execute(stmt)
            except Exception:
                # Ignore incompatible alterations; table may already have these defined differently
                pass

        # Indexes (best-effort)
        try:
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_qdrant_resumes_vector_id ON {table}(vector_id)")
        except Exception:
            pass
        try:
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_qdrant_resumes_email ON {table}(email)")
        except Exception:
            pass
        try:
            await conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS uq_qdrant_resumes_id ON {table}(id)")
        except Exception:
            pass
        try:
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_qdrant_resumes_owner ON {table}(owner_user_id)")
        except Exception:
            pass
        try:
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_qdrant_resumes_is_shortlisted ON {table}(is_shortlisted)")
        except Exception:
            pass
        # Full-text index on common text fields (best-effort)
        try:
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_qdrant_resumes_full_text ON {table} "
                "USING gin(to_tsvector('english', "
                f"COALESCE(name,'') || ' ' || COALESCE(summary,'') || ' ' || COALESCE(current_position,'') || ' ' || COALESCE(location,'') || ' ' || COALESCE(email,'') ))"
            )
        except Exception:
            pass
        # Add FK to users(id) for owner_user_id (best-effort, not validated to avoid legacy data issues)
        try:
            await conn.execute(
                f"ALTER TABLE {table} "
                "ADD CONSTRAINT qdrant_resumes_owner_user_id_fkey "
                "FOREIGN KEY (owner_user_id) REFERENCES public.users(id) "
                "ON DELETE SET NULL NOT VALID"
            )
        except Exception:
            # constraint may already exist or users table not present; ignore
            pass

    async def _ensure_interview_schema(self, conn: asyncpg.Connection) -> None:
        """Ensure the interview_questions, interviews, call_records, and interview_sessions tables exist."""
        # Create interview_questions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS public.interview_questions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id VARCHAR(255) NOT NULL,
                question_text TEXT NOT NULL,
                expected_answer TEXT,
                welcome_message TEXT,
                category VARCHAR(100),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
            );
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_interview_questions_user_id ON public.interview_questions(user_id);")

        # Create interviews table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS public.interviews (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id VARCHAR(255) NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                welcome_message TEXT,
                question_ids UUID[] NOT NULL,
                candidate_ids UUID[],
                status VARCHAR(50) DEFAULT 'draft' NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT fk_interview_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
            );
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_interviews_user_id ON public.interviews(user_id);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_interviews_status ON public.interviews(status);")

        # Create call_records table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS public.call_records (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                resume_id UUID NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                call_status VARCHAR(50) NOT NULL,
                initiated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                notes TEXT,
                CONSTRAINT fk_resume FOREIGN KEY (resume_id) REFERENCES public.parsed_resumes(id) ON DELETE CASCADE,
                CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
            );
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_call_records_resume_id ON public.call_records(resume_id);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_call_records_user_id ON public.call_records(user_id);")

        # Create interview_sessions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS public.interview_sessions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id VARCHAR(255) NOT NULL,
                session_type VARCHAR(50) NOT NULL,
                question_ids JSONB NOT NULL,
                candidate_ids JSONB,
                current_question_index INTEGER DEFAULT 0,
                status VARCHAR(50) DEFAULT 'active',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT fk_user_session FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
            );
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_interview_sessions_user_id ON public.interview_sessions(user_id);")

        logger.info("[PG] Ensured interview schema (questions, interviews, calls, sessions).")

    async def list_resumes(
        self,
        offset: int = 0,
        limit: int = 20,
        search: Optional[str] = None,
        name: Optional[str] = None,
        email: Optional[str] = None,
        location: Optional[str] = None,
        job_title: Optional[str] = None,
        role_category: Optional[str] = None,
        company: Optional[str] = None,
        owner_user_id: Optional[str] = None,
        order_by: str = "-created_at",
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """List resumes from qdrant_resumes with simple filters and pagination.

        Returns (total_count, rows) where rows are dicts.
        """
        if not await self.connect():
            return 0, []

        assert self._pool is not None

        # Build where clause
        where = []
        args: List[Any] = []

        def add_ilike(column: str, value: Optional[str]):
            if value is None:
                return
            val = str(value).strip()
            if not val:
                return
            args.append(f"%{val}%")
            where.append(f"{column} ILIKE ${len(args)}")

        if search:
            s = str(search).strip()
            if s:
                like = f"%{s}%"
                args.extend([like, like, like, like, like])
                where.append(
                    "(name ILIKE $%d OR current_position ILIKE $%d OR summary ILIKE $%d OR location ILIKE $%d OR email ILIKE $%d)"
                    % (len(args) - 4, len(args) - 3, len(args) - 2, len(args) - 1, len(args))
                )

        add_ilike("name", name)
        add_ilike("email", email)
        add_ilike("location", location)
        add_ilike("current_position", job_title)
        add_ilike("role_category", role_category)

        # Owner filter (exact match preferred)
        if owner_user_id and str(owner_user_id).strip():
            args.append(str(owner_user_id).strip())
            where.append(f"owner_user_id = ${len(args)}")

        # Company filter from work_history JSONB (any element with company ILIKE '%value%')
        if company is not None and str(company).strip():
            args.append(f"%{str(company).strip()}%")
            where.append(
                f"EXISTS (SELECT 1 FROM jsonb_array_elements(work_history) w WHERE w->>'company' ILIKE ${len(args)})"
            )

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        # Whitelist sortable columns
        sortable = {
            "created_at": "created_at",
            "embedding_generated_at": "embedding_generated_at",
            "upload_timestamp": "upload_timestamp",
            "name": "name",
            "current_position": "current_position",
        }
        sort_dir = "DESC"
        sort_key = order_by
        if order_by.startswith("-"):
            sort_key = order_by[1:]
            sort_dir = "DESC"
        elif order_by.startswith("+"):
            sort_key = order_by[1:]
            sort_dir = "ASC"
        else:
            sort_dir = "ASC"
        order_column = sortable.get(sort_key, "created_at")
        order_sql = f"ORDER BY {order_column} {sort_dir} NULLS LAST"

        # Count and fetch
        sql_count = f"SELECT COUNT(*) FROM {self._table} {where_sql}"
        sql_rows = f"""
            SELECT id, vector_id, owner_user_id, name, email, phone, location, linkedin_url,
                   current_position, summary, total_experience, role_category, seniority,
                   best_role, skills, work_history, projects, education,
                   role_classification, recommended_roles, original_filename,
                   upload_timestamp, embedding_model, embedding_generated_at, created_at, is_shortlisted
            FROM {self._table}
            {where_sql}
            {order_sql}
            LIMIT {max(1, int(limit))} OFFSET {max(0, int(offset))}
        """

        async with self._pool.acquire() as conn:
            total = await conn.fetchval(sql_count, *args)
            rows = await conn.fetch(sql_rows, *args)

        # Convert Record to dict
        items: List[Dict[str, Any]] = [dict(r) for r in rows]
        return int(total or 0), items

    # --- Shortlisting Functions ---

    async def get_shortlisted_resumes(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all shortlisted resumes for a given user."""
        if not await self.connect():
            return []
        assert self._pool is not None
        sql = f"SELECT * FROM {self._table} WHERE owner_user_id = $1 AND is_shortlisted = TRUE ORDER BY upload_timestamp DESC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, user_id)
        
        # Convert UUID objects to strings for JSON serialization
        result = []
        for row in rows:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            result.append(row_dict)
        return result

    async def update_shortlist_status(self, resume_id: uuid.UUID, is_shortlisted: bool) -> bool:
        """Update the shortlist status for a specific resume."""
        if not await self.connect():
            return False
        assert self._pool is not None
        
        # Update PostgreSQL
        sql = f"UPDATE {self._table} SET is_shortlisted = $2 WHERE id = $1"
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, resume_id, is_shortlisted)
            success = result == "UPDATE 1"
            
            if success:
                # Also update Qdrant payload to keep it in sync
                try:
                    from .qdrant_client import qdrant_client
                    if qdrant_client._connected:
                        # Get current payload from Qdrant
                        current_payload = await qdrant_client.get_resume_by_id(str(resume_id))
                        if current_payload:
                            # Update the shortlist status
                            current_payload['is_shortlisted'] = is_shortlisted
                            # Upsert back to Qdrant (this will update the payload)
                            # Note: We need the vector, but since we're only updating payload, 
                            # we can use a dummy vector or get the existing one
                            # For now, we'll skip Qdrant update as it requires the vector
                            # The shortlist status will be merged from PostgreSQL in search results
                            pass
                except Exception as e:
                    logger.warning(f"Could not update Qdrant payload for shortlist status: {e}")
            
            return success

    async def get_shortlist_status(self, resume_id: uuid.UUID) -> bool:
        """Check if a specific resume is shortlisted."""
        if not await self.connect():
            return False
        assert self._pool is not None
        sql = f"SELECT is_shortlisted FROM {self._table} WHERE id = $1"
        async with self._pool.acquire() as conn:
            status = await conn.fetchval(sql, resume_id)
        return status is True

    # --- Interview Question CRUD Functions ---

    async def create_interview_question(self, question_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new interview question."""
        if not await self.connect():
            return None
        assert self._pool is not None
        sql = """
            INSERT INTO public.interview_questions (user_id, question_text, expected_answer, welcome_message, category)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *;
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                question_data['user_id'],
                question_data['question_text'],
                question_data.get('expected_answer'),
                question_data.get('welcome_message'),
                question_data.get('category')
            )
        if row:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            return row_dict
        return None

    async def get_interview_questions(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all interview questions for a user."""
        if not await self.connect():
            return []
        assert self._pool is not None
        sql = "SELECT * FROM public.interview_questions WHERE user_id = $1 ORDER BY created_at DESC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, user_id)
        
        # Convert UUID objects to strings for JSON serialization
        result = []
        for row in rows:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            result.append(row_dict)
        return result

    async def update_interview_question(self, question_id: uuid.UUID, question_update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing interview question."""
        if not await self.connect() or not question_update:
            return None
        assert self._pool is not None
        
        # Dynamically build the SET clause
        set_parts = []
        args = []
        for i, (key, value) in enumerate(question_update.items()):
            if key in ('question_text', 'expected_answer', 'welcome_message', 'category'): # Whitelist of updatable columns
                set_parts.append(f"{key} = ${i + 1}")
                args.append(value)
        
        if not set_parts:
            return None # Nothing to update

        set_clause = ", ".join(set_parts)
        args.append(question_id)
        
        sql = f"""
            UPDATE public.interview_questions
            SET {set_clause}, updated_at = NOW()
            WHERE id = ${len(args)}
            RETURNING *;
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        if row:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            return row_dict
        return None

    async def get_interview_question(self, question_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get a single interview question by ID."""
        if not await self.connect():
            return None
        assert self._pool is not None
        sql = "SELECT * FROM public.interview_questions WHERE id = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, question_id)
        if row:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            return row_dict
        return None

    async def delete_interview_question(self, question_id: uuid.UUID) -> bool:
        """Delete an interview question."""
        if not await self.connect():
            return False
        assert self._pool is not None
        sql = "DELETE FROM public.interview_questions WHERE id = $1"
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, question_id)
            return result == "DELETE 1"

    # --- Call Logging Functions ---

    async def log_interview_call(self, resume_id: uuid.UUID, user_id: str, status: str, notes: Optional[str]) -> Optional[Dict[str, Any]]:
        """Log an initiated interview call to the database."""
        if not await self.connect():
            return None
        assert self._pool is not None
        sql = """
            INSERT INTO public.call_records (resume_id, user_id, call_status, notes)
            VALUES ($1, $2, $3, $4)
            RETURNING *;
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, resume_id, user_id, status, notes)
        if row:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            return row_dict
        return None

    async def get_resume(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """Get a single resume by id from qdrant_resumes."""
        if not await self.connect():
            return None
        assert self._pool is not None
        sql = f"""
            SELECT id, vector_id, owner_user_id, name, email, phone, location, linkedin_url,
                   current_position, summary, total_experience, role_category, seniority,
                   best_role, skills, work_history, projects, education,
                   role_classification, recommended_roles, original_filename,
                   upload_timestamp, embedding_model, embedding_generated_at, created_at, is_shortlisted
            FROM {self._table}
            WHERE id = $1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, resume_id)
        if row:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            return row_dict
        return None

    async def _ensure_prompts_schema(self, conn: asyncpg.Connection) -> None:
        """Ensure the user_search_prompts table and indexes exist."""
        table = self._prompts_table
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id uuid PRIMARY KEY,
            user_id varchar(255) NOT NULL,
            prompt text NOT NULL,
            route text DEFAULT 'search-resumes-intent-based',
            liked boolean,
            asked_at timestamptz DEFAULT now(),
            response_meta jsonb DEFAULT '{{}}'::jsonb,
            created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now()
        );
        """
        await conn.execute(create_sql)
        # Indexes
        try:
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_user_prompts_user_id ON {table}(user_id)")
        except Exception:
            pass
        try:
            await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_user_prompts_asked_at ON {table}(asked_at DESC)")
        except Exception:
            pass

    async def insert_user_search_prompt(
        self,
        user_id: str,
        prompt: str,
        route: str = "search-resumes-intent-based",
        liked: Optional[bool] = None,
        asked_at: Optional[datetime] = None,
        response_meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not await self.connect():
            return None
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            if not self._prompts_schema_ensured:
                try:
                    await self._ensure_prompts_schema(conn)
                    self._prompts_schema_ensured = True
                except Exception as ee:
                    logger.info(f"[PG] Prompts schema ensure skipped: {ee}")
            pid = str(uuid.uuid4())
            sql = f"""
                INSERT INTO {self._prompts_table} (id, user_id, prompt, route, liked, asked_at, response_meta)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                RETURNING id
            """
            rm = response_meta or {}
            try:
                new_id = await conn.fetchval(sql, pid, user_id, prompt, route, liked, asked_at or datetime.utcnow(), json.dumps(rm))
                return str(new_id) if new_id else pid
            except Exception as e:
                logger.error(f"[PG] Failed to insert user_search_prompt: {e}")
                return None

    async def update_user_search_prompt_feedback(self, prompt_id: str, liked: Optional[bool]) -> bool:
        if not await self.connect():
            return False
        assert self._pool is not None
        sql = f"""
            UPDATE {self._prompts_table}
            SET liked = $2, updated_at = NOW()
            WHERE id = $1
        """
        async with self._pool.acquire() as conn:
            try:
                res = await conn.execute(sql, prompt_id, liked)
                return res and res.upper().startswith("UPDATE")
            except Exception as e:
                logger.error(f"[PG] Failed to update prompt feedback {prompt_id}: {e}")
                return False

    async def get_dashboard_metrics(self, owner_user_id: str) -> Dict[str, Any]:
        """Compute dashboard metrics for a given owner_user_id."""
        if not await self.connect():
            return {}
        assert self._pool is not None

        async with self._pool.acquire() as conn:
            # Total candidates overall
            total_all = await conn.fetchval(
                f"SELECT COUNT(*) FROM {self._table} WHERE owner_user_id = $1",
                owner_user_id,
            )

            # This month and last month counts
            month_cur = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM {self._table}
                WHERE owner_user_id = $1
                  AND created_at >= date_trunc('month', now())
                """,
                owner_user_id,
            )
            month_prev = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM {self._table}
                WHERE owner_user_id = $1
                  AND created_at >= date_trunc('month', now() - interval '1 month')
                  AND created_at <  date_trunc('month', now())
                """,
                owner_user_id,
            )

            # This week and last week processed
            week_cur = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM {self._table}
                WHERE owner_user_id = $1
                  AND created_at >= date_trunc('week', now())
                """,
                owner_user_id,
            )
            week_prev = await conn.fetchval(
                f"""
                SELECT COUNT(*) FROM {self._table}
                WHERE owner_user_id = $1
                  AND created_at >= date_trunc('week', now()) - interval '1 week'
                  AND created_at <  date_trunc('week', now())
                """,
                owner_user_id,
            )

            # Avg processing time (embedding_generated_at - upload_timestamp) this week / last week
            avg_proc_cur = await conn.fetchval(
                f"""
                SELECT AVG(EXTRACT(EPOCH FROM (embedding_generated_at - upload_timestamp)))
                FROM {self._table}
                WHERE owner_user_id = $1
                  AND upload_timestamp IS NOT NULL
                  AND embedding_generated_at IS NOT NULL
                  AND created_at >= date_trunc('week', now())
                """,
                owner_user_id,
            )
            avg_proc_prev = await conn.fetchval(
                f"""
                SELECT AVG(EXTRACT(EPOCH FROM (embedding_generated_at - upload_timestamp)))
                FROM {self._table}
                WHERE owner_user_id = $1
                  AND upload_timestamp IS NOT NULL
                  AND embedding_generated_at IS NOT NULL
                  AND created_at >= date_trunc('week', now()) - interval '1 week'
                  AND created_at <  date_trunc('week', now())
                """,
                owner_user_id,
            )

            # Match accuracy from user_search_prompts (last 30 days vs previous 30 days)
            prompts_30 = await conn.fetchrow(
                f"""
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE liked IS TRUE) AS liked
                FROM {self._prompts_table}
                WHERE user_id = $1
                  AND route = 'search-resumes-intent-based'
                  AND asked_at >= now() - interval '30 days'
                """,
                owner_user_id,
            )
            prompts_prev_30 = await conn.fetchrow(
                f"""
                SELECT COUNT(*) AS total,
                       COUNT(*) FILTER (WHERE liked IS TRUE) AS liked
                FROM {self._prompts_table}
                WHERE user_id = $1
                  AND route = 'search-resumes-intent-based'
                  AND asked_at >= now() - interval '60 days'
                  AND asked_at <  now() - interval '30 days'
                """,
                owner_user_id,
            )

        def pct_change(cur: Optional[float], prev: Optional[float]) -> float:
            try:
                c = float(cur or 0)
                p = float(prev or 0)
                if p == 0:
                    return 100.0 if c > 0 else 0.0
                return ((c - p) / p) * 100.0
            except Exception:
                return 0.0

        month_change = pct_change(month_cur, month_prev)
        week_change = pct_change(week_cur, week_prev)
        # For processing time, lower is better, we still compute delta (negative means faster)
        proc_change = 0.0
        try:
            proc_change = pct_change(avg_proc_cur, avg_proc_prev)
        except Exception:
            proc_change = 0.0

        acc_cur = 0.0
        acc_prev = 0.0
        try:
            tcur = float((prompts_30 or {}).get('total', 0))
            lcur = float((prompts_30 or {}).get('liked', 0))
            acc_cur = (lcur / tcur * 100.0) if tcur > 0 else 0.0
        except Exception:
            acc_cur = 0.0
        try:
            tprev = float((prompts_prev_30 or {}).get('total', 0))
            lprev = float((prompts_prev_30 or {}).get('liked', 0))
            acc_prev = (lprev / tprev * 100.0) if tprev > 0 else 0.0
        except Exception:
            acc_prev = 0.0
        acc_change = pct_change(acc_cur, acc_prev)

        return {
            "total_candidates": int(total_all or 0),
            "total_candidates_vs_last_month_percent": month_change,
            "resumes_processed_this_week": int(week_cur or 0),
            "resumes_processed_vs_prev_week_percent": week_change,
            "avg_processing_time_seconds_this_week": float(avg_proc_cur or 0.0),
            "avg_processing_time_vs_prev_week_percent": proc_change,
            "match_accuracy_percent_last_30_days": acc_cur,
            "match_accuracy_vs_prev_30_days_percent": acc_change,
        }

    async def get_recent_activity(self, owner_user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Assemble recent activity timeline from resumes + user prompts."""
        if not await self.connect():
            return []
        assert self._pool is not None

        items: List[Dict[str, Any]] = []
        async with self._pool.acquire() as conn:
            # Recent uploads
            rows_up = await conn.fetch(
                f"""
                SELECT id, name, current_position, created_at
                FROM {self._table}
                WHERE owner_user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                owner_user_id, limit,
            )
            for r in rows_up:
                items.append({
                    "type": "resume_uploaded",
                    "title": "New resume uploaded",
                    "subtitle": f"{(r['name'] or 'Unknown')} - {(r['current_position'] or 'Unknown')}",
                    "timestamp": (r["created_at"].isoformat() if r["created_at"] else None),
                    "meta": {"resume_id": str(r["id"])},
                })

            # Recent searches (from prompts)
            rows_pr = await conn.fetch(
                f"""
                SELECT id, prompt, asked_at, response_meta
                FROM {self._prompts_table}
                WHERE user_id = $1 AND route = 'search-resumes-intent-based'
                ORDER BY asked_at DESC
                LIMIT $2
                """,
                owner_user_id, limit,
            )
            import json as _json
            from collections.abc import Mapping
            for r in rows_pr:
                raw_meta = r["response_meta"]
                meta: Dict[str, Any] = {}
                if raw_meta:
                    try:
                        if isinstance(raw_meta, Mapping):
                            meta = dict(raw_meta)
                        elif isinstance(raw_meta, str):
                            # Try parse JSON string
                            try:
                                parsed = _json.loads(raw_meta)
                                if isinstance(parsed, Mapping):
                                    meta = dict(parsed)
                            except Exception:
                                meta = {}
                        else:
                            # Last resort: attempt dict() only if iterable of pairs
                            try:
                                meta = dict(raw_meta)  # may raise
                            except Exception:
                                meta = {}
                    except Exception:
                        meta = {}
                result_count = meta.get("result_count")
                key_reqs = meta.get("key_requirements_count")
                top_match_score = meta.get("top_match_score")
                top_match_role = meta.get("top_match_role")
                # Generic search completed
                if result_count is not None:
                    items.append({
                        "type": "search_completed",
                        "title": "Search completed",
                        "subtitle": f"'{r['prompt']}' returned {result_count} results",
                        "timestamp": (r["asked_at"].isoformat() if r["asked_at"] else None),
                        "meta": {"prompt_id": str(r["id"]), **meta},
                    })
                else:
                    items.append({
                        "type": "search_completed",
                        "title": "Search completed",
                        "subtitle": f"Query '{r['prompt']}' executed",
                        "timestamp": (r["asked_at"].isoformat() if r["asked_at"] else None),
                        "meta": {"prompt_id": str(r["id"]), **meta},
                    })
                # Query analyzed
                if key_reqs is not None:
                    items.append({
                        "type": "query_analyzed",
                        "title": "Query analyzed",
                        "subtitle": f"AI extracted {key_reqs} key requirements",
                        "timestamp": (r["asked_at"].isoformat() if r["asked_at"] else None),
                        "meta": {"prompt_id": str(r["id"]), **meta},
                    })
                # High match found
                try:
                    if top_match_score is not None and float(top_match_score) >= 0.90:
                        role_text = f"{top_match_role} - " if top_match_role else ""
                        items.append({
                            "type": "high_match_found",
                            "title": "High match found",
                            "subtitle": f"{role_text}{int(float(top_match_score)*100)}% match",
                            "timestamp": (r["asked_at"].isoformat() if r["asked_at"] else None),
                            "meta": {"prompt_id": str(r["id"]), **meta},
                        })
                except Exception:
                    pass

        # Sort combined list by timestamp desc and cap to limit
        def ts_key(it: Dict[str, Any]) -> float:
            try:
                return (datetime.fromisoformat(it.get("timestamp")) if it.get("timestamp") else datetime.min).timestamp() # type: ignore
            except Exception:
                return 0.0
        items.sort(key=ts_key, reverse=True)
        return items[:limit]

    async def get_recent_prompt(self, owner_user_id: str, days: int = 30) -> Optional[str]:
        """Return the most recent search prompt text for this user within the last N days."""
        if not await self.connect():
            return None
        assert self._pool is not None
        sql = f"""
            SELECT prompt FROM {self._prompts_table}
            WHERE user_id = $1 AND route = 'search-resumes-intent-based'
              AND asked_at >= now() - ($2 || ' days')::interval
            ORDER BY asked_at DESC
            LIMIT 1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, owner_user_id, days)
        return str(row["prompt"]) if row and row.get("prompt") else None

    async def get_top_role_category(self, owner_user_id: str) -> Optional[str]:
        """Return the most frequent role_category for this user."""
        if not await self.connect():
            return None
        assert self._pool is not None
        sql = f"""
            SELECT role_category, COUNT(*) AS c
            FROM {self._table}
            WHERE owner_user_id = $1 AND role_category IS NOT NULL AND role_category <> ''
            GROUP BY role_category
            ORDER BY c DESC
            LIMIT 1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, owner_user_id)
        return str(row["role_category"]) if row and row.get("role_category") else None

    async def check_user_resume_limit(self, user_id: str, requested_count: int = 1) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if user can upload the requested number of resumes within their limit.

        Args:
            user_id: User identifier
            requested_count: Number of resumes they want to upload (default: 1)

        Returns:
            Tuple of (can_upload: bool, limit_info: Dict)
            limit_info contains current usage, limits, and available slots
        """
        if not await self.connect():
            return False, {"error": "Database connection failed"}

        assert self._pool is not None

        async with self._pool.acquire() as conn:
            # Get user's current limits and usage
            sql = """
                SELECT
                    total_resumes_uploaded,
                    resume_limit,
                    tokens_used,
                    token_limit,
                    last_resume_uploaded_at
                FROM public.user_resume_limits
                WHERE user_id = $1
            """

            row = await conn.fetchrow(sql, user_id)

            if not row:
                # User doesn't exist in limits table - try to initialize
                init_success = await self.init_user_resume_limits(user_id)
                if not init_success:
                    # If initialization fails, return safe defaults but allow upload
                    logger.warning(f"[PG] Could not initialize limits for {user_id}, using defaults")
                    return True, {
                        "current_resumes": 0,
                        "resume_limit": 10,
                        "available_slots": 10,
                        "requested_count": requested_count,
                        "can_upload": True,
                        "tokens_used": 0,
                        "token_limit": 1000000,
                        "last_uploaded_at": None,
                        "warning": "Using default limits - user limits could not be initialized"
                    }
                # Get the newly created row
                row = await conn.fetchrow(sql, user_id)

            current_count = row['total_resumes_uploaded'] if row else 0
            limit = row['resume_limit'] if row else 10
            available = max(0, limit - current_count)

            can_upload = available >= requested_count

            limit_info = {
                "current_resumes": current_count,
                "resume_limit": limit,
                "available_slots": available,
                "requested_count": requested_count,
                "can_upload": can_upload,
                "tokens_used": row['tokens_used'] if row else 0,
                "token_limit": row['token_limit'] if row else 1000000,
                "last_uploaded_at": row['last_resume_uploaded_at'].isoformat() if row and row['last_resume_uploaded_at'] else None
            }

            return can_upload, limit_info

    async def init_user_resume_limits(self, user_id: str) -> bool:
        """Initialize user_resume_limits entry for new user with default values."""
        if not await self.connect():
            return False

        assert self._pool is not None

        try:
            async with self._pool.acquire() as conn:
                # First check if user exists in users table
                user_exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM public.users WHERE id = $1)",
                    user_id
                )

                if not user_exists:
                    # Create user entry first
                    await conn.execute(
                        """
                        INSERT INTO public.users (id, email, name, created_at, updated_at)
                        VALUES ($1, $2, $3, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        user_id,
                        f"user_{user_id}@temp.com",  # Temporary email
                        f"User {user_id}"  # Temporary name
                    )
                    logger.info(f"[PG] Created temporary user entry for: {user_id}")

                # Now create the resume limits entry
                await conn.execute(
                    """
                    INSERT INTO public.user_resume_limits (user_id, total_resumes_uploaded, resume_limit, tokens_used, token_limit)
                    VALUES ($1, 0, 10, 0, 1000000)
                    ON CONFLICT (user_id) DO NOTHING
                    """,
                    user_id
                )
            logger.info(f"[PG] Initialized resume limits for user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"[PG] Failed to initialize resume limits for {user_id}: {e}")
            return False

    async def increment_user_resume_count(self, user_id: str, count: int = 1, tokens_used: int = 0) -> bool:
        """
        Increment user's resume upload count and token usage.

        Args:
            user_id: User identifier
            count: Number of resumes to add (default: 1)
            tokens_used: Tokens consumed in processing

        Returns:
            bool: Success status
        """
        if not await self.connect():
            return False

        assert self._pool is not None

        sql = """
            UPDATE public.user_resume_limits
            SET
                total_resumes_uploaded = total_resumes_uploaded + $2,
                tokens_used = tokens_used + $3,
                last_resume_uploaded_at = NOW(),
                updated_at = NOW()
            WHERE user_id = $1
        """

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(sql, user_id, count, tokens_used)
                if result and result.upper().startswith("UPDATE"):
                    logger.info(f"[PG] Updated resume count (+{count}) and tokens (+{tokens_used}) for user: {user_id}")
                    return True
                else:
                    # User doesn't exist, initialize first
                    if await self.init_user_resume_limits(user_id):
                        # Try the update again
                        result = await conn.execute(sql, user_id, count, tokens_used)
                        if result and result.upper().startswith("UPDATE"):
                            logger.info(f"[PG] Initialized and updated resume count for user: {user_id}")
                            return True
                    logger.warning(f"[PG] Could not initialize or update resume count for user: {user_id}")
            return False
        except Exception as e:
            logger.error(f"[PG] Failed to increment resume count for {user_id}: {e}")
            return False

    async def decrement_user_resume_count(self, user_id: str, count: int = 1, tokens_used: int = 0) -> bool:
        """
        Decrement user's resume upload count and token usage (e.g., when resumes are deleted).

        Args:
            user_id: User identifier
            count: Number of resumes to subtract (default: 1)
            tokens_used: Tokens to subtract from usage

        Returns:
            bool: Success status
        """
        if not await self.connect():
            return False

        assert self._pool is not None

        sql = """
            UPDATE public.user_resume_limits
            SET
                total_resumes_uploaded = GREATEST(0, total_resumes_uploaded - $2),
                tokens_used = GREATEST(0, tokens_used - $3),
                updated_at = NOW()
            WHERE user_id = $1
        """

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(sql, user_id, count, tokens_used)
                if result and result.upper().startswith("UPDATE"):
                    logger.info(f"[PG] Decremented resume count (-{count}) and tokens (-{tokens_used}) for user: {user_id}")
                    return True
                else:
                    logger.warning(f"[PG] No rows updated when decrementing resume count for user: {user_id}")
                    return False
        except Exception as e:
            logger.error(f"[PG] Failed to decrement resume count for {user_id}: {e}")
            return False

    async def reset_user_resume_count(self, user_id: str, new_count: int = 0, new_tokens: int = 0) -> bool:
        """
        Reset user's resume count and token usage to specific values.

        Args:
            user_id: User identifier
            new_count: New resume count (default: 0)
            new_tokens: New token count (default: 0)

        Returns:
            bool: Success status
        """
        if not await self.connect():
            return False

        assert self._pool is not None

        sql = """
            UPDATE public.user_resume_limits
            SET
                total_resumes_uploaded = $2,
                tokens_used = $3,
                updated_at = NOW()
            WHERE user_id = $1
        """

        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(sql, user_id, new_count, new_tokens)
                if result and result.upper().startswith("UPDATE"):
                    logger.info(f"[PG] Reset resume count to {new_count} and tokens to {new_tokens} for user: {user_id}")
                    return True
                else:
                    logger.warning(f"[PG] No rows updated when resetting resume count for user: {user_id}")
                    return False
        except Exception as e:
            logger.error(f"[PG] Failed to reset resume count for {user_id}: {e}")
            return False

    async def get_user_resume_limits(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's current resume limits and usage."""
        if not await self.connect():
            return None

        assert self._pool is not None

        sql = """
            SELECT
                total_resumes_uploaded,
                resume_limit,
                tokens_used,
                token_limit,
                created_at,
                updated_at,
                last_resume_uploaded_at
            FROM public.user_resume_limits
            WHERE user_id = $1
        """

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, user_id)

        if not row:
            return None

        return {
            "user_id": user_id,
            "total_resumes_uploaded": row['total_resumes_uploaded'],
            "resume_limit": row['resume_limit'],
            "tokens_used": row['tokens_used'],
            "token_limit": row['token_limit'],
            "created_at": row['created_at'].isoformat() if row['created_at'] else None,
            "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None,
            "last_resume_uploaded_at": row['last_resume_uploaded_at'].isoformat() if row['last_resume_uploaded_at'] else None,
            "available_resume_slots": max(0, row['resume_limit'] - row['total_resumes_uploaded']),
            "available_token_slots": max(0, row['token_limit'] - row['tokens_used'])
        }

    # --- Interview CRUD Functions ---

    async def create_interview(self, interview_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a new interview."""
        if not await self.connect():
            return None
        assert self._pool is not None
        sql = """
            INSERT INTO public.interviews (user_id, title, description, welcome_message, question_ids, candidate_ids)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING *;
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                sql,
                interview_data['user_id'],
                interview_data['title'],
                interview_data.get('description'),
                interview_data.get('welcome_message'),
                interview_data['question_ids'],
                interview_data.get('candidate_ids')
            )
        if row:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            return row_dict
        return None

    async def get_user_interviews(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all interviews for a user."""
        if not await self.connect():
            return []
        assert self._pool is not None
        sql = "SELECT * FROM public.interviews WHERE user_id = $1 ORDER BY created_at DESC"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, user_id)
        
        # Convert UUID objects to strings for JSON serialization
        result = []
        for row in rows:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            result.append(row_dict)
        return result

    async def update_interview(self, interview_id: uuid.UUID, interview_update: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update an existing interview."""
        if not await self.connect() or not interview_update:
            return None
        assert self._pool is not None
        
        # Dynamically build the SET clause
        set_parts = []
        args = []
        for i, (key, value) in enumerate(interview_update.items()):
            if key in ('title', 'description', 'welcome_message', 'question_ids', 'candidate_ids', 'status'): # Whitelist of updatable columns
                set_parts.append(f"{key} = ${i + 1}")
                args.append(value)
        
        if not set_parts:
            return None # Nothing to update

        set_clause = ", ".join(set_parts)
        args.append(interview_id)
        
        sql = f"""
            UPDATE public.interviews
            SET {set_clause}, updated_at = NOW()
            WHERE id = ${len(args)}
            RETURNING *;
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
        if row:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            return row_dict
        return None

    async def delete_interview(self, interview_id: uuid.UUID) -> bool:
        """Delete an interview."""
        if not await self.connect():
            return False
        assert self._pool is not None
        sql = "DELETE FROM public.interviews WHERE id = $1"
        async with self._pool.acquire() as conn:
            result = await conn.execute(sql, interview_id)
            return result == "DELETE 1"

    async def get_interview_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        ok = await self.connect()
        if not ok:
            return None
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM public.interview_sessions WHERE id = $1",
                session_id
            )
            if row:
                row_dict = dict(row)
                # Parse question_ids from JSON string to list
                if 'question_ids' in row_dict and row_dict['question_ids'] is not None:
                    if isinstance(row_dict['question_ids'], str):
                        try:
                            row_dict['question_ids'] = json.loads(row_dict['question_ids'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['question_ids'] = []
                    elif not isinstance(row_dict['question_ids'], list):
                        row_dict['question_ids'] = []
                else:
                    row_dict['question_ids'] = []

                # Parse candidate_ids from JSON string to list
                if 'candidate_ids' in row_dict and row_dict['candidate_ids'] is not None:
                    if isinstance(row_dict['candidate_ids'], str):
                        try:
                            row_dict['candidate_ids'] = json.loads(row_dict['candidate_ids'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['candidate_ids'] = []
                    elif not isinstance(row_dict['candidate_ids'], list):
                        row_dict['candidate_ids'] = []
                else:
                    row_dict['candidate_ids'] = None

                return row_dict
            return None

    async def save_interview_response(
        self,
        user_id: str,
        question_id: uuid.UUID,
        answer_text: Optional[str] = None,
        audio_file_path: Optional[str] = None,
        audio_duration: Optional[float] = None,
        response_time_seconds: Optional[float] = None
    ) -> Optional[str]:
        """Save an interview response (text and/or audio)."""
        if not await self.connect():
            return None
        assert self._pool is not None
        
        sql = """
            INSERT INTO public.interview_responses (
                user_id, question_id, answer_text, audio_file_path, 
                audio_duration, response_time_seconds
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
        """
        
        async with self._pool.acquire() as conn:
            try:
                response_id = await conn.fetchval(
                    sql,
                    user_id,
                    question_id,
                    answer_text,
                    audio_file_path,
                    audio_duration,
                    response_time_seconds
                )
                return str(response_id) if response_id else None
            except Exception as e:
                logger.error(f"[PG] Failed to save interview response: {e}")
                return None


    async def create_interview_session(self, user_id: str, session_type: str, question_ids: List[uuid.UUID], candidate_ids: Optional[List[uuid.UUID]] = None) -> Optional[Dict[str, Any]]:
        """Create a new interview session."""
        ok = await self.connect()
        if not ok:
            return None
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            # Convert UUIDs to strings for JSONB storage
            import json
            question_ids_json = json.dumps([str(uid) for uid in question_ids]) if question_ids else json.dumps([])
            candidate_ids_json = json.dumps([str(uid) for uid in candidate_ids]) if candidate_ids else None
            
            row = await conn.fetchrow(
                """
                INSERT INTO public.interview_sessions (user_id, session_type, question_ids, candidate_ids)
                VALUES ($1, $2, $3, $4)
                RETURNING id, user_id, session_type, question_ids, candidate_ids, current_question_index, status, created_at, updated_at
                """,
                user_id, session_type, question_ids_json, candidate_ids_json
            )
            if row:
                row_dict = dict(row)
                # Parse JSONB columns back to Python objects
                if 'question_ids' in row_dict and row_dict['question_ids'] is not None:
                    if isinstance(row_dict['question_ids'], str):
                        try:
                            row_dict['question_ids'] = json.loads(row_dict['question_ids'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['question_ids'] = []
                    elif not isinstance(row_dict['question_ids'], list):
                        row_dict['question_ids'] = []
                else:
                    row_dict['question_ids'] = []

                if 'candidate_ids' in row_dict and row_dict['candidate_ids'] is not None:
                    if isinstance(row_dict['candidate_ids'], str):
                        try:
                            row_dict['candidate_ids'] = json.loads(row_dict['candidate_ids'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['candidate_ids'] = []
                    elif not isinstance(row_dict['candidate_ids'], list):
                        row_dict['candidate_ids'] = []
                else:
                    row_dict['candidate_ids'] = None
                return row_dict
        return None

    async def get_interview_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        ok = await self.connect()
        if not ok:
            return None
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM public.interview_sessions WHERE id = $1",
                session_id
            )
            if row:
                row_dict = dict(row)
                # Parse JSONB columns back to Python objects
                if 'question_ids' in row_dict and row_dict['question_ids'] is not None:
                    if isinstance(row_dict['question_ids'], str):
                        try:
                            row_dict['question_ids'] = json.loads(row_dict['question_ids'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['question_ids'] = []
                    elif not isinstance(row_dict['question_ids'], list):
                        row_dict['question_ids'] = []
                else:
                    row_dict['question_ids'] = []

                if 'candidate_ids' in row_dict and row_dict['candidate_ids'] is not None:
                    if isinstance(row_dict['candidate_ids'], str):
                        try:
                            row_dict['candidate_ids'] = json.loads(row_dict['candidate_ids'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['candidate_ids'] = []
                    elif not isinstance(row_dict['candidate_ids'], list):
                        row_dict['candidate_ids'] = []
                else:
                    row_dict['candidate_ids'] = None
                return row_dict
            return None

    async def get_user_interview_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all interview sessions for a user."""
        ok = await self.connect()
        if not ok:
            return []
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM public.interview_sessions WHERE user_id = $1 ORDER BY created_at DESC",
                user_id
            )
            # Parse JSONB columns back to Python objects
            result = []
            for row in rows:
                row_dict = dict(row)
                # Parse question_ids from JSON string to list
                if 'question_ids' in row_dict and row_dict['question_ids'] is not None:
                    if isinstance(row_dict['question_ids'], str):
                        try:
                            row_dict['question_ids'] = json.loads(row_dict['question_ids'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['question_ids'] = []
                    elif not isinstance(row_dict['question_ids'], list):
                        row_dict['question_ids'] = []
                else:
                    row_dict['question_ids'] = []

                # Parse candidate_ids from JSON string to list
                if 'candidate_ids' in row_dict and row_dict['candidate_ids'] is not None:
                    if isinstance(row_dict['candidate_ids'], str):
                        try:
                            row_dict['candidate_ids'] = json.loads(row_dict['candidate_ids'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['candidate_ids'] = []
                    elif not isinstance(row_dict['candidate_ids'], list):
                        row_dict['candidate_ids'] = []
                else:
                    row_dict['candidate_ids'] = None

                result.append(row_dict)
            return result

    async def get_candidate_interview_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all interview sessions where the user is a candidate."""
        ok = await self.connect()
        if not ok:
            return []
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            # For JSONB arrays, we need to check if the user_id exists in the array
            rows = await conn.fetch(
                "SELECT * FROM public.interview_sessions WHERE candidate_ids IS NOT NULL AND candidate_ids::jsonb ? $1 ORDER BY created_at DESC",
                user_id
            )
            # Parse JSONB columns back to Python objects
            result = []
            for row in rows:
                row_dict = dict(row)
                # Parse question_ids from JSON string to list
                if 'question_ids' in row_dict and row_dict['question_ids'] is not None:
                    if isinstance(row_dict['question_ids'], str):
                        try:
                            row_dict['question_ids'] = json.loads(row_dict['question_ids'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['question_ids'] = []
                    elif not isinstance(row_dict['question_ids'], list):
                        row_dict['question_ids'] = []
                else:
                    row_dict['question_ids'] = []

                # Parse candidate_ids from JSON string to list
                if 'candidate_ids' in row_dict and row_dict['candidate_ids'] is not None:
                    if isinstance(row_dict['candidate_ids'], str):
                        try:
                            row_dict['candidate_ids'] = json.loads(row_dict['candidate_ids'])
                        except (json.JSONDecodeError, TypeError):
                            row_dict['candidate_ids'] = []
                    elif not isinstance(row_dict['candidate_ids'], list):
                        row_dict['candidate_ids'] = []
                else:
                    row_dict['candidate_ids'] = None

                result.append(row_dict)
            return result

    async def update_interview_session_status(self, session_id: str, status: str, current_question_index: Optional[int] = None) -> bool:
        """Update interview session status and optionally current question index."""
        ok = await self.connect()
        if not ok:
            return False
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            if current_question_index is not None:
                result = await conn.execute(
                    "UPDATE public.interview_sessions SET status = $1, current_question_index = $2, updated_at = NOW() WHERE id = $3",
                    status, current_question_index, session_id
                )
            else:
                result = await conn.execute(
                    "UPDATE public.interview_sessions SET status = $1, updated_at = NOW() WHERE id = $2",
                    status, session_id
                )
            return result and result.upper().startswith("UPDATE")

    async def save_interview_transcript(self, session_id: uuid.UUID, question_id: uuid.UUID, candidate_response: Optional[str] = None, ai_evaluation: Optional[str] = None, audio_file_path: Optional[str] = None, audio_duration: Optional[float] = None, response_time_seconds: Optional[float] = None, is_match: Optional[bool] = None) -> Optional[str]:
        """Save an interview transcript entry."""
        if not await self.connect():
            return None
        assert self._pool is not None
        
        sql = """
            INSERT INTO public.interview_transcripts (
                session_id, question_id, candidate_response, ai_evaluation, audio_file_path, 
                audio_duration, response_time_seconds, is_match
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """
        
        async with self._pool.acquire() as conn:
            try:
                transcript_id = await conn.fetchval(
                    sql,
                    session_id,
                    question_id,
                    candidate_response,
                    ai_evaluation,
                    audio_file_path,
                    audio_duration,
                    response_time_seconds,
                    is_match
                )
                return str(transcript_id) if transcript_id else None
            except Exception as e:
                logger.error(f"[PG] Failed to save interview transcript: {e}")
                return None

    async def get_interview_transcripts(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all transcripts for an interview session."""
        ok = await self.connect()
        if not ok:
            return []
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT t.*, q.question_text, q.expected_answer, q.category
                FROM public.interview_transcripts t
                LEFT JOIN public.interview_questions q ON t.question_id = q.id
                WHERE t.session_id = $1::uuid
                ORDER BY t.created_at ASC
                """,
                session_id
            )
            return [dict(row) for row in rows]

    async def save_interviewer_decision(self, session_id: uuid.UUID, resume_id: uuid.UUID, decision: str, notes: Optional[str] = None) -> Optional[str]:
        """Save an interviewer decision."""
        if not await self.connect():
            return None
        assert self._pool is not None
        
        sql = """
            INSERT INTO public.interviewer_decisions (session_id, resume_id, decision, notes)
            VALUES ($1, $2, $3, $4)
            RETURNING id
        """
        
        async with self._pool.acquire() as conn:
            try:
                decision_id = await conn.fetchval(sql, session_id, resume_id, decision, notes)
                return str(decision_id) if decision_id else None
            except Exception as e:
                logger.error(f"[PG] Failed to save interviewer decision: {e}")
                return None

    async def get_interviewer_decisions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all interviewer decisions for a session."""
        ok = await self.connect()
        if not ok:
            return []
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM public.interviewer_decisions WHERE session_id = $1::uuid ORDER BY created_at DESC",
                session_id
            )
            return [dict(row) for row in rows]

    async def get_user_interview_decisions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all interviewer decisions for a user's sessions."""
        ok = await self.connect()
        if not ok:
            return []
        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT d.*, s.session_type, r.name as candidate_name, r.current_position
                FROM public.interviewer_decisions d
                JOIN public.interview_sessions s ON d.session_id = s.id
                JOIN public.parsed_resumes r ON d.resume_id = r.id
                WHERE s.user_id = $1
                ORDER BY d.created_at DESC
                """,
                user_id
            )
            return [dict(row) for row in rows]

    async def get_interview_questions_by_ids(self, question_ids: List[uuid.UUID]) -> List[Dict[str, Any]]:
        """Get interview questions by their IDs."""
        if not await self.connect():
            return []
        assert self._pool is not None
        
        if not question_ids:
            return []
        
        # Create placeholders for the IN clause
        placeholders = ', '.join(f'${i+1}' for i in range(len(question_ids)))
        sql = f"SELECT * FROM public.interview_questions WHERE id IN ({placeholders})"
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *question_ids)
            
            # Convert UUID objects to strings for JSON serialization
            result = []
            for row in rows:
                row_dict = dict(row)
                for key, value in row_dict.items():
                    if isinstance(value, uuid.UUID):
                        row_dict[key] = str(value)
                    elif isinstance(value, datetime):
                        row_dict[key] = value.isoformat()
                result.append(row_dict)
            
            # Sort results to match the order of question_ids
            id_to_index = {str(qid): i for i, qid in enumerate(question_ids)}
            result.sort(key=lambda q: id_to_index.get(q['id'], 999))
            
            return result

    async def get_interview(self, interview_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get a single interview by ID."""
        if not await self.connect():
            return None
        assert self._pool is not None
        sql = "SELECT * FROM public.interviews WHERE id = $1"
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(sql, interview_id)
        if row:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, uuid.UUID):
                    row_dict[key] = str(value)
                elif isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
            return row_dict
        return None

    async def update_transcript_evaluation(self, transcript_id: str, evaluation: Dict[str, Any]) -> bool:
        """Update the AI evaluation of an interview transcript."""
        if not await self.connect():
            return False
        assert self._pool is not None
        
        sql = """
            UPDATE public.interview_transcripts
            SET ai_evaluation = $2, updated_at = NOW()
            WHERE id = $1
        """
        
        async with self._pool.acquire() as conn:
            try:
                result = await conn.execute(sql, transcript_id, json.dumps(evaluation))
                return result and result.upper().startswith("UPDATE")
            except Exception as e:
                logger.error(f"[PG] Failed to update transcript evaluation: {e}")
                return False


pg_client = PostgresClient()










