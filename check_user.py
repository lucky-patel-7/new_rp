import asyncio
import asyncpg
from config.settings import settings

async def check_user():
    try:
        # Connect to database
        conn = await asyncpg.connect(
            host=settings.postgres.host,
            port=settings.postgres.port,
            user=settings.postgres.username,
            password=settings.postgres.password,
            database=settings.postgres.database
        )

        # Check if user exists in users table
        user_row = await conn.fetchrow(
            "SELECT id, email, name FROM public.users WHERE id = $1",
            "106333593103872410028"
        )

        if user_row:
            print(f"User found: {dict(user_row)}")
        else:
            print("User not found in users table")

        # Check if user has any resumes
        resume_count = await conn.fetchval(
            "SELECT COUNT(*) FROM qdrant_resumes WHERE owner_user_id = $1",
            "106333593103872410028"
        )
        print(f"Total resumes for user: {resume_count}")

        # Check shortlisted resumes
        shortlisted_count = await conn.fetchval(
            "SELECT COUNT(*) FROM qdrant_resumes WHERE owner_user_id = $1 AND is_shortlisted = TRUE",
            "106333593103872410028"
        )
        print(f"Shortlisted resumes for user: {shortlisted_count}")

        # Check user limits
        limits_row = await conn.fetchrow(
            "SELECT * FROM public.user_resume_limits WHERE user_id = $1",
            "106333593103872410028"
        )

        if limits_row:
            print(f"User limits: {dict(limits_row)}")
        else:
            print("User limits not found")

        await conn.close()

    except Exception as e:
        print(f"Database error: {e}")

asyncio.run(check_user())