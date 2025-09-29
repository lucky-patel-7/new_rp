import asyncio
import asyncpg
from config.settings import settings

async def debug_query():
    try:
        # Connect to database
        conn = await asyncpg.connect(
            host=settings.postgres.host,
            port=settings.postgres.port,
            user=settings.postgres.username,
            password=settings.postgres.password,
            database=settings.postgres.database
        )

        user_id = "106333593103872410028"

        # Try the exact query from get_shortlisted_resumes
        sql = "SELECT * FROM qdrant_resumes WHERE owner_user_id = $1 AND is_shortlisted = TRUE ORDER BY upload_timestamp DESC"

        print(f"Executing query: {sql}")
        print(f"With user_id: {user_id}")

        rows = await conn.fetch(sql, user_id)

        print(f"Query returned {len(rows)} rows")

        if rows:
            print("First row keys:", list(rows[0].keys()))
            print("First row sample data:")
            for key, value in list(rows[0].items())[:5]:  # First 5 fields
                print(f"  {key}: {type(value)} = {value}")

        # Try to convert UUIDs like the method does
        result = []
        for row in rows:
            row_dict = dict(row)
            for key, value in row_dict.items():
                if isinstance(value, asyncpg.pgproto.pgproto.UUID):
                    row_dict[key] = str(value)
            result.append(row_dict)

        print(f"Successfully processed {len(result)} rows")

        await conn.close()

    except Exception as e:
        print(f"Database error: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(debug_query())