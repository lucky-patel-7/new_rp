import asyncpg
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def check_schema():
    conn = await asyncpg.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432')),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'password'),
        database=os.getenv('DB_NAME', 'rp_plus_ai')
    )

    # Check interview_sessions table structure
    result = await conn.fetch("""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'interview_sessions'
        ORDER BY ordinal_position
    """)

    print('interview_sessions table structure:')
    for row in result:
        print(f'  {row["column_name"]}: {row["data_type"]} (nullable: {row["is_nullable"]})')

    await conn.close()

if __name__ == "__main__":
    asyncio.run(check_schema())