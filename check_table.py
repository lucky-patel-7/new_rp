import asyncpg
import asyncio

async def check_table():
    try:
        conn = await asyncpg.connect('postgresql://postgres:password@localhost:5432/resume_parser')
        result = await conn.fetch("SELECT column_name FROM information_schema.columns WHERE table_name = 'interview_questions' ORDER BY ordinal_position")
        print('Columns:', [row['column_name'] for row in result])
        await conn.close()
    except Exception as e:
        print(f"Error: {e}")

asyncio.run(check_table())