import asyncpg
import asyncio

async def check_interviews_table():
    try:
        conn = await asyncpg.connect('postgresql://postgres:Initial0@103.180.31.22:5432/resume_parser')
        result = await conn.fetchval("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'interviews')")
        print('Interviews table exists:', result)
        await conn.close()
    except Exception as e:
        print('Error:', e)

asyncio.run(check_interviews_table())