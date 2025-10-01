import asyncio
import sys
sys.path.insert(0, '.')
from src.resume_parser.database.postgres_client import pg_client

async def check_data():
    await pg_client.connect()

    # Check raw interviews table
    print('=== Raw interviews table ===')
    try:
        rows = await pg_client._pool.fetch("SELECT * FROM public.interviews WHERE user_id = $1 ORDER BY created_at DESC", '106333593103872410028')
        print(f'Raw Interviews for user: {len(rows)}')
        for row in rows:
            print(f'  {dict(row)}')
    except Exception as e:
        print(f'Error: {e}')

    # Check if interviews table exists
    print('\n=== Table existence ===')
    try:
        exists = await pg_client._pool.fetchval("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'interviews')")
        print(f'Interviews table exists: {exists}')
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(check_data())