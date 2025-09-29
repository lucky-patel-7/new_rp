import asyncio
import asyncpg

async def main():
    try:
        conn = await asyncpg.connect(
            host='103.180.31.22',
            port=5432,
            database='resume_parser',
            user='postgres',
            password='Initial0'
        )
        
        # Check if the user from resumes exists
        user = await conn.fetchrow('SELECT id, email FROM public.users WHERE id = $1', '108849531532929381403')
        if user:
            print(f'User exists: {dict(user)}')
        else:
            print('User 108849531532929381403 does not exist in users table')
            
        await conn.close()
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    asyncio.run(main())