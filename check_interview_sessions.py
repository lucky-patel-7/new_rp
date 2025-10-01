import asyncio
from src.resume_parser.database.postgres_client import pg_client

async def check_table():
    ok = await pg_client.connect()
    if not ok:
        print('Failed to connect to database')
        return

    try:
        async with pg_client._pool.acquire() as conn:
            # Check if interview_sessions table exists
            result = await conn.fetchrow("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'interview_sessions')")
            print(f"interview_sessions table exists: {result[0]}")

            if result[0]:
                # Get table structure
                columns = await conn.fetch("SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'interview_sessions' ORDER BY ordinal_position")
                print("Table structure:")
                for col in columns:
                    print(f"  {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")

                # Check recent sessions
                sessions = await conn.fetch("SELECT id, user_id, session_type, question_ids, candidate_ids, status, created_at FROM interview_sessions ORDER BY created_at DESC LIMIT 5")
                print(f"\nRecent sessions ({len(sessions)}):")
                for session in sessions:
                    print(f"  ID: {session['id']}, User: {session['user_id']}, Type: {session['session_type']}, Status: {session['status']}, Created: {session['created_at']}")
                    print(f"    Question IDs: {session['question_ids']}")
                    print(f"    Candidate IDs: {session['candidate_ids']}")
            else:
                print("interview_sessions table does not exist!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await pg_client.close()

if __name__ == "__main__":
    asyncio.run(check_table())</content>
<parameter name="filePath">c:\Users\25080402\Desktop\RP_Plus_AI\new_rp\check_interview_sessions.py