import asyncio
from src.resume_parser.database.postgres_client import pg_client

async def check_questions():
    await pg_client.connect()
    # Try to get some interview questions
    try:
        questions = await pg_client.get_interview_questions(user_id="test_user")
        print(f'Found {len(questions)} questions')
        if questions:
            print(f'First question ID: {questions[0]["id"]}')
            print(f'First question text: {questions[0]["question_text"][:50]}...')
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(check_questions())