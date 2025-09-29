import httpx
import asyncio

async def test_interview_questions():
    try:
        async with httpx.AsyncClient() as client:
            # Test getting interview questions
            response = await client.get('http://localhost:8000/users/108849531532929381403/questions')
            print(f'Interview questions status: {response.status_code}')
            if response.status_code == 200:
                data = response.json()
                print(f'Success: Retrieved {len(data)} questions')
                if data:
                    print(f'First question: {data[0].get("question_text", "N/A")[:50]}...')
            else:
                print(f'Error: {response.text}')
    except Exception as e:
        print(f'Test failed: {e}')

asyncio.run(test_interview_questions())