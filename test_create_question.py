import httpx
import asyncio

async def test_create_question():
    try:
        async with httpx.AsyncClient() as client:
            # Test creating an interview question
            question_data = {
                "user_id": "106333593103872410028",
                "question_text": "What is your experience with Python?",
                "category": "Technical"
            }
            response = await client.post('http://localhost:8000/users/106333593103872410028/questions', json=question_data)
            print(f'Create question status: {response.status_code}')
            if response.status_code == 201:
                data = response.json()
                print(f'Success: Created question with ID: {data.get("id")}')
                print(f'Question: {data.get("question_text")}')
            else:
                print(f'Error: {response.text}')
    except Exception as e:
        print(f'Test failed: {e}')

asyncio.run(test_create_question())