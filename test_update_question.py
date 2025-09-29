import httpx
import asyncio

async def test_update_question():
    try:
        async with httpx.AsyncClient() as client:
            # First create a question
            question_data = {
                "user_id": "106333593103872410028",
                "question_text": "What is your experience with Java?",
                "category": "Technical"
            }
            create_response = await client.post('http://localhost:8000/users/106333593103872410028/questions', json=question_data)
            if create_response.status_code == 201:
                created_question = create_response.json()
                question_id = created_question.get("id")
                print(f'Created question with ID: {question_id}')

                # Now update the question
                update_data = {
                    "question_text": "What is your experience with Java and Spring Framework?",
                    "category": "Backend Development"
                }
                update_response = await client.put(f'http://localhost:8000/questions/{question_id}', json=update_data)
                print(f'Update question status: {update_response.status_code}')
                if update_response.status_code == 200:
                    updated_question = update_response.json()
                    print(f'Success: Updated question: {updated_question.get("question_text")}')
                    print(f'Category: {updated_question.get("category")}')
                else:
                    print(f'Error: {update_response.text}')
            else:
                print(f'Failed to create question: {create_response.text}')
    except Exception as e:
        print(f'Test failed: {e}')

asyncio.run(test_update_question())