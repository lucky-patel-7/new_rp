import requests

# Test creating a test interview session
url = 'http://localhost:8002/users/test_user/interview-sessions'
data = {
    "user_id": "test_user",
    "session_type": "test",
    "question_ids": ["ad37fe9d-46e6-4e88-beea-0f04f5e6542c"],  # Real question ID from database
    "candidate_ids": []  # Empty array for test interviews
}

response = requests.post(url, json=data)
print(f'Status: {response.status_code}')
if response.status_code == 201:
    data = response.json()
    print(f'Success: Interview session created')
    print(f'Session ID: {data["id"]}')
    print(f'Session Type: {data["session_type"]}')
    print(f'Candidate IDs: {data["candidate_ids"]}')
else:
    print(f'Error: {response.text}')