import requests
import json

# Test the interview action endpoint with skip action
url = 'http://localhost:8000/interview-sessions/test-session-123/action'
data = {
    "session_id": "test-session-123",  # Use a test session ID
    "action": "skip",
    "question_id": "test-question-123"
}

print("Testing interview action endpoint with skip action...")
print(f"URL: {url}")
print(f"Data: {json.dumps(data, indent=2)}")

try:
    response = requests.post(url, json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")

    if response.status_code == 200:
        response_data = response.json()
        print("\n✅ Success! Response structure:")
        print(json.dumps(response_data, indent=2))

        # Check if the expected fields are present
        expected_fields = ['success', 'action', 'message', 'next_question', 'question_number', 'total_questions']
        print(f"\nChecking for expected fields:")
        for field in expected_fields:
            if field in response_data:
                print(f"✅ {field}: {response_data[field]}")
            else:
                print(f"❌ {field}: Missing")

    else:
        print(f"\n❌ Request failed with status {response.status_code}")

except Exception as e:
    print(f"\n❌ Test failed with error: {e}")