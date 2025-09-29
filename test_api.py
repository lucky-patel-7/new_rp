import requests

# Test the /resumes endpoint with the user that has resumes
url = 'http://localhost:8000/resumes'
params = {'user_id': '108849531532929381403', 'page': 1, 'page_size': 10}

response = requests.get(url, params=params)
print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    print(f'Success: {data["success"]}')
    print(f'Total resumes: {data["total"]}')
    print(f'Items returned: {len(data["items"])}')
    if data["items"]:
        print(f'First resume: {data["items"][0]["name"]}')
else:
    print(f'Error: {response.text}')