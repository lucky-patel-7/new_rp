import httpx

# Test the shortlisted endpoint
url = 'http://localhost:8000/resumes/shortlisted'
params = {'user_id': '106333593103872410028'}

async def test_endpoint():
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        print(f'Status: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'Success: {len(data.get("shortlisted_candidates", []))} shortlisted candidates')
        else:
            print(f'Error: {response.text}')

# Run the async test
import asyncio
asyncio.run(test_endpoint())