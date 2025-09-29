import httpx
import asyncio

async def check_health():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('http://localhost:8000/health')
            print(f'Health check: {response.status_code}')
            if response.status_code == 200:
                data = response.json()
                print('Services status:')
                for service, status in data.get('services', {}).items():
                    print(f'  {service}: {status}')
    except Exception as e:
        print(f'Health check failed: {e}')

asyncio.run(check_health())