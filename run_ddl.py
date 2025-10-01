import asyncpg
import asyncio

async def run_ddl():
    conn = await asyncpg.connect(
        host='103.180.31.22',
        port=5432,
        user='postgres',
        password='Initial0',
        database='resume_parser'
    )

    # Execute the DDL
    with open('minimal_ddl.sql', 'r') as f:
        ddl_sql = f.read()

    try:
        await conn.execute(ddl_sql)
        print("DDL executed successfully")
    except Exception as e:
        print(f"Error executing DDL: {e}")

    # Check if tables exist
    tables = await conn.fetch("""
        SELECT table_schema, table_name FROM information_schema.tables 
        WHERE table_type = 'BASE TABLE' AND table_schema = 'public'
    """)
    print("Existing tables in public:", [row['table_name'] for row in tables])

    await conn.close()

if __name__ == "__main__":
    asyncio.run(run_ddl())