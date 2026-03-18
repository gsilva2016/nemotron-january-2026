import mariadb
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Access variables using os.getenv()
DB_PASSWD = os.getenv("DB_PASSWD")

# 1. Database connection parameters (use a user with CREATE privileges)
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',  # e.g., 'root'
    'password': DB_PASSWD,
}

# The database name you want to create
new_db_name = "inv_db"

conn = None
cursor = None

try:
    # 2. Establish a connection to the MariaDB server (without specifying a default database)
    conn = mariadb.connect(**db_config)
    
    # 3. Create a cursor object to execute SQL queries
    cursor = conn.cursor()
    
    # 4. Execute the CREATE DATABASE command
    create_db_query = f"CREATE DATABASE IF NOT EXISTS {new_db_name}"
    cursor.execute(create_db_query)
    
    print(f"Database '{new_db_name}' created successfully or already exists.")

    # Optional: Verify the database was created
    cursor.execute("SHOW DATABASES")
    print("\nList of databases:")
    for db in cursor:
        print(db[0])

except mariadb.Error as err:
    print(f"Error: {err}")
    sys.exit(1)

finally:
    # 5. Close the cursor and connection to free resources
    if cursor:
        cursor.close()
    if conn:
        conn.close()