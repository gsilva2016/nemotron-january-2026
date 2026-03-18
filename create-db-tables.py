import mariadb
import sys
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

# Access variables
DBPASSWD = os.getenv('DB_PASSWD')

# 1. Database Connection Parameters
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': DBPASSWD,
    'database': 'inv_db'
}

# SQL statement to create a new table
# The IF NOT EXISTS clause prevents an error if the table already exists
create_table_sql_query = """
CREATE TABLE inv_db.inventory (
	id BIGINT auto_increment NOT NULL,
	category varchar(100) NOT NULL,
	description varchar(100) NOT NULL,
	ingredients varchar(100) NULL,
	coo varchar(100) NULL,
	brand varchar(100) NULL,
	price FLOAT DEFAULT 0.00 NOT NULL,
	quantity BIGINT DEFAULT 0 NULL,
	CONSTRAINT inventory_pk PRIMARY KEY (id)
)
ENGINE=InnoDB
"""

insert_table_sql_query = """
INSERT INTO inv_db.inventory (category,description,ingredients,coo,brand,price,quantity) VALUES
	 ('Beverage','Coke','Caffeine, Sugar, Water','USA','Coca-Cola',2.5,10),
	 ('Sandwich','Turkey sandwich. Deliciously packed protein.','Turkey, Lettuce,American Cheese, Bread','USA','Fry''s',5.56,18),
	 ('Beverage','Lemonade','Water, Sugar, Brazilian Lemons','BRAZIL','Pepsi',3.3,22),
	 ('Beverage','Iced coffee','Coffee, water, sugar','COLUMBIA','COLUMBIAN INC',2.5,9),
	 ('Icecream','Vanilla icecream sandwich','Chocolate, vanilla, milk, water','USA','Fat Boy',9.99,100),
	 ('Sandwich','Chicken sandwich with lettuce and peppers of your choice.','Chicken, hot peppers, sweet peppers','USA','Fry''s',8.4,23);
"""

conn = None
cursor = None

try:
    # 2. Establish the connection
    conn = mariadb.connect(**db_config)
    print("Connection successful!")

    # 3. Create a cursor object
    cursor = conn.cursor()

    # 4. Execute the CREATE TABLE statement
    cursor.execute(create_table_sql_query)
    print("Table created successfully (if it did not already exist).")

    cursor.execute(insert_table_sql_query)
    print("Table data created successfully (if it did not already exist).")

    # 5. Commit the changes
    conn.commit()

except mariadb.Error as e:
    print(f"Error connecting to or interacting with MariaDB: {e}")
    sys.exit(1)

finally:
    # 6. Close the cursor and connection to free resources
    if cursor:
        cursor.close()
    if conn:
        conn.close()
    print("Connection closed.")