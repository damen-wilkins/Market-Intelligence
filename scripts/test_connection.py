import psycopg
from database.connection import get_connection_string

with psycopg.connect(get_connection_string()) as conn:
    print("Connected successfully!")