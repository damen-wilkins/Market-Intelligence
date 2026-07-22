import os
from dotenv import load_dotenv

load_dotenv()

def get_connection_string():
    return (
        f"postgresql://"
        f"{os.getenv('DB_USER')}:"
        f"{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:"
        f"{os.getenv('DB_PORT')}/"
        f"{os.getenv('DB_NAME')}"
    )
def get_env(name: str):
    return os.getenv(name)