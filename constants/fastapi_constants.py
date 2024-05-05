import os

SERVER_PORT: int = int(os.getenv("SERVER_PORT"))
SERVER_HOST: str = os.getenv("SERVER_HOST")
SERVER_WORKERS: int = int(os.getenv("SERVER_WORKERS"))
