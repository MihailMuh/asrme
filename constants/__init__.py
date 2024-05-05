from pathlib import Path

from dotenv import load_dotenv

BASE_DIR: Path = Path(__file__).parent.parent

load_dotenv(BASE_DIR / ".env")
