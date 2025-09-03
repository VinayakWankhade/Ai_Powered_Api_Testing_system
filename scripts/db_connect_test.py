import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure data dir exists
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

from src.database.connection import create_tables, get_db_session  # noqa: E402
from sqlalchemy import text  # noqa: E402


def main():
    try:
        create_tables()
        db = get_db_session()
        db.execute(text("SELECT 1"))
        db.close()
        print("DB_OK")
    except Exception as e:
        print(f"DB_ERROR: {e}")


if __name__ == "__main__":
    main()

