from pathlib import Path
import sqlite3
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "student_data.db"


class SQLiteDatabase:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute_update(self, query, params=()):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid

    def fetch_all(self, query, params=()):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def fetch_dataframe(self, query, params=()):
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=list(params))