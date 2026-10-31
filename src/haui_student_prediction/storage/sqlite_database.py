from pathlib import Path
import sqlite3

import pandas as pd

from haui_student_prediction.app_config import get_project_root


class SQLiteDatabase:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = (
            Path(db_path).expanduser().resolve()
            if db_path is not None
            else (get_project_root() / "data" / "student_data.db").resolve()
        )
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