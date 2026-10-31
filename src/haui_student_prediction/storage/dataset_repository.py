from pathlib import Path
import logging
import re
import sqlite3

import pandas as pd

from haui_student_prediction.app_config import get_project_root

logger = logging.getLogger(__name__)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DataRepository:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = (
            Path(db_path).expanduser().resolve()
            if db_path is not None
            else get_project_root() / "data" / "session_student.db"
        ).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_table_name(table_name: str) -> str:
        if not _IDENTIFIER_PATTERN.fullmatch(table_name):
            raise ValueError(f"Invalid SQLite table name: {table_name!r}")
        return table_name

    def save_session_data(
        self,
        df: pd.DataFrame,
        table_name: str = "current_session",
    ) -> bool:
        if df is None or df.empty:
            logger.warning("Không lưu phiên SQLite vì dataset rỗng.")
            return False
        try:
            table_name = self._validate_table_name(table_name)
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)
            return True
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            logger.exception("Lỗi lưu phiên SQLite: %s", exc)
            return False

    def get_session_data(
        self,
        table_name: str = "current_session",
    ) -> pd.DataFrame:
        try:
            table_name = self._validate_table_name(table_name)
            if not self.db_path.exists():
                return pd.DataFrame()
            with sqlite3.connect(self.db_path) as conn:
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
                    ("table", table_name),
                ).fetchone()
                if exists is None:
                    return pd.DataFrame()
                return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        except (OSError, sqlite3.Error, ValueError, TypeError):
            logger.exception("Không đọc được phiên SQLite.")
            return pd.DataFrame()
