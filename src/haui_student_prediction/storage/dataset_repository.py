from pathlib import Path
import pandas as pd
import logging
import sqlite3

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "session_student.db"


class DataRepository:
    def __init__(self, db_path: Path | str | None = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def save_session_data(self, df: pd.DataFrame, table_name: str = "current_session") -> bool:
        if df.empty:
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(name=table_name, con=conn, if_exists="replace", index=False)
            return True
        except Exception as e:
            logger.error("Lỗi lưu phiên SQLite: %s", e)
            return False

    def get_session_data(self, table_name: str = "current_session") -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception as e:
            logger.warning("Không đọc được phiên SQLite: %s", e)
            return pd.DataFrame()