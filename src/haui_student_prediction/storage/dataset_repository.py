import pandas as pd
import logging
import sqlite3
import os

logger = logging.getLogger(__name__)

class DataRepository:
    def __init__(self, db_path: str = "data/session_student.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

    def save_session_data(self, df: pd.DataFrame, table_name: str = 'current_session') -> bool:
        """Lưu DataFrame vào phiên làm việc hiện tại của SQLite."""
        if df.empty:
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Ghi đè (replace) hoặc tạo mới theo từng phiên mở máy/nhập file
                df.to_sql(name=table_name, con=conn, if_exists='replace', index=False)
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu phiên SQLite: {e}")
            return False

    def get_session_data(self, table_name: str = 'current_session') -> pd.DataFrame:
        """Đọc dữ liệu từ phiên làm việc hiện tại."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception:
            return pd.DataFrame()