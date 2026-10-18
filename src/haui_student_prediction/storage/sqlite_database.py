import sqlite3
import pandas as pd
import os

class SQLiteDatabase:
    def __init__(self, db_path="data/student_data.db"):
        self.db_path = db_path
        # Tự động tạo thư mục chứa database nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def get_connection(self):
        """Khởi tạo kết nối. Sử dụng row_factory để trả về dữ liệu dạng dictionary."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute_update(self, query, params=()):
        """Dùng cho các lệnh INSERT, UPDATE, DELETE, CREATE TABLE"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.lastrowid # Trả về ID của dòng vừa được insert

    def fetch_all(self, query, params=()):
        """Dùng cho lệnh SELECT, trả về danh sách các dictionary (phù hợp cho API hoặc UI)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def fetch_dataframe(self, query, params=()):
        """Dùng cho lệnh SELECT, trả về trực tiếp Pandas DataFrame (Tối ưu cho Module EDA & ML)"""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)