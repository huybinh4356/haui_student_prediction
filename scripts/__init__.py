import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from storage.sqlite_database import SQLiteDatabase

def init_database():
    db = SQLiteDatabase("data/student_data.db")
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS student_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT UNIQUE NOT NULL,
        full_name TEXT NOT NULL,
        absence_percentage REAL NOT NULL,
        lms_learning_percentage REAL NOT NULL,
        tx1_score REAL NOT NULL,
        midterm_score REAL NOT NULL,
        tx2_score REAL,
        has_tx2 INTEGER DEFAULT 0,
        interaction_score REAL,
        stress_score REAL,
        motivation_score REAL,
        works_part_time INTEGER,
        actual_final_exam_score REAL,
        date_of_birth TEXT,
        birth_year INTEGER,
        hometown TEXT,
        major TEXT,
        course_id TEXT,
        semester TEXT,
        snapshot_date TEXT
    );
    """
    db.execute_update(create_table_query)
    print("Đã tạo bảng 'student_records' theo chuẩn StudentRecord.")

    csv_path = "data/synthetic_haui_sample.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        with db.get_connection() as conn:
            df.to_sql('student_records', conn, if_exists='replace', index=False)
            print(f"Đã nạp {len(df)} dòng dữ liệu từ CSV vào SQLite.")
    else:
        print(f"Không tìm thấy {csv_path}, hãy đảm bảo file nằm đúng thư mục.")

if __name__ == "__main__":
    init_database()