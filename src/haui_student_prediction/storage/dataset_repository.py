import logging
from typing import Optional
import pandas as pd

# Chèn thêm 2 import mới này
from storage.sqlite_database import SQLiteDatabase
# from models.student_record import StudentRecord  # Bỏ comment và sửa lại đường dẫn nếu cần

# Giữ nguyên phần khai báo logger cũ
logger = logging.getLogger(__name__)
class DataRepository:
    def __init__(self):
        self.db = SQLiteDatabase()

    def get_student_by_id(self, student_id: str) -> Optional[StudentRecord]:
        query = "SELECT * FROM student_records WHERE student_id = ?"
        result = self.db.fetch_all(query, (student_id,))
        
        if not result:
            return None
            
        row = result[0]
        return StudentRecord(
            student_id=row['student_id'],
            full_name=row['full_name'],
            absence_percentage=row['absence_percentage'],
            lms_learning_percentage=row['lms_learning_percentage'],
            tx1_score=row['tx1_score'],
            midterm_score=row['midterm_score'],
            tx2_score=row['tx2_score'],
            has_tx2=bool(row['has_tx2']),
            interaction_score=row['interaction_score'],
            stress_score=row['stress_score'],
            motivation_score=row['motivation_score'],
            works_part_time=bool(row['works_part_time']) if row['works_part_time'] is not None else None,
            actual_final_exam_score=row['actual_final_exam_score'],
            date_of_birth=row['date_of_birth'],
            birth_year=row['birth_year'],
            hometown=row['hometown'],
            major=row['major'],
            course_id=row['course_id'],
            semester=row['semester'],
            snapshot_date=row['snapshot_date']
        )