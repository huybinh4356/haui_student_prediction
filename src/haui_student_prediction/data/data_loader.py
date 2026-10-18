from storage.sqlite_database import SQLiteDatabase
import pandas as pd
from typing import Optional
# Nhớ import class StudentRecord từ file khai báo của dự án
# Ví dụ: from models.student_record import StudentRecord 

class StudentDataLoader:
    def __init__(self):
        # Khởi tạo kết nối db
        self.db = SQLiteDatabase()

    def get_student_by_id(self, student_id: str) -> Optional[StudentRecord]:
        """Lấy thông tin một sinh viên cụ thể và trả về trực tiếp object StudentRecord"""
        query = "SELECT * FROM student_records WHERE student_id = ?"
        result = self.db.fetch_all(query, (student_id,))
        
        if not result:
            return None
            
        row = result[0]
        # Mapping trực tiếp dữ liệu từ dictionary sang dataclass
        return StudentRecord(
            student_id=row['student_id'],
            full_name=row['full_name'],
            absence_percentage=row['absence_percentage'],
            lms_learning_percentage=row['lms_learning_percentage'],
            tx1_score=row['tx1_score'],
            midterm_score=row['midterm_score'],
            tx2_score=row['tx2_score'],
            has_tx2=bool(row['has_tx2']),  # Ép kiểu int 0/1 từ SQLite về bool
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

    def get_all_students_df(self) -> pd.DataFrame:
        """Lấy toàn bộ dữ liệu sinh viên trả về DataFrame để Model ML sử dụng"""
        # Đổi tên bảng thành student_records cho khớp với init_db
        query = "SELECT * FROM student_records"
        return self.db.fetch_dataframe(query)