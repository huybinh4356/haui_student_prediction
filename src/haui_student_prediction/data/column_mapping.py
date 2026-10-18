from typing import Dict

def get_default_column_mapping() -> Dict[str, str]:
    return {
        "Mã SV": "student_id",
        "Họ và tên": "full_name",
        "Ngày sinh": "date_of_birth",
        "Năm sinh": "birth_year",
        "Quê quán": "hometown",
        "Ngành": "major",
        "Tỷ lệ nghỉ": "absence_percentage",
        "Tỷ lệ học LMS": "lms_learning_percentage",
        "Điểm TX1": "tx1_score",
        "Điểm TX2": "tx2_score",
        "Điểm GK": "midterm_score",
        "Điểm thi": "actual_final_exam_score",
        "Đi làm thêm": "works_part_time",
        "Mức độ stress": "stress_score",
        "Động lực học": "motivation_score",
        "Điểm tương tác": "interaction_score",
        "Mã môn": "course_id",
        "Học kỳ": "semester"
    }
