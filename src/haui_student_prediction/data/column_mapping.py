import re
import unicodedata
from typing import Dict


def normalize_column_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).strip().lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("đ", "d")
    return re.sub(r"[^a-z0-9]", "", text)


def is_training_target_column(column_name: object) -> bool:
    normalized = normalize_column_name(column_name)
    prediction_markers = (
        "predicted",
        "prediction",
        "forecast",
        "expected",
        "estimate",
        "estimated",
        "dukien",
        "dudoan",
        "uocluong",
    )
    if any(marker in normalized for marker in prediction_markers):
        return False

    aliases = {
        "targetfinalexamscore",
        "finalexamscore",
        "actualfinalexamscore",
        "diemthi",
        "diemthicuoiky",
        "diemcuoiky",
        "diemck",
        "diemthucte",
        "diemthicuoikythucte",
        "diemcuoikythucte",
        "diemckthucte",
    }
    if normalized in aliases:
        return True
    if "final" in normalized and "exam" in normalized and "score" in normalized:
        return True
    return (
        "diemthi" in normalized
        or "diemcuoiky" in normalized
        or normalized.startswith("diemck")
    )


def get_default_column_mapping() -> Dict[str, str]:
    return {
        "Mã SV": "student_id",
        "Mã sinh viên": "student_id",
        "MSSV": "student_id",
        "MSV": "student_id",
        "Student ID": "student_id",
        "student_id": "student_id",
        "Họ và tên": "full_name",
        "Họ tên": "full_name",
        "Tên sinh viên": "full_name",
        "Full Name": "full_name",
        "full_name": "full_name",
        "Ngày sinh": "date_of_birth",
        "Date of Birth": "date_of_birth",
        "date_of_birth": "date_of_birth",
        "Năm sinh": "birth_year",
        "Birth Year": "birth_year",
        "birth_year": "birth_year",
        "Quê quán": "hometown",
        "Quê": "hometown",
        "Hometown": "hometown",
        "hometown": "hometown",
        "Ngành": "major",
        "Ngành học": "major",
        "Major": "major",
        "major": "major",
        "Tỷ lệ nghỉ": "absence_percentage",
        "Tỷ lệ nghỉ học": "absence_percentage",
        "Tỷ lệ vắng mặt": "absence_percentage",
        "Tỷ lệ nghỉ (%)": "absence_percentage",
        "Absence Percentage": "absence_percentage",
        "absence_percentage": "absence_percentage",
        "Tỷ lệ học LMS": "lms_learning_percentage",
        "Tỷ lệ LMS": "lms_learning_percentage",
        "Tỷ lệ thời gian học trên hệ thống": "lms_learning_percentage",
        "LMS Percentage": "lms_learning_percentage",
        "LMS Learning Percentage": "lms_learning_percentage",
        "lms_learning_percentage": "lms_learning_percentage",
        "Điểm TX1": "tx1_score",
        "TX1": "tx1_score",
        "TX1 Score": "tx1_score",
        "tx1_score": "tx1_score",
        "Điểm TX2": "tx2_score",
        "TX2": "tx2_score",
        "TX2 Score": "tx2_score",
        "tx2_score": "tx2_score",
        "Điểm GK": "midterm_score",
        "Điểm giữa kỳ": "midterm_score",
        "Giữa kỳ": "midterm_score",
        "Midterm Score": "midterm_score",
        "midterm_score": "midterm_score",
        "Đi làm thêm": "works_part_time",
        "Có đi làm thêm": "works_part_time",
        "Làm thêm": "works_part_time",
        "Works Part Time": "works_part_time",
        "works_part_time": "works_part_time",
        "Mức độ stress": "stress_score",
        "Stress": "stress_score",
        "Stress Score": "stress_score",
        "stress_score": "stress_score",
        "Động lực học": "motivation_score",
        "Động lực học tập": "motivation_score",
        "Motivation": "motivation_score",
        "Motivation Score": "motivation_score",
        "motivation_score": "motivation_score",
        "Điểm tương tác": "interaction_score",
        "Mức độ tương tác": "interaction_score",
        "Interaction": "interaction_score",
        "Interaction Score": "interaction_score",
        "interaction_score": "interaction_score",
        "Mã môn": "course_id",
        "Mã môn học": "course_id",
        "Mã học phần": "course_id",
        "Course ID": "course_id",
        "course_id": "course_id",
        "Học kỳ": "semester",
        "Kỳ học": "semester",
        "Semester": "semester",
        "semester": "semester",
    }
