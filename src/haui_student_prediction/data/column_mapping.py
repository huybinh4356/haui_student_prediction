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
    aliases = {
        "targetfinalexamscore",
        "finalexamscore",
        "diemthi",
        "diemthicuoiky",
        "diemcuoiky",
        "diemck",
    }
    if normalized in aliases:
        return True
    if "predicted" in normalized:
        return False
    if "final" in normalized and "exam" in normalized and "score" in normalized:
        return True
    return "diem" in normalized and ("thi" in normalized or "cuoiky" in normalized)


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
        "Đi làm thêm": "works_part_time",
        "Mức độ stress": "stress_score",
        "Động lực học": "motivation_score",
        "Điểm tương tác": "interaction_score",
        "Mã môn": "course_id",
        "Học kỳ": "semester",
    }
