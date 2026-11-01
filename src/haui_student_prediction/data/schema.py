from dataclasses import dataclass
from typing import Optional

@dataclass
class StudentRecord:
    student_id: str
    full_name: str
    absence_percentage: float
    lms_learning_percentage: float
    tx1_score: float
    midterm_score: float
    tx2_score: Optional[float] = None
    has_tx2: bool = False
    interaction_score: Optional[float] = None
    stress_score: Optional[float] = None
    motivation_score: Optional[float] = None
    works_part_time: Optional[bool] = None
    date_of_birth: Optional[str] = None
    birth_year: Optional[int] = None
    hometown: Optional[str] = None
    major: Optional[str] = None
    course_id: Optional[str] = None
    semester: Optional[str] = None
    snapshot_date: Optional[str] = None
