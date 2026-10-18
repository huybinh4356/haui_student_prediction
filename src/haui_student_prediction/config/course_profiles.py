from dataclasses import dataclass
from typing import Dict, List
from haui_student_prediction.scoring.final_course_score import CourseProfile


@dataclass
class CourseProfileConfig:
    profile_id: str
    name: str
    tx1_weight: float
    tx2_weight: float
    midterm_weight: float
    final_exam_weight: float
    has_tx2: bool

    def to_scoring_profile(self) -> CourseProfile:
        return CourseProfile(
            tx1_weight=self.tx1_weight,
            tx2_weight=self.tx2_weight,
            midterm_weight=self.midterm_weight,
            final_exam_weight=self.final_exam_weight,
            has_tx2=self.has_tx2,
        )


DEFAULT_COURSE_PROFILES: Dict[str, CourseProfileConfig] = {
    "no_tx2": CourseProfileConfig(
        profile_id="no_tx2",
        name="Không có TX2",
        tx1_weight=0.20,
        tx2_weight=0.0,
        midterm_weight=0.20,
        final_exam_weight=0.60,
        has_tx2=False,
    ),
    "with_tx2": CourseProfileConfig(
        profile_id="with_tx2",
        name="Có TX2",
        tx1_weight=0.15,
        tx2_weight=0.15,
        midterm_weight=0.20,
        final_exam_weight=0.50,
        has_tx2=True,
    ),
}

DEFAULT_COURSES_WITH_TX2: List[str] = []
