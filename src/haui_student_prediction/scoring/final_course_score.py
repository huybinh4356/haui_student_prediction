from dataclasses import dataclass
from typing import Optional

@dataclass
class CourseProfile:
    tx1_weight: float
    tx2_weight: float
    midterm_weight: float
    final_exam_weight: float
    has_tx2: bool

class ScoringService:
    def __init__(self, profiles: dict[str, CourseProfile]):
        self.profiles = profiles

    def calculate_score(
        self,
        profile_id: str,
        tx1: float,
        midterm: float,
        predicted_final: float,
        tx2: Optional[float] = None
    ) -> float:
        if profile_id not in self.profiles:
            raise ValueError("Invalid profile_id")

        profile = self.profiles[profile_id]

        weight_sum = (
            profile.tx1_weight + profile.tx2_weight
            + profile.midterm_weight + profile.final_exam_weight
        )
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        if profile.has_tx2:
            if tx2 is None:
                raise ValueError("TX2 is missing")
            return (
                profile.tx1_weight * tx1 +
                profile.tx2_weight * tx2 +
                profile.midterm_weight * midterm +
                profile.final_exam_weight * predicted_final
            )

        return (
            profile.tx1_weight * tx1 +
            profile.midterm_weight * midterm +
            profile.final_exam_weight * predicted_final
        )
