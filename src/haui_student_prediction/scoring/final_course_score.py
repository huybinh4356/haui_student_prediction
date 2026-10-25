from dataclasses import dataclass
import math
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

        weights = (
            profile.tx1_weight,
            profile.tx2_weight,
            profile.midterm_weight,
            profile.final_exam_weight,
        )
        if any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in weights):
            raise ValueError("Weights must be finite and non-negative")
        weight_sum = sum(float(weight) for weight in weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        if not profile.has_tx2 and abs(float(profile.tx2_weight)) > 1e-6:
            raise ValueError("A course profile without TX2 must use a zero TX2 weight")

        components = {
            "TX1": tx1,
            "midterm": midterm,
            "predicted final exam": predicted_final,
        }
        if profile.has_tx2:
            if tx2 is None:
                raise ValueError("TX2 is missing")
            components["TX2"] = tx2

        normalized = {}
        for name, value in components.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be a finite number")
            normalized[name] = number

        if profile.has_tx2:
            return (
                profile.tx1_weight * normalized["TX1"] +
                profile.tx2_weight * normalized["TX2"] +
                profile.midterm_weight * normalized["midterm"] +
                profile.final_exam_weight * normalized["predicted final exam"]
            )

        return (
            profile.tx1_weight * normalized["TX1"] +
            profile.midterm_weight * normalized["midterm"] +
            profile.final_exam_weight * normalized["predicted final exam"]
        )
