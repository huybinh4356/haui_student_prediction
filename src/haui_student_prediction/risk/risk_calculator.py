from dataclasses import dataclass
import math
from typing import Tuple

@dataclass
class RiskResult:
    absence_status: str
    weighted_score: float
    risk_category: str
    is_override: bool

class RiskCalculator:
    def __init__(self):
        self.weights = {
            'absence': 0.75,
            'academic': 0.08,
            'lms': 0.08,
            'interaction': 0.04,
            'work': 0.02,
            'stress': 0.015,
            'motivation': 0.015
        }

    def _normalize_absence(self, percentage: float) -> Tuple[float, str]:
        if percentage > 75:
            return 100.0, "REPEAT_REQUIRED"
        if percentage >= 20:
            return 80.0, "high"
        if percentage >= 10:
            return 55.0, "medium"
        if percentage >= 5:
            return 30.0, "low"
        return 0.0, "very low"

    def _normalize_lms(self, percentage: float) -> float:
        if percentage < 5:
            return 100.0
        if percentage < 10:
            return 75.0
        if percentage < 20:
            return 50.0
        if percentage < 30:
            return 25.0
        return 0.0

    def _normalize_academic(self, predicted_course_score: float) -> float:
        return max(0.0, min(100.0, (10.0 - predicted_course_score) * 10.0))

    def _normalize_interaction(self, score: float, max_score: float) -> float:
        return max(0.0, min(100.0, 100.0 - (score / max_score * 100.0)))

    def _normalize_stress(self, score: float, max_score: float) -> float:
        return max(0.0, min(100.0, (score / max_score * 100.0)))

    def _normalize_motivation(self, score: float, max_score: float) -> float:
        return max(0.0, min(100.0, 100.0 - (score / max_score * 100.0)))

    def _get_category(self, score: float) -> str:
        if score >= 80:
            return "Very high"
        if score >= 60:
            return "High"
        if score >= 40:
            return "Medium"
        if score >= 20:
            return "Low"
        return "Very low"

    def calculate_risk(
        self,
        absence_pct: float,
        lms_pct: float,
        predicted_course_score: float,
        interaction: float,
        interaction_max: float,
        stress: float,
        stress_max: float,
        motivation: float,
        motivation_max: float,
        work_risk_points: float = 50.0
    ) -> RiskResult:
        inputs = {
            "absence_pct": absence_pct,
            "lms_pct": lms_pct,
            "predicted_course_score": predicted_course_score,
            "interaction": interaction,
            "interaction_max": interaction_max,
            "stress": stress,
            "stress_max": stress_max,
            "motivation": motivation,
            "motivation_max": motivation_max,
            "work_risk_points": work_risk_points,
        }
        for name, value in inputs.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must be a finite number")
            inputs[name] = number

        absence_pct = inputs["absence_pct"]
        lms_pct = inputs["lms_pct"]
        predicted_course_score = inputs["predicted_course_score"]
        interaction = inputs["interaction"]
        interaction_max = inputs["interaction_max"]
        stress = inputs["stress"]
        stress_max = inputs["stress_max"]
        motivation = inputs["motivation"]
        motivation_max = inputs["motivation_max"]
        work_risk_points = inputs["work_risk_points"]

        if not (0 <= absence_pct <= 100 and 0 <= lms_pct <= 100):
            raise ValueError("Percentages must be within 0 and 100")
        if interaction_max <= 0 or stress_max <= 0 or motivation_max <= 0:
            raise ValueError("Survey scale maxima must be greater than zero")
        if not 0 <= work_risk_points <= 100:
            raise ValueError("work_risk_points must be within 0 and 100")

        if any(not math.isfinite(float(weight)) or float(weight) < 0 for weight in self.weights.values()):
            raise ValueError("Risk weights must be finite and non-negative")
        if abs(sum(float(weight) for weight in self.weights.values()) - 1.0) > 1e-6:
            raise ValueError("Risk weights must sum to 1.0")

        absence_risk, status = self._normalize_absence(absence_pct)
        lms_risk = self._normalize_lms(lms_pct)
        acad_risk = self._normalize_academic(predicted_course_score)
        int_risk = self._normalize_interaction(interaction, interaction_max)
        stress_risk = self._normalize_stress(stress, stress_max)
        mot_risk = self._normalize_motivation(motivation, motivation_max)

        total_risk = (
            self.weights['absence'] * absence_risk +
            self.weights['academic'] * acad_risk +
            self.weights['lms'] * lms_risk +
            self.weights['interaction'] * int_risk +
            self.weights['work'] * work_risk_points +
            self.weights['stress'] * stress_risk +
            self.weights['motivation'] * mot_risk
        )

        return RiskResult(
            absence_status=status,
            weighted_score=total_risk,
            risk_category=self._get_category(total_risk),
            is_override=(absence_pct > 75)
        )
