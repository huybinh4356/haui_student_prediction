import numpy as np
import pandas as pd

from haui_student_prediction.data.validator import FIELD_LABELS
from haui_student_prediction.risk.risk_calculator import RiskCalculator, RiskResult


RISK_INPUT_RANGES = {
    "absence_percentage": (0.0, 100.0),
    "lms_learning_percentage": (0.0, 100.0),
    "predicted_course_score": (0.0, 10.0),
    "interaction_score": (1.0, 5.0),
    "stress_score": (1.0, 5.0),
    "motivation_score": (1.0, 5.0),
}


class RiskPredictionService:
    def __init__(self, calculator: RiskCalculator):
        self.calculator = calculator

    def validate_student_record(self, student_record: pd.Series) -> list[str]:
        issues: list[str] = []
        for column, (minimum, maximum) in RISK_INPUT_RANGES.items():
            label = FIELD_LABELS.get(column, column)
            if column not in student_record.index:
                issues.append(f"{label} ({column}): thiếu cột")
                continue

            value = student_record.get(column)
            if value is None or pd.isna(value):
                issues.append(f"{label} ({column}): chưa có dữ liệu")
                continue

            try:
                number = float(value)
            except (TypeError, ValueError):
                issues.append(f"{label} ({column}): giá trị không phải số")
                continue

            if not np.isfinite(number):
                issues.append(f"{label} ({column}): giá trị không hữu hạn")
            elif not minimum <= number <= maximum:
                issues.append(
                    f"{label} ({column}): phải nằm trong khoảng "
                    f"{minimum:g}–{maximum:g}, hiện tại là {number:g}"
                )

        return issues

    def evaluate_student_risk(self, student_record: pd.Series) -> RiskResult:
        issues = self.validate_student_record(student_record)
        if issues:
            raise ValueError(
                "Không thể đánh giá nguy cơ: " + "; ".join(issues)
            )

        absence = self._require_float(student_record, "absence_percentage")
        lms = self._require_float(student_record, "lms_learning_percentage")
        predicted_score = self._require_float(student_record, "predicted_course_score")
        interaction = self._require_float(student_record, "interaction_score")
        stress = self._require_float(student_record, "stress_score")
        motivation = self._require_float(student_record, "motivation_score")

        return self.calculator.calculate_risk(
            absence_pct=absence,
            lms_pct=lms,
            predicted_course_score=predicted_score,
            interaction=interaction,
            interaction_max=5.0,
            stress=stress,
            stress_max=5.0,
            motivation=motivation,
            motivation_max=5.0,
            work_risk_points=50.0,
        )

    def _require_float(self, row: pd.Series, col: str) -> float:
        val = row.get(col)
        if val is None or pd.isna(val):
            raise ValueError(f"Missing required field '{col}' for risk calculation")
        try:
            number = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Field '{col}' cannot be converted to number: {val!r}"
            ) from exc
        if not np.isfinite(number):
            raise ValueError(f"Field '{col}' must be finite for risk calculation")
        return number
