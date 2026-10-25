import pandas as pd
import numpy as np
from haui_student_prediction.risk.risk_calculator import RiskCalculator, RiskResult


class RiskPredictionService:
    def __init__(self, calculator: RiskCalculator):
        self.calculator = calculator

    def evaluate_student_risk(self, student_record: pd.Series) -> RiskResult:
        absence = self._require_float(student_record, "absence_percentage")
        lms = self._require_float(student_record, "lms_learning_percentage")
        predicted_score = self._require_float(student_record, "predicted_course_score")
        interaction = self._require_float(student_record, "interaction_score")
        stress = self._require_float(student_record, "stress_score")
        motivation = self._require_float(student_record, "motivation_score")
        self._parse_bool(student_record, "works_part_time")

        work_points = 50.0

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
            work_risk_points=work_points,
        )

    def _require_float(self, row: pd.Series, col: str) -> float:
        val = row.get(col)
        if val is None or pd.isna(val):
            raise ValueError(f"Missing required field '{col}' for risk calculation")
        try:
            number = float(val)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Field '{col}' cannot be converted to number: {val!r}") from e
        if not np.isfinite(number):
            raise ValueError(f"Field '{col}' must be finite for risk calculation")
        return number

    def _parse_bool(self, row: pd.Series, col: str) -> bool:
        val = row.get(col)
        if val is None or pd.isna(val):
            raise ValueError(f"Missing required field '{col}' for risk calculation")
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            if not np.isfinite(float(val)):
                raise ValueError(f"Field '{col}' has unrecognised boolean value: {val!r}")
            return bool(val)
        s = str(val).strip().lower()
        if s in {"true", "1", "yes", "có", "co", "đúng", "dung"}:
            return True
        if s in {"false", "0", "no", "không", "khong", "sai"}:
            return False
        raise ValueError(f"Field '{col}' has unrecognised boolean value: {val!r}")
