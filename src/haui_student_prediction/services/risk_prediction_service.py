import pandas as pd
from typing import Dict, Any
from haui_student_prediction.risk.risk_calculator import RiskCalculator, RiskResult

class RiskPredictionService:
    def __init__(self, calculator: RiskCalculator):
        self.calculator = calculator

    def evaluate_student_risk(self, student_record: pd.Series) -> RiskResult:
        try:
            absence = float(student_record.get('absence_percentage', 0))
            lms = float(student_record.get('lms_learning_percentage', 0))
            predicted_score = float(student_record.get('predicted_course_score', 0))
            interaction = float(student_record.get('interaction_score', 0))
            stress = float(student_record.get('stress_score', 0))
            motivation = float(student_record.get('motivation_score', 0))
            work_status = bool(student_record.get('works_part_time', False))
            
            work_points = 50.0 if work_status else 0.0

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
                work_risk_points=work_points
            )
        except Exception as e:
            raise ValueError("Invalid student data for risk calculation") from e