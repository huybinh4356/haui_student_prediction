import pandas as pd

from haui_student_prediction.risk.risk_calculator import RiskCalculator
from haui_student_prediction.services.risk_prediction_service import RiskPredictionService


class StudentRiskService(RiskPredictionService):
    def __init__(self, calculator: RiskCalculator | None = None):
        super().__init__(calculator or RiskCalculator())

    def evaluate_student(self, student_record: pd.Series):
        return self.evaluate_student_risk(student_record)
