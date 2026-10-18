import pytest
from haui_student_prediction.risk.risk_calculator import RiskCalculator

def test_absence_override():
    calc = RiskCalculator()
    res = calc.calculate_risk(31.0, 100.0, 10.0, 10.0, 10.0, 0.0, 10.0, 10.0, 10.0, 0.0)
    assert res.is_override is True
    assert res.absence_status == "REPEAT_REQUIRED"

def test_weights_sum_to_one():
    calc = RiskCalculator()
    total = sum(calc.weights.values())
    assert abs(total - 1.0) < 1e-6

def test_invalid_percentage():
    calc = RiskCalculator()
    with pytest.raises(ValueError):
         calc.calculate_risk(105.0, 100.0, 10.0, 10.0, 10.0, 0.0, 10.0, 10.0, 10.0, 0.0)
