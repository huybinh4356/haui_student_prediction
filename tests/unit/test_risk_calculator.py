import pytest

from haui_student_prediction.risk.risk_calculator import RiskCalculator


def _calculate(calc, absence):
    return calc.calculate_risk(
        absence_pct=absence,
        lms_pct=100.0,
        predicted_course_score=10.0,
        interaction=5.0,
        interaction_max=5.0,
        stress=0.0,
        stress_max=5.0,
        motivation=5.0,
        motivation_max=5.0,
        work_risk_points=50.0,
    )


def test_absence_override_only_above_75_percent():
    calc = RiskCalculator()

    at_threshold = _calculate(calc, 75.0)
    above_threshold = _calculate(calc, 76.0)

    assert at_threshold.is_override is False
    assert at_threshold.absence_status == "high"
    assert above_threshold.is_override is True
    assert above_threshold.absence_status == "REPEAT_REQUIRED"


def test_weights_sum_to_one():
    calc = RiskCalculator()
    assert abs(sum(calc.weights.values()) - 1.0) < 1e-6


def test_invalid_percentage():
    calc = RiskCalculator()
    with pytest.raises(ValueError):
        _calculate(calc, 105.0)
