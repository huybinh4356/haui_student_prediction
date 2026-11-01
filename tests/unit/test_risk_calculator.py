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


def test_absence_override_only_above_30_percent():
    calc = RiskCalculator()

    at_threshold = _calculate(calc, 30.0)
    just_above_threshold = _calculate(calc, 30.01)

    assert at_threshold.is_override is False
    assert at_threshold.absence_status == "high"
    assert just_above_threshold.is_override is True
    assert just_above_threshold.absence_status == "REPEAT_REQUIRED"


def test_weights_sum_to_one():
    calc = RiskCalculator()
    assert abs(sum(calc.weights.values()) - 1.0) < 1e-6


def test_invalid_percentage():
    calc = RiskCalculator()
    with pytest.raises(ValueError):
        _calculate(calc, 105.0)



def test_non_finite_component_is_rejected():
    calc = RiskCalculator()
    with pytest.raises(ValueError, match="finite number"):
        calc.calculate_risk(
            absence_pct=10.0,
            lms_pct=50.0,
            predicted_course_score=float("nan"),
            interaction=3.0,
            interaction_max=5.0,
            stress=2.0,
            stress_max=5.0,
            motivation=4.0,
            motivation_max=5.0,
        )


def test_zero_survey_scale_maximum_is_rejected():
    calc = RiskCalculator()
    with pytest.raises(ValueError, match="greater than zero"):
        calc.calculate_risk(
            absence_pct=10.0,
            lms_pct=50.0,
            predicted_course_score=7.0,
            interaction=3.0,
            interaction_max=0.0,
            stress=2.0,
            stress_max=5.0,
            motivation=4.0,
            motivation_max=5.0,
        )


def test_invalid_risk_weights_are_rejected():
    calc = RiskCalculator()
    calc.weights["absence"] = float("nan")
    with pytest.raises(ValueError, match="finite and non-negative"):
        _calculate(calc, 10.0)
