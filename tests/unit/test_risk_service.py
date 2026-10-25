import pytest
import pandas as pd
import numpy as np
from haui_student_prediction.risk.risk_calculator import RiskCalculator
from haui_student_prediction.services.risk_prediction_service import RiskPredictionService
from haui_student_prediction.services.risk_service import StudentRiskService


@pytest.fixture
def service():
    return RiskPredictionService(RiskCalculator())


def _row(**kwargs):
    defaults = {
        "absence_percentage": 10.0,
        "lms_learning_percentage": 50.0,
        "predicted_course_score": 7.0,
        "interaction_score": 3.0,
        "stress_score": 2.0,
        "motivation_score": 4.0,
        "works_part_time": False,
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


def test_missing_absence_raises(service):
    with pytest.raises(ValueError, match="absence_percentage"):
        service.evaluate_student_risk(_row(absence_percentage=np.nan))


def test_missing_works_part_time_raises(service):
    with pytest.raises(ValueError, match="works_part_time"):
        service.evaluate_student_risk(_row(works_part_time=np.nan))


def test_nan_works_part_time_not_true(service):
    """NaN must not be silently treated as True."""
    with pytest.raises(ValueError):
        service.evaluate_student_risk(_row(works_part_time=np.nan))


def test_works_part_time_string_false(service):
    r = service.evaluate_student_risk(_row(works_part_time="False"))
    assert r.weighted_score >= 0


def test_works_part_time_string_true(service):
    r = service.evaluate_student_risk(_row(works_part_time="True"))
    assert r.weighted_score >= 0


def test_absence_no_override_at_75(service):
    r = service.evaluate_student_risk(_row(absence_percentage=75))
    assert r.is_override is False
    assert r.absence_status == "high"


def test_absence_override_at_76(service):
    r = service.evaluate_student_risk(_row(absence_percentage=76))
    assert r.is_override is True
    assert r.absence_status == "REPEAT_REQUIRED"


def test_absence_no_override_at_30(service):
    r = service.evaluate_student_risk(_row(absence_percentage=30))
    assert r.is_override is False


def test_absence_boundaries(service):
    r0 = service.evaluate_student_risk(_row(absence_percentage=0))
    assert r0.absence_status == "very low"
    r5 = service.evaluate_student_risk(_row(absence_percentage=5))
    assert r5.absence_status == "low"
    r10 = service.evaluate_student_risk(_row(absence_percentage=10))
    assert r10.absence_status == "medium"
    r20 = service.evaluate_student_risk(_row(absence_percentage=20))
    assert r20.absence_status == "high"
    r30 = service.evaluate_student_risk(_row(absence_percentage=30))
    assert r30.absence_status == "high"
    r75 = service.evaluate_student_risk(_row(absence_percentage=75))
    assert r75.absence_status == "high"


def test_lms_boundaries(service):
    r0 = service.evaluate_student_risk(_row(lms_learning_percentage=0))
    r5 = service.evaluate_student_risk(_row(lms_learning_percentage=5))
    r10 = service.evaluate_student_risk(_row(lms_learning_percentage=10))
    r20 = service.evaluate_student_risk(_row(lms_learning_percentage=20))
    r30 = service.evaluate_student_risk(_row(lms_learning_percentage=30))
    r100 = service.evaluate_student_risk(_row(lms_learning_percentage=100))
    assert r0.weighted_score > r5.weighted_score
    assert r5.weighted_score > r10.weighted_score
    assert r10.weighted_score > r20.weighted_score
    assert r20.weighted_score > r30.weighted_score
    assert r30.weighted_score == r100.weighted_score


def test_weights_sum_to_one():
    calc = RiskCalculator()
    total = sum(calc.weights.values())
    assert abs(total - 1.0) < 1e-6


def test_risk_score_in_range(service):
    r = service.evaluate_student_risk(_row())
    assert 0 <= r.weighted_score <= 100



def test_part_time_status_uses_neutral_risk_mapping(service):
    not_working = service.evaluate_student_risk(_row(works_part_time=False))
    working = service.evaluate_student_risk(_row(works_part_time=True))

    assert working.weighted_score == not_working.weighted_score



def test_legacy_risk_service_uses_real_record_inputs():
    service = StudentRiskService()
    result = service.evaluate_student(_row(absence_percentage=76))

    assert result.is_override is True
    assert result.absence_status == "REPEAT_REQUIRED"
