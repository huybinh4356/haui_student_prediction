import pytest
from haui_student_prediction.scoring.final_course_score import CourseProfile, ScoringService

def test_course_score_no_tx2():
    profiles = {
        "no_tx2": CourseProfile(0.20, 0.0, 0.20, 0.60, False)
    }
    service = ScoringService(profiles)
    score = service.calculate_score("no_tx2", tx1=8.0, midterm=7.0, predicted_final=9.0)
    assert abs(score - 8.4) < 1e-6

def test_course_score_with_tx2():
    profiles = {
        "with_tx2": CourseProfile(0.15, 0.15, 0.20, 0.50, True)
    }
    service = ScoringService(profiles)
    score = service.calculate_score("with_tx2", tx1=8.0, midterm=7.0, predicted_final=9.0, tx2=8.0)
    assert abs(score - 8.3) < 1e-6

def test_missing_tx2_raises_error():
    profiles = {
        "with_tx2": CourseProfile(0.15, 0.15, 0.20, 0.50, True)
    }
    service = ScoringService(profiles)
    with pytest.raises(ValueError):
        service.calculate_score("with_tx2", tx1=8.0, midterm=7.0, predicted_final=9.0)