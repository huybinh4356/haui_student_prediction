import pandas as pd
import pytest

from haui_student_prediction.ml.data_splitting import split_train_test


def test_repeated_students_are_kept_in_only_one_split():
    dataframe = pd.DataFrame(
        {
            "student_id": [f"SV{index:03d}" for index in range(12) for _ in range(3)],
            "target_final_exam_score": [float(index % 10) for index in range(36)],
            "tx1_score": [5.0 + index % 5 for index in range(36)],
        }
    )

    train_data, test_data = split_train_test(dataframe, test_size=0.2, random_state=42)

    train_students = set(train_data["student_id"])
    test_students = set(test_data["student_id"])
    assert train_students.isdisjoint(test_students)
    assert len(train_data) >= 3
    assert len(test_data) >= 2


def test_repeated_students_with_missing_ids_are_rejected():
    dataframe = pd.DataFrame(
        {
            "student_id": ["SV001", "SV001", "SV002", "SV003", None, "SV004"],
            "target_final_exam_score": [7.0, 8.0, 6.0, 9.0, 5.0, 7.5],
        }
    )

    with pytest.raises(ValueError, match="thiếu student_id"):
        split_train_test(dataframe, test_size=0.2, random_state=42)


def test_single_repeated_student_cannot_be_split_independently():
    dataframe = pd.DataFrame(
        {
            "student_id": ["SV001"] * 10,
            "target_final_exam_score": [float(index) for index in range(10)],
        }
    )

    with pytest.raises(ValueError, match="một student_id"):
        split_train_test(dataframe, test_size=0.2, random_state=42)


def test_unique_students_use_regular_train_test_split():
    dataframe = pd.DataFrame(
        {
            "student_id": [f"SV{index:03d}" for index in range(20)],
            "target_final_exam_score": [float(index % 10) for index in range(20)],
        }
    )

    train_data, test_data = split_train_test(dataframe, test_size=0.2, random_state=42)

    assert len(train_data) == 16
    assert len(test_data) == 4

def test_student_id_whitespace_and_case_variants_cannot_leak_between_splits():
    dataframe = pd.DataFrame(
        {
            "student_id": ["SV001", " sv001 ", "sv002", "SV002", "SV003", "sv003"] * 4,
            "target_final_exam_score": [float(index % 10) for index in range(24)],
        }
    )

    train_data, test_data = split_train_test(dataframe, test_size=0.2, random_state=42)

    normalize = lambda values: {str(value).strip().casefold() for value in values}
    assert normalize(train_data["student_id"]).isdisjoint(normalize(test_data["student_id"]))

