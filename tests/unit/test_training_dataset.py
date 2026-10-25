import pandas as pd
import pytest

from haui_student_prediction.data.column_mapping import is_training_target_column
from haui_student_prediction.data.training_dataset import (
    TARGET_COLUMN,
    load_historical_training_dataset,
)


def test_load_historical_training_dataset_uses_label_only_for_training(tmp_path):
    rows = []
    for index in range(12):
        rows.append(
            {
                "Mã SV": f"SV{index + 1:03d}",
                "Tỷ lệ nghỉ": index * 2,
                "Tỷ lệ học LMS": 100 - index * 3,
                "Điểm TX1": 5 + index * 0.2,
                "Điểm GK": 4.5 + index * 0.25,
                "Điểm thi": 4 + index * 0.3,
                "Điểm tương tác": 2 + index % 4,
                "Mức độ stress": 1 + index % 5,
                "Động lực học": 3 + index % 3,
                "Đi làm thêm": index % 2 == 0,
            }
        )
    source = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(source, index=False, encoding="utf-8-sig")

    loaded = load_historical_training_dataset(str(source))

    assert TARGET_COLUMN in loaded.columns
    assert "Điểm thi" not in loaded.columns
    assert loaded[TARGET_COLUMN].notna().all()
    assert len(loaded) == 12
    assert "tx1_score" in loaded.columns
    assert "midterm_score" in loaded.columns


def test_training_dataset_requires_one_historical_label(tmp_path):
    source = tmp_path / "unlabeled.csv"
    pd.DataFrame(
        {
            "Mã SV": ["SV001", "SV002"],
            "Điểm TX1": [7.0, 8.0],
            "Điểm GK": [6.0, 9.0],
        }
    ).to_csv(source, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="đúng một cột nhãn"):
        load_historical_training_dataset(str(source))



@pytest.mark.parametrize(
    "column_name",
    [
        "Điểm thi cuối kỳ thực tế",
        "Điểm thi thực tế",
        "Điểm CK thực tế",
        "actual_final_exam_score",
        "target_final_exam_score",
    ],
)
def test_historical_target_aliases_are_recognized(column_name):
    assert is_training_target_column(column_name)


@pytest.mark.parametrize(
    "column_name",
    [
        "predicted_final_exam_score",
        "Điểm thi dự kiến",
        "Điểm cuối kỳ dự đoán",
        "estimated_final_exam_score",
    ],
)
def test_predicted_score_columns_are_not_historical_targets(column_name):
    assert not is_training_target_column(column_name)


def test_training_dataset_rejects_infinite_target_values(tmp_path):
    rows = []
    for index in range(12):
        rows.append(
            {
                "Mã SV": f"SV{index + 1:03d}",
                "Điểm TX1": 7.0,
                "Điểm GK": 7.5,
                "Điểm thi": float("inf") if index == 0 else 8.0,
            }
        )
    source = tmp_path / "history_with_infinite_target.csv"
    pd.DataFrame(rows).to_csv(source, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="hữu hạn"):
        load_historical_training_dataset(str(source))


@pytest.mark.parametrize(
    "column_name",
    [
        "Điểm thi giữa kỳ",
        "Điểm thi học phần",
        "Điểm thi dự báo",
        "Estimated final exam score",
    ],
)
def test_non_final_exam_columns_are_not_historical_targets(column_name):
    assert not is_training_target_column(column_name)
