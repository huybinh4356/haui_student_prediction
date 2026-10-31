from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from haui_student_prediction.data import uci_student_performance as uci_data
from haui_student_prediction.data.uci_student_performance import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    load_uci_student_performance,
)
from haui_student_prediction.ml.uci_model_comparison import compare_uci_models


def _make_uci_source(rows=24):
    data = []
    for index in range(rows):
        data.append(
            {
                "G1": 10 + index % 8,
                "G2": 11 + index % 8,
                "G3": 12 + index % 8,
                "studytime": 1 + index % 4,
                "failures": index % 3,
                "absences": index,
                "schoolsup": "yes" if index % 2 else "no",
                "famsup": "yes" if index % 3 else "no",
                "paid": "no" if index % 2 else "yes",
                "higher": "yes",
                "internet": "yes" if index % 4 else "no",
                "sex": "F" if index % 2 else "M",
                "age": 15 + index % 5,
                "address": "U" if index % 2 else "R",
            }
        )
    return pd.DataFrame(data)


def test_uci_loader_selects_only_documented_features_and_target(tmp_path):
    source_path = tmp_path / "student-por.csv"
    _make_uci_source().to_csv(source_path, index=False, sep=";", encoding="utf-8")

    result = load_uci_student_performance(
        subject="por",
        data_file=source_path,
    )

    expected = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    assert result.columns.tolist() == expected
    assert "sex" not in result.columns
    assert "age" not in result.columns
    assert "address" not in result.columns
    assert len(result) == 24


def test_uci_loader_normalizes_grade_scale_without_changing_absence_count(tmp_path):
    source_path = tmp_path / "student-por.csv"
    _make_uci_source().to_csv(source_path, index=False, sep=";", encoding="utf-8")

    result = load_uci_student_performance(
        subject="por",
        data_file=source_path,
    )

    assert result.loc[0, "period1_grade"] == 5.0
    assert result.loc[0, "period2_grade"] == 5.5
    assert result.loc[0, TARGET_COLUMN] == 6.0
    assert result.loc[10, "absence_count"] == 10


def test_uci_loader_rejects_unknown_subject():
    with pytest.raises(ValueError, match="subject"):
        load_uci_student_performance(subject="history")


def test_uci_loader_requires_target_and_feature_columns(tmp_path):
    source_path = tmp_path / "incomplete.csv"
    _make_uci_source().drop(columns=["G3"]).to_csv(
        source_path,
        index=False,
        sep=";",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="G3"):
        load_uci_student_performance(
            subject="por",
            data_file=source_path,
        )


def test_uci_loader_rejects_out_of_range_grade(tmp_path):
    source = _make_uci_source()
    source.loc[0, "G3"] = 21
    source_path = tmp_path / "invalid_grade.csv"
    source.to_csv(source_path, index=False, sep=";", encoding="utf-8")

    with pytest.raises(ValueError, match="thang điểm 0–20"):
        load_uci_student_performance(
            subject="por",
            data_file=source_path,
        )



def test_uci_model_comparison_uses_cross_validation_inside_training_partition(tmp_path):
    source_path = tmp_path / "student-por.csv"
    _make_uci_source(rows=60).to_csv(
        source_path,
        index=False,
        sep=";",
        encoding="utf-8",
    )
    dataframe = load_uci_student_performance(
        subject="por",
        data_file=source_path,
    )

    metrics, predictions = compare_uci_models(
        dataframe,
        test_size=0.2,
        random_state=42,
        cv_folds=5,
    )

    assert set(metrics["model"]) == {"XGBoost", "Random Forest", "CatBoost"}
    assert (metrics["cv_folds"] == 5).all()
    assert (metrics["train_rows"] == 48).all()
    assert (metrics["test_rows"] == 12).all()
    assert len(predictions) == 12
    assert {
        "cv_mae_mean",
        "cv_mae_std",
        "cv_rmse_mean",
        "cv_rmse_std",
        "cv_r2_mean",
        "cv_r2_std",
        "mae",
        "rmse",
        "r2",
    }.issubset(metrics.columns)
    assert metrics["cv_mae_mean"].notna().all()


def test_uci_model_comparison_rejects_invalid_cv_folds(tmp_path):
    source_path = tmp_path / "student-por.csv"
    _make_uci_source(rows=30).to_csv(
        source_path,
        index=False,
        sep=";",
        encoding="utf-8",
    )
    dataframe = load_uci_student_performance(
        subject="por",
        data_file=source_path,
    )

    with pytest.raises(ValueError, match="cv_folds"):
        compare_uci_models(dataframe, cv_folds=1)



class _FakeUciResponse(BytesIO):
    headers = {"Content-Type": "application/zip"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def geturl(self):
        return uci_data.UCI_DOWNLOAD_URL


def _make_uci_zip():
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("student/student-por.csv", "G1;G2;G3")
        archive.writestr("student/student-mat.csv", "G1;G2;G3")
    return buffer.getvalue()


def test_uci_download_uses_encoded_url_and_replaces_invalid_cached_archive(
    tmp_path,
    monkeypatch,
):
    raw_dir = tmp_path / "uci_cache"
    raw_dir.mkdir()
    cached_archive = raw_dir / "student+performance.zip"
    cached_archive.write_bytes(b"not a zip archive")
    downloaded_bytes = _make_uci_zip()

    def fake_urlopen(request, timeout):
        assert "%2B" in request.full_url
        assert timeout == 45
        return _FakeUciResponse(downloaded_bytes)

    monkeypatch.setattr(uci_data, "urlopen", fake_urlopen)

    archive_path = uci_data._download_archive(raw_dir)

    assert archive_path == cached_archive
    assert ZipFile(archive_path).namelist() == [
        "student/student-por.csv",
        "student/student-mat.csv",
    ]


def test_uci_download_reports_non_zip_server_response(tmp_path, monkeypatch):
    class HtmlResponse(_FakeUciResponse):
        headers = {"Content-Type": "text/html; charset=utf-8"}

    def fake_urlopen(request, timeout):
        return HtmlResponse(b"<html>Access denied</html>")

    monkeypatch.setattr(uci_data, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Content-Type 'text/html"):
        uci_data._download_archive(tmp_path)
