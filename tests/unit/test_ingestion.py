"""Unit tests for the data ingestion pipeline (Giai đoạn 2).

Covers: file-not-found, unsupported format, missing columns, percentage out
of range, uncoercible numbers, boolean string parsing (incl. "False" != True),
duplicates, valid file, and TX2 determination.
"""
import pytest
import pandas as pd
from pathlib import Path

from haui_student_prediction.data.cleaning import DataCleaner
from haui_student_prediction.data.validator import DataValidator
from haui_student_prediction.data.ingestion_service import IngestionService
from haui_student_prediction.data.column_mapping import get_default_column_mapping


# ---- fixtures ---------------------------------------------------------------

@pytest.fixture
def cleaner():
    return DataCleaner(get_default_column_mapping())


@pytest.fixture
def validator():
    return DataValidator()


@pytest.fixture
def service(cleaner, validator):
    return IngestionService(cleaner, validator)


REQUIRED_COLS = ["student_id", "absence_percentage", "tx1_score", "midterm_score"]
COURSES_TX2 = ["IT101", "IT102"]


def _write_csv(tmp_path: Path, name: str, rows: list[dict]) -> str:
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
    return str(p)


def _valid_rows():
    # Two valid rows; course IT101 has TX2.
    return [
        {
            "Mã SV": "SV01", "Học kỳ": "2024.1", "Mã môn": "IT101",
            "Tỷ lệ nghỉ": "10", "Tỷ lệ học LMS": "80",
            "Điểm TX1": "8.0", "Điểm TX2": "7.5", "Điểm GK": "7.0",
            "Đi làm thêm": "False",
        },
        {
            "Mã SV": "SV02", "Học kỳ": "2024.1", "Mã môn": "IT999",
            "Tỷ lệ nghỉ": "5", "Tỷ lệ học LMS": "90",
            "Điểm TX1": "9.0", "Điểm GK": "8.0",
            "Đi làm thêm": "yes",
        },
    ]


# ---- 1. file not found ------------------------------------------------------

def test_file_not_found_raises(service, tmp_path):
    with pytest.raises(FileNotFoundError):
        service.load_and_validate(str(tmp_path / "nope.csv"), REQUIRED_COLS, COURSES_TX2)


# ---- 2. unsupported format --------------------------------------------------

def test_unsupported_format_raises(service, tmp_path):
    p = tmp_path / "data.txt"
    p.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        service.load_and_validate(str(p), REQUIRED_COLS, COURSES_TX2)


# ---- 3. missing required columns --------------------------------------------

def test_missing_required_columns(service, tmp_path):
    # Only student_id present -> missing absence_percentage, tx1_score, midterm_score
    rows = [{"Mã SV": "SV01", "Học kỳ": "2024.1", "Mã môn": "IT101"}]
    path = _write_csv(tmp_path, "missing.csv", rows)
    res = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)
    assert not res.is_success
    assert "absence_percentage" in res.missing_columns
    assert "tx1_score" in res.missing_columns
    assert "midterm_score" in res.missing_columns
    assert res.data is None


# ---- 4. percentage out of range --------------------------------------------

def test_percentage_out_of_range(service, tmp_path):
    rows = _valid_rows()
    rows[0]["Tỷ lệ nghỉ"] = "150"   # > 100
    path = _write_csv(tmp_path, "pct.csv", rows)
    res = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)
    assert not res.is_success
    pct_errors = [e for e in res.errors if e.get("kind") == "percentage_out_of_range"]
    assert len(pct_errors) == 1
    assert pct_errors[0]["column"] == "absence_percentage"
    assert pct_errors[0]["row"] == 0


def test_percentage_negative(service, tmp_path):
    rows = _valid_rows()
    rows[1]["Tỷ lệ học LMS"] = "-5"
    path = _write_csv(tmp_path, "negpct.csv", rows)
    res = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)
    assert not res.is_success
    neg = [e for e in res.errors if e.get("kind") == "percentage_out_of_range"]
    assert len(neg) == 1
    assert neg[0]["column"] == "lms_learning_percentage"


# ---- 5. non-numeric value (uncoercible) -------------------------------------

def test_non_numeric_value_blocks(service, tmp_path):
    rows = _valid_rows()
    rows[0]["Điểm TX1"] = "abc"   # not a number
    path = _write_csv(tmp_path, "badnum.csv", rows)
    res = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)
    assert not res.is_success
    conv = [e for e in res.errors if e.get("kind") == "numeric_conversion"]
    assert len(conv) == 1
    assert conv[0]["column"] == "tx1_score"
    assert conv[0]["row"] == 0


def test_infinite_numeric_value_blocks(service, tmp_path):
    rows = _valid_rows()
    rows[0]["Điểm TX1"] = "inf"
    path = _write_csv(tmp_path, "infinite.csv", rows)

    result = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)

    assert not result.is_success
    errors = [error for error in result.errors if error.get("kind") == "numeric_conversion"]
    assert any(error["column"] == "tx1_score" and error["row"] == 0 for error in errors)


def test_empty_cell_is_legitimate_missing(service, tmp_path):
    """An empty cell must become NaN, NOT 0, and must NOT block success."""
    rows = _valid_rows()
    rows[1]["Điểm TX1"] = ""   # empty
    path = _write_csv(tmp_path, "empty.csv", rows)
    res = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)
    assert res.is_success
    assert res.data is not None
    val = res.data.loc[res.data["student_id"] == "SV02", "tx1_score"].iloc[0]
    assert pd.isna(val), "empty cell must remain NaN, not become 0"


# ---- 6. boolean string parsing ---------------------------------------------

def test_boolean_false_string_not_true(cleaner):
    """Requirement #3: the string 'False' must NOT be interpreted as True."""
    df = pd.DataFrame({
        "works_part_time": [
            "False", "True", "0", "1", "yes", "no", "không", "có", "xyz", "",
        ],
    })
    out = cleaner.convert_types(df)
    vals = out["works_part_time"].tolist()
    assert bool(vals[0]) is False   # "False"
    assert bool(vals[1]) is True    # "True"
    assert bool(vals[2]) is False   # "0"
    assert bool(vals[3]) is True    # "1"
    assert bool(vals[4]) is True    # "yes"
    assert bool(vals[5]) is False   # "no"
    assert bool(vals[6]) is False   # "không"
    assert bool(vals[7]) is True    # "có"
    assert pd.isna(vals[8])         # "xyz" unrecognised -> NaN + error
    assert pd.isna(vals[9])         # "" -> legitimate missing


def test_boolean_unrecognised_recorded(cleaner):
    df = pd.DataFrame({"works_part_time": ["maybe"]})
    out = cleaner.convert_types(df)
    assert pd.isna(out["works_part_time"].iloc[0])
    errs = cleaner.collect_conversion_errors()
    assert len(errs) == 1
    assert errs[0]["kind"] == "boolean_conversion"
    assert errs[0]["row"] == 0


def test_boolean_blocks_ingestion(service, tmp_path):
    rows = _valid_rows()
    rows[0]["Đi làm thêm"] = "maybe"
    path = _write_csv(tmp_path, "badbool.csv", rows)
    res = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)
    assert not res.is_success
    berrs = [e for e in res.errors if e.get("kind") == "boolean_conversion"]
    assert len(berrs) == 1


# ---- 7. duplicates ----------------------------------------------------------

def test_duplicates_block_success(service, tmp_path):
    rows = _valid_rows()
    # Make row 2 a duplicate of row 1 on (student_id, course_id, semester)
    rows[1]["Mã SV"] = "SV01"
    rows[1]["Mã môn"] = "IT101"
    rows[1]["Học kỳ"] = "2024.1"
    path = _write_csv(tmp_path, "dup.csv", rows)
    res = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)
    assert not res.is_success, "duplicates must block success"
    assert len(res.duplicate_indices) == 2
    assert 0 in res.duplicate_indices
    assert 1 in res.duplicate_indices
    assert res.data is None, "duplicate data must not be silently returned"
    dup_errs = [e for e in res.errors if e.get("kind") == "duplicate"]
    assert len(dup_errs) == 1


# ---- 8. valid file ----------------------------------------------------------

def test_valid_file_succeeds(service, tmp_path):
    rows = _valid_rows()
    path = _write_csv(tmp_path, "valid.csv", rows)
    res = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)
    assert res.is_success
    assert res.data is not None
    assert len(res.data) == 2
    assert res.data["student_id"].tolist() == ["SV01", "SV02"]
    # "False" string must survive as False boolean, not True
    assert bool(res.data["works_part_time"].iloc[0]) is False
    # "yes" -> True
    assert bool(res.data["works_part_time"].iloc[1]) is True


# ---- 9. TX2 determination ---------------------------------------------------

def test_tx2_determination(service, tmp_path):
    rows = _valid_rows()
    path = _write_csv(tmp_path, "tx2.csv", rows)
    res = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)
    assert res.is_success
    df = res.data
    # SV01 course IT101 -> has_tx2 True; SV02 IT999 -> False
    r0 = df[df["student_id"] == "SV01"].iloc[0]
    r1 = df[df["student_id"] == "SV02"].iloc[0]
    assert bool(r0["has_tx2"]) is True
    assert bool(r1["has_tx2"]) is False


# ---- 10. result consistency -------------------------------------------------

def test_result_consistency_no_success_returns_none_data(service, tmp_path):
    """When blocked, data must be None even though errors list is populated."""
    rows = _valid_rows()
    rows[0]["Tỷ lệ nghỉ"] = "200"
    path = _write_csv(tmp_path, "consistency.csv", rows)
    res = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)
    assert not res.is_success
    assert res.data is None
    assert len(res.errors) > 0
    # duplicate_indices present (possibly empty) and typed list[int]
    assert isinstance(res.duplicate_indices, list)
    # missing_columns present and typed list[str]
    assert isinstance(res.missing_columns, list)



def test_historical_target_column_is_removed_from_prediction_import(service, tmp_path):
    rows = _valid_rows()
    rows[0]["Điểm thi"] = "9.0"
    path = _write_csv(tmp_path, "prediction_input.csv", rows)

    result = service.load_and_validate(path, REQUIRED_COLS, COURSES_TX2)

    assert result.is_success
    assert result.data is not None
    assert "Điểm thi" not in result.data.columns
    assert "target_final_exam_score" not in result.data.columns
    assert any(
        warning.get("kind") == "training_target_excluded"
        for warning in result.warnings
    )
