import sqlite3

import pandas as pd

from haui_student_prediction.storage.dataset_repository import DataRepository


def test_save_and_read_session_data(tmp_path):
    db_path = tmp_path / "session.db"
    repository = DataRepository(db_path)
    source = pd.DataFrame({"student_id": ["S001", "S002"], "tx1_score": [7.5, 8.0]})

    assert repository.save_session_data(source)

    actual = repository.get_session_data()
    pd.testing.assert_frame_equal(actual, source)


def test_read_missing_database_returns_empty_dataframe(tmp_path):
    repository = DataRepository(tmp_path / "missing.db")

    actual = repository.get_session_data()

    assert actual.empty


def test_read_missing_table_returns_empty_dataframe(tmp_path):
    repository = DataRepository(tmp_path / "session.db")
    repository.save_session_data(pd.DataFrame({"student_id": ["S001"]}))

    actual = repository.get_session_data("not_created")

    assert actual.empty


def test_empty_dataset_is_not_saved(tmp_path):
    repository = DataRepository(tmp_path / "session.db")

    assert not repository.save_session_data(pd.DataFrame())
    assert not (tmp_path / "session.db").exists()


def test_invalid_table_name_is_rejected(tmp_path):
    repository = DataRepository(tmp_path / "session.db")
    source = pd.DataFrame({"student_id": ["S001"]})

    assert not repository.save_session_data(source, "current_session; DROP TABLE students")


def test_import_replaces_previous_session(tmp_path):
    repository = DataRepository(tmp_path / "session.db")
    repository.save_session_data(pd.DataFrame({"student_id": ["S001"]}))
    repository.save_session_data(pd.DataFrame({"student_id": ["S002"]}))

    actual = repository.get_session_data()

    assert actual["student_id"].tolist() == ["S002"]

def test_default_database_path_uses_current_project_root(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_BASE_DIR", str(tmp_path))

    repository = DataRepository()

    assert repository.db_path == (tmp_path / "data" / "session_student.db").resolve()

