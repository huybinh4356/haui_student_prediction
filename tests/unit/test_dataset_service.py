import pandas as pd

from haui_student_prediction.data.ingestion_service import IngestionResult
from haui_student_prediction.services.dataset_service import DatasetService


class FakeIngestionService:
    def __init__(self, result):
        self.result = result

    def load_and_validate(self, file_path, required_cols, courses_with_tx2):
        return self.result


class FakeRepository:
    def __init__(self, save_result):
        self.save_result = save_result

    def save_session_data(self, data):
        return self.save_result


def make_result():
    return IngestionResult(
        is_success=True,
        data=pd.DataFrame({"student_id": ["S001"]}),
        errors=[],
        missing_columns=[],
        duplicate_indices=[],
        warnings=[],
    )


def test_import_reports_sqlite_failure():
    service = DatasetService(FakeIngestionService(make_result()))
    service._repository = FakeRepository(False)

    result = service.import_dataset("students.csv", ["student_id"], [])

    assert not result.is_success
    assert result.data is None
    assert any(error["kind"] == "storage_error" for error in result.errors)


def test_import_keeps_success_when_sqlite_save_succeeds():
    service = DatasetService(FakeIngestionService(make_result()))
    service._repository = FakeRepository(True)

    result = service.import_dataset("students.csv", ["student_id"], [])

    assert result.is_success
    assert result.data is not None
