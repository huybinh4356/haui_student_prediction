from typing import List

from haui_student_prediction.data.ingestion_service import IngestionResult, IngestionService
from haui_student_prediction.storage.dataset_repository import DataRepository


class DatasetService:
    def __init__(self, ingestion_service: IngestionService):
        self.ingestion_service = ingestion_service
        self._repository = DataRepository()

    def import_dataset(
        self,
        file_path: str,
        required_cols: List[str],
        courses_with_tx2: List[str],
    ) -> IngestionResult:
        result = self.ingestion_service.load_and_validate(
            file_path, required_cols, courses_with_tx2
        )
        if not result.is_success or result.data is None:
            return result

        if not self._repository.save_session_data(result.data):
            return IngestionResult(
                is_success=False,
                data=None,
                errors=[
                    *result.errors,
                    {
                        "kind": "storage_error",
                        "error": (
                            "Không thể lưu dataset vào SQLite. "
                            "Kiểm tra quyền ghi và đường dẫn database."
                        ),
                    },
                ],
                missing_columns=result.missing_columns,
                duplicate_indices=result.duplicate_indices,
                warnings=result.warnings,
            )
        return result

    def process_and_import_file(
        self,
        file_path: str,
        required_cols: List[str],
        courses_with_tx2: List[str],
    ) -> IngestionResult:
        return self.import_dataset(file_path, required_cols, courses_with_tx2)

    def get_session_data(self):
        return self._repository.get_session_data()
