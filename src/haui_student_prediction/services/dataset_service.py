from typing import List
from haui_student_prediction.data.ingestion_service import IngestionService, IngestionResult
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
        if result.is_success and result.data is not None:
            self._repository.save_session_data(result.data)
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