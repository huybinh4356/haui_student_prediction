import pandas as pd
from haui_student_prediction.data.ingestion_service import IngestionService, IngestionResult

class DatasetService:
    def __init__(self, ingestion_service: IngestionService):
        self.ingestion_service = ingestion_service
        self.current_dataset = None
        self.dataset_version = None

    def import_dataset(
        self, file_path: str, required_cols: list[str],
        courses_with_tx2: list[str],
    ) -> IngestionResult:
        result = self.ingestion_service.load_and_validate(
            file_path, required_cols, courses_with_tx2,
        )
        if result.is_success and result.data is not None:
            self.current_dataset = result.data
            self.dataset_version = str(pd.util.hash_pandas_object(result.data).sum())
        return result

    def get_current_data(self) -> pd.DataFrame:
        if self.current_dataset is None:
            raise ValueError("No dataset loaded")
        return self.current_dataset.copy()
