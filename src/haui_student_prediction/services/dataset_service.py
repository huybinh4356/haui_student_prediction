from haui_student_prediction.data.ingestion_service import IngestionService
from haui_student_prediction.data.validator import ValidatorService
from haui_student_prediction.data.cleaning import CleaningService
from haui_student_prediction.data.column_mapping import get_default_column_mapping
from haui_student_prediction.storage.dataset_repository import DataRepository

class DatasetService:
    @staticmethod
    def process_and_import_file(file_path: str):
        """Pipeline xử lý file và lưu vào bộ đệm phiên làm việc."""
        df_raw = IngestionService.load_file(file_path)
        
        column_map = get_default_column_mapping()
        df_mapped = df_raw.rename(columns=column_map)
        
        df_valid, errors = ValidatorService.validate(df_mapped)
        if df_valid.empty:
            return None, errors
            
        df_cleaned = CleaningService.clean(df_valid)
        
        # Lưu vào SQLite làm phiên tạm thời
        repository = DataRepository()
        repository.save_session_data(df_cleaned)
        
        return df_cleaned, errors