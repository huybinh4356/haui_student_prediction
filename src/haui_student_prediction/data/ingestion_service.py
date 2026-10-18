import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from haui_student_prediction.data.cleaning import DataCleaner
from haui_student_prediction.data.validator import DataValidator

@dataclass
class IngestionResult:
    is_success: bool
    data: Optional[pd.DataFrame]
    errors: List[Dict[str, Any]]
    missing_columns: List[str]
    duplicate_indices: List[int]

class IngestionService:
    def __init__(self, cleaner: DataCleaner, validator: DataValidator):
        self.cleaner = cleaner
        self.validator = validator

    def load_and_validate(self, file_path: str, required_cols: List[str], courses_with_tx2: List[str]) -> IngestionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix == '.csv':
            df_raw = pd.read_csv(path)
        elif path.suffix in ['.xls', '.xlsx']:
            df_raw = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file format")

        df_mapped = self.cleaner.rename_columns(df_raw)
        df_typed = self.cleaner.convert_types(df_mapped)
        df_final = self.cleaner.determine_tx2_applicability(df_typed, courses_with_tx2)

        missing_cols = self.validator.validate_required_columns(df_final, required_cols)
        
        percentage_cols = [col for col in ["absence_percentage", "lms_learning_percentage"] if col in df_final.columns]
        perc_errors = self.validator.validate_percentages(df_final, percentage_cols)
        
        duplicate_indices = self.validator.check_duplicates(df_final, ["student_id", "course_id", "semester"])

        is_success = len(missing_cols) == 0 and len(perc_errors) == 0

        return IngestionResult(
            is_success=is_success,
            data=df_final if is_success else None,
            errors=perc_errors,
            missing_columns=missing_cols,
            duplicate_indices=duplicate_indices
        )