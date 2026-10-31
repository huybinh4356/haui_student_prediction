import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from haui_student_prediction.data.cleaning import DataCleaner, PERCENTAGE_COLUMNS
from haui_student_prediction.data.column_mapping import is_training_target_column
from haui_student_prediction.data.validator import (
    FIELD_LABELS,
    RISK_REQUIRED_INPUTS,
    DataValidator,
)

@dataclass
class IngestionResult:
    is_success: bool
    data: Optional[pd.DataFrame]
    errors: List[Dict[str, Any]]                # hard errors (block success)
    missing_columns: List[str]
    duplicate_indices: List[int]
    warnings: List[Dict[str, Any]] = field(
        default_factory=list,
    )  # non-blocking notes

class IngestionService:
    """Policy summary (requirement #5 & #6):

    * DUPLICATES: a duplicate on (student_id, course_id, semester) is a blocking
      error -> is_success becomes False. Data is NOT silently dropped and is NOT
      returned; the caller (UI) must surface duplicate_indices to the user so
      they can fix the source file. We never auto-delete rows.
    * MISSING VALUES: empty cells are legitimate missing values (NaN) and do not
      block success. Uncoercible numeric / boolean tokens ARE blocking errors.
    * PERCENTAGES out of [0,100]: blocking error.
    * SCORE SCALE: no hard threshold (no explicit config); flagged as a warning
      note for operator confirmation (requirement #4).
    * is_success is True only when: required columns present AND no percentage
      errors AND no conversion errors AND no duplicates.
    """
    def __init__(self, cleaner: DataCleaner, validator: DataValidator):
        self.cleaner = cleaner
        self.validator = validator

    def load_and_validate(
        self, file_path: str, required_cols: List[str],
        courses_with_tx2: List[str],
    ) -> IngestionResult:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == '.csv':
            try:
                df_raw = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df_raw = pd.read_csv(path, encoding="cp1258")
        elif suffix in ['.xls', '.xlsx']:
            df_raw = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: '{suffix}'. Supported: .csv, .xls, .xlsx")

        excluded_target_columns = [
            column for column in df_raw.columns
            if is_training_target_column(column)
        ]
        if excluded_target_columns:
            df_raw = df_raw.drop(columns=excluded_target_columns)

        df_mapped = self.cleaner.rename_columns(df_raw)
        df_typed = self.cleaner.convert_types(df_mapped)
        df_final = self.cleaner.determine_tx2_applicability(df_typed, courses_with_tx2)

        # --- collect all error/warning categories ---
        missing_cols = self.validator.validate_required_columns(df_final, required_cols)

        percentage_cols = [c for c in PERCENTAGE_COLUMNS if c in df_final.columns]
        perc_errors = self.validator.validate_percentages(df_final, percentage_cols)

        conversion_errors = self.cleaner.collect_conversion_errors()
        range_errors = self.validator.validate_numeric_ranges(df_final)

        risk_data_warnings: List[Dict[str, Any]] = []
        for column in RISK_REQUIRED_INPUTS:
            if column not in df_final.columns:
                missing_rows = len(df_final)
                missing_column = True
            else:
                missing_rows = int(df_final[column].isna().sum())
                missing_column = False
            if missing_rows:
                risk_data_warnings.append(
                    {
                        "column": column,
                        "label": FIELD_LABELS.get(column, column),
                        "missing_rows": missing_rows,
                        "total_rows": len(df_final),
                        "missing_column": missing_column,
                        "kind": "risk_data_incomplete",
                        "note": (
                            "Thiếu cột đầu vào cần thiết để đánh giá nguy cơ."
                            if missing_column
                            else "Có bản ghi thiếu giá trị đầu vào để đánh giá nguy cơ."
                        ),
                    }
                )

        duplicate_key_columns = ["student_id", "course_id", "semester"]
        unavailable_duplicate_keys = [
            column for column in duplicate_key_columns if column not in df_final.columns
        ]
        incomplete_duplicate_key_rows = 0
        if not unavailable_duplicate_keys:
            key_data = df_final[duplicate_key_columns]
            complete_key_mask = key_data.notna().all(axis=1)
            for column in duplicate_key_columns:
                complete_key_mask &= key_data[column].astype(str).str.strip().ne("")
            incomplete_duplicate_key_rows = int((~complete_key_mask).sum())
        duplicate_indices = self.validator.check_duplicates(
            df_final, duplicate_key_columns
        )

        all_hard_errors: List[Dict[str, Any]] = []
        all_hard_errors.extend(perc_errors)
        all_hard_errors.extend(conversion_errors)
        all_hard_errors.extend(range_errors)
        # Duplicates are blocking: represent them as explicit error entries too.
        if duplicate_indices:
            all_hard_errors.append({
                "column": "student_id,course_id,semester",
                "rows": [int(i) for i in duplicate_indices],
                "count": len(duplicate_indices),
                "error": (
                    f"{len(duplicate_indices)} duplicate record(s) on "
                    "(student_id, course_id, semester)"
                ),
                "kind": "duplicate",
            })

        # is_success: False if ANY blocking condition is true.
        is_success = (
            len(missing_cols) == 0
            and len(perc_errors) == 0
            and len(conversion_errors) == 0
            and len(range_errors) == 0
            and len(duplicate_indices) == 0
        )

        return IngestionResult(
            is_success=is_success,
            data=df_final if is_success else None,
            errors=all_hard_errors,
            missing_columns=missing_cols,
            duplicate_indices=[int(i) for i in duplicate_indices],
            warnings=[
                *risk_data_warnings,
                *(
                    [{
                        "column": ",".join(unavailable_duplicate_keys),
                        "note": (
                            "Không thể kiểm tra trùng theo khóa đầy đủ "
                            "(student_id, course_id, semester) vì file thiếu cột khóa. "
                            "Bổ sung mã môn và học kỳ để tăng khả năng phát hiện bản ghi trùng."
                        ),
                        "kind": "duplicate_check_incomplete",
                    }]
                    if unavailable_duplicate_keys else []
                ),
                *(
                    [{
                        "column": ",".join(duplicate_key_columns),
                        "rows_with_missing_key": incomplete_duplicate_key_rows,
                        "note": (
                            f"{incomplete_duplicate_key_rows} dòng thiếu ít nhất một giá trị khóa "
                            "trùng (student_id, course_id, semester), vì vậy các dòng này không "
                            "được dùng để kết luận trùng."
                        ),
                        "kind": "duplicate_check_incomplete_values",
                    }]
                    if incomplete_duplicate_key_rows else []
                ),
                *[
                    {
                        "column": str(column),
                        "note": "Nhãn điểm lịch sử đã được loại khỏi dữ liệu sinh viên dùng để dự đoán.",
                        "kind": "training_target_excluded",
                    }
                    for column in excluded_target_columns
                ],
            ],
        )
