from pathlib import Path

import numpy as np
import pandas as pd

from haui_student_prediction.data.cleaning import DataCleaner
from haui_student_prediction.data.column_mapping import (
    get_default_column_mapping,
    is_training_target_column,
)
from haui_student_prediction.data.validator import DataValidator


TARGET_COLUMN = "target_final_exam_score"


def _read_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="cp1258")
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Định dạng tệp không được hỗ trợ: {suffix}")


def load_historical_training_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {file_path}")

    raw = _read_dataset(path)
    target_columns = [
        column for column in raw.columns
        if is_training_target_column(column)
    ]
    if len(target_columns) != 1:
        raise ValueError(
            "Tệp huấn luyện phải có đúng một cột nhãn điểm cuối kỳ "
            "(ví dụ: target_final_exam_score, Điểm thi hoặc Điểm cuối kỳ)."
        )

    source_target = raw[target_columns[0]]
    target = pd.to_numeric(source_target, errors="coerce")
    nonblank = source_target.notna() & source_target.astype(str).str.strip().ne("")
    invalid = nonblank & target.isna()
    if invalid.any():
        rows = [int(index) + 2 for index in source_target.index[invalid][:8]]
        raise ValueError(
            f"Cột nhãn có giá trị không phải số tại các dòng: {rows}"
        )

    non_finite = target.notna() & ~np.isfinite(target.astype(float))
    if non_finite.any():
        rows = [int(index) + 2 for index in source_target.index[non_finite][:8]]
        raise ValueError(
            f"Cột nhãn chứa giá trị không hữu hạn tại các dòng: {rows}"
        )

    feature_data = raw.drop(columns=target_columns)
    cleaner = DataCleaner(get_default_column_mapping())
    feature_data = cleaner.rename_columns(feature_data)
    feature_data = cleaner.convert_types(feature_data)
    conversion_errors = cleaner.collect_conversion_errors()
    if conversion_errors:
        details = "; ".join(
            f"dòng {error['row'] + 2}, cột {error['column']}: {error['value']}"
            for error in conversion_errors[:8]
        )
        raise ValueError(f"Dữ liệu huấn luyện có giá trị không hợp lệ: {details}")

    target_values = target.to_numpy(dtype=float, na_value=np.nan, copy=True)
    target_values[~np.isfinite(target_values)] = np.nan
    feature_data[TARGET_COLUMN] = target_values

    range_errors = DataValidator().validate_numeric_ranges(feature_data)
    if range_errors:
        details = "; ".join(
            f"dòng {error['row'] + 2}, cột {error['column']}: {error['error']}"
            for error in range_errors[:10]
        )
        remainder = len(range_errors) - 10
        suffix = f"; và {remainder} lỗi khác" if remainder > 0 else ""
        raise ValueError(
            "Dữ liệu lịch sử có giá trị ngoài thang cho phép: "
            f"{details}{suffix}"
        )

    valid = np.isfinite(target_values)
    if int(valid.sum()) < 10:
        raise ValueError(
            f"Chỉ có {int(valid.sum())} nhãn điểm hợp lệ. Cần tối thiểu 10 bản ghi lịch sử; "
            "nhiều dữ liệu đa dạng hơn thường giúp đánh giá mô hình đáng tin cậy hơn."
        )

    return feature_data.reset_index(drop=True)
