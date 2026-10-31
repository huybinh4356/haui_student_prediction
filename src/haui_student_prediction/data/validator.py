import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple


FIELD_RANGES = {
    "absence_percentage": (0.0, 100.0),
    "lms_learning_percentage": (0.0, 100.0),
    "tx1_score": (0.0, 10.0),
    "tx2_score": (0.0, 10.0),
    "midterm_score": (0.0, 10.0),
    "target_final_exam_score": (0.0, 10.0),
    "interaction_score": (1.0, 5.0),
    "stress_score": (1.0, 5.0),
    "motivation_score": (1.0, 5.0),
}

RISK_REQUIRED_INPUTS = (
    "absence_percentage",
    "lms_learning_percentage",
    "interaction_score",
    "stress_score",
    "motivation_score",
)

FIELD_LABELS = {
    "absence_percentage": "Tỷ lệ nghỉ học (%)",
    "lms_learning_percentage": "Tỷ lệ học tập trên LMS (%)",
    "tx1_score": "Điểm thường xuyên 1",
    "tx2_score": "Điểm thường xuyên 2",
    "midterm_score": "Điểm giữa kỳ",
    "target_final_exam_score": "Điểm thi cuối kỳ thực tế",
    "predicted_course_score": "Điểm học phần dự đoán",
    "interaction_score": "Mức độ tương tác",
    "stress_score": "Mức độ căng thẳng",
    "motivation_score": "Động lực học tập",
}


class DataValidator:
    def validate_percentages(self, df: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
        errors = []
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col]
            invalid_mask = series.notna() & ((series < 0) | (series > 100))
            for idx in series.index[invalid_mask]:
                errors.append({
                    "row": int(idx),
                    "column": col,
                    "value": str(series.loc[idx]),
                    "error": f"Percentage in column '{col}' must be between 0 and 100",
                    "kind": "percentage_out_of_range",
                })
        return errors

    def validate_numeric_ranges(
        self,
        df: pd.DataFrame,
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[Dict[str, Any]]:
        configured_ranges = FIELD_RANGES if ranges is None else ranges
        errors: List[Dict[str, Any]] = []
        for column, (minimum, maximum) in configured_ranges.items():
            if column not in df.columns:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            present = df[column].notna()
            finite = pd.Series(
                np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
                index=df.index,
            )
            invalid = present & (
                ~numeric.notna()
                | ~finite
                | (numeric < minimum)
                | (numeric > maximum)
            )
            for index in df.index[invalid]:
                errors.append(
                    {
                        "row": int(index),
                        "column": column,
                        "value": str(df.at[index, column]),
                        "minimum": minimum,
                        "maximum": maximum,
                        "error": (
                            f"{FIELD_LABELS.get(column, column)} phải nằm trong "
                            f"khoảng {minimum:g}–{maximum:g}."
                        ),
                        "kind": "numeric_out_of_range",
                    }
                )
        return errors

    def validate_required_columns(self, df: pd.DataFrame, required_columns: List[str]) -> List[str]:
        return [col for col in required_columns if col not in df.columns]

    def check_duplicates(self, df: pd.DataFrame, subset: List[str]) -> List[int]:
        if not all(col in df.columns for col in subset):
            return []

        key_data = df[subset]
        complete_mask = key_data.notna().all(axis=1)
        for column in subset:
            if key_data[column].dtype == object:
                complete_mask &= key_data[column].astype(str).str.strip().ne("")

        complete_data = df.loc[complete_mask]
        duplicates = complete_data[complete_data.duplicated(subset=subset, keep=False)]
        return duplicates.index.tolist()

    def validate_score_scale(
        self, df: pd.DataFrame, columns: List[str], max_score: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        errors: List[Dict[str, Any]] = []
        notes: List[Dict[str, Any]] = []
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col]
            present = series.notna()
            if max_score is not None:
                out = present & ((series < 0) | (series > max_score))
                for idx in series.index[out]:
                    errors.append({
                        "row": int(idx),
                        "column": col,
                        "value": str(series.loc[idx]),
                        "error": f"Score in column '{col}' must be between 0 and {max_score}",
                        "kind": "score_out_of_range",
                    })
            else:
                plausible_out = present & ((series < 0) | (series > 10))
                for idx in series.index[plausible_out]:
                    notes.append({
                        "row": int(idx),
                        "column": col,
                        "value": str(series.loc[idx]),
                        "note": (
                            f"Score in column '{col}' is outside the reference 0-10 band "
                            "implied by the risk module; scale needs confirmation"
                        ),
                        "kind": "score_scale_needs_confirmation",
                    })
        return errors, notes
