import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Canonical numeric columns that must be coerced to float.
# NOTE on score scales (requirement #4):
#   - absence_percentage / lms_learning_percentage are percentages -> [0, 100].
#   - tx1_score / tx2_score / midterm_score / target_final_exam_score are course
#     scores. The risk module (_normalize_academic in risk_calculator.py) computes
#     (10.0 - score) * 10, which implies a 0-10 scale, but there is NO explicit
#     configuration file declaring this scale. We therefore DO NOT enforce an
#     arbitrary score range here. Range validation for scores is intentionally left
#     as a "needs confirmation" note (see validator.validate_score_scale) rather
#     than hard-failing on an invented threshold.
NUMERIC_COLUMNS = [
    "absence_percentage",
    "lms_learning_percentage",
    "tx1_score",
    "tx2_score",
    "midterm_score",
    "target_final_exam_score",
    "stress_score",
    "motivation_score",
    "interaction_score",
]

PERCENTAGE_COLUMNS = ["absence_percentage", "lms_learning_percentage"]

# Truthy / falsy string tokens recognised for boolean parsing.
# Vietnamese tokens included because source data uses Vietnamese headers.
_TRUE_TOKENS = {"true", "t", "yes", "y", "1", "có", "co", "đúng", "dung"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0", "không", "khong", "sai"}


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


class DataCleaner:
    def __init__(self, column_mapping: Dict[str, str]):
        self.column_mapping = column_mapping
        # Collected during convert_types(); read by collect_conversion_errors().
        self._conversion_errors: List[Dict[str, Any]] = []

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.column_mapping)

    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to canonical types.

        Distinguishes three kinds of cells (requirement #2):
          * empty/blank cell             -> NaN  (legitimate missing value)
          * non-empty, unparsable number -> NaN  + recorded conversion error
          * unrecognised boolean token   -> NaN  + recorded conversion error

        No bad value is silently turned into 0.
        """
        self._conversion_errors = []
        df_clean = df.copy()

        # Normalise empty-looking strings to NaN across object columns so that
        # legitimately-empty cells are distinguishable from bad-value cells.
        for col in df_clean.columns:
            if df_clean[col].dtype == object:
                df_clean[col] = df_clean[col].apply(
                    lambda v: np.nan if _is_blank(v) else v
                )

        for col in NUMERIC_COLUMNS:
            if col not in df_clean.columns:
                continue
            original = df_clean[col]
            converted = pd.Series(pd.to_numeric(original, errors="coerce"), index=df_clean.index)
            # A cell is a conversion error iff: original non-empty AND converted NaN.
            not_blank_mask = original.apply(lambda v: not _is_blank(v)).astype(bool)
            failed_mask = converted.isna().astype(bool) & not_blank_mask
            for idx in df_clean.index[failed_mask]:
                self._conversion_errors.append({
                    "row": int(idx),
                    "column": col,
                    "value": str(original.loc[idx]),
                    "error": f"Cannot convert value to number in column '{col}'",
                    "kind": "numeric_conversion",
                })
            df_clean[col] = converted

        if "works_part_time" in df_clean.columns:
            original = df_clean["works_part_time"]
            parsed = pd.Series(original.apply(self._parse_bool_strict), index=df_clean.index)
            # Record unrecognised boolean tokens (non-empty original that parsed to NaN).
            not_blank_mask = original.apply(lambda v: not _is_blank(v)).astype(bool)
            failed_mask = parsed.isna().astype(bool) & not_blank_mask
            for idx in df_clean.index[failed_mask]:
                self._conversion_errors.append({
                    "row": int(idx),
                    "column": "works_part_time",
                    "value": str(original.loc[idx]),
                    "error": "Unrecognised boolean value in column 'works_part_time'",
                    "kind": "boolean_conversion",
                })
            df_clean["works_part_time"] = parsed

        return df_clean

    def _parse_bool_strict(self, value: Any) -> Any:
        """Parse a boolean strictly (requirement #3).

        Accepts real bool, numbers (0/non-zero), and common strings:
        true/false/yes/no/1/0 plus Vietnamese tokens (có/không/đúng/sai).
        EMPTY cell -> NaN (legitimate missing). UNRECOGNISED non-empty token
        -> NaN (caller records a conversion error). The string "False" must
        NOT be interpreted as True.
        """
        if _is_blank(value):
            return np.nan
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not (isinstance(value, float) and pd.isna(value)):
            return False if value == 0 else True
        s = str(value).strip().lower()
        if s == "":
            return np.nan
        if s in _TRUE_TOKENS:
            return True
        if s in _FALSE_TOKENS:
            return False
        # Unrecognised -> NaN (failure recorded by caller via failed_mask)
        return np.nan

    def collect_conversion_errors(self) -> List[Dict[str, Any]]:
        """Return conversion errors recorded during the last convert_types() call."""
        return list(self._conversion_errors)

    def determine_tx2_applicability(
        self, df: pd.DataFrame, courses_with_tx2: list[str],
    ) -> pd.DataFrame:
        df_clean = df.copy()
        if "course_id" in df_clean.columns:
            df_clean["has_tx2"] = df_clean["course_id"].isin(courses_with_tx2)
        else:
            df_clean["has_tx2"] = False
        return df_clean
