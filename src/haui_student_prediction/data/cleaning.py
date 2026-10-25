import pandas as pd
import numpy as np
from typing import Dict, List, Any

from haui_student_prediction.data.column_mapping import normalize_column_name

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
        normalized_mapping = {
            normalize_column_name(source): target
            for source, target in self.column_mapping.items()
        }
        original_columns = list(df.columns)
        renamed_columns = [
            normalized_mapping.get(normalize_column_name(source), source)
            for source in original_columns
        ]
        duplicates = sorted({
            column for column in renamed_columns
            if renamed_columns.count(column) > 1
        })
        if duplicates:
            sources = {
                target: [
                    str(source)
                    for source, mapped in zip(original_columns, renamed_columns)
                    if mapped == target
                ]
                for target in duplicates
            }
            raise ValueError(
                f"Multiple input columns map to the same field: {sources}"
            )
        return df.set_axis(renamed_columns, axis="columns")

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

        for column in ("student_id", "course_id"):
            if column in df_clean.columns:
                df_clean[column] = df_clean[column].apply(self._normalize_student_id)
        if "semester" in df_clean.columns:
            df_clean["semester"] = df_clean["semester"].apply(self._normalize_identifier_text)

        for col in NUMERIC_COLUMNS:
            if col not in df_clean.columns:
                continue
            original = df_clean[col]
            converted = pd.Series(pd.to_numeric(original, errors="coerce"), index=df_clean.index)
            not_blank_mask = original.apply(lambda v: not _is_blank(v)).astype(bool)
            non_finite_mask = converted.notna() & ~np.isfinite(converted.astype(float))
            failed_mask = (converted.isna() & not_blank_mask) | non_finite_mask
            for idx in df_clean.index[failed_mask]:
                self._conversion_errors.append({
                    "row": int(idx),
                    "column": col,
                    "value": str(original.loc[idx]),
                    "error": f"Invalid numeric value in column '{col}'",
                    "kind": "numeric_conversion",
                })
            df_clean[col] = converted.mask(non_finite_mask)

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

    def _normalize_student_id(self, value: Any) -> Any:
        if _is_blank(value):
            return np.nan
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return str(value)
        if isinstance(value, (float, np.floating)) and np.isfinite(value):
            if float(value).is_integer():
                return str(int(value))
        return str(value).strip().upper()

    def _normalize_identifier_text(self, value: Any) -> Any:
        if _is_blank(value):
            return np.nan
        return str(value).strip()

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
        if courses_with_tx2 and "course_id" in df_clean.columns:
            normalized_tx2_courses = {
                self._normalize_student_id(course_id)
                for course_id in courses_with_tx2
                if not _is_blank(course_id)
            }
            applicable = df_clean["course_id"].isin(normalized_tx2_courses)
            applicable = applicable.mask(df_clean["course_id"].isna(), pd.NA)
            df_clean["has_tx2"] = applicable.astype("boolean")
        elif "tx2_score" in df_clean.columns and "course_id" in df_clean.columns:
            course_ids = df_clean["course_id"]
            has_course_id = (
                course_ids.notna()
                & course_ids.astype("string").str.strip().ne("")
            )
            applicable = pd.Series(pd.NA, index=df_clean.index, dtype="boolean")
            known_course_scores = df_clean.loc[has_course_id].groupby(
                "course_id"
            )["tx2_score"].transform(lambda values: values.notna().any())
            applicable.loc[has_course_id] = known_course_scores.astype("boolean")
            missing_course_id_with_score = (~has_course_id) & df_clean["tx2_score"].notna()
            applicable.loc[missing_course_id_with_score] = True
            df_clean["has_tx2"] = applicable
        elif "tx2_score" in df_clean.columns:
            df_clean["has_tx2"] = pd.Series(
                df_clean["tx2_score"].notna().where(df_clean["tx2_score"].notna(), pd.NA),
                index=df_clean.index,
                dtype="boolean",
            )
        else:
            df_clean["has_tx2"] = pd.Series(
                pd.NA,
                index=df_clean.index,
                dtype="boolean",
            )
        return df_clean
