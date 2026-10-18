import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

class DataValidator:
    def validate_percentages(self, df: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
        """Validate percentage columns are within [0, 100].

        NaN values (legitimate missing) are NOT flagged as errors -- only values
        that are present and out of range. Row indices refer to the DataFrame
        index so the caller can point at the exact row + column.
        """
        errors = []
        for col in columns:
            if col not in df.columns:
                continue
            series = df[col]
            # Only flag non-NaN values that are out of [0, 100].
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

    def validate_required_columns(self, df: pd.DataFrame, required_columns: List[str]) -> List[str]:
        return [col for col in required_columns if col not in df.columns]

    def check_duplicates(self, df: pd.DataFrame, subset: List[str]) -> List[int]:
        """Return indices of all rows that participate in a duplicate group
        (keep=False) on the given subset. Empty list if subset columns absent."""
        if not all(col in df.columns for col in subset):
            return []
        duplicates = df[df.duplicated(subset=subset, keep=False)]
        return duplicates.index.tolist()

    def validate_score_scale(
        self, df: pd.DataFrame, columns: List[str], max_score: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Validate score columns against an optional scale.

        Requirement #4: we must NOT invent a score scale. If `max_score` is None
        (no explicit configuration), we do NOT hard-fail. Instead we return a
        list of "needs_confirmation" notes for any score that looks out of a
        plausible 0-10 band (derived from risk_calculator._normalize_academic
        which assumes 0-10). These notes are informational and do not block
        is_success -- they are surfaced so the operator can confirm the scale.

        If `max_score` is provided (explicit config), out-of-range scores become
        hard errors (returned in the first list).

        Returns (errors, notes) where errors block success and notes do not.
        """
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
                # No explicit config: flag plausible outliers as notes only.
                # Use 0-10 as a *soft* reference band (derived, not invented as a rule).
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
