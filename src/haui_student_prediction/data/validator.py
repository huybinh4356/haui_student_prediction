import pandas as pd
from typing import List, Dict, Any

class DataValidator:
    def validate_percentages(self, df: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
        errors = []
        for col in columns:
            if col in df.columns:
                invalid_mask = (df[col] < 0) | (df[col] > 100)
                invalid_indices = df[invalid_mask].index.tolist()
                for idx in invalid_indices:
                    errors.append({
                        "row": idx,
                        "column": col,
                        "error": "Percentage must be between 0 and 100"
                    })
        return errors

    def validate_required_columns(self, df: pd.DataFrame, required_columns: List[str]) -> List[str]:
        missing = [col for col in required_columns if col not in df.columns]
        return missing

    def check_duplicates(self, df: pd.DataFrame, subset: List[str]) -> List[int]:
        if not all(col in df.columns for col in subset):
            return []
        duplicates = df[df.duplicated(subset=subset, keep=False)]
        return duplicates.index.tolist()