import pandas as pd
from typing import Dict, Any

class EDAStatistics:
    def __init__(self):
        self.excluded_columns = ["student_id", "full_name"]

    def get_numeric_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        cols_to_analyze = [c for c in df.columns if c not in self.excluded_columns]
        df_filtered = df[cols_to_analyze]
        raw_dict = df_filtered.describe().to_dict()
        return {str(k): v for k, v in raw_dict.items()}

    def get_missingness_report(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {str(col): 0.0 for col in df.columns}
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        return {str(k): float(v) for k, v in missing_percentages.items()}

    def get_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_analyze = [c for c in df.columns if c not in self.excluded_columns]
        df_numeric = df[cols_to_analyze].select_dtypes(include=['number'])
        return df_numeric.corr()