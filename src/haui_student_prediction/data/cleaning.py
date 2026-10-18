import pandas as pd
from typing import Dict

class DataCleaner:
    def __init__(self, column_mapping: Dict[str, str]):
        self.column_mapping = column_mapping

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.column_mapping)

    def convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        numeric_cols = [
            "absence_percentage", "lms_learning_percentage", 
            "tx1_score", "tx2_score", "midterm_score", 
            "actual_final_exam_score", "stress_score", 
            "motivation_score", "interaction_score"
        ]
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        if "works_part_time" in df_clean.columns:
            df_clean["works_part_time"] = df_clean["works_part_time"].astype(bool)
            
        return df_clean

    def determine_tx2_applicability(self, df: pd.DataFrame, courses_with_tx2: list[str]) -> pd.DataFrame:
        df_clean = df.copy()
        if "course_id" in df_clean.columns:
            df_clean["has_tx2"] = df_clean["course_id"].isin(courses_with_tx2)
        else:
            df_clean["has_tx2"] = False
        return df_clean