import pandas as pd
from haui_student_prediction.features.feature_contract import FeatureContract

class FeatureBuilder:
    def __init__(self, contract: FeatureContract):
        self.contract = contract

    def extract_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        columns_to_keep = self.contract.get_all_features()
        if is_training:
            columns_to_keep.append(self.contract.target_column)

        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df[columns_to_keep].copy()