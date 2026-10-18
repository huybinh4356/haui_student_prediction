import pandas as pd
from haui_student_prediction.features.feature_contract import (
    FeatureContract,
    IDENTIFIER_COLUMNS,
)


class FeatureBuilder:
    """Extracts only contracted feature columns from a DataFrame.

    Guarantees (requirement #1 & #6):
      * The target column is NEVER included in features (only appended when
        is_training=True, returned separately from the feature matrix).
      * Identifier columns (student_id, full_name, ...) are never part of
        features even if a caller mistakenly adds them to the contract -- the
        contract itself is validated at construction time.
    """

    def __init__(self, contract: FeatureContract):
        # Validate the contract once at construction so bad contracts fail early.
        contract.validate()
        self.contract = contract

    def extract_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Return the feature matrix (X).

        When is_training=True the target column MUST be present in df (it is
        used to form y separately, NOT included in X). The returned DataFrame
        contains features only.
        """
        feature_cols = self.contract.get_all_features()

        # Double-guard: even if the contract was tampered with, strip identifiers.
        safe_cols = [c for c in feature_cols if c not in IDENTIFIER_COLUMNS
                     and c != self.contract.target_column]

        missing_features = [c for c in safe_cols if c not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")

        if is_training:
            if self.contract.target_column not in df.columns:
                raise ValueError(
                    f"Target column '{self.contract.target_column}' is missing for training"
                )

        X = df[safe_cols].copy()

        # sklearn's SimpleImputer/OneHotEncoder do not accept bool dtype; cast
        # categorical columns to string/object so they are treated as
        # categorical text. Bool True/False -> "True"/"False".
        for col in self.contract.categorical_features:
            if col in X.columns and X[col].dtype == bool:
                X[col] = X[col].astype(str)

        return X

    def extract_target(self, df: pd.DataFrame) -> pd.Series:
        """Return the target vector (y). Only valid during training."""
        if self.contract.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.contract.target_column}' is missing"
            )
        return df[self.contract.target_column].copy()
