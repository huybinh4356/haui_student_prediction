from dataclasses import dataclass, field
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from haui_student_prediction.features.feature_contract import FeatureContract


@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    r2: float
    cv_mae_mean: float = float("nan")
    cv_mae_std: float = float("nan")
    cv_rmse_mean: float = float("nan")
    cv_rmse_std: float = float("nan")
    baseline_mae: float = float("nan")
    baseline_rmse: float = float("nan")
    actual_mean: float = float("nan")
    actual_std: float = float("nan")
    actual_min: float = float("nan")
    actual_max: float = float("nan")
    predicted_mean: float = float("nan")
    predicted_std: float = float("nan")
    predicted_min: float = float("nan")
    predicted_max: float = float("nan")
    prediction_spread_ratio: float = float("nan")
    prediction_unique_count: int = 0
    beats_baseline: bool = False
    selected_candidate: str = ""
    cv_strategy: str = "KFold"
    feature_effects: dict[str, dict[str, float]] = field(default_factory=dict)
    cv_r2_mean: float = float("nan")
    cv_spread_ratio: float = float("nan")
    max_abs_feature_correlation: float = float("nan")
    data_signal_warning: str = ""


def train_and_evaluate(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    contract: FeatureContract,
) -> Tuple[Pipeline, ModelMetrics]:
    from haui_student_prediction.ml.xgboost_regressor import train_xgboost_and_evaluate

    return train_xgboost_and_evaluate(df_train, df_test, contract)
