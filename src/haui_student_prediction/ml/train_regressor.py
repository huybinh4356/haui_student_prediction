from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.pipeline import Pipeline

from haui_student_prediction.features.feature_contract import FeatureContract


@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    r2: float


def train_and_evaluate(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    contract: FeatureContract,
) -> Tuple[Pipeline, ModelMetrics]:
    from haui_student_prediction.ml.xgboost_regressor import train_xgboost_and_evaluate

    return train_xgboost_and_evaluate(df_train, df_test, contract)
