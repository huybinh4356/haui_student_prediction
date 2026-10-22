from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from haui_student_prediction.features.feature_builder import FeatureBuilder
from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.ml.preprocessing import build_preprocessing_pipeline
from haui_student_prediction.ml.train_regressor import ModelMetrics


XGBOOST_MODEL_VERSION = "xgboost-1.0.0"


def _new_pipeline(contract: FeatureContract) -> Pipeline:
    preprocessor = build_preprocessing_pipeline(
        contract.numerical_features,
        contract.categorical_features,
    )
    regressor = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=120,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=1,
        tree_method="hist",
        eval_metric="rmse",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )


def _prepare_labeled_data(
    dataframe: pd.DataFrame,
    target_column: str,
    minimum_rows: int,
    dataset_name: str,
) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        raise ValueError(f"{dataset_name} data is empty")
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' is missing from {dataset_name} data")

    target = pd.to_numeric(dataframe[target_column], errors="coerce")
    target_values = target.to_numpy(dtype=float, na_value=np.nan)
    valid_mask = np.isfinite(target_values)
    prepared = dataframe.loc[valid_mask].copy()
    prepared[target_column] = target_values[valid_mask]

    if len(prepared) < minimum_rows:
        raise ValueError(
            f"{dataset_name} data has {len(prepared)} valid target rows; "
            f"at least {minimum_rows} are required"
        )

    return prepared


def train_xgboost_and_evaluate(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    contract: FeatureContract,
) -> Tuple[Pipeline, ModelMetrics]:
    contract.validate()
    target_column = contract.target_column

    train_data = _prepare_labeled_data(
        df_train,
        target_column,
        minimum_rows=3,
        dataset_name="Training",
    )
    test_data = _prepare_labeled_data(
        df_test,
        target_column,
        minimum_rows=2,
        dataset_name="Test",
    )

    builder = FeatureBuilder(contract)
    evaluation_model = _new_pipeline(contract)
    X_train = builder.extract_features(train_data, is_training=True)
    y_train = train_data[target_column]
    evaluation_model.fit(X_train, y_train)

    X_test = builder.extract_features(test_data, is_training=False)
    y_test = test_data[target_column]
    predictions = np.asarray(evaluation_model.predict(X_test), dtype=float)

    metrics = ModelMetrics(
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=float(root_mean_squared_error(y_test, predictions)),
        r2=float(r2_score(y_test, predictions)) if len(y_test) >= 2 else float("nan"),
    )

    full_data = pd.concat([train_data, test_data], axis=0)
    full_data = _prepare_labeled_data(
        full_data,
        target_column,
        minimum_rows=3,
        dataset_name="Combined",
    )
    final_model = _new_pipeline(contract)
    X_full = builder.extract_features(full_data, is_training=True)
    y_full = full_data[target_column]
    final_model.fit(X_full, y_full)

    return final_model, metrics
