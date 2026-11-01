from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from haui_student_prediction.features.feature_builder import FeatureBuilder
from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.ml.model_registry import ACTIVE_MODEL_VERSION
from haui_student_prediction.ml.preprocessing import build_preprocessing_pipeline
from haui_student_prediction.ml.train_regressor import ModelMetrics


XGBOOST_MODEL_VERSION = ACTIVE_MODEL_VERSION

CANDIDATE_CONFIGS = {
    "conservative": {
        "n_estimators": 120,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
    },
    "balanced": {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 2,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
    },
    "more_flexible": {
        "n_estimators": 300,
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 1,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
    },
}


def _new_pipeline(
    contract: FeatureContract,
    model_parameters: dict | None = None,
) -> Pipeline:
    preprocessor = build_preprocessing_pipeline(
        contract.numerical_features,
        contract.categorical_features,
    )
    parameters = {
        "objective": "reg:squarederror",
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": 1,
        "tree_method": "hist",
        "eval_metric": "rmse",
    }
    parameters.update(model_parameters or {})
    regressor = XGBRegressor(**parameters)
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
    if target.isna().all():
        raise ValueError(f"{dataset_name} target '{target_column}' is entirely NaN")
    target_values = target.to_numpy(dtype=float, na_value=np.nan)
    valid_mask = np.isfinite(target_values)
    prepared = dataframe.loc[valid_mask].copy()
    prepared[target_column] = target_values[valid_mask]

    if len(prepared) < minimum_rows:
        raise ValueError(
            f"{dataset_name} data is too small: {len(prepared)} valid target rows; "
            f"at least {minimum_rows} are required"
        )

    return prepared


def _cross_validate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    contract: FeatureContract,
    groups: pd.Series | None = None,
) -> tuple[str, dict, dict[str, float], str]:
    if len(X_train) < 10:
        raise ValueError(
            "Cần ít nhất 10 bản ghi trong tập Training để thực hiện 5-fold "
            "Cross-validation. Hãy bổ sung dữ liệu lịch sử."
        )

    group_values = None
    has_repeated_students = False
    if groups is not None:
        normalized_groups = groups.astype("string").str.strip().str.casefold()
        has_repeated_students = (
            normalized_groups.notna().all()
            and normalized_groups.ne("").all()
            and normalized_groups.duplicated().any()
        )

    if has_repeated_students:
        unique_groups = int(normalized_groups.nunique())
        if unique_groups < 5:
            raise ValueError(
                "Dữ liệu có sinh viên xuất hiện nhiều lần nhưng chỉ có "
                f"{unique_groups} mã sinh viên duy nhất. Cần ít nhất 5 nhóm "
                "sinh viên để Cross-validation không đưa cùng sinh viên vào "
                "cả phần huấn luyện và xác thực."
            )
        splitter = GroupKFold(n_splits=5)
        group_values = normalized_groups.to_numpy()
        cv_strategy = "GroupKFold"
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_strategy = "KFold"

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    candidate_results = []

    for candidate_name, parameters in CANDIDATE_CONFIGS.items():
        pipeline = _new_pipeline(contract, parameters)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=splitter,
            groups=group_values,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        mae_scores = -np.asarray(scores["test_mae"], dtype=float)
        rmse_scores = -np.asarray(scores["test_rmse"], dtype=float)
        summary = {
            "cv_mae_mean": float(np.mean(mae_scores)),
            "cv_mae_std": float(np.std(mae_scores, ddof=1)),
            "cv_rmse_mean": float(np.mean(rmse_scores)),
            "cv_rmse_std": float(np.std(rmse_scores, ddof=1)),
        }
        candidate_results.append((candidate_name, parameters, summary))

    best = min(
        candidate_results,
        key=lambda item: (item[2]["cv_mae_mean"], item[2]["cv_rmse_mean"]),
    )
    return best[0], best[1], best[2], cv_strategy


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
        minimum_rows=10,
        dataset_name="Training",
    )

    builder = FeatureBuilder(contract)
    X_train = builder.extract_features(train_data, is_training=True)
    y_train = train_data[target_column]

    training_groups = (
        train_data["student_id"]
        if "student_id" in train_data.columns
        else None
    )
    selected_name, selected_parameters, cv_metrics, cv_strategy = _cross_validate_candidates(
        X_train,
        y_train,
        contract,
        groups=training_groups,
    )
    evaluation_model = _new_pipeline(contract, selected_parameters)
    evaluation_model.fit(X_train, y_train)

    if df_test is None:
        raise ValueError("Test data is None")

    if df_test.empty:
        metrics = ModelMetrics(
            mae=float("nan"),
            rmse=float("nan"),
            r2=float("nan"),
            **cv_metrics,
            selected_candidate=selected_name,
            cv_strategy=cv_strategy,
        )
        return evaluation_model, metrics

    test_data = _prepare_labeled_data(
        df_test,
        target_column,
        minimum_rows=1,
        dataset_name="Test",
    )
    X_test = builder.extract_features(test_data, is_training=False)
    y_test = test_data[target_column]
    predictions = np.asarray(evaluation_model.predict(X_test), dtype=float)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_predictions = np.asarray(baseline.predict(X_test), dtype=float)

    actual_values = y_test.to_numpy(dtype=float)
    actual_std = float(np.std(actual_values))
    predicted_std = float(np.std(predictions))
    spread_ratio = (
        predicted_std / actual_std if actual_std > 1e-12 else float("nan")
    )
    baseline_mae = float(mean_absolute_error(y_test, baseline_predictions))
    mae = float(mean_absolute_error(y_test, predictions))

    metrics = ModelMetrics(
        mae=mae,
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        r2=float(r2_score(y_test, predictions)) if len(y_test) >= 2 else float("nan"),
        **cv_metrics,
        baseline_mae=baseline_mae,
        baseline_rmse=float(np.sqrt(mean_squared_error(y_test, baseline_predictions))),
        actual_mean=float(np.mean(actual_values)),
        actual_std=actual_std,
        actual_min=float(np.min(actual_values)),
        actual_max=float(np.max(actual_values)),
        predicted_mean=float(np.mean(predictions)),
        predicted_std=predicted_std,
        predicted_min=float(np.min(predictions)),
        predicted_max=float(np.max(predictions)),
        prediction_spread_ratio=spread_ratio,
        prediction_unique_count=int(np.unique(np.round(predictions, decimals=8)).size),
        beats_baseline=bool(mae <= baseline_mae),
        selected_candidate=selected_name,
        cv_strategy=cv_strategy,
    )

    full_data = pd.concat([train_data, test_data], axis=0)
    final_model = _new_pipeline(contract, selected_parameters)
    X_full = builder.extract_features(full_data, is_training=True)
    y_full = full_data[target_column]
    final_model.fit(X_full, y_full)

    return final_model, metrics
