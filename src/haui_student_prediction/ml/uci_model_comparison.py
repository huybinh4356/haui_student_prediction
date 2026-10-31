from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from haui_student_prediction.data.uci_student_performance import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from haui_student_prediction.ml.preprocessing import build_preprocessing_pipeline


def compare_uci_models(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataframe is None or dataframe.empty:
        raise ValueError("UCI dataframe rỗng.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size phải nằm trong khoảng 0.1 đến 0.4.")

    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    required_columns = feature_columns + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"UCI dataframe thiếu cột cần thiết: {missing}")
    if len(dataframe) < 20:
        raise ValueError("Cần tối thiểu 20 bản ghi để so sánh mô hình.")

    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    target_values = target.to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(target_values)
    if not valid.all():
        raise ValueError("UCI target chứa giá trị thiếu hoặc không hữu hạn.")

    features = dataframe[feature_columns].copy()
    labels = pd.Series(target_values, index=dataframe.index, name=TARGET_COLUMN)
    row_indices = np.arange(len(dataframe))
    train_indices, test_indices = train_test_split(
        row_indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    X_train = features.iloc[train_indices]
    X_test = features.iloc[test_indices]
    y_train = labels.iloc[train_indices]
    y_test = labels.iloc[test_indices]

    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài CatBoost. Chạy: python -m pip install -e ."
        ) from exc

    estimators = {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=5.0,
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
            eval_metric="rmse",
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_features=1.0,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=400,
            learning_rate=0.03,
            depth=5,
            loss_function="RMSE",
            l2_leaf_reg=5.0,
            random_seed=random_state,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        ),
    }

    metric_rows = []
    prediction_data = {
        "source_row": test_indices + 2,
        "actual_course_grade": y_test.to_numpy(dtype=float),
    }

    for model_name, estimator in estimators.items():
        preprocessor = build_preprocessing_pipeline(
            NUMERICAL_FEATURES,
            CATEGORICAL_FEATURES,
        )
        preprocessor.sparse_threshold = 0.0
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", estimator),
            ]
        )

        started_at = perf_counter()
        model.fit(X_train, y_train)
        training_seconds = perf_counter() - started_at
        predictions = np.asarray(model.predict(X_test), dtype=float)

        safe_name = model_name.lower().replace(" ", "_")
        prediction_data[f"predicted_{safe_name}"] = predictions
        prediction_data[f"absolute_error_{safe_name}"] = np.abs(
            y_test.to_numpy(dtype=float) - predictions
        )

        metric_rows.append(
            {
                "model": model_name,
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "r2": float(r2_score(y_test, predictions)),
                "training_seconds": float(training_seconds),
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values(
        by=["mae", "rmse"],
        ascending=True,
    ).reset_index(drop=True)
    predictions_frame = pd.DataFrame(prediction_data).sort_values(
        by="source_row"
    ).reset_index(drop=True)

    return metrics, predictions_frame
