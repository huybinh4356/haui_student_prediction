from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from haui_student_prediction.data.uci_student_performance import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from haui_student_prediction.ml.preprocessing import build_preprocessing_pipeline


def _build_estimators(random_state: int) -> dict[str, object]:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài CatBoost. Chạy: python -m pip install -e ."
        ) from exc

    return {
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


def _build_model(estimator: object) -> Pipeline:
    preprocessor = build_preprocessing_pipeline(
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
    )
    preprocessor.sparse_threshold = 0.0
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )


def compare_uci_models(
    dataframe: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataframe is None or dataframe.empty:
        raise ValueError("UCI dataframe rỗng.")
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size phải nằm trong khoảng 0.1 đến 0.4.")
    if cv_folds < 2:
        raise ValueError("cv_folds phải lớn hơn hoặc bằng 2.")

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

    if cv_folds > len(X_train):
        raise ValueError("cv_folds không được lớn hơn số bản ghi trong tập Training.")

    estimators = _build_estimators(random_state)
    cv_splitter = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    metric_rows = []
    prediction_data = {
        "source_row": test_indices + 2,
        "actual_course_grade": y_test.to_numpy(dtype=float),
    }

    for model_name, estimator in estimators.items():
        model = _build_model(estimator)

        cv_started_at = perf_counter()
        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        cv_training_seconds = perf_counter() - cv_started_at

        started_at = perf_counter()
        model.fit(X_train, y_train)
        training_seconds = perf_counter() - started_at
        predictions = np.asarray(model.predict(X_test), dtype=float)

        safe_name = model_name.lower().replace(" ", "_")
        prediction_data[f"predicted_{safe_name}"] = predictions
        prediction_data[f"absolute_error_{safe_name}"] = np.abs(
            y_test.to_numpy(dtype=float) - predictions
        )

        cv_mae = -cv_results["test_mae"]
        cv_rmse = -cv_results["test_rmse"]
        cv_r2 = cv_results["test_r2"]

        metric_rows.append(
            {
                "model": model_name,
                "cv_folds": int(cv_folds),
                "cv_mae_mean": float(np.mean(cv_mae)),
                "cv_mae_std": float(np.std(cv_mae, ddof=1)),
                "cv_rmse_mean": float(np.mean(cv_rmse)),
                "cv_rmse_std": float(np.std(cv_rmse, ddof=1)),
                "cv_r2_mean": float(np.mean(cv_r2)),
                "cv_r2_std": float(np.std(cv_r2, ddof=1)),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "r2": float(r2_score(y_test, predictions)),
                "cv_training_seconds": float(cv_training_seconds),
                "training_seconds": float(training_seconds),
                "train_rows": int(len(train_indices)),
                "test_rows": int(len(test_indices)),
            }
        )

    metrics = pd.DataFrame(metric_rows).sort_values(
        by=["cv_mae_mean", "cv_rmse_mean"],
        ascending=True,
    ).reset_index(drop=True)
    predictions_frame = pd.DataFrame(prediction_data).sort_values(
        by="source_row"
    ).reset_index(drop=True)

    return metrics, predictions_frame


def train_and_register_uci_transfer_model(
    subject: str = "mat",
    version: str = "xgboost-2.2.0",
    random_state: int = 42,
) -> tuple[Pipeline, object, object]:
    """
    Huấn luyện mô hình chuyển giao quan hệ hành vi và học thuật từ dữ liệu chuẩn quốc tế UCI
    và đăng ký vào ModelRegistry để làm mô hình dự đoán chính thức cho HaUI.
    """
    import hashlib
    from pathlib import Path
    from haui_student_prediction.app_config import load_config
    from haui_student_prediction.data.uci_student_performance import load_uci_student_performance
    from haui_student_prediction.features.feature_contract import FeatureContract
    from haui_student_prediction.features.feature_builder import FeatureBuilder
    from haui_student_prediction.ml.model_registry import ModelRegistry
    from haui_student_prediction.ml.data_splitting import split_train_test
    from haui_student_prediction.ml.xgboost_regressor import train_xgboost_and_evaluate
    from haui_student_prediction.ml.accuracy_evaluator import compute_all_accuracy_metrics

    project_root = Path(__file__).resolve().parents[3]
    raw_dir = project_root / "data" / "raw" / "uci_student_performance"
    uci_df = load_uci_student_performance(subject=subject, raw_dir=raw_dir)

    n = len(uci_df)
    mapped_df = pd.DataFrame()
    mapped_df["student_id"] = [f"UCI_{subject.upper()}_{i+1:03d}" for i in range(n)]
    mapped_df["full_name"] = [f"Student {subject.upper()} {i+1}" for i in range(n)]
    mapped_df["major"] = "Khoa học máy tính"
    mapped_df["course_id"] = "ITP101"
    mapped_df["semester"] = "2026-1"

    # G1 / 2.0 -> tx1_score (0-10)
    mapped_df["tx1_score"] = np.round(uci_df["period1_grade"].astype(float) / 2.0, 1)
    # G2 / 2.0 -> midterm_score (0-10)
    mapped_df["midterm_score"] = np.round(uci_df["period2_grade"].astype(float) / 2.0, 1)

    # absences -> absence_percentage (approx 2.5% per absence, capped at 100%)
    mapped_df["absence_percentage"] = np.round(np.clip(uci_df["absence_count"].astype(float) * 2.5, 0.0, 100.0), 1)

    # studytime (1-4) -> lms_learning_percentage (25% to 95%)
    mapped_df["lms_learning_percentage"] = np.round(np.clip(uci_df["weekly_study_time_level"].astype(float) * 22.0 + 10.0, 0.0, 100.0), 1)

    # Motivation & behavior
    higher_intent = uci_df["higher_education_intent"].astype(str).str.lower().isin(["yes", "1", "true"])
    mapped_df["motivation_score"] = np.where(higher_intent, 4.0, 2.0)
    mapped_df["stress_score"] = 3.0
    mapped_df["interaction_score"] = 3.5
    mapped_df["works_part_time"] = False

    # Target: G3 / 2.0 -> target_final_exam_score (0-10)
    target_col = "target_final_exam_score"
    mapped_df[target_col] = np.round(uci_df["target_course_grade"].astype(float) / 2.0, 1)

    numeric_features = [
        "tx1_score",
        "midterm_score",
        "absence_percentage",
        "lms_learning_percentage",
        "interaction_score",
        "stress_score",
        "motivation_score",
    ]
    categorical_features = ["works_part_time", "major", "course_id"]

    contract = FeatureContract(
        numerical_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target_col,
    )

    train_data, test_data = split_train_test(mapped_df, test_size=0.2, random_state=random_state)
    model, metrics = train_xgboost_and_evaluate(train_data, test_data, contract)

    builder = FeatureBuilder(contract)
    X_test = builder.extract_features(test_data, is_training=False)
    y_test_pred = model.predict(X_test)
    y_test_true = test_data[target_col].values
    sample_ids = test_data["student_id"].astype(str).tolist()
    eval_result = compute_all_accuracy_metrics(y_test_true, y_test_pred, sample_ids=sample_ids)

    config = load_config()
    registry = ModelRegistry(str(config.model_registry_path), str(config.model_artifacts_dir))
    dataset_hash = hashlib.sha256(
        pd.util.hash_pandas_object(mapped_df, index=True).values.tobytes()
    ).hexdigest()

    registry.save_model(
        model=model,
        version=version,
        dataset_hash=dataset_hash,
        metrics=metrics,
        contract=contract,
    )

    return model, metrics, eval_result
