from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from haui_student_prediction.features.feature_builder import FeatureBuilder
from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.ml.data_splitting import split_train_test
from haui_student_prediction.ml.preprocessing import build_preprocessing_pipeline

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    model_metrics: pd.DataFrame
    cross_validation_results: pd.DataFrame
    test_predictions: pd.DataFrame
    residual_analysis: pd.DataFrame
    feature_importance: pd.DataFrame
    best_model_name: str
    acceptance_status: str


def _build_candidate_pipelines(contract: FeatureContract, random_state: int = 42) -> dict[str, Pipeline]:
    """Construct Scikit-Learn pipelines tailored to each algorithm's requirements."""
    # Preprocessor WITH numeric scaling for linear regularized models (Ridge, Lasso)
    linear_preprocessor = build_preprocessing_pipeline(
        contract.numerical_features,
        contract.categorical_features,
        scale_numeric=True,
    )
    linear_preprocessor.sparse_threshold = 0.0

    # Preprocessor WITHOUT numeric scaling for tree-based models (RF, XGBoost, CatBoost)
    tree_preprocessor = build_preprocessing_pipeline(
        contract.numerical_features,
        contract.categorical_features,
        scale_numeric=False,
    )
    tree_preprocessor.sparse_threshold = 0.0

    return {
        "Baseline (Mean)": Pipeline([
            ("preprocessor", linear_preprocessor),
            ("regressor", DummyRegressor(strategy="mean")),
        ]),
        "Ridge": Pipeline([
            ("preprocessor", linear_preprocessor),
            ("regressor", RidgeCV(alphas=np.logspace(-3, 3, 20))),
        ]),
        "Lasso": Pipeline([
            ("preprocessor", linear_preprocessor),
            ("regressor", LassoCV(alphas=np.logspace(-4, 2, 20), random_state=random_state, max_iter=5000)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("regressor", XGBRegressor(
                objective="reg:squarederror",
                n_estimators=350,
                learning_rate=0.06,
                max_depth=5,
                min_child_weight=2,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.05,
                reg_lambda=1.5,
                random_state=random_state,
                n_jobs=1,
                tree_method="hist",
                eval_metric="rmse",
            )),
        ]),
        "CatBoost": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("regressor", CatBoostRegressor(
                iterations=350,
                learning_rate=0.06,
                depth=5,
                loss_function="RMSE",
                l2_leaf_reg=3.0,
                random_seed=random_state,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
            )),
        ]),
    }


def run_model_benchmark(
    dataframe: pd.DataFrame,
    contract: FeatureContract,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
    sample_id_col: str | None = "student_id",
) -> BenchmarkResult:
    """Benchmark multiple algorithms on a common dataset using identical splits and contracts."""
    contract.validate()
    target_col = contract.target_column

    # Data split: strictly 80% train, 20% independent holdout test
    train_data, test_data = split_train_test(dataframe, test_size=test_size, random_state=random_state)

    builder = FeatureBuilder(contract)
    X_train = builder.extract_features(train_data, is_training=True)
    y_train = train_data[target_col].to_numpy(dtype=float)
    X_test = builder.extract_features(test_data, is_training=False)
    y_test = test_data[target_col].to_numpy(dtype=float)

    # Sample identifiers
    test_ids = (
        test_data[sample_id_col].astype(str).tolist()
        if sample_id_col and sample_id_col in test_data.columns
        else [f"SAMP_{i:04d}" for i in range(len(test_data))]
    )

    actual_std_test = float(np.std(y_test))
    actual_mean_test = float(np.mean(y_test))
    actual_std_train = float(np.std(y_train))

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    folds = list(kf.split(X_train, y_train))

    pipelines = _build_candidate_pipelines(contract, random_state=random_state)

    metrics_rows = []
    cv_detail_rows = []
    test_pred_dict = {
        "sample_id": test_ids,
        "actual_final_exam_score": y_test,
    }
    residual_rows = []
    importance_records = []

    baseline_mae = float("nan")

    for model_name, pipeline in pipelines.items():
        start_time = perf_counter()

        # 1. 5-Fold Cross-Validation on training set
        oof_preds = np.empty(len(y_train), dtype=float)
        fold_maes = []
        fold_rmses = []

        for fold_idx, (fit_idx, val_idx) in enumerate(folds, 1):
            fold_pipe = _build_candidate_pipelines(contract, random_state=random_state)[model_name]
            fold_pipe.fit(X_train.iloc[fit_idx], y_train[fit_idx])
            val_preds = np.asarray(fold_pipe.predict(X_train.iloc[val_idx]), dtype=float).reshape(-1)
            oof_preds[val_idx] = val_preds

            f_mae = float(mean_absolute_error(y_train[val_idx], val_preds))
            f_rmse = float(np.sqrt(mean_squared_error(y_train[val_idx], val_preds)))
            fold_maes.append(f_mae)
            fold_rmses.append(f_rmse)

            cv_detail_rows.append({
                "model_name": model_name,
                "fold": fold_idx,
                "val_mae": f_mae,
                "val_rmse": f_rmse,
            })

        cv_mae_mean = float(np.mean(fold_maes))
        cv_mae_std = float(np.std(fold_maes, ddof=1))
        cv_rmse_mean = float(np.mean(fold_rmses))
        cv_rmse_std = float(np.std(fold_rmses, ddof=1))
        cv_r2 = float(r2_score(y_train, oof_preds))
        oof_std = float(np.std(oof_preds))
        cv_spread_ratio = oof_std / actual_std_train if actual_std_train > 1e-12 else 0.0

        # 2. Fit on full training set and evaluate on independent Holdout Test set
        pipeline.fit(X_train, y_train)
        elapsed_time = perf_counter() - start_time

        test_preds = np.asarray(pipeline.predict(X_test), dtype=float).reshape(-1)
        test_mae = float(mean_absolute_error(y_test, test_preds))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
        test_r2 = float(r2_score(y_test, test_preds))
        pred_std = float(np.std(test_preds))
        pred_mean = float(np.mean(test_preds))
        spread_ratio = pred_std / actual_std_test if actual_std_test > 1e-12 else 0.0

        if model_name == "Baseline (Mean)":
            baseline_mae = test_mae

        beats_baseline = bool(test_mae <= baseline_mae) if not np.isnan(baseline_mae) else False

        metrics_rows.append({
            "model_name": model_name,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_r2": cv_r2,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "actual_std": actual_std_test,
            "spread_ratio": spread_ratio,
            "cv_spread_ratio": cv_spread_ratio,
            "beats_baseline": beats_baseline,
            "fit_seconds": round(elapsed_time, 2),
        })

        test_pred_dict[f"pred__{model_name}"] = test_preds

        # Residuals
        residuals = y_test - test_preds
        abs_errors = np.abs(residuals)
        for sid, act, prd, res, abe in zip(test_ids, y_test, test_preds, residuals, abs_errors):
            residual_rows.append({
                "sample_id": sid,
                "model_name": model_name,
                "actual_score": act,
                "predicted_score": round(prd, 3),
                "residual": round(res, 3),
                "absolute_error": round(abe, 3),
            })

        # Feature importances
        prep = pipeline.named_steps["preprocessor"]
        reg = pipeline.named_steps["regressor"]
        raw_feature_names = list(prep.get_feature_names_out())
        source_features = contract.get_all_features()
        ordered_sources = sorted(source_features, key=len, reverse=True)

        if hasattr(reg, "coef_"):
            raw_weights = np.abs(reg.coef_)
        elif hasattr(reg, "feature_importances_"):
            raw_weights = reg.feature_importances_
        else:
            raw_weights = np.zeros(len(raw_feature_names))

        source_weights = {feat: 0.0 for feat in source_features}
        for pos, tname in enumerate(raw_feature_names):
            s_name = tname.split("__", 1)[-1]
            matched = next((s for s in ordered_sources if s_name == s or s_name.startswith(s + "_")), None)
            if matched:
                source_weights[matched] += float(raw_weights[pos])

        # Normalize to percentage
        total_w = sum(source_weights.values())
        for feat, val in source_weights.items():
            importance_records.append({
                "model_name": model_name,
                "feature": feat,
                "importance_weight": val,
                "importance_percentage": round((val / total_w * 100.0) if total_w > 0 else 0.0, 2),
            })

    metrics_df = pd.DataFrame(metrics_rows)
    cv_df = pd.DataFrame(cv_detail_rows)
    test_pred_df = pd.DataFrame(test_pred_dict)
    residual_df = pd.DataFrame(residual_rows)
    importance_df = pd.DataFrame(importance_records)

    # Acceptance logic: must beat baseline MAE and have positive R2
    non_baseline = metrics_df[metrics_df["model_name"] != "Baseline (Mean)"]
    qualified = non_baseline[(non_baseline["test_mae"] < baseline_mae) & (non_baseline["test_r2"] > 0.0)]

    if qualified.empty:
        best_model_name = "None"
        acceptance_status = "NO_ACCEPTABLE_MODEL (Không có mô hình nào vượt baseline hoặc R² âm)"
    else:
        # Sort by lowest MAE, tiebreak by highest spread ratio
        best_row = qualified.sort_values(by=["test_mae", "spread_ratio"], ascending=[True, False]).iloc[0]
        best_model_name = str(best_row["model_name"])
        acceptance_status = f"ACCEPTED (Mô hình tốt nhất: {best_model_name}, MAE = {best_row['test_mae']:.4f}, R² = {best_row['test_r2']:.4f})"

    return BenchmarkResult(
        model_metrics=metrics_df,
        cross_validation_results=cv_df,
        test_predictions=test_pred_df,
        residual_analysis=residual_df,
        feature_importance=importance_df,
        best_model_name=best_model_name,
        acceptance_status=acceptance_status,
    )

