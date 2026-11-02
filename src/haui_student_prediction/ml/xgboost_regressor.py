from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from haui_student_prediction.features.feature_builder import FeatureBuilder
from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.ml.model_explainability import tree_feature_contributions
from haui_student_prediction.ml.model_registry import ACTIVE_MODEL_VERSION
from haui_student_prediction.ml.preprocessing import build_preprocessing_pipeline
from haui_student_prediction.ml.train_regressor import ModelMetrics


XGBOOST_MODEL_VERSION = ACTIVE_MODEL_VERSION

# ---------------------------------------------------------------------------
# Candidate hyper-parameter configurations
#
# The previous set was **over-regularised**: every candidate used very shallow
# trees (max_depth 2-3), very high L2 (reg_lambda 2-8), and a tiny learning
# rate — which forced the model to predict near the training-set mean.
#
# The new set keeps a few conservative candidates (to avoid over-fitting on
# small datasets) but adds candidates that are **flexible enough** to capture
# real grade variation.
# ---------------------------------------------------------------------------
CANDIDATE_CONFIGS = {
    # --- Flexible candidates (address the 6-7 clustering problem) -----------
    "flexible_deep": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    "flexible_mid": {
        "n_estimators": 400,
        "learning_rate": 0.08,
        "max_depth": 5,
        "min_child_weight": 2,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    "balanced_v2": {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 4,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
    },
    "deep_low_lr": {
        "n_estimators": 800,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.01,
        "reg_lambda": 1.5,
    },
    # --- Moderate candidates (good general-purpose balance) ------------------
    "moderate_depth4": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 2,
        "gamma": 0.05,
        "reg_alpha": 0.05,
        "reg_lambda": 2.0,
    },
    "moderate_depth5": {
        "n_estimators": 350,
        "learning_rate": 0.06,
        "max_depth": 5,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    # --- Conservative candidates (safer for very small datasets) -------------
    "conservative": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
    },
    "conservative_shallow": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 5,
        "gamma": 0.2,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
    },
}


def _new_pipeline(
    contract: FeatureContract,
    model_parameters: dict | None = None,
) -> Pipeline:
    # Tree-based models do NOT need StandardScaler.  Passing
    # scale_numeric=False (default) removes the scaler so that
    # regularisation parameters (reg_lambda, reg_alpha) work on raw
    # feature magnitudes rather than z-scored values — which previously
    # caused extreme shrinkage toward the mean.
    preprocessor = build_preprocessing_pipeline(
        contract.numerical_features,
        contract.categorical_features,
    )
    parameters = {
        "objective": "reg:squarederror",
        "subsample": 0.8,
        "colsample_bytree": 0.8,
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

    y_values = y_train.to_numpy(dtype=float)
    y_train_std = float(np.std(y_values))
    folds = list(splitter.split(X_train, y_train, groups=group_values))
    candidate_results = []

    for candidate_name, parameters in CANDIDATE_CONFIGS.items():
        # True out-of-fold (OOF) predictions: every training row is predicted
        # by a model that never saw it.  Spread measured on OOF predictions
        # reflects real generalisation; in-sample spread would reward models
        # that simply memorise noise.
        oof_predictions = np.empty(len(y_values), dtype=float)
        fold_mae = []
        fold_rmse = []
        for fit_index, validation_index in folds:
            pipeline = _new_pipeline(contract, parameters)
            pipeline.fit(X_train.iloc[fit_index], y_train.iloc[fit_index])
            fold_predictions = np.asarray(
                pipeline.predict(X_train.iloc[validation_index]), dtype=float
            )
            oof_predictions[validation_index] = fold_predictions
            fold_truth = y_values[validation_index]
            fold_mae.append(float(mean_absolute_error(fold_truth, fold_predictions)))
            fold_rmse.append(float(np.sqrt(mean_squared_error(fold_truth, fold_predictions))))

        oof_std = float(np.std(oof_predictions))
        summary = {
            "cv_mae_mean": float(np.mean(fold_mae)),
            "cv_mae_std": float(np.std(fold_mae, ddof=1)),
            "cv_rmse_mean": float(np.mean(fold_rmse)),
            "cv_rmse_std": float(np.std(fold_rmse, ddof=1)),
            "cv_r2_mean": float(r2_score(y_values, oof_predictions)),
            "cv_spread_ratio": oof_std / y_train_std if y_train_std > 1e-12 else 0.0,
        }
        candidate_results.append((candidate_name, parameters, summary))

    # --- Selection -----------------------------------------------------------
    # 1. Accuracy first: keep only candidates whose CV MAE is within 6% of the
    #    best CV MAE (statistically indistinguishable on typical data sizes).
    # 2. Among those, prefer the candidate with the largest OOF prediction
    #    spread, so a model that genuinely separates weak and strong students
    #    wins over one that hugs the mean with the same accuracy.
    # Accuracy is never sacrificed for spread: on data without signal the
    # most accurate (mean-like) model is still chosen and a warning is raised.
    best_mae = min(item[2]["cv_mae_mean"] for item in candidate_results)
    best_cv_r2 = max(item[2]["cv_r2_mean"] for item in candidate_results)
    if best_cv_r2 < LOW_SIGNAL_CV_R2:
        # No candidate generalises: any extra spread would be noise, so pick
        # the most accurate candidate outright.
        best = min(
            candidate_results,
            key=lambda item: (item[2]["cv_mae_mean"], item[2]["cv_rmse_mean"]),
        )
        return best[0], best[1], best[2], cv_strategy
    tolerance = best_mae * 1.06
    shortlisted = [item for item in candidate_results if item[2]["cv_mae_mean"] <= tolerance]
    best = max(
        shortlisted,
        key=lambda item: (item[2]["cv_spread_ratio"], -item[2]["cv_mae_mean"]),
    )
    return best[0], best[1], best[2], cv_strategy


def _summarize_feature_effects(
    contributions: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    summaries = {}
    for feature in feature_names:
        values = contributions[feature].to_numpy(dtype=float)
        summaries[feature] = {
            "mean_absolute_points": float(np.mean(np.abs(values))),
            "mean_signed_points": float(np.mean(values)),
        }
    return summaries


LOW_SIGNAL_CV_R2 = 0.05
LOW_SIGNAL_CORRELATION = 0.10


def _max_abs_feature_correlation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    contract: FeatureContract,
) -> float:
    """Largest |Pearson r| between any numeric feature and the target."""
    target = pd.to_numeric(y_train, errors="coerce")
    best = 0.0
    for column in contract.numerical_features:
        if column not in X_train.columns:
            continue
        values = pd.to_numeric(X_train[column], errors="coerce")
        mask = values.notna() & target.notna()
        if mask.sum() < 3 or values[mask].nunique() < 2 or target[mask].nunique() < 2:
            continue
        correlation = float(np.corrcoef(values[mask], target[mask])[0, 1])
        if np.isfinite(correlation):
            best = max(best, abs(correlation))
    return best


def _data_signal_warning(cv_r2: float, max_abs_correlation: float) -> str:
    """Explain when the training labels are not predictable from the features.

    If out-of-fold R² is ~0, no hyper-parameter choice can produce meaningful,
    well-spread predictions: the best possible estimate is the average score,
    which is exactly the "everything is 6-7 points" symptom.
    """
    if not np.isfinite(cv_r2) or cv_r2 >= LOW_SIGNAL_CV_R2:
        return ""
    message = (
        f"Dữ liệu lịch sử gần như không có tín hiệu dự đoán: R² cross-validation = "
        f"{cv_r2:.3f} (mô hình không giải thích được biến thiên của điểm cuối kỳ). "
    )
    if max_abs_correlation < LOW_SIGNAL_CORRELATION:
        message += (
            f"Tương quan lớn nhất giữa một biến đầu vào và điểm cuối kỳ chỉ là "
            f"{max_abs_correlation:.3f}, tức điểm cuối kỳ trong tệp gần như ngẫu nhiên "
            "so với TX1, giữa kỳ, chuyên cần, LMS... "
        )
    message += (
        "Khi đó dự đoán tốt nhất về mặt thống kê là điểm trung bình, nên kết quả sẽ dồn "
        "về một khoảng hẹp. Cần dữ liệu lịch sử thực tế (hoặc dữ liệu mô phỏng có quan hệ "
        "hợp lý giữa điểm quá trình và điểm cuối kỳ) để mô hình phân biệt được sinh viên."
    )
    return message


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
    max_abs_correlation = _max_abs_feature_correlation(X_train, y_train, contract)
    signal_warning = _data_signal_warning(
        cv_metrics["cv_r2_mean"],
        max_abs_correlation,
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
            max_abs_feature_correlation=max_abs_correlation,
            data_signal_warning=signal_warning,
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

    test_contributions = tree_feature_contributions(
        evaluation_model,
        X_test,
        contract.get_all_features(),
    )
    feature_effects = _summarize_feature_effects(
        test_contributions,
        contract.get_all_features(),
    )

    metrics = ModelMetrics(
        mae=mae,
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        r2=float(r2_score(y_test, predictions)) if len(y_test) >= 2 else float("nan"),
        cv_mae_mean=cv_metrics["cv_mae_mean"],
        cv_mae_std=cv_metrics["cv_mae_std"],
        cv_rmse_mean=cv_metrics["cv_rmse_mean"],
        cv_rmse_std=cv_metrics["cv_rmse_std"],
        cv_r2_mean=cv_metrics["cv_r2_mean"],
        cv_spread_ratio=cv_metrics["cv_spread_ratio"],
        max_abs_feature_correlation=max_abs_correlation,
        data_signal_warning=signal_warning,
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
        feature_effects=feature_effects,
    )

    full_data = pd.concat([train_data, test_data], axis=0)
    final_model = _new_pipeline(contract, selected_parameters)
    X_full = builder.extract_features(full_data, is_training=True)
    y_full = full_data[target_column]
    final_model.fit(X_full, y_full)

    return final_model, metrics
