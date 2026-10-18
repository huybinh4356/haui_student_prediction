import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from haui_student_prediction.ml.preprocessing import build_preprocessing_pipeline
from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.features.feature_builder import FeatureBuilder
from dataclasses import dataclass
from typing import Tuple


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
    """Train a LinearRegression model on df_train and evaluate on df_test.

    Requirement #3: uses LinearRegression as the primary model (NOT Ridge by
    default). Ridge/Lasso comparison is intentionally not included here.

    Requirement #2: preprocessing lives inside the sklearn Pipeline so it is
    fit ONLY on the train split -- no leakage of test statistics into imputer
    / scaler / encoder.

    Requirement #4: validates target non-empty, data not too small, and handles
    edge cases explicitly rather than producing meaningless metrics.

    Requirement #5: metrics (MAE, RMSE, R²) are computed only when there is a
    non-empty test set. When R² is undefined (n_test < 2), it is reported as
    NaN rather than a fabricated number.
    """
    # --- validate contract ---
    contract.validate()

    num_cols = contract.numerical_features
    cat_cols = contract.categorical_features
    target_col = contract.target_column

    # --- validate inputs ---
    if df_train is None or len(df_train) == 0:
        raise ValueError("Training dataframe is empty")
    if df_test is None:
        raise ValueError("Test dataframe is None")
    if target_col not in df_train.columns:
        raise ValueError(f"Target column '{target_col}' missing in training data")
    if target_col not in df_test.columns:
        raise ValueError(f"Target column '{target_col}' missing in test data")

    # Minimum sample sizes: need at least 2 train rows to fit a regressor with
    # an intercept, and at least 1 feature. Require >= 3 train rows to have any
    # meaningful fit.
    if len(df_train) < 3:
        raise ValueError(
            f"Training data too small ({len(df_train)} rows); need at least 3"
        )

    # Target must not be entirely NaN -- otherwise training is meaningless.
    y_train = df_train[target_col]
    if y_train.isna().all():
        raise ValueError(f"Training target '{target_col}' is entirely NaN")

    # Drop rows with NaN target from training (cannot learn from them).
    train_mask = y_train.notna()
    df_train_fit = df_train.loc[train_mask]
    if len(df_train_fit) < 3:
        raise ValueError(
            f"After dropping NaN targets, only {len(df_train_fit)} "
            "usable training rows; need at least 3"
        )

    builder = FeatureBuilder(contract)

    # Extract features (excludes target & identifiers by construction).
    X_train = builder.extract_features(df_train_fit, is_training=True)
    y_train_fit = df_train_fit[target_col]

    # Build preprocessing + regressor in a single Pipeline (leakage-safe).
    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])

    model.fit(X_train, y_train_fit)

    # --- evaluation ---
    if len(df_test) == 0:
        # Empty test set: cannot compute any metric. Report NaN explicitly
        # rather than fabricating a score.
        return model, ModelMetrics(mae=float("nan"), rmse=float("nan"), r2=float("nan"))

    y_test = df_test[target_col]
    if y_test.isna().all():
        raise ValueError(f"Test target '{target_col}' is entirely NaN")

    # Drop NaN-target test rows for metric computation.
    test_mask = y_test.notna()
    df_test_eval = df_test.loc[test_mask]
    if len(df_test_eval) == 0:
        raise ValueError("No test rows with non-NaN target to evaluate")

    X_test = builder.extract_features(df_test_eval, is_training=False)
    y_test_eval = df_test_eval[target_col]

    raw_predictions = model.predict(X_test)
    if isinstance(raw_predictions, tuple):
        predictions = raw_predictions[0]
    else:
        predictions = raw_predictions
    pred_array = np.asarray(predictions)

    mae = float(mean_absolute_error(y_test_eval, pred_array))
    rmse = float(root_mean_squared_error(y_test_eval, pred_array))

    # R² is only defined when n_test >= 2. With a single test point, r2_score
    # would raise or return an undefined value; report NaN instead.
    if len(y_test_eval) >= 2:
        r2 = float(r2_score(y_test_eval, pred_array))
    else:
        r2 = float("nan")

    return model, ModelMetrics(mae=mae, rmse=rmse, r2=r2)
