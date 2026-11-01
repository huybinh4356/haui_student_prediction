import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from haui_student_prediction.app_config import load_config
from haui_student_prediction.data.training_dataset import (
    TARGET_COLUMN,
    load_historical_training_dataset,
)
from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.ml.model_registry import ACTIVE_MODEL_VERSION, ModelRegistry
from haui_student_prediction.ml.data_splitting import split_train_test
from haui_student_prediction.ml.xgboost_regressor import train_xgboost_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the HaUI final-exam prediction model using historical labeled data.",
    )
    parser.add_argument(
        "file",
        help="CSV/Excel historical dataset with a final-exam label column.",
    )
    parser.add_argument(
        "--version",
        default=ACTIVE_MODEL_VERSION,
        help="Model version to register.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of records reserved for the test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split.",
    )
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.4:
        parser.error("--test-size phải nằm trong khoảng 0.1 đến 0.4")

    dataframe = load_historical_training_dataset(args.file)
    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    valid = target.notna() & np.isfinite(target.astype(float))
    dataframe = dataframe.loc[valid].copy()
    dataframe[TARGET_COLUMN] = target.loc[valid].astype(float)

    if len(dataframe) < 10:
        raise SystemExit(f"Cần tối thiểu 10 bản ghi có nhãn hợp lệ; hiện có {len(dataframe)}.")

    numeric_candidates = [
        "tx1_score",
        "tx2_score",
        "midterm_score",
        "absence_percentage",
        "lms_learning_percentage",
        "interaction_score",
        "stress_score",
        "motivation_score",
    ]
    categorical_candidates = ["works_part_time", "major", "course_id"]
    numeric_features = [
        column for column in numeric_candidates
        if column in dataframe.columns and dataframe[column].notna().any()
    ]
    categorical_features = [
        column for column in categorical_candidates
        if column in dataframe.columns and dataframe[column].notna().any()
        and dataframe[column].nunique(dropna=True) > 1
    ]

    if not numeric_features and not categorical_features:
        raise SystemExit("Không có đặc trưng hợp lệ để huấn luyện.")

    contract = FeatureContract(
        numerical_features=numeric_features,
        categorical_features=categorical_features,
        target_column=TARGET_COLUMN,
    )
    train_data, test_data = split_train_test(dataframe, args.test_size, args.random_state)
    if len(train_data) < 3 or len(test_data) < 2:
        raise SystemExit("Tập train/test không đủ số bản ghi.")

    model, metrics = train_xgboost_and_evaluate(train_data, test_data, contract)
    print(f"Train: {len(train_data)} rows; Test: {len(test_data)} rows")
    print(f"MAE:  {metrics.mae:.4f}")
    print(f"RMSE: {metrics.rmse:.4f}")
    print(f"R2:   {metrics.r2:.4f}" if np.isfinite(metrics.r2) else "R2:   N/A")

    dataset_hash = hashlib.sha256(
        pd.util.hash_pandas_object(dataframe, index=True).values.tobytes()
    ).hexdigest()
    config = load_config()
    registry = ModelRegistry(str(config.model_registry_path), str(config.model_artifacts_dir))
    registry.save_model(
        model=model,
        version=args.version,
        dataset_hash=dataset_hash,
        metrics=metrics,
        contract=contract,
    )
    print(f"Model saved: {args.version}")
    print(f"Registry: {config.model_registry_path}")
    print(f"Artifacts: {config.model_artifacts_dir}")
    print(f"Verified: {registry.has_model(args.version)}")


if __name__ == "__main__":
    main()
