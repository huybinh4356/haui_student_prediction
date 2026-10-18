import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np

from haui_student_prediction.app_config import load_config
from haui_student_prediction.data.column_mapping import get_default_column_mapping
from haui_student_prediction.data.cleaning import DataCleaner
from haui_student_prediction.data.validator import DataValidator
from haui_student_prediction.data.ingestion_service import IngestionService
from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.ml.train_regressor import train_and_evaluate, ModelMetrics
from haui_student_prediction.ml.model_registry import ModelRegistry
from haui_student_prediction.config.course_profiles import DEFAULT_COURSES_WITH_TX2


def load_and_validate_dataset(file_path: str) -> pd.DataFrame:
    cleaner = DataCleaner(get_default_column_mapping())
    validator = DataValidator()
    ingestion = IngestionService(cleaner, validator)

    required_cols = [
        "student_id", "absence_percentage", "tx1_score",
        "midterm_score", "actual_final_exam_score",
    ]
    result = ingestion.load_and_validate(
        file_path, required_cols, DEFAULT_COURSES_WITH_TX2,
    )

    if not result.is_success:
        print("ERROR: Data validation failed.")
        if result.missing_columns:
            print(f"  Missing columns: {result.missing_columns}")
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
        sys.exit(1)

    if result.data is None:
        print("ERROR: No data returned from ingestion.")
        sys.exit(1)

    return result.data


def split_train_test(
    df: pd.DataFrame, test_size: float, random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    n_test = max(1, int(len(df) * test_size))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Train final exam prediction model from labeled data",
    )
    parser.add_argument(
        "file",
        help="Path to CSV/Excel with actual_final_exam_score labels",
    )
    parser.add_argument(
        "--version", default="1.0.0",
        help="Model version to register",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for test set (default 0.2)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for train/test split",
    )
    args = parser.parse_args()

    config = load_config()

    df = load_and_validate_dataset(args.file)

    if "actual_final_exam_score" not in df.columns:
        print(
            "ERROR: Target column 'actual_final_exam_score' not found. "
            "Cannot train without historical labels.",
        )
        sys.exit(1)

    n_total = len(df)
    n_valid_target = df["actual_final_exam_score"].notna().sum()
    print(f"Loaded {n_total} rows, {n_valid_target} with target labels")

    if n_valid_target < 10:
        print(
            "ERROR: Need at least 10 rows with target labels. "
            f"Only {n_valid_target} available.",
        )
        sys.exit(1)

    cat_cols = ["works_part_time"] if "works_part_time" in df.columns else []
    num_cols = [
        c for c in [
            "tx1_score", "midterm_score",
            "absence_percentage", "lms_learning_percentage",
            "interaction_score", "stress_score", "motivation_score",
        ]
        if c in df.columns
    ]

    if len(num_cols) == 0:
        print("ERROR: No numerical features available for training.")
        sys.exit(1)

    contract = FeatureContract(
        numerical_features=num_cols,
        categorical_features=cat_cols,
        target_column="actual_final_exam_score",
    )

    df_train, df_test = split_train_test(
        df, args.test_size, args.random_state,
    )
    print(f"Train: {len(df_train)} rows, Test: {len(df_test)} rows")

    if len(df_train) < 3:
        print("ERROR: Training set too small after split.")
        sys.exit(1)

    model, metrics = train_and_evaluate(df_train, df_test, contract)

    print(f"MAE:  {metrics.mae:.4f}")
    print(f"RMSE: {metrics.rmse:.4f}")
    if np.isnan(metrics.r2):
        print("R2:   N/A (test set too small)")
    else:
        print(f"R2:   {metrics.r2:.4f}")

    dataset_hash = str(pd.util.hash_pandas_object(df).sum())

    registry = ModelRegistry(
        str(config.model_registry_path),
        str(config.model_artifacts_dir),
    )
    registry.save_model(
        model=model,
        version=args.version,
        dataset_hash=dataset_hash,
        metrics=metrics,
        contract=contract,
    )

    print(f"Model saved as version {args.version}")
    print(f"Registry: {config.model_registry_path}")
    print(f"Artifacts: {config.model_artifacts_dir}")

    verify = registry.has_model(args.version)
    print(f"Verification: has_model('{args.version}') = {verify}")


if __name__ == "__main__":
    main()
