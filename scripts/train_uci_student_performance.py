import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from haui_student_prediction.data.uci_student_performance import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    UCI_DATASET_DOI,
    UCI_DATASET_LICENSE,
    UCI_DATASET_NAME,
    load_uci_student_performance,
)
from haui_student_prediction.ml.uci_model_comparison import compare_uci_models


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare XGBoost, Random Forest, and CatBoost "
            "on the UCI Student Performance benchmark."
        )
    )
    parser.add_argument(
        "--subject",
        choices=["por", "mat"],
        default="por",
        help="por = Portuguese (649 rows); mat = Mathematics (395 rows).",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional local student-por.csv or student-mat.csv file.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw/uci_student_performance",
        help="Directory used to cache the downloaded UCI ZIP archive.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="outputs/uci_model_comparison",
        help="Directory for the comparison metrics and row-level predictions.",
    )
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.4:
        parser.error("--test-size phải nằm trong khoảng 0.1 đến 0.4")

    data_file = resolve_project_path(args.data_file) if args.data_file else None
    raw_dir = resolve_project_path(args.raw_dir)
    output_dir = resolve_project_path(args.output_dir)

    dataframe = load_uci_student_performance(
        subject=args.subject,
        data_file=data_file,
        raw_dir=raw_dir,
    )
    metrics, predictions = compare_uci_models(
        dataframe,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"uci_{args.subject}_model_metrics.csv"
    predictions_path = output_dir / f"uci_{args.subject}_test_predictions.csv"
    metadata_path = output_dir / f"uci_{args.subject}_training_metadata.txt"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    metadata_path.write_text(
        "\n".join(
            [
                f"dataset={UCI_DATASET_NAME}",
                "dataset_id=320",
                f"subject={args.subject}",
                f"dataset_rows={len(dataframe)}",
                "target=G3 final course grade, normalized from 0-20 to 0-10",
                f"numerical_features={','.join(NUMERICAL_FEATURES)}",
                f"categorical_features={','.join(CATEGORICAL_FEATURES)}",
                f"test_size={args.test_size}",
                f"random_state={args.random_state}",
                f"license={UCI_DATASET_LICENSE}",
                f"doi={UCI_DATASET_DOI}",
                "production_registry=not_updated",
                "note=G3 is a final course grade, not a separately identified final-exam score.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Dataset: {UCI_DATASET_NAME} ({args.subject})")
    print(f"Records: {len(dataframe)}")
    print(f"Features: {len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)}")
    print("Target: G3 final course grade (0-10 scale)")
    print("\nModel comparison:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nBest model by MAE: {metrics.iloc[0]['model']}")
    print(f"Metrics: {metrics_path}")
    print(f"Test predictions: {predictions_path}")
    print(f"Metadata: {metadata_path}")
    print("The UCI benchmark models were not added to the HaUI production model registry.")


if __name__ == "__main__":
    main()
