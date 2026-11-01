import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from haui_student_prediction.data.column_mapping import get_default_column_mapping
from haui_student_prediction.data.cleaning import DataCleaner
from haui_student_prediction.data.validator import DataValidator
from haui_student_prediction.data.ingestion_service import IngestionService


def main():
    parser = argparse.ArgumentParser(description="Validate a dataset file")
    parser.add_argument("file", help="Path to CSV or Excel file")
    parser.add_argument("--required-cols", nargs="+", default=["student_id", "absence_percentage", "tx1_score", "midterm_score"])
    parser.add_argument("--courses-tx2", nargs="+", default=[])
    args = parser.parse_args()

    cleaner = DataCleaner(get_default_column_mapping())
    validator = DataValidator()
    service = IngestionService(cleaner, validator)
    result = service.load_and_validate(args.file, args.required_cols, args.courses_tx2)

    print(f"Success: {result.is_success}")
    print(f"Rows: {len(result.data) if result.data is not None else 0}")
    print(f"Missing columns: {result.missing_columns}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Duplicates: {len(result.duplicate_indices)}")
    if result.errors:
        for err in result.errors[:10]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
