import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from haui_student_prediction.eda.charts import ChartService


def main():
    parser = argparse.ArgumentParser(description="Run EDA on a dataset")
    parser.add_argument("file", help="Path to CSV or Excel file")
    parser.add_argument("--output", default="outputs/eda", help="Output directory for charts")
    args = parser.parse_args()

    df = pd.read_csv(args.file) if args.file.endswith(".csv") else pd.read_excel(args.file)
    service = ChartService(output_dir=args.output)

    print(f"Dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")

    stats = service.statistics.get_numeric_summary(df)
    print("\nNumeric summary:")
    for col, values in stats.items():
        print(f"  {col}: mean={values.get('mean', 'N/A'):.2f}")

    missing = service.statistics.get_missingness_report(df)
    print("\nMissingness (%):")
    for col, pct in missing.items():
        if pct > 0:
            print(f"  {col}: {pct:.1f}%")

    print(f"\nCharts saved to: {args.output}")


if __name__ == "__main__":
    main()
