import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from haui_student_prediction.risk.risk_calculator import RiskCalculator
from haui_student_prediction.services.risk_prediction_service import RiskPredictionService


def main():
    parser = argparse.ArgumentParser(description="Export risk assessment report")
    parser.add_argument("file", help="Path to CSV/Excel with predictions")
    parser.add_argument("--output", default="outputs/reports/risk_report.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.file) if args.file.endswith(".csv") else pd.read_excel(args.file)
    service = RiskPredictionService(RiskCalculator())

    results = []
    for _, row in df.iterrows():
        try:
            r = service.evaluate_student_risk(row)
            results.append({
                "student_id": row.get("student_id"),
                "absence_status": r.absence_status,
                "risk_score": round(r.weighted_score, 2),
                "risk_category": r.risk_category,
                "override": r.is_override,
            })
        except ValueError as e:
            results.append({
                "student_id": row.get("student_id"),
                "error": str(e),
            })

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False)
    print(f"Report saved to {args.output} ({len(out)} rows)")


if __name__ == "__main__":
    main()
