import pandas as pd
from pathlib import Path
from typing import Optional

class ReportService:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_prediction_results(self, df: pd.DataFrame, filename: str, include_pii: bool = False) -> str:
        export_df = df.copy()
        
        if not include_pii:
            cols_to_drop = [col for col in ['student_id', 'full_name'] if col in export_df.columns]
            export_df = export_df.drop(columns=cols_to_drop)

        output_path = self.output_dir / filename
        export_df.to_csv(output_path, index=False)
        return str(output_path)

    def export_eda_summary(self, summary_data: dict, filename: str) -> str:
        df_summary = pd.DataFrame.from_dict(summary_data, orient='index')
        output_path = self.output_dir / filename
        df_summary.to_csv(output_path, index=True)
        return str(output_path)