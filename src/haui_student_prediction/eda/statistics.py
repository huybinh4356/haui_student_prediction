from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any

from haui_student_prediction.data.column_mapping import get_column_vietnamese_label


class EDAStatistics:
    def __init__(self):
        self.excluded_columns = ["student_id", "full_name"]

    def get_numeric_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        cols_to_analyze = [c for c in df.columns if c not in self.excluded_columns]
        df_filtered = df[cols_to_analyze]
        raw_dict = df_filtered.describe().to_dict()
        return {str(k): v for k, v in raw_dict.items()}

    def get_missingness_report(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {str(col): 0.0 for col in df.columns}
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        return {str(k): float(v) for k, v in missing_percentages.items()}

    def get_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_analyze = [c for c in df.columns if c not in self.excluded_columns]
        df_numeric = df[cols_to_analyze].select_dtypes(include=["number"])
        return df_numeric.corr()

    def get_detailed_dataframe_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo bảng thống kê mô tả chuyên sâu đa chỉ số dạng ngang với tiêu đề Tiếng Việt."""
        excluded = set(self.excluded_columns)
        numeric_cols = [c for c in df.select_dtypes(include="number").columns if c not in excluded]

        rows = []
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.dropna()
            total = len(df)
            valid_count = len(valid)
            missing_pct = ((total - valid_count) / total * 100) if total > 0 else 0.0

            if valid_count > 0:
                mean_val = valid.mean()
                std_val = valid.std() if valid_count > 1 else 0.0
                min_val = valid.min()
                q25_val = valid.quantile(0.25)
                median_val = valid.median()
                q75_val = valid.quantile(0.75)
                max_val = valid.max()
                skew_val = valid.skew() if valid_count > 2 else 0.0
            else:
                mean_val = std_val = min_val = q25_val = median_val = q75_val = max_val = skew_val = 0.0

            rows.append(
                {
                    "Chỉ số học tập": get_column_vietnamese_label(col),
                    "Mã biến": col,
                    "Số lượng": f"{valid_count:,}",
                    "Trung bình": f"{mean_val:.2f}",
                    "Độ lệch chuẩn": f"{std_val:.2f}",
                    "Nhỏ nhất": f"{min_val:.1f}",
                    "Q1 (25%)": f"{q25_val:.1f}",
                    "Trung vị (50%)": f"{median_val:.1f}",
                    "Q3 (75%)": f"{q75_val:.1f}",
                    "Lớn nhất": f"{max_val:.1f}",
                    "Độ xiên (Skew)": f"{skew_val:+.2f}",
                    "Khuyết thiếu (%)": f"{missing_pct:.1f}%",
                }
            )

        return pd.DataFrame(rows)

    def generate_pedagogical_insights(self, df: pd.DataFrame) -> list[str]:
        """Tự động phân tích và đưa ra các nhận định sư phạm chi tiết từ bộ dữ liệu."""
        insights = []
        n = len(df)
        if n == 0:
            return ["Chưa có dữ liệu để phân tích."]

        # 1. Tổng quan quy mô
        insights.append(f"Quy mô dữ liệu: Đã phân tích {n:,} hồ sơ sinh viên.")

        # 2. Phân tích điểm học thuật
        if "midterm_score" in df.columns:
            mid = pd.to_numeric(df["midterm_score"], errors="coerce").dropna()
            if not mid.empty:
                good_count = (mid >= 7.0).sum()
                bad_count = (mid < 4.0).sum()
                insights.append(
                    f"Điểm giữa kỳ: Trung bình {mid.mean():.2f}/10. "
                    f"Có {good_count} bạn ({good_count/len(mid)*100:.1f}%) đạt khá giỏi (≥7.0), "
                    f"và {bad_count} bạn ({bad_count/len(mid)*100:.1f}%) có nguy cơ thi lại (<4.0)."
                )

        # 3. Phân tích chuyên cần & quy chế vắng
        if "absence_percentage" in df.columns:
            abs_s = pd.to_numeric(df["absence_percentage"], errors="coerce").dropna()
            if not abs_s.empty:
                over_30 = (abs_s > 30.0).sum()
                warning_20 = ((abs_s > 20.0) & (abs_s <= 30.0)).sum()
                if over_30 > 0:
                    insights.append(
                        f"Cảnh báo quy chế vắng học HaUI: Có {over_30} sinh viên ({over_30/len(abs_s)*100:.1f}%) "
                        f"nghỉ học quá 30% số tiết (buộc phải học lại). Có {warning_20} bạn đang ở mức chạm ngưỡng nguy cơ (20-30%)."
                    )
                else:
                    insights.append(
                        f"Tỷ lệ chuyên cần tốt: Toàn bộ sinh viên đều duy trì tỷ lệ nghỉ học dưới ngưỡng 30% theo quy chế."
                    )

        # 4. Phân tích tự học trực tuyến LMS
        if "lms_learning_percentage" in df.columns:
            lms_s = pd.to_numeric(df["lms_learning_percentage"], errors="coerce").dropna()
            if not lms_s.empty:
                low_lms = (lms_s < 30.0).sum()
                insights.append(
                    f"Học tập trên LMS: Mức độ tương tác LMS trung bình đạt {lms_s.mean():.1f}%. "
                    f"Tuy nhiên có {low_lms} sinh viên ({low_lms/len(lms_s)*100:.1f}%) ít học trực tuyến (<30%)."
                )

        # 5. Phân tích đi làm thêm
        if "works_part_time" in df.columns:
            wpt_mask = df["works_part_time"].astype(str).str.lower().isin(["true", "1", "có", "yes"])
            wpt_count = wpt_mask.sum()
            insights.append(
                f"Tình hình đi làm thêm: Có {wpt_count} sinh viên ({wpt_count/n*100:.1f}%) đi làm thêm ngoài giờ học."
            )

        # 6. Mối liên hệ tâm lý - học tập
        if "stress_score" in df.columns and "motivation_score" in df.columns:
            stress = pd.to_numeric(df["stress_score"], errors="coerce").dropna()
            mot = pd.to_numeric(df["motivation_score"], errors="coerce").dropna()
            if not stress.empty and not mot.empty:
                insights.append(
                    f"Chỉ số tâm lý học đường: Động lực học tập bình quân đạt {mot.mean():.2f}/5; "
                    f"mức độ stress trung bình {stress.mean():.2f}/5."
                )

        return insights
