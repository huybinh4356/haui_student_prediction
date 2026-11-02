from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from haui_student_prediction.data.column_mapping import get_column_vietnamese_label


class DistributionPlots:
    def __init__(self):
        self.excluded = ["student_id", "full_name"]
        sns.set_theme(style="whitegrid", palette="Blues_r")

    def _format_dataframe_for_display(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        display_df = df.copy()
        if "works_part_time" in cols and "works_part_time" in display_df.columns:
            display_df["works_part_time"] = display_df["works_part_time"].map(
                lambda val: "Có đi làm thêm" if bool(val) is True or str(val).lower() in ("true", "1", "có") else "Không đi làm thêm"
            )
        return display_df

    def plot_histogram(self, df: pd.DataFrame, column: str) -> Figure:
        if column in self.excluded or column not in df.columns:
            raise ValueError("Cột dữ liệu không hợp lệ hoặc thuộc danh sách loại trừ.")

        col_label = get_column_vietnamese_label(column)
        fig, ax = plt.subplots(figsize=(8, 5.5))

        data_series = pd.to_numeric(df[column], errors="coerce").dropna()
        sns.histplot(data=data_series, kde=True, ax=ax, color="#1D4ED8", edgecolor="white", alpha=0.7)

        mean_val = float(data_series.mean()) if not data_series.empty else 0.0
        median_val = float(data_series.median()) if not data_series.empty else 0.0

        ax.axvline(mean_val, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Trung bình: {mean_val:.2f}")
        ax.axvline(median_val, color="#16A34A", linestyle=":", linewidth=1.8, label=f"Trung vị: {median_val:.2f}")

        ax.set_title(f"Biểu đồ phân phối tần số: {col_label}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(col_label, fontsize=11, fontweight="medium")
        ax.set_ylabel("Số lượng sinh viên", fontsize=11, fontweight="medium")
        ax.legend(frameon=True, facecolor="white", framealpha=0.9)
        fig.tight_layout()
        return fig

    def plot_count(self, df: pd.DataFrame, column: str) -> Figure:
        if column in self.excluded or column not in df.columns:
            raise ValueError("Cột dữ liệu không hợp lệ hoặc thuộc danh sách loại trừ.")

        col_label = get_column_vietnamese_label(column)
        display_df = self._format_dataframe_for_display(df, [column])

        fig, ax = plt.subplots(figsize=(8, 5.5))
        order = display_df[column].value_counts().index
        sns.countplot(data=display_df, x=column, order=order, ax=ax, hue=column, palette="Blues_d", legend=False)

        total = len(display_df[column].dropna())
        for p in ax.patches:
            height = p.get_height()
            if np.isfinite(height) and height > 0:
                pct = (height / total) * 100 if total > 0 else 0
                ax.annotate(
                    f"{int(height):,}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    xytext=(0, 3),
                    textcoords="offset points",
                )

        ax.set_title(f"Biểu đồ cơ cấu số lượng sinh viên theo: {col_label}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(col_label, fontsize=11, fontweight="medium")
        ax.set_ylabel("Số lượng sinh viên", fontsize=11, fontweight="medium")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def plot_kde_comparison(self, df: pd.DataFrame, num_col: str, group_col: str) -> Figure:
        """So sánh phân phối mật độ hạt nhân (KDE) của biến số theo nhóm đối tượng."""
        if num_col in self.excluded or group_col in self.excluded:
            raise ValueError("Cột dữ liệu không hợp lệ.")

        num_label = get_column_vietnamese_label(num_col)
        group_label = get_column_vietnamese_label(group_col)
        display_df = self._format_dataframe_for_display(df, [num_col, group_col])

        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.kdeplot(
            data=display_df,
            x=num_col,
            hue=group_col,
            common_norm=False,
            fill=True,
            alpha=0.35,
            ax=ax,
            palette="Set2",
        )
        ax.set_title(f"So sánh phân phối {num_label} theo {group_label}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(num_label, fontsize=11)
        ax.set_ylabel("Mật độ xác suất", fontsize=11)
        fig.tight_layout()
        return fig

    def plot_violin(self, df: pd.DataFrame, cat_col: str, num_col: str) -> Figure:
        """Biểu đồ đàn vĩ cầm (Violin plot) kết hợp phân vị."""
        if cat_col in self.excluded or num_col in self.excluded:
            raise ValueError("Cột dữ liệu không hợp lệ.")

        cat_label = get_column_vietnamese_label(cat_col)
        num_label = get_column_vietnamese_label(num_col)
        display_df = self._format_dataframe_for_display(df, [cat_col, num_col])

        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.violinplot(data=display_df, x=cat_col, y=num_col, inner="quartile", hue=cat_col, palette="Pastel1", legend=False, ax=ax)
        ax.set_title(f"Phân bố hình học và phân vị {num_label} theo {cat_label}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(cat_label, fontsize=11)
        ax.set_ylabel(num_label, fontsize=11)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig
