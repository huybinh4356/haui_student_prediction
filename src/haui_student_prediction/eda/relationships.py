from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from haui_student_prediction.data.column_mapping import get_column_vietnamese_label


class RelationshipPlots:
    def __init__(self):
        self.excluded = ["student_id", "full_name"]
        sns.set_theme(style="whitegrid")

    def _format_dataframe_for_display(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        display_df = df.copy()
        if "works_part_time" in cols and "works_part_time" in display_df.columns:
            display_df["works_part_time"] = display_df["works_part_time"].map(
                lambda val: "Có đi làm thêm" if bool(val) is True or str(val).lower() in ("true", "1", "có") else "Không đi làm thêm"
            )
        return display_df

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str) -> Figure:
        if x_col in self.excluded or y_col in self.excluded:
            raise ValueError("Cột dữ liệu không hợp lệ hoặc thuộc danh sách loại trừ.")

        x_label = get_column_vietnamese_label(x_col)
        y_label = get_column_vietnamese_label(y_col)

        fig, ax = plt.subplots(figsize=(8.5, 6))

        valid_data = df[[x_col, y_col]].dropna()
        x_vals = pd.to_numeric(valid_data[x_col], errors="coerce")
        y_vals = pd.to_numeric(valid_data[y_col], errors="coerce")
        mask = x_vals.notna() & y_vals.notna()

        # Scatter plot with regression line
        sns.regplot(
            x=x_vals[mask],
            y=y_vals[mask],
            ax=ax,
            scatter_kws={"alpha": 0.55, "color": "#2563EB", "s": 38},
            line_kws={"color": "#DC2626", "linewidth": 2, "label": "Đường xu hướng (Hồi quy)"},
        )

        corr = float(np.corrcoef(x_vals[mask], y_vals[mask])[0, 1]) if mask.sum() > 2 else 0.0
        corr_desc = (
            "Tương quan rất mạnh" if abs(corr) >= 0.7
            else "Tương quan khá mạnh" if abs(corr) >= 0.5
            else "Tương quan trung bình" if abs(corr) >= 0.3
            else "Tương quan yếu / Không rõ rệt"
        )
        sign = "thuận" if corr > 0 else "nghịch"

        info_box = f"Hệ số tương quan r = {corr:+.2f}\n({corr_desc} {sign})"
        ax.text(
            0.05, 0.92, info_box,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1", alpha=0.95),
        )

        ax.set_title(f"Tương quan phân tán: {y_label} vs {x_label}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(x_label, fontsize=11, fontweight="medium")
        ax.set_ylabel(y_label, fontsize=11, fontweight="medium")
        ax.legend(loc="lower right", frameon=True, facecolor="white")
        fig.tight_layout()
        return fig

    def plot_boxplot(self, df: pd.DataFrame, cat_col: str, num_col: str) -> Figure:
        if cat_col in self.excluded or num_col in self.excluded:
            raise ValueError("Cột dữ liệu không hợp lệ hoặc thuộc danh sách loại trừ.")

        cat_label = get_column_vietnamese_label(cat_col)
        num_label = get_column_vietnamese_label(num_col)
        display_df = self._format_dataframe_for_display(df, [cat_col, num_col])

        fig, ax = plt.subplots(figsize=(9.5, 6))
        sns.boxplot(
            data=display_df,
            x=cat_col,
            y=num_col,
            ax=ax,
            hue=cat_col,
            palette="Blues",
            legend=False,
            boxprops=dict(alpha=0.8),
            flierprops=dict(marker="o", markerfacecolor="#EF4444", markersize=5),
        )

        # Overlay stripplot for sample distribution
        if len(display_df) <= 400:
            sns.stripplot(data=display_df, x=cat_col, y=num_col, color="#1E293B", alpha=0.35, size=4, jitter=0.2, ax=ax)

        ax.set_title(f"Phân vị & Điểm ngoại lai của {num_label} theo {cat_label}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel(cat_label, fontsize=11, fontweight="medium")
        ax.set_ylabel(num_label, fontsize=11, fontweight="medium")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        return fig

    def plot_correlation_heatmap(self, df: pd.DataFrame) -> Figure:
        """Ma trận tương quan với các biến số được dịch sang Tiếng Việt."""
        excluded = set(self.excluded)
        num_cols = [c for c in df.select_dtypes(include="number").columns if c not in excluded]
        if len(num_cols) < 2:
            raise ValueError("Cần ít nhất 2 biến số để vẽ ma trận tương quan.")

        corr_matrix = df[num_cols].corr()

        # Rename rows and columns to Vietnamese labels
        vn_labels = [get_column_vietnamese_label(c) for c in num_cols]
        corr_matrix.index = vn_labels
        corr_matrix.columns = vn_labels

        fig, ax = plt.subplots(figsize=(10, 7.5))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            linewidths=0.7,
            linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Hệ số tương quan Pearson (r)"},
            ax=ax,
        )

        ax.set_title("Ma trận hệ số tương quan đa biến giữa các yếu tố học tập", fontsize=13, fontweight="bold", pad=14)
        ax.tick_params(axis="x", rotation=35)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        return fig
