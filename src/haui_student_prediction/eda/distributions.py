import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure


class DistributionPlots:
    def __init__(self):
        self.excluded = ["student_id", "full_name"]

    def plot_histogram(self, df: pd.DataFrame, column: str) -> Figure:
        if column in self.excluded or column not in df.columns:
            raise ValueError("Invalid or excluded column")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df, x=column, kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        fig.tight_layout()
        return fig

    def plot_count(self, df: pd.DataFrame, column: str) -> Figure:
        if column in self.excluded or column not in df.columns:
            raise ValueError("Invalid or excluded column")

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=df, x=column, ax=ax)
        ax.set_title(f"Count of {column}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        return fig
