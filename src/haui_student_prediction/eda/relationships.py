import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RelationshipPlots:
    def __init__(self):
        self.excluded = ["student_id", "full_name"]

    def plot_scatter(self, df: pd.DataFrame, x_col: str, y_col: str) -> plt.Figure:
        if x_col in self.excluded or y_col in self.excluded:
            raise ValueError("Invalid or excluded column")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        ax.set_title(f"{y_col} vs {x_col}")
        fig.tight_layout()
        return fig

    def plot_boxplot(self, df: pd.DataFrame, cat_col: str, num_col: str) -> plt.Figure:
        if cat_col in self.excluded or num_col in self.excluded:
            raise ValueError("Invalid or excluded column")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
        ax.set_title(f"{num_col} by {cat_col}")
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        return fig