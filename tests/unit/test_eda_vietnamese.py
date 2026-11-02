import pytest
import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from haui_student_prediction.data.column_mapping import (
    FEATURE_VIETNAMESE_NAMES,
    get_column_vietnamese_label,
)
from haui_student_prediction.eda.distributions import DistributionPlots
from haui_student_prediction.eda.relationships import RelationshipPlots
from haui_student_prediction.eda.statistics import EDAStatistics


@pytest.fixture
def sample_eda_df():
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame({
        "student_id": [f"SV{i:03d}" for i in range(n)],
        "full_name": [f"Student {i}" for i in range(n)],
        "major": ["CNTT" if i % 2 == 0 else "KTPM" for i in range(n)],
        "tx1_score": np.clip(rng.normal(7.0, 1.2, size=n), 0.0, 10.0),
        "midterm_score": np.clip(rng.normal(6.5, 1.5, size=n), 0.0, 10.0),
        "absence_percentage": np.clip(rng.normal(10.0, 8.0, size=n), 0.0, 45.0),
        "lms_learning_percentage": np.clip(rng.normal(55.0, 20.0, size=n), 0.0, 100.0),
        "stress_score": rng.integers(1, 6, size=n),
        "motivation_score": rng.integers(1, 6, size=n),
        "works_part_time": [i % 3 == 0 for i in range(n)],
    })


def test_vietnamese_label_mapping():
    assert get_column_vietnamese_label("tx1_score") == "Điểm thường kỳ 1 (TX1)"
    assert get_column_vietnamese_label("midterm_score") == "Điểm giữa kỳ (GK)"
    assert get_column_vietnamese_label("absence_percentage") == "Tỷ lệ nghỉ học (%)"
    assert get_column_vietnamese_label("lms_learning_percentage") == "Tỷ lệ học LMS (%)"
    assert get_column_vietnamese_label("works_part_time") == "Đi làm thêm"
    assert get_column_vietnamese_label("unknown_col") == "unknown_col"


def test_distribution_plots_vietnamese(sample_eda_df):
    plots = DistributionPlots()

    # 1. Histogram
    fig_hist = plots.plot_histogram(sample_eda_df, "tx1_score")
    assert isinstance(fig_hist, Figure)
    ax = fig_hist.axes[0]
    assert "Điểm thường kỳ 1 (TX1)" in ax.get_title()
    assert "Điểm thường kỳ 1 (TX1)" in ax.get_xlabel()

    # 2. Count plot
    fig_count = plots.plot_count(sample_eda_df, "major")
    assert isinstance(fig_count, Figure)
    ax_count = fig_count.axes[0]
    assert "Ngành học" in ax_count.get_title()

    # 3. KDE Comparison
    fig_kde = plots.plot_kde_comparison(sample_eda_df, "tx1_score", "major")
    assert isinstance(fig_kde, Figure)

    # 4. Violin plot
    fig_violin = plots.plot_violin(sample_eda_df, "major", "tx1_score")
    assert isinstance(fig_violin, Figure)


def test_relationship_plots_vietnamese(sample_eda_df):
    plots = RelationshipPlots()

    # 1. Scatter plot
    fig_scatter = plots.plot_scatter(sample_eda_df, "tx1_score", "midterm_score")
    assert isinstance(fig_scatter, Figure)
    ax = fig_scatter.axes[0]
    assert "Điểm thường kỳ 1 (TX1)" in ax.get_xlabel()
    assert "Điểm giữa kỳ (GK)" in ax.get_ylabel()

    # 2. Boxplot
    fig_box = plots.plot_boxplot(sample_eda_df, "major", "midterm_score")
    assert isinstance(fig_box, Figure)

    # 3. Correlation heatmap
    fig_heat = plots.plot_correlation_heatmap(sample_eda_df)
    assert isinstance(fig_heat, Figure)
    ax_heat = fig_heat.axes[0]
    # Check that tick labels on heatmap are in Vietnamese
    xticklabels = [t.get_text() for t in ax_heat.get_xticklabels()]
    assert any("Điểm" in t or "Tỷ lệ" in t for t in xticklabels)


def test_detailed_statistics_and_pedagogical_insights(sample_eda_df):
    stats = EDAStatistics()

    # 1. Detailed dataframe summary
    summary_df = stats.get_detailed_dataframe_summary(sample_eda_df)
    assert not summary_df.empty
    assert "Chỉ số học tập" in summary_df.columns
    assert "Trung bình" in summary_df.columns
    assert "Độ lệch chuẩn" in summary_df.columns
    assert "Trung vị (50%)" in summary_df.columns

    # 2. Pedagogical insights
    insights = stats.generate_pedagogical_insights(sample_eda_df)
    assert len(insights) >= 4
    assert any("Quy mô dữ liệu" in ins for ins in insights)
    assert any("Điểm giữa kỳ" in ins for ins in insights)

