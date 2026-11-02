import numpy as np
import pytest

from haui_student_prediction.ml.accuracy_evaluator import compute_all_accuracy_metrics


def test_compute_all_accuracy_metrics():
    y_true = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 4.0, 7.5, 8.5])
    y_pred = np.array([5.2, 5.8, 7.1, 7.9, 8.8, 4.3, 7.4, 8.3])

    result = compute_all_accuracy_metrics(y_true, y_pred)

    assert result.total_samples == 8
    assert result.mae > 0.0
    assert result.rmse > 0.0
    assert result.r2 > 0.8  # high agreement
    assert result.spread_ratio > 0.8
    assert result.beats_baseline is True
    assert not result.grade_band_metrics.empty
    assert not result.all_metrics_table.empty
    assert len(result.predictions_df) == 8

