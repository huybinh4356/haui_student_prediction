import pandas as pd
import pytest

from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.ml.model_comparison import run_model_benchmark


def test_model_benchmark_runs_and_evaluates_all_algorithms(tmp_path):
    # Create small synthetic test dataset with real signal
    rows = []
    for i in range(40):
        tx1 = 5.0 + (i % 6) * 0.8
        gk = 4.0 + (i % 7) * 0.8
        lms = 50.0 + (i % 10) * 5.0
        absence = (i % 5) * 5.0
        final = 0.4 * gk + 0.3 * tx1 + 0.01 * lms - 0.02 * absence
        rows.append({
            "student_id": f"TEST_{i:03d}",
            "tx1_score": tx1,
            "midterm_score": gk,
            "lms_learning_percentage": lms,
            "absence_percentage": absence,
            "works_part_time": i % 2 == 0,
            "major": "IT" if i % 2 == 0 else "SE",
            "course_id": "IT101",
            "target_final_exam_score": round(final, 1),
        })

    df = pd.DataFrame(rows)
    contract = FeatureContract(
        numerical_features=["tx1_score", "midterm_score", "lms_learning_percentage", "absence_percentage"],
        categorical_features=["works_part_time", "major", "course_id"],
        target_column="target_final_exam_score",
    )

    result = run_model_benchmark(df, contract, test_size=0.2, random_state=42, cv_folds=3)

    assert not result.model_metrics.empty
    expected_models = {"Baseline (Mean)", "Ridge", "Lasso", "Random Forest", "XGBoost", "CatBoost"}
    actual_models = set(result.model_metrics["model_name"].tolist())
    assert expected_models.issubset(actual_models)

    assert not result.cross_validation_results.empty
    assert not result.test_predictions.empty
    assert not result.residual_analysis.empty
    assert not result.feature_importance.empty

    # Top models must beat baseline on data with strong signal
    baseline_mae = result.model_metrics.loc[
        result.model_metrics["model_name"] == "Baseline (Mean)", "test_mae"
    ].iloc[0]
    ridge_mae = result.model_metrics.loc[
        result.model_metrics["model_name"] == "Ridge", "test_mae"
    ].iloc[0]
    assert ridge_mae < baseline_mae

