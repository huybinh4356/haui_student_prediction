import numpy as np
import pandas as pd

from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.ml.model_registry import ACTIVE_MODEL_VERSION
from haui_student_prediction.ml.xgboost_regressor import (
    XGBOOST_MODEL_VERSION,
    train_xgboost_and_evaluate,
)


def test_xgboost_regressor_trains_evaluates_and_predicts():
    values = np.arange(30, dtype=float)
    dataframe = pd.DataFrame(
        {
            "tx1_score": 3.0 + values * 0.2,
            "midterm_score": 4.0 + values * 0.15,
            "actual_final_exam_score": np.clip(2.5 + values * 0.18, 0.0, 10.0),
        }
    )
    contract = FeatureContract(
        numerical_features=["tx1_score", "midterm_score"],
        categorical_features=[],
        target_column="actual_final_exam_score",
    )

    model, metrics = train_xgboost_and_evaluate(
        dataframe.iloc[:24].copy(),
        dataframe.iloc[24:].copy(),
        contract,
    )

    predictions = model.predict(dataframe[["tx1_score", "midterm_score"]])

    assert XGBOOST_MODEL_VERSION == ACTIVE_MODEL_VERSION
    assert predictions.shape == (30,)
    assert np.isfinite(predictions).all()
    assert np.isfinite(metrics.mae)
    assert np.isfinite(metrics.rmse)
    assert np.isfinite(metrics.r2)
    assert metrics.mae >= 0
    assert metrics.rmse >= 0
