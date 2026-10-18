import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from haui_student_prediction.ml.preprocessing import build_preprocessing_pipeline
from dataclasses import dataclass

@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    r2: float

def train_and_evaluate(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    target_col: str, 
    num_cols: list[str], 
    cat_cols: list[str]
) -> tuple[Pipeline, ModelMetrics]:
    X_train = df_train[num_cols + cat_cols]
    y_train = df_train[target_col]
    X_test = df_test[num_cols + cat_cols]
    y_test = df_test[target_col]

    preprocessor = build_preprocessing_pipeline(num_cols, cat_cols)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)
    
    raw_predictions = model.predict(X_test)
    if isinstance(raw_predictions, tuple):
        predictions = raw_predictions[0]
    else:
        predictions = raw_predictions
        
    pred_array = np.asarray(predictions)
    
    metrics = ModelMetrics(
        mae=float(mean_absolute_error(y_test, pred_array)),
        rmse=float(root_mean_squared_error(y_test, pred_array)),
        r2=float(r2_score(y_test, pred_array))
    )
    
    return model, metrics