import pandas as pd
import numpy as np
from haui_student_prediction.ml.model_registry import ModelRegistry
from haui_student_prediction.scoring.final_course_score import ScoringService
from haui_student_prediction.features.feature_builder import FeatureBuilder
from haui_student_prediction.features.feature_contract import FeatureContract

class PredictionService:
    def __init__(self, registry: ModelRegistry, scoring_service: ScoringService):
        self.registry = registry
        self.scoring_service = scoring_service

    def predict_student_results(
        self, 
        student_records: pd.DataFrame, 
        model_version: str,
        course_profile_id: str
    ) -> pd.DataFrame:
        model, metadata = self.registry.load_model(model_version)
        
        contract = FeatureContract(**metadata["features"])
        builder = FeatureBuilder(contract)
        
        X_infer = builder.extract_features(student_records, is_training=False)
        
        raw_predictions = model.predict(X_infer)
        if isinstance(raw_predictions, tuple):
            predictions = raw_predictions[0]
        else:
            predictions = raw_predictions
            
        results_df = student_records.copy()
        results_df['predicted_final_exam_score'] = np.asarray(predictions)
        
        course_totals = []
        for _, row in results_df.iterrows():
            tx1 = float(row['tx1_score'])
            midterm = float(row['midterm_score'])
            tx2 = row.get('tx2_score', None)
            pred_final = float(row['predicted_final_exam_score'])
            
            tx2_val = float(tx2) if pd.notna(tx2) else None

            course_total = self.scoring_service.calculate_score(
                profile_id=course_profile_id,
                tx1=tx1,
                midterm=midterm,
                predicted_final=pred_final,
                tx2=tx2_val
            )
            course_totals.append(course_total)
            
        results_df['predicted_course_score'] = course_totals
        return results_df