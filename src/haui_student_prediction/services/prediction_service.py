import pandas as pd
import numpy as np
from haui_student_prediction.ml.model_registry import ACTIVE_MODEL_VERSION, ModelRegistry
from haui_student_prediction.scoring.final_course_score import ScoringService
from haui_student_prediction.features.feature_builder import FeatureBuilder
from haui_student_prediction.features.feature_contract import FeatureContract
from haui_student_prediction.data.column_mapping import is_training_target_column


class PredictionService:
    """Orchestrates model loading, feature extraction, prediction, and course
    score computation.

    Requirement #7: when no model is registered / no artifact exists, this
    service returns a clear business-domain error -- it never fabricates
    predictions.

    Requirement #6: prediction input schema must match the training schema.
    Target and identifier columns are blocked from features even if the caller
    mistakenly passes them in the DataFrame.
    """

    def __init__(self, registry: ModelRegistry, scoring_service: ScoringService):
        self.registry = registry
        self.scoring_service = scoring_service

    def is_model_ready(self, model_version: str = ACTIVE_MODEL_VERSION) -> bool:
        """True iff a model artifact + metadata are available for the version."""
        return self.registry.has_model(model_version)

    def predict_student_results(
        self,
        student_records: pd.DataFrame,
        model_version: str,
        course_profile_id: str,
    ) -> pd.DataFrame:
        # --- Requirement #7: fail loudly when no model is available ---
        if not self.is_model_ready(model_version):
            raise RuntimeError(
                f"Model version '{model_version}' is not ready. "
                "No model artifact has been trained or registered. "
                "Train a model before requesting predictions."
            )

        if student_records is None or len(student_records) == 0:
            raise ValueError("Student records DataFrame is empty")

        model, metadata = self.registry.load_model(model_version)

        # Reconstruct the contract stored with the model.
        features_meta = metadata.get("features")
        if not features_meta:
            raise ValueError(
                f"Registered model '{model_version}' has no feature metadata; "
                "cannot determine prediction schema"
            )
        contract = FeatureContract(**features_meta)
        builder = FeatureBuilder(contract)
        excluded_columns = [
            column for column in student_records.columns
            if is_training_target_column(column)
        ]
        prediction_input = student_records.drop(columns=excluded_columns, errors="ignore").copy()

        X_infer = builder.extract_features(prediction_input, is_training=False)

        raw_predictions = model.predict(X_infer)
        if isinstance(raw_predictions, tuple):
            predictions = raw_predictions[0]
        else:
            predictions = raw_predictions

        results_df = prediction_input.copy()
        results_df["predicted_final_exam_score"] = np.asarray(predictions)

        # --- Course score computation (unchanged logic) ---
        course_totals = []
        for _, row in results_df.iterrows():
            tx1 = float(row["tx1_score"])
            midterm = float(row["midterm_score"])
            tx2 = row.get("tx2_score", None)
            pred_final = float(row["predicted_final_exam_score"])

            tx2_val = float(tx2) if pd.notna(tx2) else None

            course_total = self.scoring_service.calculate_score(
                profile_id=course_profile_id,
                tx1=tx1,
                midterm=midterm,
                predicted_final=pred_final,
                tx2=tx2_val,
            )
            course_totals.append(course_total)

        results_df["predicted_course_score"] = course_totals
        return results_df
