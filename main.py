import sys
from PyQt6.QtWidgets import QApplication
from haui_student_prediction.app_config import load_config

from haui_student_prediction.data.column_mapping import get_default_column_mapping
from haui_student_prediction.data.cleaning import DataCleaner
from haui_student_prediction.data.validator import DataValidator
from haui_student_prediction.data.ingestion_service import IngestionService
from haui_student_prediction.services.dataset_service import DatasetService

from haui_student_prediction.eda.charts import ChartService

from haui_student_prediction.scoring.final_course_score import ScoringService
from haui_student_prediction.config.course_profiles import DEFAULT_COURSE_PROFILES
from haui_student_prediction.ml.model_registry import ModelRegistry
from haui_student_prediction.services.prediction_service import PredictionService

from haui_student_prediction.risk.risk_calculator import RiskCalculator
from haui_student_prediction.services.risk_prediction_service import RiskPredictionService

from haui_student_prediction.ui.main_window import MainWindow

def main():
    config = load_config()

    cleaner = DataCleaner(get_default_column_mapping())
    validator = DataValidator()
    ingestion_service = IngestionService(cleaner, validator)
    dataset_service = DatasetService(ingestion_service)

    chart_service = ChartService(str(config.report_output_dir))

    profiles = {
        k: v.to_scoring_profile()
        for k, v in DEFAULT_COURSE_PROFILES.items()
    }
    scoring_service = ScoringService(profiles)
    model_registry = ModelRegistry(str(config.model_registry_path), str(config.model_artifacts_dir))
    prediction_service = PredictionService(model_registry, scoring_service)

    risk_calculator = RiskCalculator()
    risk_service = RiskPredictionService(risk_calculator)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow(
        dataset_service=dataset_service,
        prediction_service=prediction_service,
        chart_service=chart_service,
        risk_service=risk_service
    )
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()