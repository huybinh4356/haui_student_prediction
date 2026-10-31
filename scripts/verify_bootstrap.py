import sys
from PyQt6.QtWidgets import QApplication
from haui_student_prediction.app_config import load_config
from haui_student_prediction.data.column_mapping import get_default_column_mapping
from haui_student_prediction.data.cleaning import DataCleaner
from haui_student_prediction.data.validator import DataValidator
from haui_student_prediction.data.ingestion_service import IngestionService
from haui_student_prediction.services.dataset_service import DatasetService
from haui_student_prediction.eda.charts import ChartService
from haui_student_prediction.scoring.final_course_score import CourseProfile, ScoringService
from haui_student_prediction.ml.model_registry import ModelRegistry
from haui_student_prediction.services.prediction_service import PredictionService
from haui_student_prediction.risk.risk_calculator import RiskCalculator
from haui_student_prediction.services.risk_prediction_service import RiskPredictionService
from haui_student_prediction.ui.main_window import MainWindow

def test_bootstrap():
    config = load_config()
    cleaner = DataCleaner(get_default_column_mapping())
    validator = DataValidator()
    ingestion_service = IngestionService(cleaner, validator)
    dataset_service = DatasetService(ingestion_service)
    chart_service = ChartService(str(config.report_output_dir))
    profiles = {
        "no_tx2": CourseProfile(0.20, 0.0, 0.20, 0.60, False),
        "with_tx2": CourseProfile(0.15, 0.15, 0.20, 0.50, True)
    }
    scoring_service = ScoringService(profiles)
    model_registry = ModelRegistry(str(config.model_registry_path), str(config.model_artifacts_dir))
    prediction_service = PredictionService(model_registry, scoring_service)
    risk_calculator = RiskCalculator()
    risk_service = RiskPredictionService(risk_calculator)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = MainWindow(
        dataset_service=dataset_service,
        prediction_service=prediction_service,
        chart_service=chart_service,
        risk_service=risk_service
    )
    
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print("WINDOW TITLE:", window.windowTitle())
    print("MODEL READY CHECK:", prediction_service.is_model_ready("1.0.0"))
    print("KPI MODEL TEXT:", window.card_model.value_label.text())
    print("INIT SUCCESS!")

if __name__ == "__main__":
    test_bootstrap()
