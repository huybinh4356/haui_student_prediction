from pathlib import Path
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox

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
from haui_student_prediction.ml.train_regressor import ModelMetrics
from haui_student_prediction.ml.accuracy_evaluator import compute_all_accuracy_metrics
from haui_student_prediction.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_main_window_accuracy_evaluation_page(qapp, monkeypatch):
    monkeypatch.delenv("APP_BASE_DIR", raising=False)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)

    config = load_config()
    cleaner = DataCleaner(get_default_column_mapping())
    validator = DataValidator()
    ingestion_service = IngestionService(cleaner, validator)
    dataset_service = DatasetService(ingestion_service)
    chart_service = ChartService(str(config.report_output_dir))
    profiles = {
        "no_tx2": CourseProfile(0.20, 0.0, 0.20, 0.60, False),
        "with_tx2": CourseProfile(0.15, 0.15, 0.20, 0.50, True),
    }
    scoring_service = ScoringService(profiles)
    model_registry = ModelRegistry(str(config.model_registry_path), str(config.model_artifacts_dir))
    prediction_service = PredictionService(model_registry, scoring_service)
    risk_calculator = RiskCalculator()
    risk_service = RiskPredictionService(risk_calculator)

    window = MainWindow(
        dataset_service=dataset_service,
        prediction_service=prediction_service,
        chart_service=chart_service,
        risk_service=risk_service,
    )

    assert hasattr(window, "btn_eval_test_file")
    assert hasattr(window, "btn_eval_default")
    assert hasattr(window, "btn_train_default_fast")
    assert hasattr(window, "btn_uci_register")
    assert hasattr(window, "table_all_accuracy")
    assert hasattr(window, "table_band_accuracy")
    assert hasattr(window, "card_eval_mae")

    # Verify initial accuracy page UI controls and auto-loading / manual evaluation display
    if not window._is_model_ready():
        assert window._latest_eval_result is None
        assert window.card_eval_mae.value_label.text() == "—"
        # Simulate loading evaluation to verify table & card rendering logic in fresh CI
        y_true = np.array([3.0, 4.5, 6.0, 7.5, 8.5, 9.0])
        y_pred = np.array([3.1, 4.4, 6.1, 7.4, 8.6, 8.9])
        sample_eval = compute_all_accuracy_metrics(y_true, y_pred)
        window._display_evaluation_result(sample_eval)

    assert window._latest_eval_result is not None
    assert window.table_all_accuracy.rowCount() >= 16
    assert window.table_band_accuracy.rowCount() == 4
    assert window.card_eval_mae.value_label.text() != "—"
    assert window.card_eval_r2.value_label.text() != "—"


def test_training_succeeded_automatically_updates_accuracy_metrics(qapp, monkeypatch):
    monkeypatch.delenv("APP_BASE_DIR", raising=False)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)

    config = load_config()
    cleaner = DataCleaner(get_default_column_mapping())
    validator = DataValidator()
    ingestion_service = IngestionService(cleaner, validator)
    dataset_service = DatasetService(ingestion_service)
    chart_service = ChartService(str(config.report_output_dir))
    profiles = {
        "no_tx2": CourseProfile(0.20, 0.0, 0.20, 0.60, False),
        "with_tx2": CourseProfile(0.15, 0.15, 0.20, 0.50, True),
    }
    scoring_service = ScoringService(profiles)
    model_registry = ModelRegistry(str(config.model_registry_path), str(config.model_artifacts_dir))
    prediction_service = PredictionService(model_registry, scoring_service)
    risk_calculator = RiskCalculator()
    risk_service = RiskPredictionService(risk_calculator)

    window = MainWindow(
        dataset_service=dataset_service,
        prediction_service=prediction_service,
        chart_service=chart_service,
        risk_service=risk_service,
    )

    # Simulate newly trained model evaluation
    y_true = np.array([3.0, 5.0, 6.5, 7.5, 8.5, 9.0])
    y_pred = np.array([3.1, 4.9, 6.4, 7.6, 8.4, 8.9])
    eval_res = compute_all_accuracy_metrics(y_true, y_pred)

    dummy_metrics = ModelMetrics(
        mae=0.100,
        rmse=0.120,
        r2=0.990,
        beats_baseline=True,
        selected_candidate="xgb_default",
    )

    result_payload = {
        "version": "xgboost-2.2.0",
        "metrics": dummy_metrics,
        "eval_result": eval_res,
        "train_rows": 500,
        "test_rows": 100,
        "feature_count": 8,
        "dataset_hash": "hash123",
        "features": ["tx1_score", "midterm_score"],
    }

    # Calling _on_training_succeeded directly simulating the worker completion
    window._on_training_succeeded(result_payload)

    # Verify that the full accuracy metrics, cards and tables are immediately updated
    assert window._latest_eval_result is eval_res
    assert float(window.card_eval_mae.value_label.text()) == pytest.approx(eval_res.mae, abs=0.01)
    assert float(window.card_eval_rmse.value_label.text()) == pytest.approx(eval_res.rmse, abs=0.01)
    assert window.table_all_accuracy.rowCount() == 16
    assert window.table_band_accuracy.rowCount() == 4


def test_uci_transfer_registration_and_ui_update(qapp, monkeypatch):
    monkeypatch.delenv("APP_BASE_DIR", raising=False)
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)

    config = load_config()
    cleaner = DataCleaner(get_default_column_mapping())
    validator = DataValidator()
    ingestion_service = IngestionService(cleaner, validator)
    dataset_service = DatasetService(ingestion_service)
    chart_service = ChartService(str(config.report_output_dir))
    profiles = {
        "no_tx2": CourseProfile(0.20, 0.0, 0.20, 0.60, False),
        "with_tx2": CourseProfile(0.15, 0.15, 0.20, 0.50, True),
    }
    scoring_service = ScoringService(profiles)
    model_registry = ModelRegistry(str(config.model_registry_path), str(config.model_artifacts_dir))
    prediction_service = PredictionService(model_registry, scoring_service)
    risk_calculator = RiskCalculator()
    risk_service = RiskPredictionService(risk_calculator)

    window = MainWindow(
        dataset_service=dataset_service,
        prediction_service=prediction_service,
        chart_service=chart_service,
        risk_service=risk_service,
    )

    y_true = np.array([2.5, 4.0, 6.0, 7.5, 8.5, 9.5])
    y_pred = np.array([2.7, 4.2, 5.9, 7.3, 8.4, 9.3])
    eval_res = compute_all_accuracy_metrics(y_true, y_pred)

    window._on_uci_transfer_succeeded({
        "eval_result": eval_res,
        "subject": "mat",
    })

    assert "Đã đăng ký thành công" in window.lbl_uci_status.text()
    assert window.content_area.currentIndex() == 5  # Models & Accuracy Page
    assert window._latest_eval_result is eval_res

