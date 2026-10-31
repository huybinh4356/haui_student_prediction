from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QIcon, QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from haui_student_prediction.ml.model_registry import ACTIVE_MODEL_VERSION


class _MetricCard(QFrame):
    def __init__(self, title: str, value: str, detail: str = "", tone: str = "blue"):
        super().__init__()
        self.setObjectName("MetricCard")
        self.setMinimumHeight(118)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(7)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")
        self.detail_label = QLabel(detail)
        self.detail_label.setObjectName("MetricDetail")
        self.value_label.setProperty("tone", tone)
        self.value_label.setWordWrap(True)
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.detail_label)

    def set_value(self, value: str, detail: str | None = None, tone: str | None = None) -> None:
        self.value_label.setText(value)
        if detail is not None:
            self.detail_label.setText(detail)
        if tone is not None:
            self.value_label.setProperty("tone", tone)
            style = self.style()
            if style is not None:
                style.unpolish(self.value_label)
                style.polish(self.value_label)


class _TrainingWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, dataframe: pd.DataFrame, parent: QWidget | None = None):
        super().__init__(parent)
        self.dataframe = dataframe.copy(deep=True)

    def run(self) -> None:
        try:
            from haui_student_prediction.app_config import load_config
            from haui_student_prediction.features.feature_contract import FeatureContract
            from haui_student_prediction.ml.model_registry import ModelRegistry
            from haui_student_prediction.ml.data_splitting import split_train_test
            from haui_student_prediction.ml.xgboost_regressor import train_xgboost_and_evaluate

            target_column = "target_final_exam_score"
            if target_column not in self.dataframe.columns:
                raise ValueError("Dữ liệu không có cột target_final_exam_score.")

            data = self.dataframe.copy(deep=True)
            target = pd.to_numeric(data[target_column], errors="coerce")
            finite_target = target.notna() & np.isfinite(target.astype(float))
            data = data.loc[finite_target].copy()
            data[target_column] = target.loc[finite_target].astype(float)

            if len(data) < 10:
                raise ValueError(
                    f"Chỉ có {len(data)} bản ghi huấn luyện hợp lệ. Cần tối thiểu 10 bản ghi để có thể chia tập huấn luyện và kiểm thử sơ bộ."
                )

            numeric_candidates = [
                "tx1_score",
                "tx2_score",
                "midterm_score",
                "absence_percentage",
                "lms_learning_percentage",
                "interaction_score",
                "stress_score",
                "motivation_score",
            ]
            categorical_candidates = ["works_part_time", "major", "course_id"]
            numeric_features = [
                column
                for column in numeric_candidates
                if column in data.columns and data[column].notna().any()
                and data[column].nunique(dropna=True) > 1
            ]
            categorical_features = [
                column
                for column in categorical_candidates
                if column in data.columns and data[column].notna().any()
                and data[column].nunique(dropna=True) > 1
            ]

            if not numeric_features and not categorical_features:
                raise ValueError("Không có đặc trưng hợp lệ để huấn luyện mô hình.")

            contract = FeatureContract(
                numerical_features=numeric_features,
                categorical_features=categorical_features,
                target_column=target_column,
            )

            train_data, test_data = split_train_test(
                data,
                test_size=0.2,
                random_state=42,
            )

            model, metrics = train_xgboost_and_evaluate(train_data, test_data, contract)
            hashed_values = pd.util.hash_pandas_object(
                data,
                index=True,
            ).to_numpy(dtype=np.uint64)
            dataset_hash = hashlib.sha256(hashed_values.tobytes()).hexdigest()
            config = load_config()
            registry = ModelRegistry(
                str(config.model_registry_path),
                str(config.model_artifacts_dir),
            )
            registry.save_model(
                model=model,
                version=ACTIVE_MODEL_VERSION,
                dataset_hash=dataset_hash,
                metrics=metrics,
                contract=contract,
            )
            if not registry.has_model(ACTIVE_MODEL_VERSION):
                raise RuntimeError("Mô hình đã được lưu nhưng không thể xác minh trong kho lưu trữ.")

            self.succeeded.emit(
                {
                    "version": ACTIVE_MODEL_VERSION,
                    "metrics": metrics,
                    "train_rows": len(train_data),
                    "test_rows": len(test_data),
                    "feature_count": len(contract.get_all_features()),
                    "dataset_hash": dataset_hash,
                    "features": contract.get_all_features(),
                }
            )
        except Exception as exc:
            logging.exception("Model training failed")
            self.failed.emit(str(exc))




class _UciBenchmarkWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, subject: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.subject = subject

    def run(self) -> None:
        try:
            from haui_student_prediction.data.uci_student_performance import (
                UCI_DATASET_DOI,
                UCI_DATASET_LICENSE,
                UCI_DATASET_NAME,
                load_uci_student_performance,
            )
            from haui_student_prediction.ml.uci_model_comparison import compare_uci_models

            project_root = Path(__file__).resolve().parents[3]
            raw_dir = project_root / "data" / "raw" / "uci_student_performance"
            output_dir = project_root / "outputs" / "uci_model_comparison"

            dataframe = load_uci_student_performance(
                subject=self.subject,
                raw_dir=raw_dir,
            )
            metrics, predictions = compare_uci_models(
                dataframe,
                test_size=0.2,
                random_state=42,
                cv_folds=5,
            )

            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = output_dir / f"uci_{self.subject}_model_metrics.csv"
            predictions_path = output_dir / f"uci_{self.subject}_test_predictions.csv"
            metadata_path = output_dir / f"uci_{self.subject}_training_metadata.txt"

            metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
            predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
            metadata_path.write_text(
                "\n".join(
                    [
                        f"dataset={UCI_DATASET_NAME}",
                        "dataset_id=320",
                        f"subject={self.subject}",
                        f"dataset_rows={len(dataframe)}",
                        "target=G3 final course grade, normalized from 0-20 to 0-10",
                        "test_size=0.2",
                        "cv_folds=5",
                        "validation_strategy=k-fold cross-validation inside the training partition",
                        "model_selection_metric=mean cross-validation MAE",
                        "test_usage=final holdout evaluation only",
                        "random_state=42",
                        f"license={UCI_DATASET_LICENSE}",
                        f"doi={UCI_DATASET_DOI}",
                        "production_registry=not_updated",
                        "note=G3 is a final course grade, not a separately identified final-exam score.",
                    ]
                ),
                encoding="utf-8",
            )

            self.succeeded.emit(
                {
                    "metrics": metrics,
                    "predictions": predictions,
                    "subject": self.subject,
                    "dataset_rows": len(dataframe),
                    "train_rows": int(metrics.iloc[0]["train_rows"]),
                    "test_rows": int(metrics.iloc[0]["test_rows"]),
                    "metrics_path": str(metrics_path),
                    "predictions_path": str(predictions_path),
                }
            )
        except Exception as exc:
            logging.exception("UCI benchmark training failed")
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    card_model: _MetricCard
    training_progress: QProgressBar
    uci_progress: QProgressBar
    table_uci_metrics: QTableWidget
    settings_nav_button: QPushButton

    def __init__(self, dataset_service, prediction_service, chart_service, risk_service, report_service=None):
        super().__init__()
        self.dataset_service = dataset_service
        self.prediction_service = prediction_service
        self.chart_service = chart_service
        self.risk_service = risk_service
        self.report_service = report_service
        self.current_df: pd.DataFrame | None = None
        self.prediction_df: pd.DataFrame | None = None
        self.risk_df: pd.DataFrame | None = None
        self.current_file_path: str | None = None
        self.training_file_path: str | None = None
        self._training_worker: _TrainingWorker | None = None
        self._uci_training_worker: _UciBenchmarkWorker | None = None
        self._page_titles = [
            "Tổng quan",
            "Dữ liệu sinh viên",
            "Phân tích dữ liệu",
            "Dự đoán điểm",
            "Dự đoán nguy cơ học vụ",
            "Mô hình dự đoán",
            "Huấn luyện và so sánh",
            "Báo cáo",
            "Cấu hình",
        ]
        self._page_subtitles = [
            "Theo dõi dữ liệu, chất lượng và trạng thái dự đoán.",
            "Nhập dữ liệu, kiểm tra chất lượng và xem bản ghi đã chuẩn hóa.",
            "Khám phá phân bố, xu hướng và mối liên hệ trong dữ liệu.",
            "Ước lượng điểm thi cuối kỳ và điểm học phần dự kiến.",
            "Đánh giá nguy cơ dựa trên dữ liệu học tập hiện có.",
            "Quản lý mô hình dự đoán dành riêng cho dữ liệu HaUI.",
            "Thực nghiệm XGBoost, Random Forest và CatBoost với bộ dữ liệu UCI.",
            "Xuất dữ liệu và kết quả phân tích đang có.",
            "Xem cấu hình môi trường và đường dẫn hiện hành.",
        ]
        self.setWindowTitle("Phân tích học tập HaUI")
        self.setMinimumSize(1100, 720)
        self.resize(1440, 900)
        self._apply_theme()
        self._build_shell()
        self._build_pages()
        self._switch_page(0)
        self._update_all_status()

    def _apply_theme(self) -> None:
        from haui_student_prediction.ui.theme import APP_STYLESHEET
        self.setStyleSheet(APP_STYLESHEET)

    @staticmethod
    def _standard_icon(icon_name: QStyle.StandardPixmap) -> QIcon:
        style = QApplication.style()
        return style.standardIcon(icon_name) if style is not None else QIcon()

    def _build_shell(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(242)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(14, 22, 14, 18)
        sidebar_layout.setSpacing(5)

        brand = QWidget()
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(10, 2, 10, 22)
        brand_layout.setSpacing(5)
        brand_title = QLabel("Phân tích học tập HaUI")
        brand_title.setObjectName("BrandTitle")
        brand_title.setWordWrap(True)
        brand_caption = QLabel("HỆ THỐNG PHÂN TÍCH HỌC TẬP")
        brand_caption.setObjectName("BrandCaption")
        brand_layout.addWidget(brand_title)
        brand_layout.addWidget(brand_caption)
        sidebar_layout.addWidget(brand)

        nav_label = QLabel("KHÔNG GIAN LÀM VIỆC")
        nav_label.setObjectName("SidebarSection")
        sidebar_layout.addWidget(nav_label)

        self.nav_buttons = []
        nav_items = [
            ("Tổng quan", "SP_ComputerIcon"),
            ("Dữ liệu sinh viên", "SP_FileDialogDetailedView"),
            ("Phân tích dữ liệu", "SP_FileDialogContentsView"),
            ("Dự đoán điểm", "SP_MediaPlay"),
            ("Dự đoán nguy cơ học vụ", "SP_MessageBoxWarning"),
            ("Mô hình dự đoán", "SP_BrowserReload"),
            ("Huấn luyện và so sánh", "SP_BrowserReload"),
            ("Báo cáo", "SP_FileDialogListView"),
        ]
        standard_icons = {
            "SP_ComputerIcon": QStyle.StandardPixmap.SP_ComputerIcon,
            "SP_FileDialogDetailedView": QStyle.StandardPixmap.SP_FileDialogDetailedView,
            "SP_FileDialogContentsView": QStyle.StandardPixmap.SP_FileDialogContentsView,
            "SP_MediaPlay": QStyle.StandardPixmap.SP_MediaPlay,
            "SP_MessageBoxWarning": QStyle.StandardPixmap.SP_MessageBoxWarning,
            "SP_BrowserReload": QStyle.StandardPixmap.SP_BrowserReload,
            "SP_FileDialogListView": QStyle.StandardPixmap.SP_FileDialogListView,
        }
        for index, (title, icon_key) in enumerate(nav_items):
            button = QPushButton(title)
            button.setObjectName("NavButton")
            button.setCheckable(True)
            button.setMinimumHeight(42)
            button.setIcon(self._standard_icon(standard_icons[icon_key]))
            button.clicked.connect(lambda checked=False, i=index: self._switch_page(i))
            self.nav_buttons.append(button)
            sidebar_layout.addWidget(button)

        sidebar_layout.addStretch(1)
        settings_button = QPushButton("Cấu hình ứng dụng")
        settings_button.setObjectName("NavButton")
        settings_button.setMinimumHeight(42)
        settings_button.setCheckable(True)
        settings_button.setIcon(self._standard_icon(QStyle.StandardPixmap.SP_FileDialogInfoView))
        settings_button.clicked.connect(lambda: self._switch_page(8))
        sidebar_layout.addWidget(settings_button)
        self.settings_nav_button = settings_button

        sidebar_footer = QLabel("HaUI · Phân tích kết quả học tập")
        sidebar_footer.setObjectName("BrandCaption")
        sidebar_footer.setWordWrap(True)
        sidebar_footer.setContentsMargins(10, 12, 10, 0)
        sidebar_layout.addWidget(sidebar_footer)
        root_layout.addWidget(self.sidebar)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.topbar = QFrame()
        self.topbar.setObjectName("TopBar")
        self.topbar.setFixedHeight(66)
        top_layout = QHBoxLayout(self.topbar)
        top_layout.setContentsMargins(26, 10, 26, 10)
        top_layout.setSpacing(10)
        self.topbar_context = QLabel("KHÔNG GIAN PHÂN TÍCH")
        self.topbar_context.setStyleSheet("color:#6B7280;font-size:11px;font-weight:700;letter-spacing:1px;background:transparent;")
        top_layout.addWidget(self.topbar_context)
        top_layout.addStretch(1)
        self.lbl_dataset_badge = QLabel("Chưa có dữ liệu")
        self.lbl_dataset_badge.setStyleSheet("background:#F2F4F7;color:#475467;padding:7px 10px;border-radius:7px;font-size:11px;font-weight:600;")
        self.lbl_model_badge = QLabel("Mô hình chưa sẵn sàng")
        self.lbl_model_badge.setStyleSheet("background:#FEF3F2;color:#B42318;padding:7px 10px;border-radius:7px;font-size:11px;font-weight:600;")
        top_layout.addWidget(self.lbl_dataset_badge)
        top_layout.addWidget(self.lbl_model_badge)
        right_layout.addWidget(self.topbar)

        self.content_area = QStackedWidget()
        self.content_area.setObjectName("ContentArea")
        right_layout.addWidget(self.content_area, 1)
        root_layout.addWidget(right_panel, 1)

    def _create_page(self, title: str, subtitle: str) -> tuple[QScrollArea, QWidget, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setObjectName("PageBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(28, 24, 28, 30)
        layout.setSpacing(18)
        heading = QVBoxLayout()
        heading.setSpacing(5)
        title_label = QLabel(title)
        title_label.setObjectName("PageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("PageSubtitle")
        subtitle_label.setWordWrap(True)
        heading.addWidget(title_label)
        heading.addWidget(subtitle_label)
        layout.addLayout(heading)
        scroll.setWidget(body)
        return scroll, body, layout

    def _surface(self, title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("Surface")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        if title:
            label = QLabel(title)
            label.setObjectName("SectionTitle")
            layout.addWidget(label)
        return frame, layout

    def _primary_button(self, text: str, callback, kind: str = "primary") -> QPushButton:
        button = QPushButton(text)
        names = {
            "primary": "PrimaryButton",
            "success": "SuccessButton",
            "warning": "WarningButton",
            "danger": "DangerButton",
        }
        button.setObjectName(names.get(kind, "PrimaryButton"))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _build_pages(self) -> None:
        self.pages = []
        page_definitions = [
            ("Tổng quan học tập", self._page_subtitles[0], self._build_dashboard),
            ("Dữ liệu sinh viên", self._page_subtitles[1], self._build_data_page),
            ("Phân tích dữ liệu", self._page_subtitles[2], self._build_eda_page),
            ("Dự đoán điểm thi cuối kỳ", self._page_subtitles[3], self._build_prediction_page),
            ("Dự đoán nguy cơ học vụ", self._page_subtitles[4], self._build_risk_page),
            ("Mô hình dự đoán HaUI", self._page_subtitles[5], self._build_models_page),
            ("Huấn luyện và so sánh mô hình", self._page_subtitles[6], self._build_uci_benchmark_page),
            ("Báo cáo và xuất dữ liệu", self._page_subtitles[7], self._build_reports_page),
            ("Cấu hình ứng dụng", self._page_subtitles[8], self._build_settings_page),
        ]
        for title, subtitle, builder in page_definitions:
            scroll, body, layout = self._create_page(title, subtitle)
            builder(body, layout)
            self.content_area.addWidget(scroll)
            self.pages.append(scroll)

    def _build_dashboard(self, body: QWidget, layout: QVBoxLayout) -> None:
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self._primary_button("Nhập dữ liệu", lambda: self._switch_page(1)))
        action_row.addWidget(self._primary_button("Làm mới", self._update_all_status))
        layout.addLayout(action_row)

        cards = QGridLayout()
        cards.setSpacing(12)
        self.card_records = _MetricCard("BẢN GHI ĐÃ NẠP", "—", "Chưa có bộ dữ liệu", "blue")
        self.card_missing = _MetricCard("Ô DỮ LIỆU THIẾU", "—", "Chưa có bộ dữ liệu", "amber")
        self.card_absence = _MetricCard("VƯỢT NGƯỠNG NGHỈ HỌC", "—", "Ngưỡng cấu hình > 30%", "red")
        self.card_model = _MetricCard("MÔ HÌNH DỰ ĐOÁN", "Chưa sẵn sàng", "Cần mô hình đã huấn luyện", "violet")
        cards.addWidget(self.card_records, 0, 0)
        cards.addWidget(self.card_missing, 0, 1)
        cards.addWidget(self.card_absence, 0, 2)
        cards.addWidget(self.card_model, 0, 3)
        for col in range(4):
            cards.setColumnStretch(col, 1)
        layout.addLayout(cards)

        main_grid = QGridLayout()
        main_grid.setSpacing(14)
        chart_frame, chart_layout = self._surface("Phân bố tỷ lệ nghỉ học")
        self.dashboard_chart_layout = chart_layout
        self.dashboard_chart_placeholder = QLabel("Nhập dữ liệu để hiển thị phân tích.")
        self.dashboard_chart_placeholder.setObjectName("Muted")
        self.dashboard_chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_chart_placeholder.setMinimumHeight(230)
        chart_layout.addWidget(self.dashboard_chart_placeholder)
        summary_frame, summary_layout = self._surface("Tình trạng bộ dữ liệu")
        self.dashboard_summary = QLabel("Chưa có dữ liệu được nạp vào phiên làm việc.")
        self.dashboard_summary.setWordWrap(True)
        self.dashboard_summary.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.dashboard_summary.setStyleSheet("color:#475467;line-height:1.5;")
        summary_layout.addWidget(self.dashboard_summary)
        summary_layout.addStretch(1)
        main_grid.addWidget(chart_frame, 0, 0, 1, 2)
        main_grid.addWidget(summary_frame, 0, 2, 1, 1)
        main_grid.setColumnStretch(0, 1)
        main_grid.setColumnStretch(1, 1)
        main_grid.setColumnStretch(2, 1)
        layout.addLayout(main_grid)

        notice_frame, notice_layout = self._surface("Lưu ý vận hành")
        notice = QLabel(
            "Chỉ dự đoán khi có mô hình hợp lệ và dữ liệu đáp ứng bộ đặc trưng đầu vào. "
            "Trạng thái vượt ngưỡng nghỉ học được hiển thị riêng với điểm nguy cơ tổng hợp."
        )
        notice.setWordWrap(True)
        notice.setStyleSheet("color:#475467;")
        notice_layout.addWidget(notice)
        layout.addWidget(notice_frame)

    def _build_data_page(self, body: QWidget, layout: QVBoxLayout) -> None:
        actions_frame, actions_layout = self._surface("Nhập dữ liệu")
        actions_row = QHBoxLayout()
        self.btn_import = self._primary_button("Chọn file CSV / Excel", self._handle_import)
        self.btn_import.setMinimumHeight(40)
        actions_row.addWidget(self.btn_import)
        self.btn_refresh_table = QPushButton("Làm mới hiển thị")
        self.btn_refresh_table.clicked.connect(self._refresh_data_table)
        actions_row.addWidget(self.btn_refresh_table)
        actions_row.addStretch(1)
        actions_layout.addLayout(actions_row)
        self.lbl_import_status = QLabel("Chưa chọn file dữ liệu.")
        self.lbl_import_status.setObjectName("Muted")
        self.lbl_import_status.setWordWrap(True)
        actions_layout.addWidget(self.lbl_import_status)
        self.lbl_quality_summary = QLabel("Báo cáo chất lượng sẽ xuất hiện sau khi nhập dữ liệu.")
        self.lbl_quality_summary.setWordWrap(True)
        self.lbl_quality_summary.setStyleSheet("background:#F8FAFC;padding:11px;border-radius:8px;color:#475467;")
        actions_layout.addWidget(self.lbl_quality_summary)
        layout.addWidget(actions_frame)

        table_frame, table_layout = self._surface("Bản xem trước dữ liệu")
        self.lbl_table_count = QLabel("0 bản ghi")
        self.lbl_table_count.setObjectName("Muted")
        table_layout.addWidget(self.lbl_table_count)
        self.table_import = self._create_table()
        self.table_import.setMinimumHeight(360)
        table_layout.addWidget(self.table_import)
        layout.addWidget(table_frame, 1)

    def _build_eda_page(self, body: QWidget, layout: QVBoxLayout) -> None:
        filters_frame, filters_layout = self._surface("Thiết lập phân tích")
        row = QHBoxLayout()
        self.cb_chart_type = QComboBox()
        self.cb_chart_type.addItem("Histogram", "histogram")
        self.cb_chart_type.addItem("Biểu đồ đếm", "count")
        self.cb_chart_type.addItem("Boxplot theo nhóm", "boxplot")
        self.cb_chart_type.addItem("Scatter plot", "scatter")
        self.cb_chart_type.addItem("Correlation heatmap", "heatmap")
        self.cb_eda_col = QComboBox()
        self.cb_eda_category = QComboBox()
        self.cb_eda_x = QComboBox()
        self.cb_eda_y = QComboBox()
        self.btn_plot = self._primary_button("Tạo biểu đồ", self._handle_plot)
        self.btn_save_chart = QPushButton("Lưu biểu đồ")
        self.btn_save_chart.clicked.connect(self._save_current_chart)
        row.addWidget(QLabel("Loại biểu đồ"))
        row.addWidget(self.cb_chart_type, 1)
        row.addWidget(QLabel("Biến chính"))
        row.addWidget(self.cb_eda_col, 2)
        row.addWidget(QLabel("Nhóm"))
        row.addWidget(self.cb_eda_category, 2)
        row.addWidget(self.btn_plot)
        row.addWidget(self.btn_save_chart)
        filters_layout.addLayout(row)

        scatter_row = QHBoxLayout()
        scatter_row.addWidget(QLabel("Trục X"))
        scatter_row.addWidget(self.cb_eda_x, 1)
        scatter_row.addWidget(QLabel("Trục Y"))
        scatter_row.addWidget(self.cb_eda_y, 1)
        filters_layout.addLayout(scatter_row)
        self.cb_chart_type.currentIndexChanged.connect(self._update_eda_controls)
        layout.addWidget(filters_frame)

        chart_frame, chart_layout = self._surface("Biểu đồ")
        self.chart_container = QVBoxLayout()
        self.chart_container.setSpacing(4)
        self.chart_placeholder = QLabel("Nhập dữ liệu trước khi tạo biểu đồ.")
        self.chart_placeholder.setObjectName("Muted")
        self.chart_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chart_placeholder.setMinimumHeight(300)
        self.chart_container.addWidget(self.chart_placeholder)
        chart_layout.addLayout(self.chart_container)
        layout.addWidget(chart_frame, 1)

        stats_frame, stats_layout = self._surface("Thống kê mô tả")
        self.table_eda_stats = self._create_table()
        self.table_eda_stats.setMaximumHeight(240)
        stats_layout.addWidget(self.table_eda_stats)
        layout.addWidget(stats_frame)
        self._update_eda_controls()

    def _build_prediction_page(self, body: QWidget, layout: QVBoxLayout) -> None:
        setup_frame, setup_layout = self._surface("Thiết lập dự đoán")
        row = QHBoxLayout()
        self.cb_course_profile = QComboBox()
        self.cb_course_profile.addItem("Chọn cấu hình đánh giá", None)
        self.cb_course_profile.addItem("Học phần không có TX2", "no_tx2")
        self.cb_course_profile.addItem("Học phần có TX2", "with_tx2")
        self.btn_train_from_prediction = self._primary_button("Huấn luyện mô hình", self._handle_train_model, "warning")
        self.btn_predict = self._primary_button("Dự đoán điểm học phần", self._handle_predict, "success")
        row.addWidget(QLabel("Cấu hình đánh giá"))
        row.addWidget(self.cb_course_profile, 1)
        row.addStretch(1)
        row.addWidget(self.btn_train_from_prediction)
        row.addWidget(self.btn_predict)
        setup_layout.addLayout(row)
        self.lbl_prediction_notice = QLabel(
            "Chọn cấu hình phù hợp với cấu trúc điểm của học phần. Nếu một tệp chứa nhiều loại học phần, cần tách tệp hoặc bổ sung quy tắc ánh xạ theo học phần."
        )
        self.lbl_prediction_notice.setWordWrap(True)
        self.lbl_prediction_notice.setStyleSheet("color:#475467;background:#EFF6FF;border-radius:8px;padding:10px;")
        setup_layout.addWidget(self.lbl_prediction_notice)
        layout.addWidget(setup_frame)

        result_frame, result_layout = self._surface("Kết quả dự đoán")
        self.lbl_prediction_status = QLabel("Chưa thực hiện dự đoán.")
        self.lbl_prediction_status.setObjectName("Muted")
        result_layout.addWidget(self.lbl_prediction_status)
        self.table_predict = self._create_table()
        self.table_predict.setMinimumHeight(350)
        result_layout.addWidget(self.table_predict)
        layout.addWidget(result_frame, 1)

    def _build_risk_page(self, body: QWidget, layout: QVBoxLayout) -> None:
        setup_frame, setup_layout = self._surface("Đánh giá nguy cơ")
        row = QHBoxLayout()
        self.btn_risk = self._primary_button("Tính nguy cơ học vụ", self._handle_risk, "danger")
        self.btn_export_risk = QPushButton("Xuất kết quả nguy cơ")
        self.btn_export_risk.clicked.connect(lambda: self._export_dataframe("risk"))
        row.addWidget(self.btn_risk)
        row.addWidget(self.btn_export_risk)
        row.addStretch(1)
        setup_layout.addLayout(row)
        notice = QLabel(
            "Điểm nguy cơ là kết quả theo quy tắc cấu hình, không phải xác suất đã hiệu chuẩn. "
            "Trạng thái nghỉ học vượt 30% được thể hiện riêng và không bị điểm khảo sát làm mất hiệu lực."
        )
        notice.setWordWrap(True)
        notice.setStyleSheet("color:#7A271A;background:#FEF3F2;border-radius:8px;padding:10px;")
        setup_layout.addWidget(notice)
        layout.addWidget(setup_frame)

        cards = QGridLayout()
        cards.setSpacing(12)
        self.card_risk_evaluated = _MetricCard("ĐÃ ĐÁNH GIÁ", "—", "Chưa có kết quả", "blue")
        self.card_risk_override = _MetricCard("VƯỢT NGƯỠNG", "—", "Nghỉ học > 30%", "red")
        self.card_risk_high = _MetricCard("NGUY CƠ CAO / RẤT CAO", "—", "Theo cấu hình risk engine", "amber")
        cards.addWidget(self.card_risk_evaluated, 0, 0)
        cards.addWidget(self.card_risk_override, 0, 1)
        cards.addWidget(self.card_risk_high, 0, 2)
        for col in range(3):
            cards.setColumnStretch(col, 1)
        layout.addLayout(cards)

        table_frame, table_layout = self._surface("Kết quả theo bản ghi")
        self.lbl_risk_status = QLabel("Cần dữ liệu và các trường bắt buộc hợp lệ để tính nguy cơ.")
        self.lbl_risk_status.setObjectName("Muted")
        self.lbl_risk_status.setWordWrap(True)
        table_layout.addWidget(self.lbl_risk_status)
        self.table_risk = self._create_table()
        self.table_risk.setMinimumHeight(350)
        table_layout.addWidget(self.table_risk)
        layout.addWidget(table_frame, 1)

    def _build_models_page(self, body: QWidget, layout: QVBoxLayout) -> None:
        state_frame, state_layout = self._surface("Trạng thái mô hình")
        top = QHBoxLayout()
        self.lbl_model_page_status = QLabel("Đang kiểm tra kho mô hình…")
        self.lbl_model_page_status.setStyleSheet("font-size:16px;font-weight:700;color:#B45309;")
        top.addWidget(self.lbl_model_page_status, 1)
        self.btn_model_refresh = QPushButton("Làm mới trạng thái")
        self.btn_model_refresh.clicked.connect(self._update_model_status)
        top.addWidget(self.btn_model_refresh)
        state_layout.addLayout(top)
        self.lbl_model_details = QLabel("Chưa có thông tin mô hình.")
        self.lbl_model_details.setWordWrap(True)
        self.lbl_model_details.setObjectName("Muted")
        state_layout.addWidget(self.lbl_model_details)
        layout.addWidget(state_frame)

        train_frame, train_layout = self._surface("Huấn luyện mô hình HaUI từ dữ liệu lịch sử")
        explainer = QLabel(
            "Dữ liệu sinh viên dùng để dự đoán không cần điểm cuối kỳ. Khi huấn luyện, hãy chọn riêng một tệp dữ liệu lịch sử đã có kết quả cuối kỳ; nhãn lịch sử chỉ được dùng để học và đánh giá mô hình, không đi vào đầu vào dự đoán."
        )
        explainer.setWordWrap(True)
        explainer.setStyleSheet("color:#475467;")
        train_layout.addWidget(explainer)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 0)
        self.training_progress.setVisible(False)
        train_layout.addWidget(self.training_progress)
        train_actions = QHBoxLayout()
        self.btn_train_model = self._primary_button("Huấn luyện XGBoost", self._handle_train_model, "warning")
        self.btn_train_model.setMinimumHeight(40)
        train_actions.addWidget(self.btn_train_model)
        train_actions.addStretch(1)
        train_layout.addLayout(train_actions)
        self.lbl_training_status = QLabel("Chưa chạy huấn luyện trong phiên này.")
        self.lbl_training_status.setWordWrap(True)
        self.lbl_training_status.setObjectName("Muted")
        train_layout.addWidget(self.lbl_training_status)
        layout.addWidget(train_frame)

        metrics_frame, metrics_layout = self._surface("Kết quả đánh giá gần nhất")
        self.lbl_model_metrics = QLabel("Chưa có kết quả huấn luyện trong phiên này.")
        self.lbl_model_metrics.setWordWrap(True)
        self.lbl_model_metrics.setObjectName("Muted")
        metrics_layout.addWidget(self.lbl_model_metrics)
        layout.addWidget(metrics_frame)

    def _build_uci_benchmark_page(self, body: QWidget, layout: QVBoxLayout) -> None:
        intro_frame, intro_layout = self._surface("🧪 Thực nghiệm trên bộ dữ liệu UCI")
        intro = QLabel(
            "Huấn luyện và so sánh ba thuật toán trên dữ liệu Student Performance của UCI. "
            "Kết quả tại đây phục vụ nghiên cứu thuật toán, không thay thế mô hình dự đoán chính thức của HaUI."
        )
        intro.setWordWrap(True)
        intro.setObjectName("Muted")
        intro_layout.addWidget(intro)

        subject_row = QHBoxLayout()
        subject_label = QLabel("Bộ dữ liệu:")
        self.cb_uci_subject = QComboBox()
        self.cb_uci_subject.addItem("Tiếng Bồ Đào Nha · 649 bản ghi", "por")
        self.cb_uci_subject.addItem("Toán học · 395 bản ghi", "mat")
        self.cb_uci_subject.setMinimumWidth(280)
        self.cb_uci_subject.currentIndexChanged.connect(self._handle_uci_subject_changed)
        subject_row.addWidget(subject_label)
        subject_row.addWidget(self.cb_uci_subject)
        subject_row.addStretch(1)
        self.btn_uci_train = self._primary_button(
            "Chạy huấn luyện và đánh giá",
            self._handle_uci_training,
            "primary",
        )
        self.btn_uci_train.setIcon(
            self._standard_icon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.btn_uci_train.setMinimumHeight(40)
        subject_row.addWidget(self.btn_uci_train)
        intro_layout.addLayout(subject_row)

        self.uci_progress = QProgressBar()
        self.uci_progress.setRange(0, 0)
        self.uci_progress.setVisible(False)
        intro_layout.addWidget(self.uci_progress)
        self.lbl_uci_status = QLabel(
            "Chưa chạy thực nghiệm. Chọn môn học rồi bấm “Chạy huấn luyện và đánh giá”."
        )
        self.lbl_uci_status.setWordWrap(True)
        self.lbl_uci_status.setObjectName("Muted")
        intro_layout.addWidget(self.lbl_uci_status)
        layout.addWidget(intro_frame)

        split_frame, split_layout = self._surface("Cấu hình chia dữ liệu")
        split_cards = QGridLayout()
        split_cards.setSpacing(12)
        self.card_uci_train_split = _MetricCard("TẬP HUẤN LUYỆN", "80%", "519 bản ghi dự kiến", "blue")
        self.card_uci_test_split = _MetricCard("TẬP KIỂM THỬ", "20%", "130 bản ghi dự kiến", "amber")
        self.card_uci_cv_split = _MetricCard("KIỂM ĐỊNH CHÉO", "5 phần", "Mỗi lượt giữ lại 1 phần để xác thực", "violet")
        split_cards.addWidget(self.card_uci_train_split, 0, 0)
        split_cards.addWidget(self.card_uci_test_split, 0, 1)
        split_cards.addWidget(self.card_uci_cv_split, 0, 2)
        for column in range(3):
            split_cards.setColumnStretch(column, 1)
        split_layout.addLayout(split_cards)
        split_note = QLabel(
            "Tập kiểm thử được giữ riêng để đánh giá cuối cùng. Việc chọn mô hình dựa trên MAE trung bình của 5 lượt kiểm định chéo trong tập huấn luyện."
        )
        split_note.setWordWrap(True)
        split_note.setObjectName("Muted")
        split_layout.addWidget(split_note)
        layout.addWidget(split_frame)

        fields_frame, fields_layout = self._surface("Các thông tin được đưa vào huấn luyện")
        fields_note = QLabel(
            "Điểm giai đoạn 1 · Điểm giai đoạn 2 · Mức thời gian tự học · Số lần trượt môn trước đây · "
            "Số buổi vắng · Hỗ trợ học tập tại trường · Hỗ trợ từ gia đình · Học thêm có phí · "
            "Định hướng học lên · Khả năng truy cập Internet tại nhà."
        )
        fields_note.setWordWrap(True)
        fields_note.setObjectName("Muted")
        fields_layout.addWidget(fields_note)
        target_note = QLabel(
            "Kết quả cần dự đoán: điểm tổng kết môn học (G3). Đây là điểm tổng kết theo bộ dữ liệu UCI, "
            "không phải điểm thi cuối kỳ riêng biệt."
        )
        target_note.setWordWrap(True)
        target_note.setStyleSheet(
            "color:#1D4ED8;background:#EFF6FF;border-radius:8px;padding:10px;"
        )
        fields_layout.addWidget(target_note)
        layout.addWidget(fields_frame)

        result_frame, result_layout = self._surface("Kết quả thực nghiệm")
        result_cards = QGridLayout()
        result_cards.setSpacing(12)
        self.card_uci_best_model = _MetricCard("MÔ HÌNH TỐT NHẤT", "—", "Xếp hạng theo MAE kiểm định chéo", "green")
        self.card_uci_cv_mae = _MetricCard("MAE KIỂM ĐỊNH CHÉO", "—", "Giá trị trung bình ± độ lệch chuẩn", "blue")
        self.card_uci_test_mae = _MetricCard("MAE KIỂM THỬ", "—", "Đánh giá trên dữ liệu chưa dùng để huấn luyện", "violet")
        result_cards.addWidget(self.card_uci_best_model, 0, 0)
        result_cards.addWidget(self.card_uci_cv_mae, 0, 1)
        result_cards.addWidget(self.card_uci_test_mae, 0, 2)
        for column in range(3):
            result_cards.setColumnStretch(column, 1)
        result_layout.addLayout(result_cards)

        self.lbl_uci_result_summary = QLabel("Chưa có kết quả thực nghiệm.")
        self.lbl_uci_result_summary.setWordWrap(True)
        self.lbl_uci_result_summary.setObjectName("Muted")
        result_layout.addWidget(self.lbl_uci_result_summary)

        self.table_uci_metrics = self._create_table()
        self.table_uci_metrics.setColumnCount(8)
        self.table_uci_metrics.setHorizontalHeaderLabels(
            [
                "Thuật toán",
                "MAE kiểm định chéo (TB ± ĐLC)",
                "RMSE kiểm định chéo",
                "R² kiểm định chéo",
                "MAE kiểm thử",
                "RMSE kiểm thử",
                "R² kiểm thử",
                "Thời gian 5 lượt (giây)",
            ]
        )
        self.table_uci_metrics.setMinimumHeight(250)
        metrics_header = self.table_uci_metrics.horizontalHeader()
        if metrics_header is not None:
            metrics_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        result_layout.addWidget(self.table_uci_metrics)
        layout.addWidget(result_frame)

        output_frame, output_layout = self._surface("Lưu trữ kết quả")
        output_note = QLabel(
            "Các chỉ số tổng hợp, dự đoán trên tập kiểm thử và thông tin lần chạy được lưu tại "
            "outputs/uci_model_comparison/. Dữ liệu UCI và kết quả thực nghiệm không ghi đè kho mô hình HaUI."
        )
        output_note.setWordWrap(True)
        output_note.setObjectName("Muted")
        output_layout.addWidget(output_note)
        layout.addWidget(output_frame)
        layout.addStretch(1)
        self._update_uci_split_preview()

    def _update_uci_split_preview(self) -> None:
        if not hasattr(self, "cb_uci_subject"):
            return
        subject = self.cb_uci_subject.currentData() or "por"
        dataset_rows = 649 if subject == "por" else 395
        test_rows = int(np.ceil(dataset_rows * 0.2))
        train_rows = dataset_rows - test_rows
        validation_rows = int(np.ceil(train_rows / 5))
        self.card_uci_train_split.set_value(
            "80%",
            f"Khoảng {train_rows:,}/{dataset_rows:,} bản ghi",
        )
        self.card_uci_test_split.set_value(
            "20%",
            f"{test_rows:,}/{dataset_rows:,} bản ghi",
        )
        self.card_uci_cv_split.set_value(
            "5 phần",
            f"Mỗi lượt xác thực khoảng {validation_rows:,} bản ghi",
        )

    def _handle_uci_subject_changed(self) -> None:
        self._update_uci_split_preview()
        self._refresh_uci_benchmark_results()

    def _handle_uci_training(self) -> None:
        if self._uci_training_worker is not None and self._uci_training_worker.isRunning():
            return
        subject = self.cb_uci_subject.currentData() or "por"
        subject_label = "Tiếng Bồ Đào Nha" if subject == "por" else "Toán học"
        self.btn_uci_train.setEnabled(False)
        self.cb_uci_subject.setEnabled(False)
        self.uci_progress.setVisible(True)
        self.lbl_uci_status.setText(
            f"Đang tải dữ liệu {subject_label} và huấn luyện ba mô hình. "
            "Quá trình có thể mất một lúc do mỗi mô hình được chạy qua 5 lượt kiểm định chéo."
        )
        self._uci_training_worker = _UciBenchmarkWorker(subject, self)
        self._uci_training_worker.succeeded.connect(self._on_uci_training_succeeded)
        self._uci_training_worker.failed.connect(self._on_uci_training_failed)
        self._uci_training_worker.finished.connect(self._on_uci_training_finished)
        self._uci_training_worker.start()

    def _on_uci_training_succeeded(self, result: dict[str, Any]) -> None:
        metrics = result["metrics"]
        self._render_uci_benchmark_metrics(
            metrics,
            dataset_rows=int(result["dataset_rows"]),
            train_rows=int(result["train_rows"]),
            test_rows=int(result["test_rows"]),
        )
        subject_label = "Tiếng Bồ Đào Nha" if result["subject"] == "por" else "Toán học"
        self.lbl_uci_status.setText(
            f"Đã hoàn tất thực nghiệm trên bộ dữ liệu {subject_label}. "
            f"Kết quả được lưu tại: {result['metrics_path']}"
        )

    def _on_uci_training_failed(self, error: str) -> None:
        self.lbl_uci_status.setText(f"Huấn luyện chưa hoàn tất: {self._user_error(error)}")
        QMessageBox.warning(
            self,
            "Không thể hoàn thành huấn luyện",
            self._user_error(error),
        )

    def _on_uci_training_finished(self) -> None:
        self.uci_progress.setVisible(False)
        self.btn_uci_train.setEnabled(True)
        self.cb_uci_subject.setEnabled(True)
        self._uci_training_worker = None

    def _refresh_uci_benchmark_results(self) -> None:
        if not hasattr(self, "cb_uci_subject"):
            return
        if self._uci_training_worker is not None and self._uci_training_worker.isRunning():
            return
        subject = self.cb_uci_subject.currentData() or "por"
        project_root = Path(__file__).resolve().parents[3]
        metrics_path = (
            project_root
            / "outputs"
            / "uci_model_comparison"
            / f"uci_{subject}_model_metrics.csv"
        )
        if not metrics_path.is_file():
            self.table_uci_metrics.setRowCount(0)
            self.lbl_uci_status.setText(
                "Chưa có kết quả đã lưu cho bộ dữ liệu này. Bấm “Chạy huấn luyện và đánh giá” để bắt đầu."
            )
            self.lbl_uci_result_summary.setText("Chưa có kết quả thực nghiệm.")
            self.card_uci_best_model.set_value("—", "Chưa có kết quả", "green")
            self.card_uci_cv_mae.set_value("—", "Chưa có kết quả", "blue")
            self.card_uci_test_mae.set_value("—", "Chưa có kết quả", "violet")
            return
        try:
            metrics = pd.read_csv(metrics_path)
            required = {
                "cv_folds",
                "cv_mae_mean",
                "cv_mae_std",
                "cv_rmse_mean",
                "cv_r2_mean",
                "mae",
                "rmse",
                "r2",
                "cv_training_seconds",
                "train_rows",
                "test_rows",
            }
            if not required.issubset(metrics.columns):
                raise ValueError(
                    "Kết quả cũ chưa có đầy đủ chỉ số kiểm định chéo. Hãy chạy lại thực nghiệm."
                )
            if not pd.to_numeric(metrics["cv_folds"], errors="coerce").eq(5).all():
                raise ValueError(
                    "Kết quả đã lưu không sử dụng kiểm định chéo 5 phần. Hãy chạy lại từ trang này."
                )
            dataset_rows = 649 if subject == "por" else 395
            train_rows = int(metrics.iloc[0]["train_rows"])
            test_rows = int(metrics.iloc[0]["test_rows"])
            self._render_uci_benchmark_metrics(
                metrics,
                dataset_rows=dataset_rows,
                train_rows=train_rows,
                test_rows=test_rows,
            )
            self.lbl_uci_status.setText(
                f"Đang hiển thị kết quả thực nghiệm đã lưu: {metrics_path}"
            )
        except Exception as exc:
            logging.exception("Could not load saved UCI benchmark results")
            self.table_uci_metrics.setRowCount(0)
            self.lbl_uci_result_summary.setText("Chưa có kết quả thực nghiệm hợp lệ để hiển thị.")
            self.card_uci_best_model.set_value("—", "Chưa có kết quả hợp lệ", "green")
            self.card_uci_cv_mae.set_value("—", "Chưa có kết quả hợp lệ", "blue")
            self.card_uci_test_mae.set_value("—", "Chưa có kết quả hợp lệ", "violet")
            self.lbl_uci_status.setText(
                f"Không thể đọc kết quả đã lưu: {self._user_error(exc)}"
            )

    def _render_uci_benchmark_metrics(
        self,
        metrics: pd.DataFrame,
        dataset_rows: int,
        train_rows: int,
        test_rows: int,
    ) -> None:
        if metrics.empty:
            raise ValueError("Không có kết quả để hiển thị.")
        ordered = metrics.sort_values(
            by=["cv_mae_mean", "cv_rmse_mean"],
            ascending=True,
        ).reset_index(drop=True)
        best = ordered.iloc[0]
        self.card_uci_best_model.set_value(
            str(best["model"]),
            "Được chọn theo MAE trung bình của 5 lượt kiểm định chéo",
            "green",
        )
        self.card_uci_cv_mae.set_value(
            f"{best['cv_mae_mean']:.3f}",
            f"Độ lệch chuẩn: {best['cv_mae_std']:.3f}",
            "blue",
        )
        self.card_uci_test_mae.set_value(
            f"{best['mae']:.3f}",
            f"RMSE: {best['rmse']:.3f} · R²: {best['r2']:.3f}",
            "violet",
        )
        self.lbl_uci_result_summary.setText(
            f"Đã sử dụng {dataset_rows:,} bản ghi: {train_rows:,} bản ghi huấn luyện "
            f"và {test_rows:,} bản ghi kiểm thử. Ba thuật toán dùng chung 5 phần dữ liệu "
            "kiểm định chéo; tập kiểm thử chỉ được dùng ở bước đánh giá cuối cùng."
        )

        headers = [
            "Thuật toán",
            "MAE kiểm định chéo (TB ± ĐLC)",
            "RMSE kiểm định chéo",
            "R² kiểm định chéo",
            "MAE kiểm thử",
            "RMSE kiểm thử",
            "R² kiểm thử",
            "Thời gian 5 lượt (giây)",
        ]
        self.table_uci_metrics.setSortingEnabled(False)
        self.table_uci_metrics.setColumnCount(len(headers))
        self.table_uci_metrics.setHorizontalHeaderLabels(headers)
        self.table_uci_metrics.setRowCount(len(ordered))
        for row_index, (_, row) in enumerate(ordered.iterrows()):
            values = [
                str(row["model"]),
                f"{row['cv_mae_mean']:.3f} ± {row['cv_mae_std']:.3f}",
                f"{row['cv_rmse_mean']:.3f}",
                f"{row['cv_r2_mean']:.3f}",
                f"{row['mae']:.3f}",
                f"{row['rmse']:.3f}",
                f"{row['r2']:.3f}",
                f"{row['cv_training_seconds']:.2f}",
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table_uci_metrics.setItem(row_index, column_index, item)
        self.table_uci_metrics.setSortingEnabled(True)

    def _build_reports_page(self, body: QWidget, layout: QVBoxLayout) -> None:
        frame, frame_layout = self._surface("Xuất dữ liệu và kết quả")
        intro = QLabel("Chọn loại dữ liệu đang có trong phiên làm việc rồi xuất ra CSV hoặc Excel.")
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        frame_layout.addWidget(intro)
        row = QHBoxLayout()
        self.cb_report_type = QComboBox()
        self.cb_report_type.addItem("Bộ dữ liệu đã nhập", "dataset")
        self.cb_report_type.addItem("Kết quả dự đoán điểm", "prediction")
        self.cb_report_type.addItem("Kết quả nguy cơ học vụ", "risk")
        row.addWidget(self.cb_report_type, 1)
        self.btn_export_report = self._primary_button("Xuất file", self._handle_export_report)
        row.addWidget(self.btn_export_report)
        frame_layout.addLayout(row)
        self.lbl_report_status = QLabel("Chưa xuất báo cáo nào trong phiên này.")
        self.lbl_report_status.setWordWrap(True)
        self.lbl_report_status.setObjectName("Muted")
        frame_layout.addWidget(self.lbl_report_status)
        layout.addWidget(frame)

        info_frame, info_layout = self._surface("Quy tắc xuất")
        info = QLabel(
            "Dữ liệu nguồn không bị sửa đổi khi xuất kết quả. Kiểm tra file đích trước khi chia sẻ vì báo cáo có thể chứa mã sinh viên hoặc thông tin học tập."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color:#475467;")
        info_layout.addWidget(info)
        layout.addWidget(info_frame)
        layout.addStretch(1)

    def _build_settings_page(self, body: QWidget, layout: QVBoxLayout) -> None:
        frame, frame_layout = self._surface("Môi trường và đường dẫn")
        self.settings_text = QLabel("Đang đọc cấu hình…")
        self.settings_text.setWordWrap(True)
        self.settings_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        frame_layout.addWidget(self.settings_text)
        self.btn_settings_refresh = QPushButton("Đọc lại cấu hình")
        self.btn_settings_refresh.clicked.connect(self._refresh_settings)
        frame_layout.addWidget(self.btn_settings_refresh, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(frame)

        rules_frame, rules_layout = self._surface("Nguyên tắc nghiệp vụ")
        rules = QLabel(
            "Công thức không TX2: 0,2 × TX1 + 0,2 × giữa kỳ + 0,6 × điểm cuối kỳ dự đoán.\n"
            "Công thức có TX2: 0,15 × TX1 + 0,15 × TX2 + 0,2 × giữa kỳ + 0,5 × điểm cuối kỳ dự đoán.\n"
            "Trọng số nguy cơ: nghỉ học 75%, điểm số 8%, LMS 8%, tương tác 4%, đi làm thêm 2%, stress 1,5%, động lực 1,5%.\n"
            "Tỷ lệ nghỉ học trên 30% bật trạng thái override riêng."
        )
        rules.setWordWrap(True)
        rules.setStyleSheet("color:#475467;line-height:1.5;")
        rules_layout.addWidget(rules)
        layout.addWidget(rules_frame)

    def _create_table(self) -> QTableWidget:
        table = QTableWidget()
        table.setAlternatingRowColors(True)
        table.setShowGrid(False)
        table.setWordWrap(False)
        table.setSortingEnabled(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        vertical_header = table.verticalHeader()
        if vertical_header is not None:
            vertical_header.setVisible(False)
        horizontal_header = table.horizontalHeader()
        if horizontal_header is not None:
            horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
            horizontal_header.setStretchLastSection(True)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return table

    def _switch_page(self, index: int) -> None:
        if index < 0 or index >= self.content_area.count():
            return
        self.content_area.setCurrentIndex(index)
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
        self.settings_nav_button.setChecked(index == 8)
        self.topbar_context.setText(self._page_titles[index].upper())
        if index == 0:
            self._update_dashboard()
        elif index == 2:
            self._refresh_eda_options()
        elif index == 5:
            self._update_model_status()
        elif index == 6:
            self._refresh_uci_benchmark_results()
        elif index == 8:
            self._refresh_settings()

    def _update_all_status(self) -> None:
        self._update_dashboard()
        self._update_model_status()
        self._update_global_badges()

    def _update_global_badges(self) -> None:
        if self.current_df is None:
            self.lbl_dataset_badge.setText("Chưa có dữ liệu")
            self.lbl_dataset_badge.setStyleSheet("background:#F2F4F7;color:#475467;padding:7px 10px;border-radius:7px;font-size:11px;font-weight:600;")
        else:
            filename = Path(self.current_file_path).name if self.current_file_path else "Bộ dữ liệu đã nạp"
            self.lbl_dataset_badge.setText(f"{filename} · {len(self.current_df):,} dòng")
            self.lbl_dataset_badge.setStyleSheet("background:#ECFDF3;color:#027A48;padding:7px 10px;border-radius:7px;font-size:11px;font-weight:600;")
        ready = self._is_model_ready()
        if ready:
            self.lbl_model_badge.setText(f"XGBoost {ACTIVE_MODEL_VERSION} sẵn sàng")
            self.lbl_model_badge.setStyleSheet("background:#ECFDF3;color:#027A48;padding:7px 10px;border-radius:7px;font-size:11px;font-weight:600;")
        else:
            self.lbl_model_badge.setText("Mô hình chưa sẵn sàng")
            self.lbl_model_badge.setStyleSheet("background:#FEF3F2;color:#B42318;padding:7px 10px;border-radius:7px;font-size:11px;font-weight:600;")

    def _is_model_ready(self) -> bool:
        try:
            checker = getattr(self.prediction_service, "is_model_ready", None)
            return bool(checker(ACTIVE_MODEL_VERSION)) if callable(checker) else False
        except Exception:
            logging.exception("Unable to check model readiness")
            return False

    def _update_dashboard(self) -> None:
        dataframe = self.current_df
        if dataframe is None:
            self.card_records.set_value("0", "Chưa có bộ dữ liệu", "blue")
            self.card_missing.set_value("—", "Chưa có bộ dữ liệu", "amber")
            self.card_absence.set_value("—", "Ngưỡng cấu hình > 30%", "red")
            self.card_risk_override_value = None
            self.dashboard_summary.setText("Chưa có dữ liệu được nạp vào phiên làm việc.")
            self._show_dashboard_placeholder("Nhập dữ liệu để hiển thị phân tích.")
        else:
            df = dataframe
            missing_cells = int(df.isna().sum().sum())
            if "absence_percentage" in df.columns:
                absence = pd.to_numeric(df["absence_percentage"], errors="coerce")
                override_count = int((absence > 30).sum())
                self.card_absence.set_value(f"{override_count:,}", "Tỷ lệ nghỉ học > 30%", "red")
            else:
                override_count = 0
                self.card_absence.set_value("—", "Không có cột tỷ lệ nghỉ học", "amber")
            self.card_records.set_value(f"{len(df):,}", f"{len(df.columns)} cột trong bộ dữ liệu", "blue")
            self.card_missing.set_value(f"{missing_cells:,}", "Ô thiếu trong dữ liệu đang nạp", "amber" if missing_cells else "green")
            path_text = Path(self.current_file_path).name if self.current_file_path else "Bộ dữ liệu hiện tại"
            summary_lines = [
                f"Nguồn dữ liệu: {path_text}",
                f"Số bản ghi: {len(df):,}",
                f"Số cột: {len(df.columns):,}",
                f"Ô dữ liệu thiếu: {missing_cells:,}",
                f"Bản ghi có tỷ lệ nghỉ học trên 30%: {override_count:,}",
            ]
            if self.prediction_df is not None:
                summary_lines.append(f"Bản ghi đã dự đoán điểm: {len(self.prediction_df):,}")
            if self.risk_df is not None:
                summary_lines.append(f"Bản ghi có kết quả nguy cơ: {len(self.risk_df):,}")
            self.dashboard_summary.setText("\n".join(summary_lines))
            self._render_dashboard_chart()
        ready = self._is_model_ready()
        self.card_model.set_value(
            "Sẵn sàng" if ready else "Chưa sẵn sàng",
            "Tệp mô hình hợp lệ" if ready else "Cần huấn luyện và lưu mô hình hợp lệ",
            "green" if ready else "violet",
        )
        self._update_global_badges()

    def _show_dashboard_placeholder(self, message: str) -> None:
        self._clear_layout(self.dashboard_chart_layout)
        label = QLabel(message)
        label.setObjectName("Muted")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumHeight(220)
        self.dashboard_chart_layout.addWidget(label)

    def _render_dashboard_chart(self) -> None:
        if self.current_df is None or "absence_percentage" not in self.current_df.columns:
            self._show_dashboard_placeholder("Bộ dữ liệu chưa có cột tỷ lệ nghỉ học.")
            return
        try:
            figure = self.chart_service.dist_plots.plot_histogram(self.current_df, "absence_percentage")
            figure.set_size_inches(7.5, 3.6, forward=True)
            self._clear_layout(self.dashboard_chart_layout)
            canvas = FigureCanvas(figure)
            self.dashboard_chart_layout.addWidget(canvas)
        except Exception as exc:
            logging.exception("Dashboard chart failed")
            self._show_dashboard_placeholder(f"Chưa thể tạo biểu đồ: {exc}")

    def _sync_course_profile_from_dataset(self, dataframe: pd.DataFrame) -> None:
        profile_id = None
        notice = (
            "Không thể tự xác định học phần có TX2 hay không. "
            "Hãy chọn cấu hình đánh giá đúng theo học phần trước khi dự đoán; "
            "không xem cột TX2 vắng hoặc trống là bằng chứng môn không có TX2."
        )

        if "tx2_score" in dataframe.columns:
            tx2_values = pd.to_numeric(dataframe["tx2_score"], errors="coerce")
            if not tx2_values.empty and tx2_values.notna().all():
                profile_id = "with_tx2"
                notice = (
                    "Đã chọn cấu hình có TX2 vì tất cả bản ghi đều có điểm TX2. "
                    "Hãy xác nhận cấu hình này phù hợp với quy chế đánh giá của học phần."
                )
            elif tx2_values.notna().any():
                notice = (
                    "Dữ liệu có TX2 ở một số dòng nhưng thiếu ở dòng khác. "
                    "Hãy tách file theo cấu hình học phần hoặc xử lý các điểm TX2 bị thiếu "
                    "trước khi dự đoán để tránh tính sai điểm học phần."
                )

        if profile_id is None:
            self.cb_course_profile.setCurrentIndex(0)
        else:
            profile_index = self.cb_course_profile.findData(profile_id)
            if profile_index >= 0:
                self.cb_course_profile.setCurrentIndex(profile_index)
        self.lbl_prediction_notice.setText(notice)

    def _handle_import(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn dữ liệu sinh viên",
            "",
            "Dữ liệu CSV/Excel (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls)",
        )
        if not file_path:
            return
        self.btn_import.setEnabled(False)
        self.lbl_import_status.setText("Đang đọc và kiểm tra dữ liệu…")
        QApplication.processEvents()
        try:
            from haui_student_prediction.config.course_profiles import DEFAULT_COURSES_WITH_TX2

            required_columns = ["student_id", "absence_percentage", "tx1_score", "midterm_score"]
            result = self.dataset_service.import_dataset(
                file_path,
                required_columns,
                list(DEFAULT_COURSES_WITH_TX2),
            )
            imported_data = result.data
            if not result.is_success or imported_data is None:
                error_lines = []
                if result.missing_columns:
                    error_lines.append("Thiếu cột bắt buộc: " + ", ".join(result.missing_columns))
                if result.duplicate_indices:
                    error_lines.append(f"Bản ghi trùng: {len(result.duplicate_indices)}")
                if result.errors:
                    error_lines.append("Chi tiết lỗi:")
                    for error in result.errors[:8]:
                        column = error.get("column", "")
                        message = error.get("error", "Dữ liệu không hợp lệ")
                        rows = error.get("rows", [])
                        row_text = f"; dòng: {rows[:8]}" if rows else ""
                        error_lines.append(f"• {column}: {message}{row_text}")
                detail = "\n".join(error_lines) if error_lines else "File không vượt qua kiểm tra dữ liệu."
                self.lbl_import_status.setText("Nhập dữ liệu thất bại. Dữ liệu hiện tại được giữ nguyên.")
                QMessageBox.warning(self, "Không thể nhập dữ liệu", detail)
                return

            current_df = imported_data.copy(deep=True)
            self.current_df = current_df
            self.current_file_path = file_path
            self._sync_course_profile_from_dataset(current_df)
            self.prediction_df = None
            self.risk_df = None
            warnings = getattr(result, "warnings", []) or []
            warning_text = f" Có {len(warnings)} cảnh báo cần xem lại." if warnings else ""
            self.lbl_import_status.setText(
                f"Đã nhập {len(current_df):,} bản ghi từ {Path(file_path).name}.{warning_text}"
            )
            self.lbl_quality_summary.setText(self._build_quality_summary(current_df, warnings))
            self.lbl_quality_summary.setStyleSheet(
                "background:#ECFDF3;color:#027A48;padding:11px;border-radius:8px;" if not warnings
                else "background:#FFFAEB;color:#B54708;padding:11px;border-radius:8px;"
            )
            self._refresh_data_table()
            self._refresh_eda_options()
            self._update_all_status()
            self._refresh_report_options()
            self._refresh_model_page_info()
            if warnings:
                QMessageBox.information(
                    self,
                    "Đã nhập dữ liệu kèm cảnh báo",
                    "Dữ liệu đã được nạp nhưng cần xem các cảnh báo thang điểm hoặc chất lượng trước khi phân tích tiếp.",
                )
        except Exception as exc:
            logging.exception("Dataset import failed")
            self.lbl_import_status.setText("Nhập dữ liệu thất bại. Dữ liệu hiện tại được giữ nguyên.")
            QMessageBox.critical(self, "Lỗi nhập dữ liệu", self._user_error(exc))
        finally:
            self.btn_import.setEnabled(True)

    def _build_quality_summary(self, dataframe: pd.DataFrame, warnings: list[Any]) -> str:
        missing_cells = int(dataframe.isna().sum().sum())
        missing_columns = int(dataframe.isna().any(axis=0).sum())
        lines = [
            f"Bản ghi: {len(dataframe):,} · Cột: {len(dataframe.columns):,}",
            f"Ô dữ liệu thiếu: {missing_cells:,} · Cột có giá trị thiếu: {missing_columns:,}",
            f"Cảnh báo validator: {len(warnings):,}",
        ]
        if "absence_percentage" in dataframe.columns:
            values = pd.to_numeric(dataframe["absence_percentage"], errors="coerce")
            lines.append(f"Bản ghi nghỉ học trên 30%: {int((values > 30).sum()):,}")
        return "\n".join(lines)

    def _refresh_data_table(self) -> None:
        dataframe = self.current_df
        if dataframe is None:
            self.table_import.clear()
            self.table_import.setRowCount(0)
            self.table_import.setColumnCount(0)
            self.lbl_table_count.setText("0 bản ghi")
            return
        self._populate_table(self.table_import, dataframe)
        displayed = min(len(dataframe), 1000)
        self.lbl_table_count.setText(
            f"Hiển thị {displayed:,}/{len(dataframe):,} bản ghi để giữ giao diện phản hồi tốt."
        )

    def _display_column_name(self, column: Any) -> str:
        labels = {
            "student_id": "Mã sinh viên",
            "full_name": "Họ và tên",
            "date_of_birth": "Ngày sinh",
            "birth_year": "Năm sinh",
            "hometown": "Quê quán",
            "major": "Ngành học",
            "course_id": "Mã học phần",
            "semester": "Học kỳ",
            "absence_percentage": "Tỷ lệ nghỉ học (%)",
            "lms_learning_percentage": "Tỷ lệ học tập trên LMS (%)",
            "tx1_score": "Điểm thường xuyên 1",
            "tx2_score": "Điểm thường xuyên 2",
            "midterm_score": "Điểm giữa kỳ",
            "target_final_exam_score": "Điểm thi cuối kỳ thực tế",
            "predicted_final_exam_score": "Điểm thi cuối kỳ dự đoán",
            "predicted_course_score": "Điểm học phần dự đoán",
            "interaction_score": "Mức độ tương tác",
            "stress_score": "Mức độ căng thẳng",
            "motivation_score": "Động lực học tập",
            "works_part_time": "Đi làm thêm",
            "absence_status": "Trạng thái nghỉ học",
            "weighted_score": "Điểm nguy cơ",
            "risk_category": "Mức nguy cơ học vụ",
            "override_triggered": "Kích hoạt ngưỡng bắt buộc",
            "evaluation_status": "Trạng thái đánh giá",
            "academic_risk_score": "Điểm nguy cơ học vụ",
            "academic_risk_category": "Mức nguy cơ học vụ",
            "academic_risk_override": "Trạng thái vượt ngưỡng nghỉ học",
            "academic_risk_status": "Trạng thái đánh giá nguy cơ",
            "snapshot_date": "Ngày chốt dữ liệu",
            "source_file": "Tệp dữ liệu nguồn",
            "dataset_version": "Phiên bản bộ dữ liệu",
            "weekly_study_time_level": "Mức thời gian tự học",
            "past_class_failures": "Số lần trượt môn trước đây",
            "absence_count": "Số buổi vắng",
            "extra_school_support": "Được hỗ trợ học tập tại trường",
            "family_education_support": "Được gia đình hỗ trợ học tập",
            "paid_subject_classes": "Tham gia lớp học thêm có phí",
            "higher_education_intent": "Dự định học lên bậc cao hơn",
            "home_internet_access": "Có Internet tại nhà",
            "period1_grade": "Điểm giai đoạn 1",
            "period2_grade": "Điểm giai đoạn 2",
            "target_course_grade": "Điểm tổng kết môn học",
        }
        return labels.get(str(column), str(column).replace("_", " ").capitalize())

    def _populate_table(self, table: QTableWidget, dataframe: pd.DataFrame, max_rows: int = 1000) -> None:
        table.setSortingEnabled(False)
        preview = dataframe.head(max_rows).copy()
        table.clear()
        table.setRowCount(len(preview))
        table.setColumnCount(len(preview.columns))
        table.setHorizontalHeaderLabels(
            [self._display_column_name(column) for column in preview.columns]
        )
        boolean_columns = {
            "works_part_time",
            "override_triggered",
            "academic_risk_override",
        }
        for row_index in range(len(preview)):
            for column_index, column in enumerate(preview.columns):
                value = preview.iat[row_index, column_index]
                if pd.isna(value):
                    text = "Chưa có dữ liệu"
                elif column in boolean_columns and isinstance(value, (bool, np.bool_)):
                    text = "Có" if bool(value) else "Không"
                elif column == "absence_status" and str(value) == "HOC_LAI":
                    text = "HỌC LẠI"
                elif isinstance(value, (float, np.floating)):
                    text = f"{float(value):.3f}" if np.isfinite(value) else "Không hợp lệ"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                table.setItem(row_index, column_index, item)
        table.resizeColumnsToContents()
        header = table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        table.setSortingEnabled(True)

    def _refresh_eda_options(self) -> None:
        dataframe = self.current_df
        selectors = [self.cb_eda_col, self.cb_eda_category, self.cb_eda_x, self.cb_eda_y]
        for selector in selectors:
            selector.blockSignals(True)
            selector.clear()
        if dataframe is not None:
            excluded = {"student_id", "full_name"}
            numeric_columns = [
                str(column)
                for column in dataframe.select_dtypes(include="number").columns
                if str(column) not in excluded and dataframe[column].nunique(dropna=True) > 1
            ]
            categorical_columns = [
                str(column)
                for column in dataframe.columns
                if str(column) not in excluded
                and (
                    pd.api.types.is_object_dtype(dataframe[column])
                    or pd.api.types.is_bool_dtype(dataframe[column])
                    or isinstance(dataframe[column].dtype, pd.CategoricalDtype)
                )
                and dataframe[column].nunique(dropna=True) > 1
            ]
            self.cb_eda_col.addItems(numeric_columns + categorical_columns)
            self.cb_eda_category.addItems(categorical_columns)
            self.cb_eda_x.addItems(numeric_columns)
            self.cb_eda_y.addItems(numeric_columns)
        for selector in selectors:
            selector.blockSignals(False)
        self._update_eda_controls()
        self._refresh_eda_summary()

    def _update_eda_controls(self) -> None:
        if not hasattr(self, "cb_chart_type"):
            return
        kind = self.cb_chart_type.currentData()
        self.cb_eda_category.setEnabled(kind == "boxplot")
        self.cb_eda_x.setEnabled(kind == "scatter")
        self.cb_eda_y.setEnabled(kind == "scatter")
        self.cb_eda_col.setEnabled(kind in {"histogram", "count", "boxplot"})
        if kind == "count":
            self.btn_plot.setText("Tạo biểu đồ đếm")
        elif kind == "boxplot":
            self.btn_plot.setText("Tạo boxplot")
        elif kind == "scatter":
            self.btn_plot.setText("Tạo scatter")
        elif kind == "heatmap":
            self.btn_plot.setText("Tạo heatmap")
        else:
            self.btn_plot.setText("Tạo histogram")

    def _handle_plot(self) -> None:
        if self.current_df is None or self.current_df.empty:
            QMessageBox.warning(self, "Chưa có dữ liệu", "Hãy nhập một bộ dữ liệu hợp lệ trước khi tạo biểu đồ.")
            return
        kind = self.cb_chart_type.currentData()
        figure = None
        try:
            if kind == "histogram":
                column = self.cb_eda_col.currentText()
                if not column:
                    raise ValueError("Chọn một biến số để tạo histogram.")
                figure = self.chart_service.dist_plots.plot_histogram(self.current_df, column)
            elif kind == "count":
                column = self.cb_eda_col.currentText()
                if not column:
                    raise ValueError("Chọn một biến để tạo biểu đồ đếm.")
                figure = self.chart_service.dist_plots.plot_count(self.current_df, column)
            elif kind == "boxplot":
                category = self.cb_eda_category.currentText()
                numeric = self.cb_eda_col.currentText()
                if not category or not numeric:
                    raise ValueError("Chọn biến số và biến nhóm để tạo boxplot.")
                figure = self.chart_service.relationship_plots.plot_boxplot(self.current_df, category, numeric)
            elif kind == "scatter":
                x_column = self.cb_eda_x.currentText()
                y_column = self.cb_eda_y.currentText()
                if not x_column or not y_column:
                    raise ValueError("Chọn đủ biến X và biến Y.")
                if x_column == y_column:
                    raise ValueError("Biến X và Y phải khác nhau.")
                figure = self.chart_service.relationship_plots.plot_scatter(self.current_df, x_column, y_column)
            elif kind == "heatmap":
                matrix = self.chart_service.statistics.get_correlation_matrix(self.current_df)
                if matrix.empty:
                    raise ValueError("Bộ dữ liệu không có đủ biến số để tạo biểu đồ tương quan.")
                import matplotlib.pyplot as plt
                import seaborn as sns

                figure, axis = plt.subplots(figsize=(9, 6))
                sns.heatmap(matrix, annot=matrix.shape[0] <= 12, cmap="Blues", vmin=-1, vmax=1, center=0, ax=axis)
                axis.set_title("Ma trận tương quan")
                figure.tight_layout()
            else:
                raise ValueError("Loại biểu đồ không được hỗ trợ.")
            self._display_chart(figure)
        except Exception as exc:
            logging.exception("EDA plot generation failed")
            QMessageBox.warning(self, "Không thể tạo biểu đồ", self._user_error(exc))

    def _display_chart(self, figure) -> None:
        self._clear_layout(self.chart_container)
        figure.tight_layout()
        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)
        self.chart_container.addWidget(toolbar)
        self.chart_container.addWidget(canvas, 1)
        self._current_figure = figure

    def _save_current_chart(self) -> None:
        figure = getattr(self, "_current_figure", None)
        if figure is None:
            QMessageBox.information(self, "Chưa có biểu đồ", "Hãy tạo biểu đồ trước khi lưu.")
            return
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Lưu biểu đồ",
            "eda_chart.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not file_path:
            return
        suffix = Path(file_path).suffix.lower()
        if not suffix:
            extension = "svg" if "SVG" in selected_filter else "pdf" if "PDF" in selected_filter else "png"
            file_path = f"{file_path}.{extension}"
        try:
            figure.savefig(file_path, dpi=160, bbox_inches="tight")
            QMessageBox.information(self, "Đã lưu biểu đồ", f"Biểu đồ đã được lưu tại:\n{file_path}")
        except Exception as exc:
            logging.exception("Chart export failed")
            QMessageBox.critical(self, "Lỗi lưu biểu đồ", self._user_error(exc))

    def _refresh_eda_summary(self) -> None:
        if self.current_df is None:
            self.table_eda_stats.clear()
            self.table_eda_stats.setRowCount(0)
            self.table_eda_stats.setColumnCount(0)
            return
        try:
            summary = self.chart_service.statistics.get_numeric_summary(self.current_df)
            records = []
            for column, values in summary.items():
                if column in {"student_id", "full_name"}:
                    continue
                if not isinstance(values, dict):
                    continue
                for metric, value in values.items():
                    records.append({"Biến": column, "Thống kê": metric, "Giá trị": value})
            summary_df = pd.DataFrame(records)
            self._populate_table(self.table_eda_stats, summary_df, max_rows=300)
        except Exception as exc:
            logging.exception("EDA summary failed")
            self.table_eda_stats.clear()
            self.table_eda_stats.setRowCount(1)
            self.table_eda_stats.setColumnCount(1)
            self.table_eda_stats.setHorizontalHeaderLabels(["Trạng thái"])
            self.table_eda_stats.setItem(0, 0, QTableWidgetItem(self._user_error(exc)))

    def _handle_train_model(self) -> None:
        if self._training_worker is not None and self._training_worker.isRunning():
            QMessageBox.information(self, "Đang huấn luyện", "Một tác vụ huấn luyện đang chạy.")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn dữ liệu lịch sử để huấn luyện XGBoost",
            "",
            "Dữ liệu CSV/Excel (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls)",
        )
        if not file_path:
            return

        try:
            from haui_student_prediction.data.training_dataset import (
                TARGET_COLUMN,
                load_historical_training_dataset,
            )

            training_data = load_historical_training_dataset(file_path)
            target = pd.to_numeric(training_data[TARGET_COLUMN], errors="coerce")
            valid_rows = int((target.notna() & np.isfinite(target.astype(float))).sum())
            if valid_rows < 10:
                raise ValueError(
                    f"Chỉ có {valid_rows} nhãn hợp lệ. Cần ít nhất 10 bản ghi lịch sử để huấn luyện."
                )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Không thể nạp dữ liệu huấn luyện",
                self._user_error(exc),
            )
            return

        self.training_file_path = file_path
        self.btn_train_model.setEnabled(False)
        self.btn_train_from_prediction.setEnabled(False)
        self.training_progress.setVisible(True)
        self.lbl_training_status.setText(
            f"Đã nạp {valid_rows:,} nhãn từ {Path(file_path).name}; đang chia tập và huấn luyện…"
        )
        self._training_worker = _TrainingWorker(training_data, self)
        self._training_worker.succeeded.connect(self._on_training_succeeded)
        self._training_worker.failed.connect(self._on_training_failed)
        self._training_worker.finished.connect(self._on_training_finished)
        self._training_worker.start()

    def _on_training_succeeded(self, result: dict[str, Any]) -> None:
        metrics = result["metrics"]
        mae = self._format_metric(metrics.mae)
        rmse = self._format_metric(metrics.rmse)
        r2 = self._format_metric(metrics.r2)
        self.lbl_training_status.setText(
            f"Đã lưu mô hình {result['version']}. Tập huấn luyện: {result['train_rows']:,} dòng; tập kiểm thử: {result['test_rows']:,} dòng."
        )
        self.lbl_model_metrics.setText(
            f"MAE: {mae}    RMSE: {rmse}    R²: {r2}\n"
            f"Số đặc trưng: {result['feature_count']}\n"
            f"Mã nhận diện bộ dữ liệu: {result['dataset_hash']}\n"
            f"Các đặc trưng: {', '.join(result['features'])}"
        )
        self._update_all_status()
        self._refresh_model_page_info()
        QMessageBox.information(
            self,
            "Huấn luyện hoàn tất",
            f"Mô hình {result['version']} đã được lưu và xác minh trong kho mô hình.\n\nMAE: {mae}\nRMSE: {rmse}\nR²: {r2}",
        )

    def _on_training_failed(self, message: str) -> None:
        self.lbl_training_status.setText("Huấn luyện chưa hoàn tất. Xem thông báo để biết nguyên nhân.")
        QMessageBox.critical(self, "Huấn luyện thất bại", message)

    def _on_training_finished(self) -> None:
        self.btn_train_model.setEnabled(True)
        self.btn_train_from_prediction.setEnabled(True)
        self.training_progress.setVisible(False)
        self._training_worker = None
        self._update_model_status()

    def _handle_predict(self) -> None:
        dataframe = self.current_df
        if dataframe is None or dataframe.empty:
            QMessageBox.warning(self, "Chưa có dữ liệu", "Hãy nhập bộ dữ liệu trước khi dự đoán.")
            return
        if "tx2_score" in dataframe.columns:
            tx2_values = pd.to_numeric(dataframe["tx2_score"], errors="coerce")
            if tx2_values.notna().any() and tx2_values.isna().any():
                QMessageBox.warning(
                    self,
                    "Dữ liệu TX2 chưa đồng nhất",
                    "Một số bản ghi có TX2 nhưng một số bản ghi khác không có điểm TX2. "
                    "Hãy tách file theo cấu trúc đánh giá hoặc xử lý điểm TX2 bị thiếu "
                    "trước khi dự đoán. Không thể áp dụng một công thức điểm học phần chính xác "
                    "cho toàn bộ file ở trạng thái hiện tại.",
                )
                return
        profile_id = self.cb_course_profile.currentData()
        if profile_id not in {"no_tx2", "with_tx2"}:
            QMessageBox.warning(
                self,
                "Chưa xác định cấu hình TX2",
                "Hãy chọn rõ học phần có TX2 hoặc không có TX2 theo cấu trúc đánh giá thực tế. "
                "Khi cột TX2 không có hoặc toàn ô trống, ứng dụng không tự suy luận công thức điểm.",
            )
            return

        tx2_values = (
            pd.to_numeric(dataframe["tx2_score"], errors="coerce")
            if "tx2_score" in dataframe.columns
            else pd.Series(np.nan, index=dataframe.index)
        )
        if profile_id == "with_tx2":
            if tx2_values.isna().any():
                QMessageBox.warning(
                    self,
                    "Thiếu điểm TX2",
                    "Bạn đã chọn học phần có TX2 nhưng còn bản ghi chưa có điểm TX2. "
                    "Hãy bổ sung điểm hoặc tách những bản ghi chưa đủ dữ liệu trước khi dự đoán.",
                )
                return
        elif tx2_values.notna().any():
            QMessageBox.warning(
                self,
                "Cấu hình TX2 không khớp dữ liệu",
                "File đang có điểm TX2 nhưng bạn chọn công thức không có TX2. "
                "Hãy kiểm tra cấu trúc đánh giá và chọn đúng cấu hình trước khi tính điểm học phần.",
            )
            return

        if not self._is_model_ready():
            self._update_model_status()
            QMessageBox.warning(
                self,
                "Mô hình chưa sẵn sàng",
                "Chưa tìm thấy mô hình đã huấn luyện hợp lệ. Mở trang Mô hình dự đoán HaUI và chọn một tệp dữ liệu lịch sử để huấn luyện trước khi dự đoán.",
            )
            return
        try:
            self.btn_predict.setEnabled(False)
            self.lbl_prediction_status.setText("Đang dự đoán…")
            QApplication.processEvents()
            results = self.prediction_service.predict_student_results(
                dataframe.copy(deep=True),
                model_version=ACTIVE_MODEL_VERSION,
                course_profile_id=profile_id,
            )

            risk_rows = []
            for row_number, (_, row) in enumerate(results.iterrows(), start=1):
                student_id = str(row.get("student_id", f"Bản ghi {row_number}"))
                try:
                    risk_result = self.risk_service.evaluate_student_risk(row)
                    override = bool(risk_result.is_override)
                    risk_rows.append(
                        {
                            "student_id": student_id,
                            "absence_status": "HOC_LAI" if override else self._absence_label(risk_result.absence_status),
                            "weighted_score": round(float(risk_result.weighted_score), 2),
                            "risk_category": self._risk_label(risk_result.risk_category),
                            "override_triggered": override,
                            "evaluation_status": "Đã đánh giá",
                        }
                    )
                except Exception as exc:
                    risk_rows.append(
                        {
                            "student_id": student_id,
                            "absence_status": "Chưa xác định",
                            "weighted_score": np.nan,
                            "risk_category": "Chưa đủ dữ liệu",
                            "override_triggered": False,
                            "evaluation_status": self._user_error(exc),
                        }
                    )

            self.risk_df = pd.DataFrame(risk_rows)
            risk_columns = self.risk_df.drop(columns=["student_id"]).rename(
                columns={
                    "weighted_score": "academic_risk_score",
                    "risk_category": "academic_risk_category",
                    "override_triggered": "academic_risk_override",
                    "evaluation_status": "academic_risk_status",
                }
            )
            results = pd.concat(
                [results.reset_index(drop=True), risk_columns.reset_index(drop=True)],
                axis=1,
            )
            self.prediction_df = results.copy(deep=True)
            self._populate_table(self.table_predict, self.prediction_df)
            self.lbl_prediction_status.setText(
                f"Đã dự đoán {len(self.prediction_df):,} bản ghi bằng XGBoost {ACTIVE_MODEL_VERSION}. Bảng kết quả gồm điểm cuối kỳ dự kiến, điểm học phần dự kiến và đánh giá nguy cơ học vụ."
            )
            self._refresh_report_options()
            self._update_dashboard()
            QMessageBox.information(self, "Dự đoán hoàn tất", f"Đã tạo kết quả dự đoán cho {len(self.prediction_df):,} bản ghi.")
        except Exception as exc:
            logging.exception("Score prediction failed")
            self.lbl_prediction_status.setText("Dự đoán thất bại. Kiểm tra cấu trúc đặc trưng và dữ liệu đầu vào.")
            QMessageBox.critical(self, "Không thể dự đoán", self._user_error(exc))
        finally:
            self.btn_predict.setEnabled(True)

    def _handle_risk(self) -> None:
        source = self.prediction_df
        if source is None and self.current_df is not None and "predicted_course_score" in self.current_df.columns:
            source = self.current_df
        if source is None or source.empty:
            QMessageBox.warning(
                self,
                "Chưa có điểm học phần dự kiến",
                "Hãy hoàn thành dự đoán điểm học phần trước khi tính nguy cơ học vụ.",
            )
            return
        rows = []
        errors = 0
        for row_number, (_, row) in enumerate(source.iterrows(), start=1):
            student_id = str(row.get("student_id", f"Bản ghi {row_number}"))
            try:
                result = self.risk_service.evaluate_student_risk(row)
                is_override = bool(result.is_override)
                category = self._risk_label(result.risk_category)
                rows.append(
                    {
                        "student_id": student_id,
                        "absence_status": "HOC_LAI" if is_override else self._absence_label(result.absence_status),
                        "weighted_score": round(float(result.weighted_score), 2),
                        "risk_category": category,
                        "override_triggered": is_override,
                        "evaluation_status": "Đã đánh giá",
                    }
                )
            except Exception as exc:
                errors += 1
                rows.append(
                    {
                        "student_id": student_id,
                        "absence_status": "Chưa xác định",
                        "weighted_score": np.nan,
                        "risk_category": "Chưa đủ dữ liệu",
                        "override_triggered": False,
                        "evaluation_status": self._user_error(exc),
                    }
                )
        self.risk_df = pd.DataFrame(rows)
        self._populate_table(self.table_risk, self.risk_df)
        valid_scores = pd.to_numeric(self.risk_df["weighted_score"], errors="coerce")
        valid = valid_scores.notna() & np.isfinite(valid_scores.astype(float))
        overrides = int(self.risk_df.loc[valid, "override_triggered"].astype(bool).sum()) if valid.any() else 0
        high_categories = {"Cao", "Rất cao"}
        high_count = int(self.risk_df.loc[valid, "risk_category"].isin(high_categories).sum()) if valid.any() else 0
        self.card_risk_evaluated.set_value(f"{int(valid.sum()):,}", f"Tổng bản ghi: {len(self.risk_df):,}", "green")
        self.card_risk_override.set_value(f"{overrides:,}", "Trạng thái học lại", "red")
        self.card_risk_high.set_value(f"{high_count:,}", "Nguy cơ cao hoặc rất cao", "amber")
        self.lbl_risk_status.setText(
            f"Đã đánh giá {int(valid.sum()):,}/{len(self.risk_df):,} bản ghi."
            + (f" {errors:,} bản ghi chưa đủ dữ liệu hoặc gặp lỗi; xem cột evaluation_status." if errors else "")
        )
        self._refresh_report_options()
        self._update_dashboard()

    def _update_model_status(self) -> None:
        ready = self._is_model_ready()
        if hasattr(self, "lbl_model_page_status"):
            if ready:
                self.lbl_model_page_status.setText(f"XGBoost {ACTIVE_MODEL_VERSION} đã sẵn sàng")
                self.lbl_model_page_status.setStyleSheet("font-size:16px;font-weight:700;color:#027A48;")
                self.lbl_model_details.setText(f"Kho lưu trữ xác nhận có thông tin và tệp mô hình XGBoost {ACTIVE_MODEL_VERSION}.")
            else:
                self.lbl_model_page_status.setText("Chưa có mô hình dùng được")
                self.lbl_model_page_status.setStyleSheet("font-size:16px;font-weight:700;color:#B42318;")
                self.lbl_model_details.setText(
                    "Không tìm thấy mô hình đã huấn luyện hợp lệ. Chọn riêng một tệp dữ liệu lịch sử có nhãn khi bấm Huấn luyện XGBoost; tệp sinh viên cần dự đoán không cần điểm cuối kỳ."
                )
        if hasattr(self, "card_model"):
            self.card_model.set_value(
                "Sẵn sàng" if ready else "Chưa sẵn sàng",
                "Tệp mô hình hợp lệ" if ready else "Cần huấn luyện và lưu mô hình hợp lệ",
                "green" if ready else "violet",
            )
        self._update_global_badges()

    def _refresh_model_page_info(self) -> None:
        self._update_model_status()
        if self.training_file_path and self._training_worker is None:
            self.lbl_training_status.setText(
                f"Tệp lịch sử huấn luyện đã chọn: {Path(self.training_file_path).name}. "
                "Có thể chọn tệp khác để huấn luyện lại."
            )
        elif self.current_df is not None and self._training_worker is None:
            self.lbl_training_status.setText(
                "Dữ liệu sinh viên đang dự đoán được tách riêng, không cần cột điểm cuối kỳ. "
                "Bấm Huấn luyện XGBoost để chọn tệp lịch sử có nhãn."
            )

    def _refresh_settings(self) -> None:
        try:
            from haui_student_prediction.app_config import load_config

            config = load_config()
            entries = [
                ("Môi trường", config.env),
                ("Dữ liệu gốc", config.data_raw_dir),
                ("Dữ liệu đã xử lý", config.data_processed_dir),
                ("Kho lưu mô hình", config.model_registry_path),
                ("Thư mục tệp mô hình", config.model_artifacts_dir),
                ("Thư mục báo cáo", config.report_output_dir),
            ]
            self.settings_text.setText("\n".join(f"{label}: {value}" for label, value in entries))
        except Exception as exc:
            logging.exception("Could not load application settings")
            self.settings_text.setText(f"Không thể đọc cấu hình ứng dụng: {self._user_error(exc)}")

    def _refresh_report_options(self) -> None:
        if not hasattr(self, "cb_report_type"):
            return
        current = self.cb_report_type.currentData()
        for index in range(self.cb_report_type.count() - 1, -1, -1):
            key = self.cb_report_type.itemData(index)
            available = (
                (key == "dataset" and self.current_df is not None)
                or (key == "prediction" and self.prediction_df is not None)
                or (key == "risk" and self.risk_df is not None)
            )
            self.cb_report_type.setItemText(index, {
                "dataset": "Bộ dữ liệu đã nhập",
                "prediction": "Kết quả dự đoán điểm",
                "risk": "Kết quả nguy cơ học vụ",
            }.get(key, str(key)))
            model = self.cb_report_type.model()
            if isinstance(model, QStandardItemModel):
                item = model.item(index)
                if item is not None:
                    item.setEnabled(available)
        if current is not None:
            self.cb_report_type.setCurrentIndex(max(0, self.cb_report_type.findData(current)))

    def _get_report_dataframe(self, report_type: str) -> pd.DataFrame:
        mapping = {
            "dataset": self.current_df,
            "prediction": self.prediction_df,
            "risk": self.risk_df,
        }
        dataframe = mapping.get(report_type)
        if dataframe is None or dataframe.empty:
            raise ValueError("Loại báo cáo này chưa có dữ liệu trong phiên làm việc.")
        return dataframe.copy(deep=True)

    def _handle_export_report(self) -> None:
        report_type = self.cb_report_type.currentData()
        self._export_dataframe(str(report_type))

    def _export_dataframe(self, report_type: str) -> None:
        try:
            dataframe = self._get_report_dataframe(report_type)
        except Exception as exc:
            QMessageBox.information(self, "Chưa có dữ liệu để xuất", self._user_error(exc))
            return
        default_name = {
            "dataset": "haui_dataset_cleaned.xlsx",
            "prediction": "haui_score_predictions.xlsx",
            "risk": "haui_academic_risk.xlsx",
        }.get(report_type, "haui_report.xlsx")
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Xuất dữ liệu",
            default_name,
            "Excel (*.xlsx);;CSV (*.csv)",
        )
        if not file_path:
            return
        suffix = Path(file_path).suffix.lower()
        if not suffix:
            suffix = ".csv" if "CSV" in selected_filter else ".xlsx"
            file_path = f"{file_path}{suffix}"
        try:
            if suffix == ".csv":
                dataframe.to_csv(file_path, index=False, encoding="utf-8-sig")
            elif suffix in {".xlsx", ".xls"}:
                dataframe.to_excel(file_path, index=False)
            else:
                raise ValueError("Định dạng xuất không được hỗ trợ. Hãy dùng .csv hoặc .xlsx.")
            self.lbl_report_status.setText(f"Đã xuất {len(dataframe):,} bản ghi: {file_path}")
            QMessageBox.information(self, "Xuất dữ liệu thành công", f"Đã lưu file tại:\n{file_path}")
        except Exception as exc:
            logging.exception("Report export failed")
            QMessageBox.critical(self, "Lỗi xuất dữ liệu", self._user_error(exc))

    def _risk_label(self, category: str) -> str:
        labels = {
            "Very low": "Rất thấp",
            "Low": "Thấp",
            "Medium": "Trung bình",
            "High": "Cao",
            "Very high": "Rất cao",
            "very low": "Rất thấp",
            "low": "Thấp",
            "medium": "Trung bình",
            "high": "Cao",
            "very high": "Rất cao",
        }
        return labels.get(str(category), str(category))

    def _absence_label(self, status: str) -> str:
        labels = {
            "REPEAT_REQUIRED": "Vượt ngưỡng nghỉ học",
            "high": "Nguy cơ cao",
            "medium": "Nguy cơ trung bình",
            "low": "Nguy cơ thấp",
            "very low": "Nguy cơ rất thấp",
        }
        return labels.get(str(status), str(status))

    def _format_metric(self, value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{number:.4f}" if np.isfinite(number) else "N/A"

    def _clear_layout(self, layout: QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _user_error(self, exc: Exception) -> str:
        message = str(exc).strip()
        return message if message else exc.__class__.__name__

    def closeEvent(self, a0: QCloseEvent) -> None:
        active_workers = (
            self._training_worker,
            self._uci_training_worker,
        )
        if any(worker is not None and worker.isRunning() for worker in active_workers):
            QMessageBox.warning(
                self,
                "Đang huấn luyện",
                "Hãy đợi các tác vụ huấn luyện kết thúc trước khi đóng ứng dụng.",
            )
            a0.ignore()
            return
        super().closeEvent(a0)
