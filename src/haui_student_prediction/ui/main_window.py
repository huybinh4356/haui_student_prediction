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
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
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
            self.style().unpolish(self.value_label)
            self.style().polish(self.value_label)


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
            dataset_hash = hashlib.sha256(
                pd.util.hash_pandas_object(data, index=True).values.tobytes()
            ).hexdigest()
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
                raise RuntimeError("Model đã được lưu nhưng không thể xác minh trạng thái registry.")

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


class MainWindow(QMainWindow):
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
        self._page_titles = [
            "Tổng quan",
            "Dữ liệu sinh viên",
            "Phân tích dữ liệu",
            "Dự đoán điểm",
            "Dự đoán nguy cơ học vụ",
            "Mô hình ML",
            "Báo cáo",
            "Cấu hình",
        ]
        self._page_subtitles = [
            "Theo dõi dữ liệu, chất lượng và trạng thái dự đoán.",
            "Nhập dữ liệu, kiểm tra chất lượng và xem bản ghi đã chuẩn hóa.",
            "Khám phá phân bố, xu hướng và mối liên hệ trong dữ liệu.",
            "Ước lượng điểm thi cuối kỳ và điểm học phần dự kiến.",
            "Đánh giá nguy cơ dựa trên dữ liệu học tập hiện có.",
            "Theo dõi, huấn luyện và xác minh mô hình dự đoán.",
            "Xuất dữ liệu và kết quả phân tích đang có.",
            "Xem cấu hình môi trường và đường dẫn hiện hành.",
        ]
        self.setWindowTitle("HaUI Student Analytics")
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
        brand_title = QLabel("HaUI Student Analytics")
        brand_title.setObjectName("BrandTitle")
        brand_title.setWordWrap(True)
        brand_caption = QLabel("ACADEMIC INTELLIGENCE")
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
            ("Mô hình ML", "SP_BrowserReload"),
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
            button.setIcon(QApplication.style().standardIcon(standard_icons[icon_key]))
            button.clicked.connect(lambda checked=False, i=index: self._switch_page(i))
            self.nav_buttons.append(button)
            sidebar_layout.addWidget(button)

        sidebar_layout.addStretch(1)
        settings_button = QPushButton("Cấu hình ứng dụng")
        settings_button.setObjectName("NavButton")
        settings_button.setMinimumHeight(42)
        settings_button.setCheckable(True)
        settings_button.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView))
        settings_button.clicked.connect(lambda: self._switch_page(7))
        sidebar_layout.addWidget(settings_button)
        self.settings_nav_button = settings_button

        sidebar_footer = QLabel("HaUI · Student Performance")
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
        self.lbl_model_badge = QLabel("Model chưa sẵn sàng")
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
            ("Mô hình Machine Learning", self._page_subtitles[5], self._build_models_page),
            ("Báo cáo và xuất dữ liệu", self._page_subtitles[6], self._build_reports_page),
            ("Cấu hình ứng dụng", self._page_subtitles[7], self._build_settings_page),
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
        self.card_records = _MetricCard("BẢN GHI ĐÃ NẠP", "—", "Chưa có dataset", "blue")
        self.card_missing = _MetricCard("Ô DỮ LIỆU THIẾU", "—", "Chưa có dataset", "amber")
        self.card_absence = _MetricCard("VƯỢT NGƯỠNG NGHỈ HỌC", "—", "Ngưỡng cấu hình > 30%", "red")
        self.card_model = _MetricCard("MÔ HÌNH DỰ ĐOÁN", "Chưa sẵn sàng", "Cần model đã huấn luyện", "violet")
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
        summary_frame, summary_layout = self._surface("Tình trạng dataset")
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
            "Kết quả dự đoán chỉ được cung cấp khi có model hợp lệ và dữ liệu đầu vào đáp ứng feature contract. "
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
            "Chọn một profile phù hợp với cấu trúc điểm của dataset. Nếu một file chứa nhiều loại học phần, cần tách file hoặc bổ sung mapping theo học phần."
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
        self.lbl_model_page_status = QLabel("Đang kiểm tra registry…")
        self.lbl_model_page_status.setStyleSheet("font-size:16px;font-weight:700;color:#B45309;")
        top.addWidget(self.lbl_model_page_status, 1)
        self.btn_model_refresh = QPushButton("Làm mới trạng thái")
        self.btn_model_refresh.clicked.connect(self._update_model_status)
        top.addWidget(self.btn_model_refresh)
        state_layout.addLayout(top)
        self.lbl_model_details = QLabel("Chưa có metadata model.")
        self.lbl_model_details.setWordWrap(True)
        self.lbl_model_details.setObjectName("Muted")
        state_layout.addWidget(self.lbl_model_details)
        layout.addWidget(state_frame)

        train_frame, train_layout = self._surface("Huấn luyện từ dữ liệu lịch sử")
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

    def _build_reports_page(self, body: QWidget, layout: QVBoxLayout) -> None:
        frame, frame_layout = self._surface("Xuất dữ liệu và kết quả")
        intro = QLabel("Chọn loại dữ liệu đang có trong phiên làm việc rồi xuất ra CSV hoặc Excel.")
        intro.setObjectName("Muted")
        intro.setWordWrap(True)
        frame_layout.addWidget(intro)
        row = QHBoxLayout()
        self.cb_report_type = QComboBox()
        self.cb_report_type.addItem("Dataset đã nhập", "dataset")
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
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return table

    def _switch_page(self, index: int) -> None:
        if index < 0 or index >= self.content_area.count():
            return
        self.content_area.setCurrentIndex(index)
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
        self.settings_nav_button.setChecked(index == 7)
        self.topbar_context.setText(self._page_titles[index].upper())
        if index == 0:
            self._update_dashboard()
        elif index == 2:
            self._refresh_eda_options()
        elif index == 5:
            self._update_model_status()
        elif index == 7:
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
            filename = Path(self.current_file_path).name if self.current_file_path else "Dataset đã nạp"
            self.lbl_dataset_badge.setText(f"{filename} · {len(self.current_df):,} dòng")
            self.lbl_dataset_badge.setStyleSheet("background:#ECFDF3;color:#027A48;padding:7px 10px;border-radius:7px;font-size:11px;font-weight:600;")
        ready = self._is_model_ready()
        if ready:
            self.lbl_model_badge.setText(f"XGBoost {ACTIVE_MODEL_VERSION} sẵn sàng")
            self.lbl_model_badge.setStyleSheet("background:#ECFDF3;color:#027A48;padding:7px 10px;border-radius:7px;font-size:11px;font-weight:600;")
        else:
            self.lbl_model_badge.setText("Model chưa sẵn sàng")
            self.lbl_model_badge.setStyleSheet("background:#FEF3F2;color:#B42318;padding:7px 10px;border-radius:7px;font-size:11px;font-weight:600;")

    def _is_model_ready(self) -> bool:
        try:
            checker = getattr(self.prediction_service, "is_model_ready", None)
            return bool(checker(ACTIVE_MODEL_VERSION)) if callable(checker) else False
        except Exception:
            logging.exception("Unable to check model readiness")
            return False

    def _update_dashboard(self) -> None:
        if self.current_df is None:
            self.card_records.set_value("0", "Chưa có dataset", "blue")
            self.card_missing.set_value("—", "Chưa có dataset", "amber")
            self.card_absence.set_value("—", "Ngưỡng cấu hình > 30%", "red")
            self.card_risk_override_value = None
            self.dashboard_summary.setText("Chưa có dữ liệu được nạp vào phiên làm việc.")
            self._show_dashboard_placeholder("Nhập dữ liệu để hiển thị phân tích.")
        else:
            df = self.current_df
            missing_cells = int(df.isna().sum().sum())
            if "absence_percentage" in df.columns:
                absence = pd.to_numeric(df["absence_percentage"], errors="coerce")
                override_count = int((absence > 30).sum())
                self.card_absence.set_value(f"{override_count:,}", "Tỷ lệ nghỉ học > 30%", "red")
            else:
                override_count = 0
                self.card_absence.set_value("—", "Không có cột absence_percentage", "amber")
            self.card_records.set_value(f"{len(df):,}", f"{len(df.columns)} cột trong dataset", "blue")
            self.card_missing.set_value(f"{missing_cells:,}", "Ô thiếu trong dữ liệu đang nạp", "amber" if missing_cells else "green")
            path_text = Path(self.current_file_path).name if self.current_file_path else "Dataset hiện tại"
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
            "Artifact và registry hợp lệ" if ready else "Cần huấn luyện và lưu model hợp lệ",
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
            self._show_dashboard_placeholder("Dataset chưa có cột tỷ lệ nghỉ học.")
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
            if not result.is_success or result.data is None:
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

            self.current_df = result.data.copy(deep=True)
            self.current_file_path = file_path
            self._sync_course_profile_from_dataset(self.current_df)
            self.prediction_df = None
            self.risk_df = None
            warnings = getattr(result, "warnings", []) or []
            warning_text = f" Có {len(warnings)} cảnh báo cần xem lại." if warnings else ""
            self.lbl_import_status.setText(
                f"Đã nhập {len(self.current_df):,} bản ghi từ {Path(file_path).name}.{warning_text}"
            )
            self.lbl_quality_summary.setText(self._build_quality_summary(self.current_df, warnings))
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
        if self.current_df is None:
            self.table_import.clear()
            self.table_import.setRowCount(0)
            self.table_import.setColumnCount(0)
            self.lbl_table_count.setText("0 bản ghi")
            return
        self._populate_table(self.table_import, self.current_df)
        displayed = min(len(self.current_df), 1000)
        self.lbl_table_count.setText(
            f"Hiển thị {displayed:,}/{len(self.current_df):,} bản ghi để giữ giao diện phản hồi tốt."
        )

    def _populate_table(self, table: QTableWidget, dataframe: pd.DataFrame, max_rows: int = 1000) -> None:
        table.setSortingEnabled(False)
        preview = dataframe.head(max_rows).copy()
        table.clear()
        table.setRowCount(len(preview))
        table.setColumnCount(len(preview.columns))
        table.setHorizontalHeaderLabels([str(column) for column in preview.columns])
        for row_index in range(len(preview)):
            for column_index, column in enumerate(preview.columns):
                value = preview.iat[row_index, column_index]
                if pd.isna(value):
                    text = "Chưa có dữ liệu"
                elif isinstance(value, (float, np.floating)):
                    text = f"{float(value):.3f}" if np.isfinite(value) else "Không hợp lệ"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                table.setItem(row_index, column_index, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
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
                and (pd.api.types.is_object_dtype(dataframe[column]) or pd.api.types.is_bool_dtype(dataframe[column]) or pd.api.types.is_categorical_dtype(dataframe[column]))
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
            QMessageBox.warning(self, "Chưa có dữ liệu", "Hãy nhập một dataset hợp lệ trước khi tạo biểu đồ.")
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
                    raise ValueError("Dataset không có đủ biến số để tạo heatmap tương quan.")
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
            f"Đã lưu model {result['version']}. Train: {result['train_rows']:,} dòng; test: {result['test_rows']:,} dòng."
        )
        self.lbl_model_metrics.setText(
            f"MAE: {mae}    RMSE: {rmse}    R²: {r2}\n"
            f"Số đặc trưng: {result['feature_count']}\n"
            f"Dataset fingerprint: {result['dataset_hash']}\n"
            f"Features: {', '.join(result['features'])}"
        )
        self._update_all_status()
        self._refresh_model_page_info()
        QMessageBox.information(
            self,
            "Huấn luyện hoàn tất",
            f"Model {result['version']} đã được lưu và xác minh trong registry.\n\nMAE: {mae}\nRMSE: {rmse}\nR²: {r2}",
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
        if self.current_df is None or self.current_df.empty:
            QMessageBox.warning(self, "Chưa có dữ liệu", "Hãy nhập dataset trước khi dự đoán.")
            return
        if "tx2_score" in self.current_df.columns:
            tx2_values = pd.to_numeric(self.current_df["tx2_score"], errors="coerce")
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
            pd.to_numeric(self.current_df["tx2_score"], errors="coerce")
            if "tx2_score" in self.current_df.columns
            else pd.Series(np.nan, index=self.current_df.index)
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
                "Hãy kiểm tra cấu trúc đánh giá và chọn đúng profile trước khi tính điểm học phần.",
            )
            return

        if not self._is_model_ready():
            self._update_model_status()
            QMessageBox.warning(
                self,
                "Mô hình chưa sẵn sàng",
                "Chưa tìm thấy model đã huấn luyện hợp lệ. Mở trang Mô hình ML và chọn một tệp dữ liệu lịch sử để huấn luyện trước khi dự đoán.",
            )
            return
        try:
            self.btn_predict.setEnabled(False)
            self.lbl_prediction_status.setText("Đang dự đoán…")
            QApplication.processEvents()
            results = self.prediction_service.predict_student_results(
                self.current_df.copy(deep=True),
                model_version=ACTIVE_MODEL_VERSION,
                course_profile_id=profile_id,
            )

            risk_rows = []
            for row_index, row in results.iterrows():
                student_id = str(row.get("student_id", f"Bản ghi {row_index + 1}"))
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
            self.lbl_prediction_status.setText("Dự đoán thất bại. Kiểm tra feature contract và dữ liệu đầu vào.")
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
        for row_index, row in source.iterrows():
            student_id = str(row.get("student_id", f"Bản ghi {row_index + 1}"))
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
        self.card_risk_override.set_value(f"{overrides:,}", "Trạng thái HOC_LAI", "red")
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
                self.lbl_model_details.setText(f"Registry xác nhận có metadata và artifact của XGBoost {ACTIVE_MODEL_VERSION}.")
            else:
                self.lbl_model_page_status.setText("Chưa có model dùng được")
                self.lbl_model_page_status.setStyleSheet("font-size:16px;font-weight:700;color:#B42318;")
                self.lbl_model_details.setText(
                    "Không tìm thấy model đã huấn luyện hợp lệ. Chọn riêng một tệp dữ liệu lịch sử có nhãn khi bấm Huấn luyện XGBoost; tệp sinh viên cần dự đoán không cần điểm cuối kỳ."
                )
        if hasattr(self, "card_model"):
            self.card_model.set_value(
                "Sẵn sàng" if ready else "Chưa sẵn sàng",
                "Artifact và registry hợp lệ" if ready else "Cần huấn luyện và lưu model hợp lệ",
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
                ("Model registry", config.model_registry_path),
                ("Thư mục artifact", config.model_artifacts_dir),
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
                "dataset": "Dataset đã nhập",
                "prediction": "Kết quả dự đoán điểm",
                "risk": "Kết quả nguy cơ học vụ",
            }.get(key, str(key)))
            model = self.cb_report_type.model()
            item = model.item(index) if hasattr(model, "item") else None
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

    def _clear_layout(self, layout: QVBoxLayout | QHBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
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

    def closeEvent(self, event) -> None:
        worker = self._training_worker
        if worker is not None and worker.isRunning():
            QMessageBox.warning(
                self,
                "Đang huấn luyện",
                "Hãy đợi tác vụ huấn luyện kết thúc trước khi đóng ứng dụng.",
            )
            event.ignore()
            return
        super().closeEvent(event)
