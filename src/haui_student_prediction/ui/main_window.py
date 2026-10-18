import pandas as pd
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, 
    QMessageBox, QComboBox, QHeaderView, QStackedWidget, QFrame, QApplication
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class MainWindow(QMainWindow):
    def __init__(self, dataset_service, prediction_service, chart_service, risk_service):
        super().__init__()
        self.dataset_service = dataset_service
        self.prediction_service = prediction_service
        self.chart_service = chart_service
        self.risk_service = risk_service
        
        self.current_df = None
        
        self.setWindowTitle("Hệ thống Dự đoán Kết quả & Nguy cơ Học vụ HaUI")
        self.resize(1366, 768)
        self.setStyleSheet("background-color: #F8F9FA; color: #343A40;")
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self._setup_sidebar(main_layout)
        
        self.content_area = QStackedWidget()
        main_layout.addWidget(self.content_area, stretch=1)
        
        self._init_pages()
        self.content_area.setCurrentIndex(0)

    def _setup_sidebar(self, parent_layout):
        sidebar = QFrame()
        sidebar.setStyleSheet("background-color: #0A2540; color: white;")
        sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 20)
        sidebar_layout.setSpacing(10)
        
        title = QLabel("HaUI Prediction")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white; margin-bottom: 20px;")
        sidebar_layout.addWidget(title)
        
        self.nav_buttons = []
        nav_items = ["Tổng quan", "Dữ liệu sinh viên", "Phân tích EDA", "Dự đoán điểm cuối kỳ", "Dự đoán nguy cơ học vụ"]
        
        for idx, item in enumerate(nav_items):
            btn = QPushButton(item)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: white;
                    text-align: left;
                    padding: 10px;
                    border: none;
                    border-radius: 5px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #113B5E;
                }
                QPushButton:checked {
                    background-color: #0066CC;
                    font-weight: bold;
                }
            """)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, i=idx: self._switch_page(i))
            self.nav_buttons.append(btn)
            sidebar_layout.addWidget(btn)
            
        if self.nav_buttons:
            self.nav_buttons[0].setChecked(True)
            
        sidebar_layout.addStretch()
        parent_layout.addWidget(sidebar)

    def _switch_page(self, index):
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        self.content_area.setCurrentIndex(index)
        if index == 0:
            self._update_dashboard()

    def _init_pages(self):
        self.page_dashboard = QWidget()
        self.page_import = QWidget()
        self.page_eda = QWidget()
        self.page_predict = QWidget()
        self.page_risk = QWidget()
        
        self._build_dashboard_page()
        self._build_import_page()
        self._build_eda_page()
        self._build_predict_page()
        self._build_risk_page()
        
        self.content_area.addWidget(self.page_dashboard)
        self.content_area.addWidget(self.page_import)
        self.content_area.addWidget(self.page_eda)
        self.content_area.addWidget(self.page_predict)
        self.content_area.addWidget(self.page_risk)

    def _create_header(self, text):
        lbl = QLabel(text)
        lbl.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        lbl.setStyleSheet("color: #0A2540; margin-bottom: 10px;")
        return lbl

    def _create_kpi_card(self, title, value, color="#0066CC"):
        card = QFrame()
        card.setStyleSheet(f"background-color: white; border-radius: 8px; border-top: 4px solid {color}; padding: 15px;")
        layout = QVBoxLayout(card)
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #6C757D; font-size: 14px;")
        lbl_value = QLabel(value)
        lbl_value.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)
        return card, lbl_value

    def _build_dashboard_page(self):
        layout = QVBoxLayout(self.page_dashboard)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.addWidget(self._create_header("Tổng quan hệ thống"))
        
        kpi_layout = QHBoxLayout()
        card_total, self.lbl_kpi_total = self._create_kpi_card("Tổng bản ghi đã nạp", "0", "#0066CC")
        card_valid, self.lbl_kpi_valid = self._create_kpi_card("Trạng thái dữ liệu", "Chưa có dữ liệu", "#28A745")
        card_model, self.lbl_kpi_model = self._create_kpi_card("Mô hình ML", "Sẵn sàng (v1.0.0)", "#6F42C1")
        
        kpi_layout.addWidget(card_total)
        kpi_layout.addWidget(card_valid)
        kpi_layout.addWidget(card_model)
        
        layout.addLayout(kpi_layout)
        layout.addStretch()

    def _update_dashboard(self):
        if self.current_df is not None:
            self.lbl_kpi_total.setText(str(len(self.current_df)))
            self.lbl_kpi_valid.setText("Hợp lệ")
            self.lbl_kpi_valid.setStyleSheet("color: #28A745; font-size: 24px; font-weight: bold;")
        else:
            self.lbl_kpi_total.setText("0")
            self.lbl_kpi_valid.setText("Chưa nạp")
            self.lbl_kpi_valid.setStyleSheet("color: #DC3545; font-size: 24px; font-weight: bold;")

    def _build_import_page(self):
        layout = QVBoxLayout(self.page_import)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.addWidget(self._create_header("Dữ liệu sinh viên"))
        
        control_layout = QHBoxLayout()
        btn_import = QPushButton("Nhập từ file (CSV/Excel)")
        btn_import.setFixedWidth(200)
        btn_import.setStyleSheet("background-color: #0066CC; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        btn_import.clicked.connect(self._handle_import)
        
        self.lbl_import_status = QLabel("Chưa chọn file")
        self.lbl_import_status.setStyleSheet("color: #6C757D;")
        
        control_layout.addWidget(btn_import)
        control_layout.addWidget(self.lbl_import_status)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        self.table_import = QTableWidget()
        self.table_import.setStyleSheet("background-color: white; border: 1px solid #DEE2E6;")
        layout.addWidget(self.table_import)

    def _handle_import(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn file dữ liệu", "", "Data Files (*.csv *.xlsx *.xls)")
        if not file_path:
            return
            
        try:
            self.lbl_import_status.setText("Đang xử lý...")
            QApplication.processEvents()
            
            required_cols = ["student_id", "absence_percentage", "tx1_score", "midterm_score"]
            courses_tx2 = ["IT101", "IT102"] 
            result = self.dataset_service.import_dataset(file_path, required_cols, courses_tx2)
            
            if result.is_success:
                self.current_df = result.data
                self.lbl_import_status.setText(f"Thành công! Đã nạp {len(self.current_df)} bản ghi.")
                self.lbl_import_status.setStyleSheet("color: #28A745;")
                self._populate_table(self.table_import, self.current_df)
                self._update_dashboard()
                self._refresh_eda_options()
            else:
                self.lbl_import_status.setText("Lỗi kiểm tra dữ liệu")
                self.lbl_import_status.setStyleSheet("color: #DC3545;")
                QMessageBox.warning(self, "Lỗi Dữ liệu", f"Thiếu cột: {result.missing_columns}\nLỗi khác: {len(result.errors)}")
        except Exception as e:
            self.lbl_import_status.setText("Lỗi hệ thống")
            QMessageBox.critical(self, "Lỗi hệ thống", str(e))

    def _build_eda_page(self):
        layout = QVBoxLayout(self.page_eda)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.addWidget(self._create_header("Phân tích EDA"))
        
        control_layout = QHBoxLayout()
        self.cb_eda_col = QComboBox()
        self.cb_eda_col.setStyleSheet("padding: 5px; background: white;")
        
        btn_plot = QPushButton("Vẽ biểu đồ Histogram")
        btn_plot.setStyleSheet("background-color: #0066CC; color: white; padding: 5px 15px; border-radius: 3px;")
        btn_plot.clicked.connect(self._handle_plot)
        
        control_layout.addWidget(QLabel("Chọn biến phân tích:"))
        control_layout.addWidget(self.cb_eda_col)
        control_layout.addWidget(btn_plot)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        self.chart_container = QVBoxLayout()
        chart_widget = QWidget()
        chart_widget.setLayout(self.chart_container)
        chart_widget.setStyleSheet("background-color: white; border: 1px solid #DEE2E6;")
        layout.addWidget(chart_widget, stretch=1)

    def _refresh_eda_options(self):
        self.cb_eda_col.clear()
        if self.current_df is not None:
            num_cols = self.current_df.select_dtypes(include='number').columns.tolist()
            valid_cols = [c for c in num_cols if c not in ["student_id", "full_name"]]
            self.cb_eda_col.addItems(valid_cols)

    def _handle_plot(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập dữ liệu trước.")
            return
            
        col = self.cb_eda_col.currentText()
        if not col:
            return
            
        try:
            self._clear_layout(self.chart_container)
            fig = self.chart_service.dist_plots.plot_histogram(self.current_df, col)
            canvas = FigureCanvas(fig)
            self.chart_container.addWidget(canvas)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi vẽ biểu đồ", str(e))

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _build_predict_page(self):
        layout = QVBoxLayout(self.page_predict)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.addWidget(self._create_header("Dự đoán điểm cuối kỳ"))
        
        control_layout = QHBoxLayout()
        self.cb_course_profile = QComboBox()
        self.cb_course_profile.addItems(["Công thức không có TX2", "Công thức có TX2"])
        self.cb_course_profile.setStyleSheet("padding: 5px; background: white;")
        
        btn_predict = QPushButton("Dự đoán & Tính điểm học phần")
        btn_predict.setStyleSheet("background-color: #28A745; color: white; padding: 5px 15px; border-radius: 3px; font-weight: bold;")
        btn_predict.clicked.connect(self._handle_predict)
        
        control_layout.addWidget(QLabel("Cấu hình đánh giá:"))
        control_layout.addWidget(self.cb_course_profile)
        control_layout.addWidget(btn_predict)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        self.table_predict = QTableWidget()
        self.table_predict.setStyleSheet("background-color: white; border: 1px solid #DEE2E6;")
        layout.addWidget(self.table_predict)

    def _handle_predict(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng nhập dữ liệu trước.")
            return
            
        try:
            profile_idx = self.cb_course_profile.currentIndex()
            profile_id = "no_tx2" if profile_idx == 0 else "with_tx2"
            
            result_df = self.prediction_service.predict_student_results(
                self.current_df, 
                model_version="1.0.0", 
                course_profile_id=profile_id
            )
            self.current_df = result_df
            
            disp_cols = ["student_id", "tx1_score", "midterm_score", "predicted_final_exam_score", "predicted_course_score"]
            if profile_id == "with_tx2":
                disp_cols.insert(2, "tx2_score")
                
            avail_cols = [c for c in disp_cols if c in result_df.columns]
            self._populate_table(self.table_predict, result_df[avail_cols])
        except Exception as e:
            QMessageBox.critical(self, "Lỗi Dự đoán", str(e))

    def _build_risk_page(self):
        layout = QVBoxLayout(self.page_risk)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.addWidget(self._create_header("Dự đoán nguy cơ học vụ"))
        
        control_layout = QHBoxLayout()
        btn_risk = QPushButton("Phân tích Nguy cơ toàn khóa")
        btn_risk.setStyleSheet("background-color: #DC3545; color: white; padding: 5px 15px; border-radius: 3px; font-weight: bold;")
        btn_risk.clicked.connect(self._handle_risk)
        control_layout.addWidget(btn_risk)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        self.table_risk = QTableWidget()
        self.table_risk.setStyleSheet("background-color: white; border: 1px solid #DEE2E6;")
        layout.addWidget(self.table_risk)

    def _handle_risk(self):
        if self.current_df is None:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng thực hiện Dự đoán Điểm trước khi tính Nguy cơ.")
            return
            
        if 'predicted_course_score' not in self.current_df.columns:
            QMessageBox.warning(self, "Cảnh báo", "Vui lòng thực hiện Dự đoán Điểm cuối kỳ ở tab trước để có điểm học phần.")
            return
            
        try:
            records = []
            for _, row in self.current_df.iterrows():
                res = self.risk_service.evaluate_student_risk(row)
                records.append({
                    "student_id": row.get("student_id", "N/A"),
                    "absence_status": res.absence_status,
                    "weighted_score": round(res.weighted_score, 2),
                    "risk_category": res.risk_category,
                    "is_override": "Vượt ngưỡng 30%" if res.is_override else "Bình thường"
                })
            
            risk_df = pd.DataFrame(records)
            self._populate_table(self.table_risk, risk_df)
        except Exception as e:
            QMessageBox.critical(self, "Lỗi tính toán", str(e))

    def _populate_table(self, table_widget, df):
        table_widget.clear()
        table_widget.setRowCount(df.shape[0])
        table_widget.setColumnCount(df.shape[1])
        table_widget.setHorizontalHeaderLabels(df.columns.astype(str))
        
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = str(df.iat[i, j]) if pd.notna(df.iat[i, j]) else ""
                table_widget.setItem(i, j, QTableWidgetItem(val))
                
        header = table_widget.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)