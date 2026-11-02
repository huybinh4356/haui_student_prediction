# BÁO CÁO KIỂM TOÁN KIẾN TRÚC MÃ NGUỒN VÀ HỆ THỐNG MACHINE LEARNING (ML AUDIT REPORT)
**Dự án**: HaUI Student Performance & Academic Risk Prediction System  
**Nhánh Git**: `fix/prediction-quality-validation`  
**Commit HEAD**: `ab642cc2f010451aff9c1f356ff312e7389565be` ("Add direct export button to prediction page")  
**Ngày thực hiện**: 02/11/2026  
**Chuyên gia thực hiện**: Senior Machine Learning Engineer, Data Scientist & Software Architect  

---

## 1. TỔNG QUAN HỆ THỐNG VÀ CẤU TRÚC KỸ THUẬT

### 1.1. Môi trường phát triển và Thư viện phụ thuộc
- **Hệ điều hành**: Windows 11.
- **Python**: 3.11 (`.venv\Scripts\python.exe`).
- **Framework & Giao diện**: PyQt6 (Desktop Application).
- **Core ML & Data Science**:
  - `scikit-learn`: v1.4+ (Pipelines, preprocessing, metrics, validation splitters).
  - `xgboost`: v2.2.0 (Gradient boosted decision trees, TreeSHAP).
  - `catboost`: v1.2.8 (Hỗ trợ boosting trên biến phân loại).
  - `pandas`, `numpy`, `openpyxl`, `joblib`.
  - `matplotlib`, `seaborn`: Trực quan hóa dữ liệu và biểu đồ phân tích.
  - `pytest`, `pytest-qt`: Kiểm thử tự động (162 bài kiểm thử).
- **Lưu trữ cục bộ**: SQLite (`data/haui_student_prediction.db`) và File-based Model Registry (`models/registry.json`, `models/artifacts/`).

### 1.2. Cấu trúc thư mục nguồn (`src/haui_student_prediction/`)
```
src/haui_student_prediction/
├── config/              # Course profiles (no_tx2, with_tx2)
├── data/                # Ingestion, Cleaning, Column Mapping, Schema, Validator, TrainingDataset
├── eda/                 # Thống kê mô tả, phân phối, tương quan
├── features/            # FeatureContract, FeatureBuilder (chống rò rỉ target & định danh)
├── ml/                  # DataSplitting, Preprocessing, TrainRegressor, XGBoost, ModelRegistry, Explainability
├── risk/                # RiskCalculator (quy tắc vắng > 30% -> REPEAT_REQUIRED)
├── scoring/             # FinalCourseScore (Công thức tín chỉ: 20-20-60 hoặc 15-15-20-50)
├── services/            # DatasetService, PredictionService, RiskPredictionService, ReportService
├── storage/             # SQLiteDatabase, DatasetRepository
└── ui/                  # PyQt6 MainWindow, Navigation, Views, MetricCards, QThread Workers
```

---

## 2. KẾT QUẢ KIỂM TOÁN TỪNG THÀNH PHẦN (WHAT WORKS / WHAT DOESN'T / WHAT IS BROKEN)

### 2.1. Quản lý dữ liệu & Data Ingestion (`data/`)
* **Đã hoạt động tốt**:
  - Đọc được tệp CSV (hỗ trợ `utf-8-sig`, fallback `cp1258`) và Excel (`.xlsx`, `.xls`).
  - Chuẩn hóa tên cột có dấu/không dấu qua alias mapping (`column_mapping.py`).
  - Kiểm tra range điểm (0–10), tỷ lệ phần trăm (0–100%), điểm đánh giá (1–5).
  - Tách biệt kiểm tra các trường bắt buộc (`REQUIRED_SCORE_FIELDS = tx1_score, midterm_score`).
* **Điểm chưa hoàn thiện / Đang thiếu**:
  - `training_dataset.py` và `validator.py` chỉ kiểm tra giá trị số và khoảng giá trị (range), **hoàn toàn chưa kiểm tra tính nhất quán logic nghiệp vụ (Logic Consistency)** giữa các cột điểm thành phần và điểm cuối kỳ.
  - Khi người dùng nạp một tệp dữ liệu mà nhãn cuối kỳ bị sinh ngẫu nhiên độc lập (tương quan $r \approx 0$), hệ thống vẫn cho phép nạp thành công và đưa vào huấn luyện mà không đưa ra cảnh báo sớm tại tầng dữ liệu.

### 2.2. Feature Contract & Feature Engineering (`features/`)
* **Đã hoạt động tốt**:
  - `FeatureContract`: Khai báo rõ ràng `numerical_features`, `categorical_features`, `target_column`.
  - `FeatureBuilder`:
    - Chống rò rỉ dữ liệu (Data Leakage): Loại bỏ hoàn toàn cột mục tiêu (`target_column`) khỏi ma trận đặc trưng $X$.
    - Loại bỏ các cột định danh (`student_id`, `full_name`, `snapshot_date`, ...).
    - Đồng bộ mã hóa kiểu dữ liệu cho `categorical_features` sang chuỗi/object trước khi vào Scikit-Learn Pipeline.
* **Điểm cần lưu ý**:
  - Cần đảm bảo Feature Contract được lưu trữ nguyên vẹn trong Model Registry và được tái sử dụng đồng nhất 100% giữa quá trình Training và Prediction.

### 2.3. Pipeline Huấn luyện & Mô hình Machine Learning (`ml/`)
* **Đã hoạt động tốt**:
  - Lưu và nạp mô hình qua `ModelRegistry` với mã băm dataset hash (SHA-256).
  - Đánh giá Out-Of-Fold (OOF) trong Cross-Validation 5-fold.
  - TreeSHAP cục bộ và toàn cục tái cấu trúc chính xác giá trị dự đoán của mô hình (`__baseline__` + $\sum \text{contributions} = \hat{y}$).
* **Nguyên nhân cốt lõi gây ra lỗi dự đoán co cụm ở 6–7 điểm**:
  1. **Nguyên nhân chính từ Dữ liệu**: Dữ liệu huấn luyện lịch sử gốc (`du_lieu_lich_su_huan_luyen_chuan_hoa.csv`) có tương quan giữa tất cả các đặc trưng và điểm cuối kỳ xấp xỉ bằng $0$. Toán học Machine Learning bắt buộc mô hình phải dự đoán hằng số trung bình $\bar{y} \approx 6.955$.
  2. **Nguyên nhân thuật toán**:
     - Danh sách cấu hình ứng viên trước đây chỉ có các mô hình quá bảo thủ (max_depth 2-3, reg_lambda 2-8), ép mô hình co về trung bình.
     - Hệ thống chỉ hỗ trợ duy nhất XGBoost trong pipeline chính; chưa có hệ thống so sánh đa thuật toán (Multi-model Benchmark: Ridge, Lasso, Random Forest, XGBoost, CatBoost).
     - Ngưỡng dung sai `tolerance` chọn ứng viên trong CV quá hẹp ($3\%$) dẫn tới ưu tiên mô hình co cụm thay vì mô hình có độ phân tán tốt.

### 2.4. Tính điểm học phần & Đánh giá nguy cơ học vụ (`scoring/`, `risk/`)
* **Đã hoạt động tốt**:
  - `ScoringService`:
    - Không có TX2: $\text{Điểm học phần} = 0.2 \times \text{TX1} + 0.2 \times \text{Giữa kỳ} + 0.6 \times \text{Cuối kỳ dự đoán}$.
    - Có TX2: $\text{Điểm học phần} = 0.15 \times \text{TX1} + 0.15 \times \text{TX2} + 0.2 \times \text{Giữa kỳ} + 0.5 \times \text{Cuối kỳ dự đoán}$.
  - `RiskCalculator`:
    - Áp dụng triệt để quy tắc nghiệp vụ bất biến: `absence_percentage > 30` $\rightarrow$ Trạng thái `REPEAT_REQUIRED`, gán điểm vắng $100.0$ với trọng số $0.75$, đưa nguy cơ về `High` / `Very high` và kích hoạt cờ override.
    - Tuyệt đối không sử dụng ngưỡng 75%.
* **Điểm cần bảo toàn**:
  - Không được can thiệp hoặc thay đổi các công thức và quy tắc nghiệp vụ này trong quá trình tối ưu Machine Learning.

### 2.5. Giao diện PyQt6 (`ui/main_window.py`)
* **Đã hoạt động tốt**:
  - Kiến trúc phân trang đầy đủ: Tổng quan, Dữ liệu sinh viên, Phân tích dữ liệu, Dự đoán điểm, Đánh giá nguy cơ học vụ, Mô hình dự đoán, Huấn luyện và so sánh, Báo cáo, Cấu hình.
  - Xử lý tác vụ nền qua `QThread` (`_TrainingWorker`), ngăn chặn tình trạng đơ/lag giao diện.
  - Đã tích hợp nút xuất kết quả dự đoán trực tiếp ra CSV và Excel tại commit `ab642cc`.
* **Điểm cần nâng cấp**:
  - Giao diện huấn luyện và so sánh (`Huấn luyện và so sánh`) cần hiển thị bảng so sánh đa thuật toán (Multi-model comparison table) bao gồm: Ridge, Lasso, Random Forest, XGBoost, CatBoost cùng Baseline.

---

## 3. DANH SÁCH CÁC TỆP CẦN SỬA ĐỔI VÀ BỔ SUNG (REFACTORING PLAN)

| Thứ tự | Tệp mã nguồn | Mục tiêu refactor |
|---|---|---|
| **1** | `src/haui_student_prediction/ml/model_comparison.py` | Xây dựng pipeline so sánh 5 thuật toán: Ridge, Lasso, Random Forest, XGBoost, CatBoost với DummyRegressor baseline. |
| **2** | `src/haui_student_prediction/ml/xgboost_regressor.py` | Hoàn thiện tuyển chọn siêu tham số, tối ưu hóa Out-Of-Fold $R^2$, nới lỏng tolerance dung sai độ phân tán, xuất cảnh báo tín hiệu yếu. |
| **3** | `src/haui_student_prediction/data/validator.py` & `training_dataset.py` | Bổ sung hàm kiểm tra tính nhất quán logic dữ liệu (Logic Consistency Check) và cảnh báo khi target không có tương quan với quá trình học. |
| **4** | `scripts/benchmark_models.py` | Script benchmark tự động xuất các tệp CSV, đồ thị residual, bảng so sánh đa mô hình vào `outputs/model_comparison/`. |
| **5** | `src/haui_student_prediction/ui/main_window.py` | Cập nhật trang huấn luyện & so sánh hiển thị bảng so sánh các thuật toán ML, cảnh báo chất lượng dữ liệu. |
| **6** | `tests/unit/test_model_comparison.py` | Bổ sung bộ test cho hệ thống so sánh đa thuật toán, đảm bảo tính toàn vẹn 100%. |
| **7** | `docs/ML_IMPROVEMENT_REPORT.md` | Báo cáo cải tiến kỹ thuật toàn diện sau khi hoàn thành tất cả các giai đoạn. |

