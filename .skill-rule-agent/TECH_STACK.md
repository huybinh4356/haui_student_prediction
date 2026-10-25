# TECH_STACK.md — QUY ĐỊNH CÔNG NGHỆ CHO DỰ ÁN HAUI

**Dự án:** HaUI Student Performance & Academic Risk Prediction System  
**Phiên bản:** 1.1  
**Vai trò tài liệu:** Nguồn hướng dẫn công nghệ và ranh giới tích hợp hiện hành.

## 1. Quy tắc áp dụng

1. Đọc tài liệu này và kế hoạch dự án trước khi thay đổi kiến trúc hoặc thêm thư viện.
2. Ưu tiên sửa cấu trúc và dịch vụ đang có; không tạo dự án, registry, repository hoặc service trùng chức năng.
3. Không đổi công nghệ lưu trữ, công thức điểm, quy tắc nguy cơ hoặc luồng nghiệp vụ nếu không có yêu cầu rõ ràng.
4. `pyproject.toml` là nguồn khai báo package và dependency; `requirements.txt` được duy trì đồng bộ để cài đặt thuận tiện.
5. Công nghệ mới phải có lý do, đánh giá ảnh hưởng và kiểm thử đi kèm.
6. Không đưa dữ liệu sinh viên thật, database phiên làm việc, model artifact hoặc thông tin bí mật vào Git.

## 2. Stack công nghệ hiện tại

| Công nghệ | Vai trò |
|---|---|
| Python 3.10+ | Ngôn ngữ chính của ứng dụng desktop, xử lý dữ liệu và ML |
| PyQt6 | Giao diện desktop, trang chức năng, biểu mẫu, bảng và trạng thái thao tác |
| pandas, NumPy | Đọc, kiểm tra, biến đổi và phân tích bảng dữ liệu |
| scikit-learn | Tiền xử lý, pipeline, mô hình baseline, chia dữ liệu và metrics |
| XGBoost | Mô hình hồi quy đang được tích hợp làm bộ dự đoán điểm chính |
| joblib | Lưu và nạp model pipeline đã huấn luyện |
| SQLite qua `sqlite3` | Lưu dataset của phiên làm việc mà không cần dịch vụ database ngoài |
| Matplotlib, Seaborn | Biểu đồ EDA và báo cáo |
| openpyxl | Đọc/ghi Excel qua pandas |
| PyYAML, python-dotenv | Đọc cấu hình và biến môi trường khi cần |
| pytest, pytest-qt | Unit, integration và UI tests |
| Ruff | Kiểm tra chất lượng và quy tắc code |

## 3. Lưu trữ dữ liệu

- Module lưu phiên hiện tại là `src/haui_student_prediction/storage/dataset_repository.py`.
- Database mặc định của repository là `data/session_student.db`; đường dẫn dự án lấy từ `app_config.get_project_root()` hoặc biến môi trường `APP_BASE_DIR`.
- Bảng `current_session` chứa dataset nhập thành công gần nhất; lần nhập thành công mới thay thế dữ liệu phiên trước.
- Repository phải kiểm tra tên bảng, xử lý database/bảng chưa tồn tại và trả lỗi rõ ràng khi không thể lưu dữ liệu.
- Dùng parameterized query cho mọi giá trị đầu vào trong SQL. Không nối dữ liệu người dùng trực tiếp vào câu truy vấn.
- File nguồn người dùng chọn không bị sửa. Dữ liệu thiếu không được tự ý biến thành số 0.
- Các file database cục bộ phải nằm ngoài version control.

## 4. Phân chia trách nhiệm

- `data/`: đọc file, ánh xạ cột, làm sạch và kiểm tra dữ liệu.
- `storage/`: thao tác lưu/đọc dữ liệu qua repository.
- `features/`: định nghĩa hợp đồng đặc trưng và trích xuất đầu vào ML; không để mã sinh viên, họ tên hoặc target rò rỉ vào feature.
- `ml/`: tiền xử lý, huấn luyện, đánh giá và registry model.
- `scoring/`: tính điểm tổng kết theo profile học phần.
- `risk/` và `services/risk_prediction_service.py`: tính và giải thích mức nguy cơ theo quy tắc nghiệp vụ hiện có.
- `services/`: điều phối các module nghiệp vụ; UI gọi service thay vì tự thực hiện truy vấn hoặc huấn luyện.
- `ui/`: hiển thị dữ liệu, nhận thao tác và trình bày kết quả; không chứa công thức điểm hay quy tắc nguy cơ.
- `eda/`: thống kê và trực quan hóa dữ liệu.

## 5. Luồng dự đoán

1. Kiểm tra và chuẩn hóa dữ liệu đầu vào.
2. Trích xuất đúng các đặc trưng theo `FeatureContract`.
3. Huấn luyện và đánh giá XGBoost bằng dữ liệu đã chia tập; tiền xử lý chỉ được fit trên phần train trong bước đánh giá.
4. Lưu pipeline và metadata qua `ModelRegistry`.
5. Dự đoán điểm cuối kỳ bằng model được đăng ký với version `xgboost-2.0.0`.
6. Tính điểm tổng kết bằng `ScoringService`; không tạo điểm tổng kết độc lập khiến kết quả mâu thuẫn với công thức học phần.
7. Tính nguy cơ học vụ bằng risk engine hiện tại. Quy tắc nghỉ học **trên 75%** kích hoạt override học vụ theo yêu cầu hiện tại của chủ dự án.

XGBoost không mặc nhiên chính xác hơn với mọi dataset. Dữ liệu 30–50 sinh viên có thể khiến kết quả đánh giá dao động mạnh; phải báo cáo MAE, RMSE và R² trên tập kiểm thử, đồng thời diễn giải thận trọng. Không tuyên bố mô hình tốt hơn nếu chưa có đánh giá thực tế.

## 6. Dependency và cấu hình

- Runtime dependencies được khai báo trong `pyproject.toml`; giữ `requirements.txt` đồng bộ.
- Không cài thêm package chuẩn của Python như `sqlite3`, `json`, `logging` hoặc `pathlib` qua pip.
- Không hard-code đường dẫn tuyệt đối theo máy cá nhân.
- Cấu hình đường dẫn dùng `app_config.py` và biến môi trường phù hợp; không commit secret.
- Không thêm database server, ORM hoặc framework backend khi chưa có yêu cầu nghiệp vụ và căn cứ kỹ thuật.

## 7. Kiểm thử và nghiệm thu

Chạy các lệnh sau từ thư mục gốc dự án:

```powershell
python -m compileall src tests
ruff check src tests
python -m pytest tests/unit -v
```

Phần lưu trữ cần được kiểm tra với database tạm trong test: lưu/đọc, database chưa tồn tại, bảng chưa tồn tại, tên bảng không hợp lệ và lỗi ghi. Phần ML cần kiểm tra feature leakage, dữ liệu mục tiêu thiếu, huấn luyện, dự đoán và metrics.

Một test đã được viết không đồng nghĩa với test đã chạy thành công. Báo cáo nghiệm thu phải phản ánh kết quả thực tế.

## 8. Quy tắc bảo trì

- Trước khi đổi file, tìm các nơi đang import hoặc gọi file đó.
- Không xóa module đang được sử dụng chỉ vì tên có vẻ cũ.
- Không tạo lớp tương đương nếu dự án đã có service/repository đảm nhiệm cùng chức năng.
- Mọi thay đổi ảnh hưởng đến đầu ra điểm phải giữ đúng trọng số TX1/TX2/giữa kỳ/cuối kỳ đã cấu hình.
- Mọi thay đổi liên quan nguy cơ học vụ phải giữ quy tắc nghiệp vụ nghỉ học trên 75%, trừ khi chủ dự án xác nhận quy định khác.
