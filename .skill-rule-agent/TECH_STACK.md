# TECH_STACK.md — QUY ĐỊNH CÔNG NGHỆ CHO DỰ ÁN HAUI

**Dự án:** HaUI Student Performance & Academic Risk Prediction System  
**Phiên bản:** 1.0  
**Vai trò tài liệu:** Nguồn chuẩn xác định công nghệ, trách nhiệm của từng công nghệ và ranh giới tích hợp.

## 1. Quy tắc áp dụng

1. Đây là tài liệu bắt buộc đối với mọi giai đoạn triển khai. Agent phải đọc file này trước khi thêm thư viện, chọn database hoặc đổi kiến trúc.
2. `TECH_STACK.md` quyết định công nghệ; `HAUI_Student_Performance_Prediction_Project_Plan.md` quyết định mục tiêu và nghiệp vụ; `PROJECT_FILE_MANIFEST.md` quyết định danh sách file mục tiêu; các file `AGENTS.md`, `SKILL.md`, `RULES.md`, `CODE_RULES.md` và `LEGACY_REPAIR_RULES.md` quyết định quy trình và quy tắc code.
3. Không được tự đổi H2 sang SQLite, PostgreSQL, MySQL hoặc một database khác. Không được thêm Docker, web server, ORM hoặc framework backend khi chưa có yêu cầu và lý do kỹ thuật cụ thể.
4. `pyproject.toml` là nguồn duy nhất khai báo package Python và dependency. Không duy trì danh sách dependency trùng lặp trong nhiều file độc lập.
5. Nếu môi trường không có Java/JAR cần thiết cho H2, agent phải báo blocker cụ thể và vẫn thực hiện các test không phụ thuộc H2. Không âm thầm chuyển sang database khác hoặc giả lập rằng database đã hoạt động.
6. Mọi công nghệ mới ngoài tài liệu này phải được đề xuất trong báo cáo với mục đích, chi phí phụ thuộc, ảnh hưởng kiến trúc và lý do không thể dùng stack hiện tại. Không tự thêm nếu không cần.

## 2. Stack công nghệ đã chọn

| Công nghệ | Trạng thái | Dùng để làm gì | Không dùng để làm gì |
|---|---|---|---|
| Python 3.11+ | Bắt buộc | Ngôn ngữ chính cho ứng dụng desktop, xử lý dữ liệu và ML | Không viết business logic bằng Java chỉ để truy cập database |
| PyQt6 | Bắt buộc | Giao diện desktop, điều hướng, biểu mẫu, bảng, trạng thái loading/error | Không chứa công thức tính điểm, logic nguy cơ, SQL trực tiếp hoặc code huấn luyện ML |
| pandas | Bắt buộc | Đọc, chuẩn hóa, kiểm tra và biến đổi bảng dữ liệu; tạo DataFrame cho EDA/ML | Không dùng để thay cho schema validation hoặc để che lỗi dữ liệu |
| NumPy | Bắt buộc | Các phép toán số và thao tác mảng cần thiết cho pipeline | Không xây toàn bộ nghiệp vụ thành các biểu thức khó đọc |
| scikit-learn | Bắt buộc | Pipeline tiền xử lý, baseline, mô hình hồi quy, cross-validation, chia tập và metrics | Không dùng điểm tổng kết/điểm thi thực tế làm input khi đang dự đoán các điểm đó |
| Matplotlib | Bắt buộc | Vẽ biểu đồ EDA và biểu đồ báo cáo | Không tạo số liệu minh họa rồi hiển thị như dữ liệu thật |
| Seaborn | Được phép/khuyến nghị | Heatmap, phân bố và biểu đồ thống kê khi làm kết quả rõ hơn | Không bắt buộc dùng nếu Matplotlib thuần đáp ứng tốt |
| openpyxl | Bắt buộc cho XLSX | Đọc/ghi file Excel qua pandas hoặc công cụ Excel phù hợp | Không sửa trực tiếp file Excel nguồn khi import |
| `csv` / `json` của Python | Bắt buộc khi xử lý định dạng tương ứng | Đọc CSV/JSON đơn giản và serialization cấu hình/kết quả có cấu trúc | Không dùng JSON để nhét dữ liệu lớn không kiểm soát hoặc ghi đè dữ liệu nguồn |
| H2 Database | Bắt buộc cho session storage | Lưu dữ liệu đã được chấp nhận, metadata import, bản ghi phân tích/dự đoán trong phiên chạy hiện tại | Không thay thế file nguồn; không được âm thầm dùng SQLite/PostgreSQL |
| JPype1 | Bắt buộc để kết nối H2 từ Python | Khởi chạy JVM trong tiến trình ứng dụng và cung cấp cầu nối Python–Java/JDBC | Không khởi động/dừng JVM lặp lại theo từng query hoặc từng lần mở view |
| H2 JDBC JAR | Bắt buộc để dùng H2 | Cung cấp JDBC driver `org.h2.Driver` | Không tải JAR bí mật/không rõ nguồn; không tự tải trong runtime mà không báo |
| joblib | Bắt buộc khi lưu model scikit-learn | Lưu/nạp model artifact và pipeline đã huấn luyện | Không nạp artifact không tin cậy hoặc coi artifact mẫu là model đã xác minh |
| TOML + `tomllib` | Bắt buộc cho cấu hình | Đọc cấu hình ứng dụng, ánh xạ cột, course profiles, risk profiles và database settings | Không nhúng trọng số/ngưỡng rải rác trong code UI |
| `logging` của Python | Bắt buộc | Log trạng thái, lỗi kỹ thuật và luồng chạy có kiểm soát | Không log đầy đủ MSV, họ tên, hồ sơ hoặc câu trả lời khảo sát nếu không cần |
| pytest | Bắt buộc | Unit, integration, regression và contract tests | Không xem việc viết test là bằng chứng test đã chạy |
| Ruff | Bắt buộc | Lint và format Python theo cấu hình chung | Không dùng formatter để thay cho code review hoặc test |
| pytest-qt | Dùng cho UI tests khi cần | Kiểm thử tương tác PyQt6 và widget | Không bắt buộc trong các test domain không cần Qt |
| Markdown | Bắt buộc | Tài liệu kiến trúc, từ điển dữ liệu, mô hình, rủi ro, setup và kết quả kiểm thử | Không đưa giải thích vào comment/docstring trong code first-party |

## 3. Database: H2 session storage

### 3.1. Quyết định database

Dự án dùng **H2 Database ở chế độ in-memory** làm kho dữ liệu làm việc trong phiên chạy ứng dụng. Không dùng PostgreSQL hoặc SQLite trong phiên bản này.

- Định dạng đầu vào bắt buộc: CSV và XLSX.
- JSON có thể được hỗ trợ cho dữ liệu/cấu hình có cấu trúc khi phù hợp với data contract; không được tự suy đoán schema JSON không rõ ràng.
- File gốc vẫn nằm ở vị trí do người dùng chọn và phải được giữ nguyên.
- Sau khi import và validation thành công, dữ liệu được ghi vào session store H2 trong một transaction. Nếu import thất bại, rollback transaction và giữ nguyên dữ liệu session trước đó.
- H2 giữ dữ liệu được nạp, dataset/session metadata và các kết quả hiện hành cần thiết trong phiên chạy hiện tại.
- Khi ứng dụng/JVM kết thúc, database in-memory không được xem là bản lưu lâu dài. Các file báo cáo, biểu đồ và model artifacts được lưu riêng tại các thư mục output đã cấu hình.
- Không tạo chế độ H2 file-based mặc định trong MVP. Chỉ bổ sung chế độ lưu bền nếu chủ dự án yêu cầu rõ và tài liệu/test đã cập nhật.

### 3.2. Kết nối Python tới H2

H2 là database Java dùng JDBC; Python phải kết nối qua một adapter tập trung thay vì gọi Java/JDBC trực tiếp từ mọi module.

- Cầu nối: `JPype1` và API `jpype.dbapi2` nếu phù hợp với bản JPype đã pin.
- JDBC driver class: `org.h2.Driver`.
- JDBC URL mặc định: `jdbc:h2:mem:haui_session;DB_CLOSE_DELAY=-1`.
- H2 JAR được đặt tại `vendor/h2/h2-2.5.252.jar` hoặc đường dẫn do biến môi trường `HAUI_H2_JAR` chỉ định.
- Phiên bản H2 mục tiêu ban đầu: `2.5.252`. Nếu phiên bản này không tương thích với môi trường triển khai, agent phải ghi rõ lỗi kiểm chứng và đề xuất một phiên bản thay thế có căn cứ; không tự đổi phiên bản không ghi nhận.
- Môi trường cần Java/JDK tương thích với JPype; JDK 17 64-bit là lựa chọn phát triển đề xuất. Python và JVM phải tương thích kiến trúc, ví dụ đều 64-bit.
- JVM được khởi động tối đa một lần trong process. Database adapter chịu trách nhiệm tìm JAR, kiểm tra môi trường, mở/đóng connection, transaction và chuyển lỗi database thành exception rõ ràng.
- Ứng dụng không được crash không rõ nguyên nhân nếu thiếu Java/JAR. UI/service phải trả lỗi thiết lập có thể xử lý; không giả vờ đã lưu dữ liệu.

### 3.3. Module lưu trữ

Đặt toàn bộ phần tích hợp database trong `src/haui_student_prediction/storage/`:

- `h2_database.py`: tìm JAR, kiểm tra Java/JVM, tạo connection, quản lý transaction và lỗi database.
- `schema.sql`: tạo/cập nhật cấu trúc dữ liệu theo cách không phá hủy; không dùng `DROP ALL OBJECTS`, `drop_all` hoặc xóa schema trong luồng import.
- `dataset_repository.py`: quản lý session/dataset đã import, thông tin nguồn và số liệu validation.
- `student_repository.py`: truy vấn và ghi bản ghi sinh viên/học phần theo contract của schema.
- `prediction_repository.py`: lưu kết quả dự đoán và điểm nguy cơ trong session hiện tại.

Có thể gộp repository nếu implementation nhỏ và ranh giới vẫn rõ; nếu gộp, cập nhật manifest và chỉ ra file thay thế cùng test tương ứng. Không tạo file rỗng chỉ để khớp danh sách.

Các module UI, EDA, scoring và ML không được mở JDBC connection trực tiếp. Chúng nhận dữ liệu qua service/repository contracts; pandas DataFrame có thể được tạo ở ranh giới data/service khi EDA hoặc ML cần bảng dữ liệu.

### 3.4. Quy tắc an toàn cho H2

- Chạy validation trước khi commit dữ liệu import vào H2.
- Dùng prepared/parameterized statements cho mọi giá trị SQL.
- Không dựng SQL bằng cách ghép trực tiếp tên/ngày/giá trị dữ liệu từ người dùng.
- Dùng transaction; rollback khi bất kỳ bước ghi nào thất bại.
- Dùng khóa/session ID để phân biệt lần import, không xóa dữ liệu session cũ chỉ vì nạp file mới.
- Tạo schema theo cách idempotent và có version; không tự xóa bảng để giải quyết lỗi schema.
- Không ghi dữ liệu cá nhân vào log; không đưa raw database files, dump chứa dữ liệu thật hoặc report định danh vào Git.
- Unit tests có thể dùng H2 in-memory riêng và dữ liệu tổng hợp. Integration tests phải kiểm tra vòng đời JVM/connection một cách an toàn.

## 4. Luồng dữ liệu và trách nhiệm công nghệ

1. **CSV/XLSX/JSON input** → `data/ingestion.py` đọc file; file nguồn không bị sửa.
2. **Schema/validation** → `schema.py`, `column_mapping.py`, `validator.py` chuẩn hóa tên và kiểm tra kiểu/giá trị; không tự biến thiếu thành 0.
3. **H2 session storage** → repository ghi dataset đã qua validation vào H2 trong transaction; nếu lỗi, rollback.
4. **EDA** → pandas/NumPy thống kê; Matplotlib/Seaborn tạo biểu đồ; loại MSV/họ tên khỏi phân tích định lượng.
5. **Scoring** → Python domain module áp dụng course profile, độc lập H2 và UI.
6. **ML** → scikit-learn pipeline huấn luyện/dự đoán; joblib lưu model; target `actual_final_exam_score` không đi vào X.
7. **Risk engine** → Python domain module chuẩn hóa thành phần và tính risk score; ngưỡng `absence_percentage > 30` tạo override riêng.
8. **Services** → điều phối repository, EDA, scoring, ML và risk engine; là lối vào từ UI.
9. **PyQt6 UI** → chỉ hiển thị, nhận thao tác và gọi services; không chứa SQL, scoring formulas hoặc training pipeline.
10. **Reports/artifacts** → lưu theo đường dẫn cấu hình; không lưu file nguồn đè lên chính nó.

## 5. Dependency groups và quản lý môi trường

### Runtime dependencies

- `PyQt6`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `openpyxl`
- `JPype1`
- `joblib`

`csv`, `json`, `tomllib`, `logging`, `pathlib`, `sqlite3` và các thành phần standard library khác không được thêm như pip dependencies. Không dùng `sqlite3` để lưu dữ liệu dự án; nó chỉ được phép xuất hiện trong kiểm thử độc lập nếu thật sự có lý do và không thay database chính.

### Development dependencies

- `pytest`
- `pytest-qt` cho UI tests nếu được dùng
- `ruff`

Không thêm XGBoost, LightGBM, SQLAlchemy, Alembic, Docker hoặc web framework vào MVP trừ khi benchmark/yêu cầu cụ thể chứng minh cần thiết và chủ dự án chấp thuận.

### Source of truth

- `pyproject.toml` là nguồn duy nhất cho dependency và tool configuration.
- `config/app_config.toml`, `config/column_mapping.toml`, `config/course_profiles.toml`, `config/risk_profiles.toml` và `config/database.toml` chứa setting chạy được.
- Không hard-code đường dẫn tuyệt đối phụ thuộc máy cá nhân.
- Không để secret hoặc mật khẩu thật trong cấu hình được commit.

## 6. Chất lượng, kiểm thử và quy tắc code

- `pytest`: test schema, ingest, database adapter, transaction rollback, scoring, risk boundaries, EDA, ML leakage, services và UI.
- `Ruff`: lint/format, cấu hình trong `pyproject.toml`.
- Có script hoặc test để kiểm tra code first-party không có comment/docstring theo `CODE_RULES.md`.
- Dùng dữ liệu tổng hợp cho test, luôn ghi nhãn synthetic.
- H2 integration test có thể đánh dấu là cần JDK/JAR, nhưng blocker phải nêu cụ thể; không tính test bị skip là test pass.
- Chưa có dữ liệu HaUI thật thì không có metric ML thật. Pipeline vẫn phải được test kỹ với dữ liệu synthetic, nhưng report phải tách bạch hai mục này.

## 7. Thư mục công nghệ liên quan

```text
config/
├── app_config.toml
├── column_mapping.toml
├── course_profiles.toml
├── risk_profiles.toml
└── database.toml

vendor/
└── h2/
    └── h2-2.5.252.jar

src/haui_student_prediction/
└── storage/
    ├── __init__.py
    ├── h2_database.py
    ├── schema.sql
    ├── dataset_repository.py
    ├── student_repository.py
    └── prediction_repository.py
```

JAR nêu trên là dependency bên ngoài và phải được lấy từ nguồn H2 chính thức. Có thể không đưa file binary vào Git nếu chính sách repository yêu cầu; khi đó README phải hướng dẫn tải đúng bản, và ứng dụng phải thông báo rõ khi JAR chưa tồn tại.

## 8. Các kiểm tra nghiệm thu bắt buộc cho công nghệ

- `python --version`: xác minh phiên bản Python.
- `java -version`: xác minh Java/JDK và kiến trúc phù hợp.
- Kiểm tra `JPype1` import được.
- Kiểm tra file H2 JAR tồn tại và load được qua JDBC.
- Tạo database in-memory và thực hiện create/insert/select/rollback trong test.
- Import fixture CSV/XLSX và lưu dataset đã validate vào H2 session.
- Nạp file thứ hai nhưng không làm mất dữ liệu/session cũ khi lần import mới lỗi.
- Đóng connection không làm mất dữ liệu trước khi kết thúc process; khi ứng dụng/JVM kết thúc, không cam kết giữ H2 in-memory.
- Chạy test, Ruff và startup smoke test; ghi kết quả thực tế.

## 9. Tài liệu kỹ thuật tham khảo

- H2 Quickstart: https://h2database.github.io/html/quickstart.html
- H2 Features and connection modes: https://h2database.github.io/html/features.html
- JPype User Guide: https://jpype.readthedocs.io/en/stable/userguide.html
- JPype DB-API 2: https://jpype.readthedocs.io/en/latest/dbapi2.html

## Quyết định cuối cùng

Stack của MVP là **Python + PyQt6 + pandas/NumPy + scikit-learn + Matplotlib/Seaborn + H2 in-memory qua JPype/JDBC + pytest/Ruff**. H2 session storage là công nghệ database được chọn; không thay bằng PostgreSQL hoặc SQLite nếu không có quyết định mới được ghi nhận.
