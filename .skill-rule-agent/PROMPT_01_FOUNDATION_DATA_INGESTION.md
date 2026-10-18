# PROMPT 01 — FOUNDATION & DATA INGESTION

Dán nguyên prompt này cho coding agent khi bắt đầu Giai đoạn 1.

---

Bạn đang thực hiện **Giai đoạn 1: Foundation & Data Ingestion** của `HaUI Student Performance & Academic Risk Prediction System`.

Trước tiên đọc `00_GLOBAL_INSTRUCTIONS.md` trong bộ prompt, sau đó đọc đầy đủ `AGENTS.md`, `SKILL.md`, `RULES.md`, `CODE_RULES.md`, `LEGACY_REPAIR_RULES.md` và `HAUI_Student_Performance_Prediction_Project_Plan.md` nếu hiện có trong repository. Tuân thủ tuyệt đối quy tắc không có comment/docstring trong code first-party.

## Mục tiêu

Tạo nền tảng repository chạy được và module nhập dữ liệu an toàn cho CSV/Excel. Giai đoạn này không triển khai công thức điểm, EDA, ML, risk engine, service layer hoặc UI hoàn chỉnh. Chỉ xây các hợp đồng dữ liệu/exception cần thiết để giai đoạn sau tích hợp.

## Việc phải thực hiện

### 1. Kiểm tra workspace

- Kiểm tra thư mục gốc, Git status, Python version, hệ điều hành và những file hiện có.
- Không ghi đè file đã tồn tại khi chưa đọc.
- Nếu đang ở repo không rỗng, bảo toàn code và cấu hình liên quan; không khởi tạo đè repository.
- Đọc và tuân thủ instruction files. Báo rõ file nào không có hoặc không thể đọc.

### 2. Tạo nền tảng project

Thiết lập cấu trúc Python package tối thiểu, dependency và các lệnh phát triển/test trong `pyproject.toml`. Dùng package name thống nhất, ví dụ `haui_student_prediction`.

Tạo cấu hình và tài liệu cần thiết, tối thiểu gồm:

- package trong `src/`;
- `src/haui_student_prediction/data/`;
- `tests/unit/`, `tests/integration/` khi thực sự cần;
- `config/` cho column mapping và validation nếu cần;
- `data/raw/`, `data/processed/`, `data/sample/` với `.gitkeep` nếu cần;
- `.gitignore` loại trừ `.venv`, cache, build artifacts, model artifacts không chủ ý chia sẻ và dữ liệu sinh viên thực tế;
- README ngắn với setup, test và cách chạy validation.

Không tạo module rỗng chỉ để đủ cây thư mục. Dependency phải khai báo nhất quán; tránh để `pyproject.toml` và `requirements.txt` mâu thuẫn.

### 3. Schema và column mapping

Tạo schema nội bộ cho các trường đã mô tả trong chỉ thị chung: `student_id`, `full_name`, `date_of_birth` hoặc `birth_year`, `hometown`, `major`, `absence_percentage`, `lms_learning_percentage`, `tx1_score`, `tx2_score`, `midterm_score`, `interaction_score`, `part_time_work`, `stress_level`, `learning_motivation`, và `actual_final_exam_score`.

- Tách cột định danh, feature ứng viên, trường điểm đầu vào, trường target và metadata.
- Hỗ trợ mapping tên cột nguồn sang canonical names qua cấu hình hoặc API truyền vào.
- Không giả định dataset nào cũng có đủ mọi cột.
- Phân biệt TX2 không áp dụng và TX2 thiếu/chưa công bố qua course profile/metadata rõ ràng; đừng suy ra “không có TX2” chỉ vì toàn bộ ô TX2 rỗng.
- Tránh coi `date_of_birth` và `birth_year` là hai nguồn độc lập khi chúng đại diện cùng thông tin.

### 4. Data ingestion

Tạo API đọc CSV và XLSX từ đường dẫn hoặc file-like input thích hợp.

Yêu cầu:

- Không thay đổi file nguồn.
- Không ghi đè dữ liệu gốc.
- Không gọi thao tác xóa bảng hoặc xóa dữ liệu khi import.
- Báo lỗi rõ ràng khi file không tồn tại, sai định dạng, không đọc được, thiếu cột bắt buộc hoặc có cấu trúc không hỗ trợ.
- Có quy định rõ về encoding CSV, sheet Excel và tên cột trùng lặp.
- Không tự động biến giá trị thiếu thành 0.
- Xử lý phần trăm theo dữ liệu nguồn; kiểm tra miền 0–100, không tính lại từ số buổi hoặc số giờ.
- Kiểm tra điểm trong phạm vi hợp lệ theo cấu hình thang điểm, không hard-code khi thang điểm chưa được xác minh.
- Kiểm tra duplicate theo khóa/cấu hình hợp lý; không tùy tiện xóa bản ghi duplicate. Báo số lượng và chiến lược xử lý.
- Phân biệt lỗi chặn import với cảnh báo chất lượng dữ liệu.
- Trả kết quả có cấu trúc, gồm DataFrame đã đọc, validation report và warnings/errors; không phụ thuộc vào UI.

### 5. Validation report

Báo cáo tối thiểu cần có:

- số dòng đọc được;
- số cột và các cột canonical được nhận diện;
- trường bắt buộc bị thiếu;
- giá trị thiếu theo cột;
- phần trăm ngoài khoảng 0–100;
- điểm nằm ngoài khoảng cấu hình;
- bản ghi trùng theo chiến lược khóa;
- loại cảnh báo/lỗi, số lượng và thông tin dòng có thể truy dấu mà không in dữ liệu cá nhân thừa thãi.

Không log họ tên, MSV đầy đủ hoặc dòng dữ liệu nhạy cảm khi không cần thiết.

### 6. Test bắt buộc

Tạo fixture tổng hợp, không dùng dữ liệu sinh viên thật. Viết test bao gồm:

- file CSV hợp lệ;
- file XLSX hợp lệ nếu môi trường có thư viện hỗ trợ;
- file rỗng/không tồn tại/sai định dạng;
- thiếu cột cần thiết;
- tên cột alias;
- tỷ lệ `-1`, `100.01`, `0`, `100`;
- điểm thiếu và điểm ngoài thang;
- duplicate rows;
- TX2 không áp dụng so với TX2 chưa có điểm;
- đảm bảo giá trị thiếu không biến thành zero;
- đảm bảo file nguồn không thay đổi sau import;
- đảm bảo MSV/họ tên không được gắn nhãn feature ML trong metadata schema.

Test hành vi và kết quả cụ thể, không chỉ test rằng hàm không ném exception.

## Giới hạn giai đoạn

Không xây risk formula, EDA chart, ML training, service/use case hoặc PyQt6 screens ở giai đoạn này, trừ khi có đoạn interface tối thiểu cần cho kiểm thử parser. Không tạo metric hoặc kết quả dự đoán mẫu để giả làm thật.

## Điều kiện nghiệm thu

- Project/package và môi trường dependency được khai báo.
- CSV ingestion hoạt động; XLSX hoạt động khi dependency cần thiết được cài.
- Schema/mapping/validation có test.
- Dữ liệu thiếu không bị thay bằng 0.
- Import không phá hủy dữ liệu nguồn.
- Test và lint được chạy nếu môi trường cho phép.
- Không có comment/docstring trong code first-party được tạo/sửa.
- README ghi đúng các lệnh đã xác minh.

Khi xong, báo rõ file đã tạo/sửa, thiết kế schema, lệnh test/lint đã chạy, pass/fail, blocker và lý do. **Không chuyển sang Phần 2.**
