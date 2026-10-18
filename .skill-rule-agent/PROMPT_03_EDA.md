# PROMPT 03 — EXPLORATORY DATA ANALYSIS (EDA)

Dán prompt này sau khi Phần 1 và Phần 2 đã được nghiệm thu.

---

Bạn đang thực hiện **Giai đoạn 3: Exploratory Data Analysis (EDA)** của `HaUI Student Performance & Academic Risk Prediction System`.

Đọc `00_GLOBAL_INSTRUCTIONS.md` và các file luật bắt buộc. Kiểm tra API ingestion/schema/risk/scoring thực tế trước khi tích hợp. Không giả định tên cột hoặc return types nếu source hiện tại khác mô tả; điều chỉnh có kiểm soát và thêm test. Code first-party không có comment/docstring.

## Mục tiêu

Tạo EDA có thể chạy độc lập từ dữ liệu đã được validate, tạo thống kê và biểu đồ lưu ra file, sau đó để service/UI có thể gọi lại mà không đưa logic phân tích vào widget.

## 1. Module EDA

Tạo package `eda/` với các trách nhiệm tách biệt:

- chuẩn bị dataset cho phân tích;
- thống kê dataset và chất lượng dữ liệu;
- phân tích biến số;
- phân tích biến phân loại/ordinal;
- phân tích mối quan hệ với điểm thi cuối kỳ thực tế khi target có sẵn;
- sinh chart và export report.

Dùng Matplotlib và có thể dùng Seaborn cho chart phù hợp; dependency phải được khai báo. Không bắt buộc dùng Seaborn nếu biểu đồ Matplotlib thuần đã rõ ràng và dễ kiểm soát.

EDA nhận dataset đã validate hoặc gọi một API ingestion/validation rõ ràng. Không tự tạo dữ liệu thay thế nếu import thất bại.

## 2. Các loại phân tích cần có

### Dataset overview

- số dòng/cột;
- số student_id duy nhất nếu có;
- missing count/percentage theo cột;
- duplicate summary;
- dtype;
- thống kê mô tả numeric;
- phân bố categorical/ordinal;
- validation warnings.

### Điểm học tập

- phân bố `tx1_score`, `tx2_score`, `midterm_score`;
- tỷ lệ TX2 thiếu và phân biệt no-TX2 course với missing TX2 nếu metadata có sẵn;
- phân bố `actual_final_exam_score` khi có nhãn;
- các quan hệ giữa điểm thành phần và điểm cuối kỳ thực tế.

### Hoạt động học tập

- phân bố `absence_percentage`;
- phân bố `lms_learning_percentage`;
- phân bố `interaction_score`;
- số record/sinh viên theo các band absence/risk có thể tính hợp lệ;
- quan hệ giữa các biến hoạt động với outcome nếu outcome tồn tại.

### Khảo sát và thuộc tính khác

- đi làm thêm;
- stress;
- động lực học tập;
- ngành học;
- ngày sinh/năm sinh được chuyển thành biến phân tích phù hợp nếu đủ dữ liệu;
- quê quán theo nhóm tổng hợp phù hợp.

### Tương quan/quan hệ

- correlation matrix cho numeric features phù hợp;
- heatmap chỉ dùng các biến định lượng/ordinal có ý nghĩa và xử lý missing rõ ràng;
- dùng count plot/boxplot/violin hoặc biểu đồ thích hợp cho categorical vs numeric;
- không mã hóa mã sinh viên, họ tên, quê quán dạng raw string thành số rồi tính correlation.

## 3. Loại bỏ định danh khỏi phân tích định lượng

Không được đưa `student_id` hoặc `full_name` vào numeric correlation, model-like feature comparison hoặc train-ready matrix. Có thể sử dụng student_id để đếm unique/group, nhưng không hiển thị danh sách định danh trong biểu đồ tổng hợp.

Giữ chính sách privacy-by-default: EDA mặc định tổng hợp dữ liệu. Không ghi tên/MSV lên tên file, title chart hay report công khai.

## 4. Output và API

Tạo API EDA độc lập UI. Tên cụ thể tùy kiến trúc hiện có, nhưng cần có interface để:

- nhận dataset và output directory;
- tạo summary có cấu trúc;
- lưu chart với tên ổn định;
- trả danh sách artifacts và cảnh báo;
- không sửa input DataFrame tại chỗ một cách khó đoán;
- xử lý target không có mặt: vẫn chạy EDA mô tả, đồng thời bỏ qua các phân tích yêu cầu target và nêu rõ lý do.

Dùng Matplotlib non-interactive backend khi chạy trong test/headless environment nếu cần. Không gọi `plt.show()` trong service batch. Đảm bảo đóng figure để tránh leak bộ nhớ.

## 5. Test bắt buộc

- dataset numeric/categorical hỗn hợp;
- dữ liệu thiếu;
- target không tồn tại;
- cột MSV/họ tên không xuất hiện trong danh sách feature định lượng/correlation;
- schema alias hoặc dữ liệu đã chuẩn hóa theo ingestion;
- đầu ra summary có giá trị đúng;
- các biểu đồ mong đợi được tạo ra ở output directory;
- output directory không tồn tại được tạo an toàn;
- input không bị mutate ngoài hợp đồng;
- empty dataframe và cột không có dữ liệu numeric xử lý có thông báo;
- dữ liệu duplicate được thống kê chứ không âm thầm xóa.

Dùng fixture tổng hợp; không dùng EDA sample để đưa ra kết luận thật về sinh viên HaUI.

## 6. Chạy thử thủ công

Tạo script CLI hoặc entry point rõ ràng để chạy EDA trên một file được truyền vào và một thư mục output do người dùng chỉ định. Không hard-code đường dẫn cá nhân. Nêu lệnh thực tế đã chạy và liệt kê chart/report được tạo.

## Giới hạn

Không tích hợp PyQt6 widget trong giai đoạn này. Không huấn luyện ML. Không tạo biểu đồ bằng dữ liệu giả rồi mô tả như kết quả trường.

## Điều kiện nghiệm thu

EDA độc lập, có test và có đầu ra artifact kiểm tra được; phân tích định danh đúng; xử lý target thiếu minh bạch; command/lint/test chạy khi môi trường cho phép; không có comment/docstring trong code first-party tạo/sửa. **Không chuyển sang Phần 4.**
