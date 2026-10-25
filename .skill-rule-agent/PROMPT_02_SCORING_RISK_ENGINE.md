# PROMPT 02 — COURSE SCORING & ACADEMIC-RISK ENGINE

Dán prompt này sau khi Phần 1 đã được nghiệm thu.

---

Bạn đang thực hiện **Giai đoạn 2: Logic nghiệp vụ, tính điểm học phần và bộ tính điểm nguy cơ** cho `HaUI Student Performance & Academic Risk Prediction System`.

Đọc `00_GLOBAL_INSTRUCTIONS.md` và năm file luật trong repository trước khi sửa. Xem source, tests, config và báo cáo Phần 1 thực tế; không tin báo cáo cũ nếu code/test không xác nhận. Giữ quy tắc không comment/docstring.

## Mục tiêu

Triển khai module thuần Python, deterministic, không phụ thuộc PyQt6 hay UI. Không huấn luyện ML trong giai đoạn này. Mọi công thức phải có test biên và cấu hình tập trung.

## 1. Course Scoring Engine

Tạo module scoring với API có kiểu dữ liệu rõ ràng, ví dụ nhận course profile, điểm thành phần và điểm cuối kỳ dự đoán, trả structured result gồm raw calculated score, formula profile, validation state và errors/warnings.

### Profile A — Không có TX2

`course_score = 0.20 * tx1_score + 0.20 * midterm_score + 0.60 * predicted_final_exam_score`

### Profile B — Có TX2

`course_score = 0.15 * tx1_score + 0.15 * tx2_score + 0.20 * midterm_score + 0.50 * predicted_final_exam_score`

Quy tắc:

- Course profile phải rõ ràng: `course_without_tx2` hoặc `course_with_tx2` hay tên tương đương nhất quán.
- Khi profile có TX2 mà điểm TX2 thiếu/chưa công bố, trả trạng thái thiếu dữ liệu; không chuyển tự động sang Profile A.
- Không có TX2 chỉ được dùng khi metadata/profile xác định môn đó thực sự không có component này.
- Kiểm tra điểm đầu vào theo configured score scale.
- Kiểm tra tổng trọng số mỗi profile bằng 1 với tolerance rõ ràng.
- Không tự động clip raw predicted score tại tầng model/scoring. Nếu muốn hiển thị score giới hạn, giữ riêng raw và display score, và ghi rõ policy.
- Không hard-code profile trong UI.

## 2. Risk Engine

Tạo module risk độc lập với scoring và ML.

### Trọng số bắt buộc

| Thành phần | Trọng số |
|---|---:|
| `absence_risk` | 0.75 |
| `academic_score_risk` | 0.08 |
| `lms_risk` | 0.08 |
| `interaction_risk` | 0.04 |
| `part_time_work_risk` | 0.02 |
| `stress_risk` | 0.015 |
| `motivation_risk` | 0.015 |

Kiểm tra bằng code/test rằng tổng bằng 1.0 (100%). Từng component risk phải được đưa về 0–100 trước khi weighted sum. Điểm tổng hợp có giới hạn kỳ vọng 0–100 nếu các input component hợp lệ.

### Quy tắc absence

- `absence_percentage > 75`: `override_triggered = true`, `attendance_status = HOC_LAI` theo yêu cầu dự án.
- `20 <= absence_percentage <= 75`: nguy cơ attendance cao.
- `10 <= absence_percentage < 20`: trung bình.
- `5 <= absence_percentage < 10`: thấp.
- `0 <= absence_percentage < 5`: rất thấp.

Đúng biên `75.0` không kích hoạt override. `75.01` và `76.0` phải kích hoạt. Giá trị âm hoặc trên 100 phải validation error, không được âm thầm clip.

Override phải độc lập với weighted score. Một khảo sát tích cực hoặc score tổng hợp thấp không được hủy `HOC_LAI`. Trả cả weighted score/category và override status riêng.

### Quy tắc LMS

`lms_learning_percentage` có sẵn từ nguồn, không tính lại.

Ngưỡng prototype theo giả định phần trăm càng cao thì mức học trên LMS càng nhiều:

- `< 5`: rất cao;
- `5 <= x < 10`: cao;
- `10 <= x < 20`: trung bình;
- `20 <= x < 30`: thấp;
- `x >= 30`: rất thấp.

Đưa ngưỡng vào config; đánh dấu chưa được kiểm chứng bằng lịch sử HaUI.

### Chuẩn hóa các thành phần khác

Tạo function riêng để chuẩn hóa `academic_score_risk`, `interaction_risk`, `part_time_work_risk`, `stress_risk`, `motivation_risk`. Mọi hàm phải xác định chiều nghĩa của thang đo; risk 100 luôn nghĩa là rủi ro cao.

- Không mặc định đi làm thêm là xấu. Nếu chưa có bằng chứng, cấu hình ánh xạ trung tính và giải thích trong tài liệu.
- Stress cao có thể ánh xạ thành nguy cơ cao chỉ khi scale definition xác nhận chiều này.
- Motivation/interaction cao thường có thể là tín hiệu thuận lợi, nhưng phải đọc định nghĩa thang đo thật trước khi đảo chiều.
- Không hard-code penalty 50 cho `part_time_work` mà không có căn cứ.
- Không tự tạo cắt ngưỡng điểm đậu/trượt nếu chưa có định nghĩa được phê duyệt.

### Tổng điểm

`risk_score = 0.75 * absence_risk + 0.08 * academic_score_risk + 0.08 * lms_risk + 0.04 * interaction_risk + 0.02 * part_time_work_risk + 0.015 * stress_risk + 0.015 * motivation_risk`

Tạo cấu hình mapping từ risk score sang `VERY_LOW`, `LOW`, `MEDIUM`, `HIGH`, `VERY_HIGH`. Nếu dùng band bằng nhau theo từng 20 điểm để khởi tạo, phải ghi là prototype threshold, để config thay được và test mọi boundary. Risk score không phải probability.

## 3. Kết quả và explainability

Trả cấu trúc kết quả rõ ràng bao gồm:

- `risk_score`;
- `risk_category`;
- `component_scores`;
- `attendance_status`;
- `override_triggered`;
- `reasons` có thể hiển thị cho người dùng;
- `warnings` cho đầu vào thiếu hoặc mapping chưa được xác minh.

Không chỉ trả một chuỗi nhãn. Không đưa logic UI vào risk engine.

## 4. Test bắt buộc

- Hai công thức score đúng chính xác bằng ví dụ số tự chọn.
- Tổng trọng số profile bằng 1.
- Thiếu TX2 ở profile B không chuyển sang A.
- TX2 không áp dụng ở profile A vẫn tính được.
- Các boundary absence: `0`, `4.99`, `5`, `9.99`, `10`, `19.99`, `20`, `74.99`, `75`, `75.01`, `76`, `100`.
- Validation `-0.01`, `100.01`.
- Weighted risk theo bộ input biết trước.
- Weights sum 100%; component risk bounds; total risk bounds.
- Override vẫn bật khi các component còn lại cho điểm tốt.
- Đi làm thêm ánh xạ trung tính theo config hiện tại.
- LMS boundaries: `0`, `4.99`, `5`, `9.99`, `10`, `19.99`, `20`, `29.99`, `30`, `100`.
- Thiếu dữ liệu bắt buộc trả trạng thái/error phù hợp, không tạo giá trị rác.

## Giới hạn

Không viết ML models, EDA charts, UI hoặc service orchestration trong giai đoạn này. Có thể cập nhật data schema/config vừa tạo ở Phần 1 khi cần, nhưng tránh breaking changes không có lý do.

## Điều kiện nghiệm thu

Toàn bộ công thức nằm trong module domain độc lập, cấu hình tập trung, test logic và boundary đầy đủ, lệnh test/lint chạy và kết quả được ghi trung thực. Không có comment/docstring trong code first-party tạo/sửa. **Không chuyển sang Phần 3.**
