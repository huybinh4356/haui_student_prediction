# PROMPT 06 — PYQT6 USER INTERFACE & INTEGRATION

Dán prompt này sau khi Phần 1–5 đã được nghiệm thu.

---

Bạn đang thực hiện **Giai đoạn 6: PyQt6 UI và tích hợp end-to-end** cho `HaUI Student Performance & Academic Risk Prediction System`.

Đọc `00_GLOBAL_INSTRUCTIONS.md`, các file instruction bắt buộc và tài liệu architecture/service hiện có. Kiểm tra các service contracts thật trước khi viết widget. Code first-party không có comment/docstring.

## Mục tiêu

Tạo giao diện desktop hoàn chỉnh bằng PyQt6, dùng tiếng Việt, dễ hiểu và tích hợp với service layer. Không tạo standalone form không có kết nối logic. Không triển khai business logic, công thức điểm, risk weighting hoặc ML training trực tiếp trong widget.

## 1. Những màn hình cần có

Triển khai các view theo cấu trúc phù hợp với project đang có:

1. **Tổng quan (Dashboard):** số dòng đã nạp, chất lượng dữ liệu, trạng thái model, phân bố risk nếu có dataset, số override attendance và thông tin thời điểm cập nhật.
2. **Nhập dữ liệu:** chọn CSV/XLSX, xem file được chọn, chạy validation, hiển thị lỗi/cảnh báo và thống kê import; không tự sửa file nguồn.
3. **EDA:** chọn dataset đã nạp, chạy thống kê/biểu đồ, hiển thị hoặc mở các artifact; xử lý trạng thái chưa có dataset và lỗi phân tích.
4. **Dự đoán điểm cuối kỳ:** nhập/chọn record hợp lệ, chọn course profile rõ ràng, gọi `prediction_service`, hiển thị điểm cuối kỳ dự đoán và điểm tổng kết dự kiến cùng model/formula metadata.
5. **Dự đoán nguy cơ học vụ:** hiển thị weighted score/category, component scores, giải thích, absence status và override `HOC_LAI`.
6. **Model/evaluation status:** hiển thị model đã train/chưa train, version, target, split strategy và metrics thật nếu có.
7. **Báo cáo/xuất kết quả:** chọn định dạng/path, cảnh báo nếu xuất dữ liệu định danh và xử lý lỗi file.

Không hiển thị số liệu giả khi dataset hoặc model chưa có. Empty-state phải nói rõ cần làm gì tiếp theo.

## 2. Tách biệt UI và logic

- Widget chỉ xử lý sự kiện, lấy input, gọi service và trình bày result.
- Không đọc CSV/XLSX trực tiếp trong widget.
- Không đặt formula hoặc risk weights trong widget.
- Không gọi estimator `.fit()`/`.predict()` trực tiếp từ UI.
- Không copy/paste business logic vào nhiều màn hình.
- Dependency services được inject hoặc khởi tạo tại composition/root layer, không tạo lại tùy tiện trong từng callback.
- Giữ các widget có kích thước/responsibility hợp lý.

## 3. UI/UX và trạng thái

- Có trạng thái idle, loading, success, warning, error và empty-state.
- Hiện tiến trình khi tác vụ mất thời gian; không block main Qt thread khi train/EDA tác vụ nặng. Dùng QThread/QRunnable hoặc cơ chế phù hợp, quản lý signal/error/cleanup an toàn.
- Disable controls hoặc chống chạy tác vụ lặp khi công việc đang thực hiện.
- Đưa lỗi kỹ thuật thành thông báo dễ hiểu mà không che chi tiết cần thiết cho log kỹ thuật.
- Có cách xem lỗi/validation report đủ cụ thể để người dùng sửa dữ liệu.
- Hạn chế show thông tin định danh không cần thiết.
- Có thể chọn theme/layout, nhưng ưu tiên readability, nhất quán, không để trang trí lấn át thông tin.
- Đảm bảo lỗi do không có PyQt6 hoặc không có model không bị che bằng fake UI result.

## 4. Kiểm thử giao diện

Ưu tiên tests có thể tự động hóa với Qt offscreen/platform phù hợp:

- app/window khởi động;
- điều hướng view;
- import view gọi đúng service;
- hiển thị validation errors/warnings;
- prediction view xử lý model-not-ready;
- course profile A/B và TX2 missing được phản ánh đúng từ service;
- risk view hiển thị override khi absence >30;
- không tự tạo metric/prediction khi service trả unavailable;
- loading/success/error state;
- worker thread gửi result/error về UI an toàn;
- export dialog/service error.

Nếu môi trường hiện tại không hỗ trợ GUI test, vẫn viết test cho presentation/controller logic có thể cô lập, thử import/startup bằng lệnh khả dụng và ghi rõ hạn chế. Không khẳng định đã test click thủ công nếu chưa làm thật.

## 5. End-to-end acceptance

Chạy workflow bằng fixture synthetic được gắn nhãn rõ:

1. Chọn file sample.
2. Import và validation.
3. Xem data quality report.
4. Chạy EDA và tạo artifact.
5. Thử risk calculation.
6. Thử scoring với cả hai course profiles.
7. Thử prediction path khi model chưa train để xác nhận thông báo model-not-ready.
8. Nếu fixture có target và test model hợp lệ, chạy một smoke training/evaluation riêng, nhưng không coi metric synthetic là hiệu năng HaUI.
9. Export report vào thư mục test/temp, không ghi đè raw data.

## 6. Tài liệu và đóng gói

Cập nhật README với lệnh setup, start app, import dữ liệu, chạy tests và cách hiểu status. Khớp lệnh với dependency manager thật. Không commit raw student dataset, local model artifacts nhạy cảm, `.venv` hoặc secrets.

Kiểm tra entry point và lỗi khởi động trong môi trường khả dụng. Nếu cần packaging cho Windows, chỉ thiết lập cấu hình khi có thể kiểm tra và không làm mất khả năng chạy từ source.

## 7. Luật code cuối cùng

- Không comment/docstring trong code first-party tạo/sửa.
- Không để stub/todo trong luồng tính năng được tuyên bố hoàn thành.
- Không có fake metrics/fake student predictions.
- Không để widget gọi domain code ngoài service boundary.
- Chạy full test suite, linter, formatter check và startup smoke test có thể chạy được.
- Kiểm tra diff để chắc không sửa/xóa file ngoài phạm vi.

## Điều kiện nghiệm thu

Ứng dụng có màn hình tích hợp với service thật; empty/loading/error states hoạt động; các luồng chính được kiểm thử; không lặp logic nghiệp vụ trong UI; README đúng; lệnh test/lint/start được chạy hoặc blocker được nêu chính xác; không có comment/docstring trong code first-party tạo/sửa.

Cuối cùng, lập báo cáo tổng kết tách rõ `Implemented and verified`, `Implemented but not fully verified`, `Blocked`, `Not implemented`. Không tuyên bố dự án hoàn thành nếu các luồng trọng yếu chưa chạy được.
