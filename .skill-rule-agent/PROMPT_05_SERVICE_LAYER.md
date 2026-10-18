# PROMPT 05 — SERVICE LAYER & APPLICATION USE CASES

Dán prompt này sau khi Phần 1–4 đã được nghiệm thu.

---

Bạn đang thực hiện **Giai đoạn 5: Service Layer** cho `HaUI Student Performance & Academic Risk Prediction System`.

Đọc `00_GLOBAL_INSTRUCTIONS.md` và các luật bắt buộc. Kiểm tra ingestion, scoring, risk, EDA và ML contracts hiện tại. Đừng đoán function name/return type: đọc source thật và viết service theo interface thực tế hoặc sửa interface có kiểm soát, kèm tests. Code first-party không có comment/docstring.

## Mục tiêu

Tạo lớp orchestration duy nhất giữa UI và domain modules. UI không được đọc file trực tiếp, gọi model trực tiếp, tính công thức hoặc dựng risk score. UI chỉ gọi application services và hiển thị kết quả trả về.

## 1. Services/use cases cần có

Triển khai service tương ứng với trách nhiệm thực tế, ví dụ:

- `data_import_service`: import file, gọi validator, trả report và dataset reference;
- `eda_service`: chạy EDA từ dataset đã validate và trả summary/artifact list;
- `prediction_service`: validate prediction input, gọi ML model, trả final exam prediction, sau đó gọi course scoring module để tính predicted course score;
- `risk_prediction_service`: tính component risk, weighted risk, attendance override và giải thích kết quả;
- `model_training_service`: điều phối train/evaluate/registry nếu ứng dụng có nhu cầu huấn luyện từ UI/CLI;
- `report_service`: tạo các report export được phép.

Không bắt buộc dùng đúng các tên trên nếu repo đã có naming convention tốt. Không tạo service chỉ làm lớp chuyển tiếp không có giá trị; giữ module boundaries rõ ràng.

## 2. Service contracts

Mỗi service cần:

- input và output có cấu trúc, type hints phù hợp;
- validation trước khi gọi module hạ tầng;
- phân biệt validation error, missing data, model-not-ready, file error và internal error;
- không trả fake fallback result;
- không gọi UI hoặc thao tác widget;
- không tự nhân bản công thức điểm/trọng số risk;
- không sửa dataset gốc ngoài hợp đồng;
- hỗ trợ dependency injection hoặc truyền dependency ở chỗ cần test.

Nên dùng result objects/dataclasses hoặc các kiểu dữ liệu hiện có. Không over-engineer, không đưa framework mới vào chỉ để tạo service.

## 3. Luồng dự đoán điểm

Flow kỳ vọng:

1. Nhận dữ liệu feature tại thời điểm dự đoán và course profile.
2. Validate đầu vào và trạng thái model.
3. Gọi ML prediction API để lấy raw final exam prediction.
4. Gọi scoring engine với course profile phù hợp.
5. Trả kết quả gồm raw predicted final exam score, predicted course score, model metadata/version, formula profile, warnings và validation status.

Nếu model chưa được train/nạp, trả trạng thái rõ ràng; không tạo score giả. Nếu TX2 đang thiếu ở môn có TX2, báo thiếu input và không chuyển công thức sang môn không có TX2.

## 4. Luồng dự đoán nguy cơ

1. Nhận các input đã chuẩn hóa hoặc dataset record.
2. Validate phần trăm và thang khảo sát.
3. Gọi đúng risk engine.
4. Trả weighted score, risk category, component scores, reasons, absence status và override flag.
5. Bảo đảm override `HOC_LAI` khi absence percentage >30 vẫn được giữ nguyên.

Không gọi nguy cơ tổng hợp là xác suất. Không để output khảo sát tích cực che mất override.

## 5. Luồng import và EDA

- Import service gọi parser/validator, không sửa file nguồn.
- Chỉ cho phép EDA chạy trên dataset đã được xác nhận hợp lệ theo mức độ yêu cầu của EDA.
- Trả các warning/errors rõ ràng cho UI.
- Nếu target actual final score vắng, EDA mô tả vẫn chạy được; không tạo model score thay thế.

## 6. Error handling và logging

- Dùng exception domain rõ nghĩa hoặc result type có discriminant rõ ràng.
- Không dùng `except Exception` rồi chỉ trả danh sách rỗng hoặc score 0.
- Nếu cần bắt lỗi ở service boundary, phải log an toàn và chuyển thành lỗi có ý nghĩa cho UI/CLI.
- Không ghi MSV, họ tên hoặc survey response vào log trừ khi thật sự cần và được phép.
- Không giữ DataFrame lớn trong global state nếu không cần.

## 7. Test bắt buộc

Integration tests dùng dependency thật cho domain logic và fake/mock chỉ cho boundary ngoài cần cô lập, không mock toàn bộ logic đang cần kiểm chứng.

Test các luồng:

- import hợp lệ và import lỗi;
- EDA có/không có target;
- prediction khi model sẵn sàng;
- prediction khi model chưa sẵn sàng;
- course profile A và B;
- TX2 missing ở profile B;
- risk score và attendance override;
- lỗi validation được truyền đến result đúng nghĩa;
- không có fake fallback khi dependency hỏng;
- report export;
- UI không cần biết chi tiết về estimator/dataframe nội bộ.

## Giới hạn

Không làm các màn hình PyQt6 ở giai đoạn này. Có thể thêm CLI smoke entry point nếu cần để kiểm thử service end-to-end. Không thay đổi thuật toán chỉ để làm test dễ pass.

## Điều kiện nghiệm thu

Service layer điều phối được các module thật, contract và error handling rõ ràng, integration tests có ý nghĩa và chạy được, không lặp logic nghiệp vụ trong service/UI, không có comment/docstring trong code first-party tạo/sửa. **Không chuyển sang Phần 6.**
