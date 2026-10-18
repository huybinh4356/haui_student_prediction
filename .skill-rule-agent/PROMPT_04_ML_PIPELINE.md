# PROMPT 04 — MACHINE LEARNING PIPELINE

Dán prompt này sau khi Phần 1–3 đã được nghiệm thu.

---

Bạn đang thực hiện **Giai đoạn 4: Machine Learning pipeline dự đoán điểm thi cuối kỳ**.

Đọc `00_GLOBAL_INSTRUCTIONS.md` và năm file luật bắt buộc. Xem lại schema, data validation, course profile, EDA API và tests hiện tại trước khi thiết kế ML contract. Code first-party không được có comment/docstring.

## Mục tiêu

Xây dựng pipeline ML có thể huấn luyện, đánh giá, lưu và nạp lại model một cách tái lập. Target là điểm thi cuối kỳ thực tế; tuyệt đối không tạo metrics giả nếu thiếu dữ liệu có nhãn.

## 1. Feature/target contract

Target chuẩn là `actual_final_exam_score` hoặc canonical target đã được project chốt.

- `y` chỉ chứa actual final exam score hợp lệ.
- `X` chỉ chứa thông tin được phép dùng ở timestamp/snapshot dự đoán.
- Loại `student_id`, `full_name` khỏi `X`.
- Loại target và các biến được tạo trực tiếp từ target như final course score thực tế khỏi `X`.
- Không sử dụng thông tin có sau kỳ thi, dữ liệu khảo sát sau kỳ thi hoặc feature được cập nhật sau snapshot.
- Không tự coi cột cuối kỳ chưa có là 0.
- Feature set phải được lưu/version hóa để audit.

Nếu có `snapshot_date` hoặc timestamp tương đương, tạo test đảm bảo feature thời điểm dự đoán không dùng thông tin tương lai. Nếu source không có mốc thời gian, ghi rõ giới hạn và thiết kế validation contract phù hợp; không giả vờ đã chứng minh temporal leakage-free.

## 2. Pipeline tiền xử lý

Dùng `sklearn.pipeline.Pipeline` và/hoặc `ColumnTransformer` để preprocessing được fit bên trong từng training fold.

Hỗ trợ phù hợp:

- numeric features: imputation có kiểm soát và scaling cho các model cần scaling;
- categorical features: imputation/encoding phù hợp;
- giá trị missing: giữ missing semantics; không thay bằng zero chỉ để tránh exception;
- feature schema validation;
- unknown category handling;
- kiểm soát thứ tự cột khi predict.

Không fit imputer, scaler, encoder hoặc selector trên toàn bộ dataset trước khi chia train/test.

## 3. Candidate models

Triển khai và so sánh tối thiểu:

- `DummyRegressor` baseline;
- `LinearRegression`;
- `Ridge`;
- `Lasso`.

Có thể bổ sung `RandomForestRegressor` và/hoặc `HistGradientBoostingRegressor` nếu phù hợp. Không thêm XGBoost hoặc dependency nặng nếu chưa cần.

Chọn model dựa trên validation metric và tiêu chí đã nêu; không mặc định model phức tạp nhất là tốt nhất. Giữ một test set độc lập nếu quy mô dữ liệu cho phép.

## 4. Split và leakage control

- Đánh giá đặc điểm dataset trước khi chọn split.
- Nếu nhiều dòng cùng sinh viên, cân nhắc group split theo student.
- Nếu cần đánh giá generalization sang học kỳ sau, ưu tiên temporal split với date/semester hợp lệ.
- Không để cùng bản ghi trùng hoặc snapshot gần như trùng rơi vào cả train và test.
- Trong CV, mọi preprocessing phải nằm trong estimator pipeline.
- Lưu split strategy, seed và dataset fingerprint.
- Nếu dataset quá nhỏ/thiếu biến để dùng một chiến lược đáng tin cậy, báo rõ, không cố sinh ra metric có vẻ đẹp.

## 5. Metrics và model registry

Báo tối thiểu MAE, RMSE và R² nếu phù hợp. Ghi rõ số sample, chiến lược split và baseline comparison. Không hard-code số liệu.

Tạo model registry hoặc metadata manifest ghi:

- model identifier/version;
- model family và configuration;
- feature schema/version;
- target name;
- dataset fingerprint/version;
- training timestamp;
- random seed;
- split/evaluation strategy;
- metrics thực tế;
- artifact path và hash nếu dùng.

Không load artifact không đáng tin cậy một cách tùy tiện. Xử lý model artifact thiếu, hỏng hoặc không tương thích bằng lỗi cụ thể; không fallback sang kết quả giả.

## 6. Behavior khi chưa có dữ liệu HaUI thực

Nếu không có dataset với actual final exam score:

- vẫn hoàn thiện code pipeline, validation và tests bằng fixture tổng hợp nhỏ;
- không ghi synthetic metrics vào nơi dành cho kết quả chính thức;
- không đặt trạng thái model là “trained/ready” nếu chưa train thành công;
- API prediction phải trả trạng thái model chưa sẵn sàng theo hợp đồng rõ ràng, không dự đoán giả;
- ghi chính xác dữ liệu nào còn thiếu để train thật.

## 7. Prediction contract

Tạo API predict nhận feature record/batch đã validate và trả structured prediction gồm raw score, model version, trạng thái/validation và warnings. Không clip raw output ở tầng model. Nếu application cần display score trong range hợp lệ, tạo trường display riêng với policy rõ ràng.

Dự đoán điểm cuối kỳ chỉ là output của regression. Việc tính điểm tổng kết dự kiến phải gọi scoring service/module đã được xây ở Phần 2, không lặp công thức ở ML pipeline.

## 8. Test bắt buộc

- target separation: target không nằm trong X;
- ID/name không nằm trong X;
- post-snapshot feature bị từ chối/loại theo contract;
- preprocessor fit trong CV pipeline;
- group/temporal split behavior nếu được hỗ trợ;
- baseline model có thể train với fixture hợp lệ;
- metrics được tính từ output thật, không hard-code;
- train không có target hợp lệ phải ném domain error/return failure có ý nghĩa;
- save/load artifact và metadata round-trip;
- schema mismatch và missing model artifact;
- predict single/batch;
- raw prediction không bị clip ở model layer;
- reproducibility trong giới hạn hợp lý với seed cố định.

## Giới hạn

Không tạo UI trong giai đoạn này. Không kết nối dữ liệu sinh viên thực tế khi chưa được cung cấp. Không tạo classifier nguy cơ học vụ nếu chưa có nhãn lịch sử hợp lệ.

## Điều kiện nghiệm thu

ML pipeline có thể train/evaluate bằng test fixture; các lỗi leakage trọng yếu có regression tests; model registry lưu metadata thật; khi chưa có dữ liệu thực thì báo blocked đúng cách; lệnh test/lint chạy và được báo trung thực. Không có comment/docstring trong code first-party tạo/sửa. **Không chuyển sang Phần 5.**
