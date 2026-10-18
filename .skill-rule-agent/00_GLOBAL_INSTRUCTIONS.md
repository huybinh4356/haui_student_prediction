# CHỈ THỊ CHUNG BẮT BUỘC CHO MỌI GIAI ĐOẠN

Các chỉ thị trong file này áp dụng cho cả 6 prompt triển khai. Đọc file này và các file luật ở repository trước khi sửa code.

## A. Vai trò và cách làm việc

Bạn là kỹ sư phần mềm cấp cao chịu trách nhiệm xây dựng `HaUI Student Performance & Academic Risk Prediction System`. Đây là dự án mới, kiến trúc theo module, ưu tiên khả năng kiểm thử, dễ debug, dễ bảo trì và tách biệt hoàn toàn giao diện khỏi logic nghiệp vụ.

Mỗi lần nhận một prompt giai đoạn, chỉ thực hiện phạm vi của giai đoạn đó. Không nhảy sang giai đoạn kế tiếp, không viết sẵn các module rỗng chỉ để làm đẹp cấu trúc. Được tạo interface nhỏ cần thiết cho tích hợp về sau, nhưng phải ghi rõ giới hạn và không giả vờ tính năng đã hoàn thành.

Không dừng lại ở việc lập kế hoạch hoặc mô tả giải pháp. Hãy kiểm tra workspace thật, sửa/tạo code thật, viết test, chạy lệnh xác minh trong môi trường hiện có và báo cáo kết quả thật.

## B. File luật bắt buộc

Trước khi thay đổi file nào, đọc đầy đủ các file có trong repository:

- `AGENTS.md`
- `SKILL.md`
- `RULES.md`
- `CODE_RULES.md`
- `LEGACY_REPAIR_RULES.md`
- `HAUI_Student_Performance_Prediction_Project_Plan.md` nếu file này có trong repository.

Nếu file không tồn tại, báo rõ file nào thiếu. Không tự tuyên bố đã đọc file không truy cập được. Kiểm tra repository status và cấu trúc hiện tại trước khi chỉnh sửa. Không ghi đè file người dùng khi chưa kiểm tra nội dung.

## C. Luật code không comment/docstring

Trong code first-party do dự án sở hữu, không tạo hoặc giữ lại comment, docstring hay code bị comment-out trong phạm vi file đang được tạo hoặc sửa. Không thêm `# comment`, block comment, module/class/function docstring hoặc TODO comment. Đặt giải thích trong Markdown, dùng tên biến/hàm rõ nghĩa, type hint và test.

Không xóa thông báo bản quyền hoặc comment trong code bên thứ ba chỉ để thỏa luật này. Không sửa generated/vendor code tùy tiện. Sau mỗi giai đoạn, chạy hoặc tạo kiểm tra tự động để phát hiện comment/docstring trong code first-party. Nếu có ngoại lệ bắt buộc theo cú pháp hoặc công cụ, nêu rõ bằng chứng và không âm thầm bỏ qua.

## D. Quy tắc chất lượng

- Không giả lập thành công, không hard-code metric, không tạo dự đoán giả để che thiếu model/dữ liệu.
- Không nuốt exception bằng `except: pass` hoặc trả kết quả mặc định làm lỗi trông như thành công.
- Không tự động biến dữ liệu thiếu thành 0.
- Không thực hiện thao tác phá hủy dữ liệu trong quá trình import.
- Không dùng `student_id` hoặc `full_name` làm feature ML.
- UI chỉ gọi service/use case; UI không tự tính công thức điểm, trọng số nguy cơ hoặc huấn luyện ML.
- Mọi trọng số, ngưỡng, alias cột và cấu hình môn học cần được tập trung, kiểm tra tính hợp lệ và có test.
- Giữ nguyên dữ liệu gốc; chỉ ghi ra processed/output ở vị trí riêng.
- Chỉ dùng dữ liệu tổng hợp để test khi chưa có dữ liệu thật. Ghi rõ dữ liệu tổng hợp không dùng để khẳng định metric thực tế.
- Không tải dữ liệu sinh viên thật lên dịch vụ ngoài hoặc commit dữ liệu cá nhân vào Git.
- Không thêm thư viện nếu chưa có lý do và chưa cập nhật file dependency.
- Kiểm tra tương thích các module hiện có trước khi đổi interface.

## E. Đặc tả nghiệp vụ đã chốt

### Dữ liệu đầu vào

- `student_id`: mã sinh viên, chỉ định danh/liên kết, không phải feature.
- `full_name`: họ tên, chỉ hiển thị/quản lý, không phải feature.
- `date_of_birth` hoặc `birth_year`: giữ theo nguồn thực tế; không nhân đôi thành hai feature độc lập nếu một trường có thể suy ra từ trường kia.
- `hometown`: quê quán.
- `major`: ngành học.
- `absence_percentage`: phần trăm nghỉ học do nguồn cung cấp trực tiếp.
- `lms_learning_percentage`: phần trăm thời gian/hoạt động học trên hệ thống do nguồn cung cấp trực tiếp.
- `tx1_score`: điểm thường xuyên 1.
- `tx2_score`: điểm thường xuyên 2 khi học phần có thành phần này.
- `midterm_score`: điểm giữa kỳ.
- `interaction_score`: điểm tương tác, được xếp trong nhóm hoạt động học tập dù nguồn có thể lấy từ khảo sát.
- `part_time_work`: tình trạng đi làm thêm.
- `stress_level`: mức stress từ khảo sát.
- `learning_motivation`: động lực học tập từ khảo sát.
- `actual_final_exam_score`: điểm thi cuối kỳ thực tế, chỉ là target huấn luyện/evaluation và không được đưa vào feature tại thời điểm dự đoán.

Các tên trên là tên chuẩn nội bộ. Tên cột thực tế cần được ánh xạ bằng cấu hình sau khi xem file dữ liệu mẫu. Không khẳng định khảo sát luôn dùng thang 1–5 khi chưa xác minh; nếu chưa có định nghĩa thang đo, cấu hình rõ giả định và báo cần xác minh.

### Công thức điểm

Học phần không có TX2:

`predicted_course_score = 0.20 * TX1 + 0.20 * MIDTERM + 0.60 * predicted_final_exam_score`

Học phần có TX2:

`predicted_course_score = 0.15 * TX1 + 0.15 * TX2 + 0.20 * MIDTERM + 0.50 * predicted_final_exam_score`

Phân biệt học phần thật sự không có TX2 với học phần có TX2 nhưng điểm TX2 bị thiếu/chưa công bố. Không tự động đổi trường hợp thiếu điểm thành profile không có TX2. Đặt profile và trọng số trong cấu hình; kiểm tra tổng trọng số bằng 1. Các hệ số là yêu cầu dự án cung cấp, cần xác minh với cấu hình đánh giá học phần trước khi tuyên bố là chính thức cho mọi môn HaUI.

### Điểm nguy cơ có trọng số

| Thành phần | Trọng số |
|---|---:|
| Nguy cơ từ tỷ lệ nghỉ học | 75% |
| Nguy cơ từ điểm số học tập | 8% |
| Nguy cơ từ tỷ lệ học trên hệ thống | 8% |
| Nguy cơ từ điểm tương tác | 4% |
| Yếu tố đi làm thêm | 2% |
| Stress | 1.5% |
| Động lực học tập | 1.5% |
| Tổng | 100% |

Mỗi thành phần phải được chuẩn hóa về thang nguy cơ 0–100 trước khi nhân trọng số. Điểm cao hơn phải nhất quán nghĩa là nguy cơ cao hơn. Tổng trọng số phải được kiểm tra bằng test.

Ngưỡng nghỉ học theo yêu cầu dự án:

- `absence_percentage > 30`: bật override `HOC_LAI` theo quy tắc cấu hình dự án.
- `20 <= absence_percentage <= 30`: nguy cơ cao.
- `10 <= absence_percentage < 20`: nguy cơ trung bình.
- `5 <= absence_percentage < 10`: nguy cơ thấp.
- `0 <= absence_percentage < 5`: nguy cơ rất thấp.

Điều kiện là **trên 30%**, không phải lớn hơn hoặc bằng 30%. Trạng thái override độc lập với weighted score; yếu tố tích cực khác không được hủy override. Không gọi đây là quy định chính thức áp dụng mọi học phần nếu chưa có tài liệu HaUI tương ứng.

Tỷ lệ LMS được lấy trực tiếp, không tính lại. Giả sử phần trăm càng cao thì mức tham gia càng cao, các ngưỡng khởi tạo là:

- dưới 5%: nguy cơ rất cao;
- 5% đến dưới 10%: nguy cơ cao;
- 10% đến dưới 20%: nguy cơ trung bình;
- 20% đến dưới 30%: nguy cơ thấp;
- từ 30% trở lên: nguy cơ rất thấp.

Đây là ngưỡng prototype có thể cấu hình, chưa phải ngưỡng đã kiểm chứng bằng dữ liệu lịch sử.

Đi làm thêm không mặc định là yếu tố bất lợi. Khi chưa có bằng chứng, dùng ánh xạ trung tính có thể cấu hình, không tự gán mức phạt 50. Stress, động lực và tương tác phải theo đúng chiều thang đo đã xác minh.

Điểm nguy cơ tổng hợp là điểm quy tắc, không phải xác suất thống kê. Chỉ xây dựng ML classifier dự đoán nguy cơ nếu có target lịch sử hợp lệ và quy trình đánh giá phù hợp.

### EDA và ML

- EDA có thể xem xét mọi trường phù hợp, ngoại trừ không coi `student_id` và `full_name` là biến số phân tích hoặc feature dự đoán.
- Model hồi quy dự đoán `actual_final_exam_score` từ các trường đã có trước thời điểm dự đoán.
- Đánh giá ít nhất bằng MAE, RMSE và R² khi thích hợp; so sánh baseline với model khác.
- Có pipeline preprocessing để fit imputer/encoder/scaler chỉ trên train fold.
- Chọn split theo sinh viên/học kỳ/thời gian khi cần; tránh rò rỉ cùng người hoặc cùng nhóm qua train/test.
- Không báo metric ML thực tế nếu chưa có dữ liệu lịch sử có target.

## F. Quy trình hoàn thành mỗi giai đoạn

1. Đọc file luật và kiểm tra code hiện tại.
2. Nêu phạm vi thay đổi ngắn gọn, rồi thực hiện trong cùng phiên làm việc.
3. Tạo/sửa code thật và test tương ứng.
4. Chạy các lệnh test/lint/import phù hợp với môi trường.
5. Sửa lỗi do thay đổi của mình gây ra và chạy lại test liên quan.
6. Kiểm tra code comment/docstring.
7. Kiểm tra diff, file ngoài phạm vi và dữ liệu nhạy cảm.
8. Cập nhật documentation liên quan.
9. Báo cáo file đã thay đổi, lệnh đã chạy và kết quả thật, test chưa chạy được cùng nguyên nhân, vấn đề còn lại.

Không chuyển sang giai đoạn kế tiếp cho đến khi tiêu chí nghiệm thu của giai đoạn hiện tại được đáp ứng hoặc có blocker cụ thể được ghi lại. Không tuyên bố “hoàn thành” chỉ vì đã tạo file.
