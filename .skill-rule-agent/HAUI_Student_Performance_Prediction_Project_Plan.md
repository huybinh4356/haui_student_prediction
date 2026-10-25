**HAUI Student Performance & Academic Risk Prediction System**  
**Tài liệu:** Kế hoạch tổng thể dự án mới  
   
 **Phiên bản đặc tả:** 1.0  
   
 **Trạng thái:** Bản thiết kế ban đầu — cần đối chiếu schema và dữ liệu mẫu HaUI trước khi chốt triển khai  
   
 **Định hướng:** Xây dựng repository mới, tận dụng các ý tưởng/kiến trúc tốt của student_analysis, không sao chép nguyên trạng code cũ.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNBCkLfFDZwwIgHRiywEZJWQZeZ2ao9AAD+4lyruzq+ngAA8Nr1AOH0BedHjjlfAAAAAElFTkSuQmCC)  
**1. Tầm nhìn và phạm vi dự án**  
**1.1. Mục tiêu**  
Xây dựng ứng dụng phân tích và dự đoán kết quả học tập sinh viên HaUI từ dữ liệu điểm thành phần, tỷ lệ nghỉ học, tỷ lệ thời gian học trên hệ thống, điểm tương tác và khảo sát bối cảnh. Hệ thống có bốn nhóm chức năng chính:  
1. **Quản lý dữ liệu và kiểm tra chất lượng:** nhập file, chuẩn hóa dữ liệu, kiểm tra cột thiếu, sai kiểu, giá trị ngoài phạm vi và dữ liệu trùng.  
2. **EDA (Exploratory Data Analysis):** phân tích phân bố dữ liệu và mối liên hệ giữa các biến với kết quả học tập.  
3. **Dự đoán điểm thi cuối kỳ:** huấn luyện mô hình hồi quy trên dữ liệu lịch sử; dùng điểm dự đoán để tính điểm tổng kết học phần dự kiến theo công thức được cấu hình cho học phần.  
4. **Dự đoán nguy cơ học vụ:** ở phiên bản đầu tiên, dùng bộ quy tắc và điểm nguy cơ có trọng số; giữ kiến trúc mở để bổ sung mô hình phân loại ML khi có nhãn lịch sử phù hợp và được kiểm chứng.  
**1.2. Nguyên tắc thiết kế**  
- Tên hiển thị dùng **“Dự đoán kết quả học tập”** và  **“Dự đoán nguy cơ học vụ”**, không dùng “chẩn đoán”.  
- Quy tắc nghỉ học trên 30% là một **điều kiện ưu tiên riêng** theo đặc tả của dự án; điểm khảo sát hoặc điểm số tốt không được xóa trạng thái này.  
- Công thức tính điểm học phần và công thức điểm nguy cơ là hai thành phần độc lập.  
- Không tự bịa dữ liệu HaUI, không báo chỉ số mô hình từ dữ liệu cũ như chỉ số của mô hình mới.  
- Không đưa MSV và họ tên vào feature huấn luyện.  
- Giữ dữ liệu gốc; không chỉnh sửa trực tiếp file nguồn. Mọi chuẩn hóa tạo ra dữ liệu dẫn xuất có thể truy vết.  
- Không để giao diện tự tính công thức hoặc tự nạp model. UI chỉ gọi service.  
- Không dùng dữ liệu thật, thông tin cá nhân hoặc model artifact riêng tư trong repository công khai.  
- Nếu dữ liệu chưa đủ để huấn luyện hoặc trường bắt buộc còn thiếu, hiển thị rõ “chưa đủ dữ liệu” thay vì đưa ra con số giả.  
**1.3. Những gì không nằm trong phạm vi ban đầu**  
- Không tự động ra quyết định học vụ chính thức thay cho trường/giảng viên.  
- Không coi khảo sát stress là chẩn đoán sức khỏe tâm thần.  
- Không tự động thu thập dữ liệu từ tài khoản/hệ thống trường khi chưa được cấp phép; chỉ xử lý dữ liệu được cung cấp hợp lệ.  
- Không xây dựng hệ thống đa người dùng/phân quyền phức tạp nếu chưa có yêu cầu.  
- Không bắt buộc triển khai PostgreSQL/Docker ngay ở MVP; có thể thiết kế lớp lưu trữ để bổ sung sau.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSeYxKS/kJkED6bwYAVvImwJtszMVu0BAPAXx1rd1fn1BACA164HHDwF+DpPyKwAAAAASUVORK5CYII=)  
**2. Yêu cầu dữ liệu HaUI**  
**2.1. Một đơn vị dữ liệu cần xác định rõ**  
Đơn vị phù hợp cho bài toán dự đoán điểm là **một sinh viên trong một học phần ở một học kỳ, tại một thời điểm quan sát**. Không nên giả định một sinh viên chỉ có một dòng duy nhất, vì cùng một sinh viên học nhiều học phần và mỗi học phần có thể có công thức điểm khác nhau.  
Nếu nguồn hiện tại chỉ có một dòng/sinh viên, cần xác minh các điểm đó là điểm tổng hợp của môn nào, học kỳ nào và có thể liên kết với kết quả cuối kỳ thực tế hay không trước khi dùng cho ML.  
**2.2. Từ điển dữ liệu đề xuất**  
| | | | | |  
|-|-|-|-|-|  
| **Nhóm** | **Tên trường chuẩn đề xuất** | **Kiểu dữ liệu** | **Bắt buộc?** | **Mục đích / quy tắc** |   
| Định danh | student_id | Chuỗi | Có | MSV; dùng liên kết và loại trùng, không dùng làm feature |   
| Định danh | full_name | Chuỗi | Theo nguồn | Hiển thị nội bộ; không dùng làm feature; cần bảo vệ dữ liệu |   
| Nhân khẩu học | date_of_birth | Ngày | Một trong hai | Nếu có ngày sinh thì có thể tạo nhóm tuổi khi cần; không dùng đồng thời ngày sinh và năm sinh như hai feature độc lập |   
| Nhân khẩu học | birth_year | Số nguyên | Một trong hai | Dùng thay cho ngày sinh nếu nguồn chỉ cung cấp năm sinh |   
| Nhân khẩu học | hometown | Chuỗi/mã địa phương | Có nếu nguồn có | EDA tổng hợp; chuẩn hóa tên địa phương; chỉ đưa vào ML nếu có lý do và đã đánh giá bias |   
| Học tập | major | Chuỗi/mã ngành | Có nếu nguồn có | EDA; có thể là feature sau khi đánh giá số lượng mẫu và tính phù hợp |   
| Ngữ cảnh môn học | course_id | Chuỗi | Nên có | Xác định học phần và công thức đánh giá |   
| Ngữ cảnh môn học | course_name | Chuỗi | Nên có | Hiển thị và EDA; mã môn ổn định được ưu tiên cho liên kết |   
| Ngữ cảnh môn học | semester | Chuỗi | Nên có | Phân tách học kỳ, hỗ trợ đánh giá theo thời gian |   
| Ngữ cảnh thời gian | snapshot_date | Ngày/giờ | Rất nên có | Cho biết dữ liệu nào đã tồn tại ở thời điểm dự đoán; hỗ trợ chống leakage |   
| Điểm thành phần | tx1_score | Số thực | Theo đề cương | Điểm thường xuyên 1 |   
| Điểm thành phần | tx2_score | Số thực/null | Theo đề cương | TX2; null phải phân biệt với học phần không có TX2 |   
| Cấu hình môn | has_tx2 | Boolean | Có trong cấu hình | True nếu học phần có TX2; không suy ra chỉ bằng cách thấy ô TX2 trống |   
| Điểm thành phần | midterm_score | Số thực | Theo cấu trúc môn | Điểm giữa kỳ |   
| Nhãn lịch sử | actual_final_exam_score | Số thực | Có trong dữ liệu huấn luyện | Điểm cuối kỳ thực tế; dùng làm target huấn luyện/đánh giá, không xuất hiện trong input khi dự đoán trước kỳ thi |   
| Chuyên cần | absence_percentage | Số thực 0–100 | Có | Tỷ lệ nghỉ học đã được nguồn tính sẵn; không tính lại từ số buổi |   
| LMS | lms_learning_percentage | Số thực 0–100 | Có nếu nguồn có | Tỷ lệ thời gian/hoạt động học trên hệ thống đã được nguồn tính sẵn; không tính lại từ giờ học |   
| Hoạt động học tập | interaction_score | Số thực hoặc phân loại | Có nếu khảo sát có | Điểm tương tác được xếp vào nhóm hoạt động học tập trong mô hình phân tích dù nguồn thu thập có thể là khảo sát |   
| Bối cảnh | works_part_time | Boolean/phân loại | Có nếu khảo sát có | Có đi làm thêm hay không; không mặc định “có làm thêm” luôn làm tăng nguy cơ nếu chưa có bằng chứng |   
| Bối cảnh | stress_score | Số thực/thứ bậc | Có nếu khảo sát có | Cần biết thang đo và chiều ý nghĩa của điểm |   
| Bối cảnh | motivation_score | Số thực/thứ bậc | Có nếu khảo sát có | Cần biết thang đo và chiều ý nghĩa của điểm |   
| Chất lượng nguồn | survey_timestamp | Ngày/giờ | Rất nên có | Đảm bảo câu trả lời khảo sát tồn tại trước thời điểm dự đoán |   
| Phiên bản | source_file / dataset_version | Chuỗi | Hệ thống tạo | Truy xuất nguồn và phiên bản dữ liệu |   
   
**Chú ý:** Tên trường ở trên là tên chuẩn nội bộ đề xuất, không có nghĩa file Excel HaUI hiện đã sử dụng đúng các cột này. Cần có bước ánh xạ tên cột thực tế sang schema chuẩn.  
**2.3. Quy tắc xử lý dữ liệu**  
- absence_percentage và lms_learning_percentage đã là phần trăm: chỉ kiểm tra hợp lệ trong phạm vi 0–100, không tính lại.  
- Giá trị ngoài phạm vi, chuỗi không chuyển đổi được hoặc điểm ngoài thang điểm hợp lệ phải được báo lỗi/đưa vào danh sách cần rà soát, không âm thầm sửa.  
- Phân biệt TX2 không áp dụng với TX2 áp dụng nhưng chưa có điểm.  
- Không thay dữ liệu thiếu bằng 0 nếu 0 là một điểm hợp lệ.  
- Giữ raw data read-only; xử lý và chuẩn hóa ở staging/processed.  
- Không xóa bản ghi trùng trước khi xác minh khóa trùng phù hợp, ví dụ student_id + course_id + semester + snapshot_date.  
- Với nhiều lần chụp theo thời gian, mỗi bản ghi phải ghi rõ thời điểm; không trộn dữ liệu sau kỳ thi vào bản chụp trước kỳ thi.  
- Dữ liệu demo được commit lên Git phải là dữ liệu giả lập/đã ẩn danh, không phải dữ liệu sinh viên thật.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSfYxKK/kYXEkyk8WcGbCFuCLTOzVXsAAPzFuVZ3dXw9AQDgtesB/v8F8JQadPwAAAAASUVORK5CYII=)  
**3. Công thức tính điểm học phần**  
Công thức phải đặt trong cấu hình theo học phần, không rải trực tiếp trong UI hay model. Hai trường hợp khởi đầu theo yêu cầu hiện tại:  
**3.1. Học phần không có TX2**  
[  
   
 \widehat{S}*{HP}=0.2S*{TX1}+0.2S_{GK}+0.6\widehat{S}_{CK}  
   
 ]  
**3.2. Học phần có TX1, TX2 và giữa kỳ**  
[  
   
 \widehat{S}*{HP}=0.15S*{TX1}+0.15S_{TX2}+0.2S_{GK}+0.5\widehat{S}_{CK}  
   
 ]  
Trong đó:  
- S_TX1: TX1 thực tế đã có.  
- S_TX2: TX2 thực tế đã có, chỉ áp dụng khi cấu hình môn xác định có TX2.  
- S_GK: điểm giữa kỳ thực tế đã có.  
- S_CK_hat: điểm cuối kỳ do mô hình hồi quy dự đoán.  
- S_HP_hat: điểm tổng kết học phần dự kiến.  
**3.3. Các trường hợp phải xử lý rõ**  
1. **Môn không có TX2:** dùng công thức 0.2 / 0.2 / 0.6.  
2. **Môn có TX2 và điểm TX2 đã có:** dùng công thức 0.15 / 0.15 / 0.2 / 0.5.  
3. **Môn có TX2 nhưng chưa nhập điểm:** không tự động chuyển sang công thức không có TX2; báo thiếu dữ liệu hoặc chờ cập nhật theo chính sách đã chốt.  
4. **Thiếu TX1 hoặc giữa kỳ:** không đưa ra điểm tổng kết dự kiến như thể đủ dữ liệu; hiển thị trường bị thiếu.  
5. **Điểm dự đoán ngoài thang hợp lệ:** lưu raw prediction phục vụ đánh giá, đồng thời giới hạn giá trị chỉ ở tầng hiển thị/tính điểm theo quy định được chốt; không che giấu raw prediction.  
6. **Trọng số thay đổi theo học phần:** cấu hình riêng bằng course_scoring_config, có version và kiểm thử.  
Hai công thức trên được nhập theo yêu cầu của chủ dự án. Trước khi mô tả là áp dụng chung cho mọi học phần, cần xác nhận với đề cương/cấu hình đánh giá chính thức của các môn thực tế.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSNBCkLfFDZwwIgHRiywEZJWQZeZ2ao9AAD+4lyruzq+ngAA8Nr1AOH0BedHjjlfAAAAAElFTkSuQmCC)  
**4. Thiết kế ML dự đoán điểm cuối kỳ**  
**4.1. Mục tiêu**  
- **Target:**actual_final_exam_score.  
- **Input:** chỉ dùng biến có thể biết tại snapshot_date trước khi diễn ra thi cuối kỳ.  
- Không đưa student_id, full_name, actual_final_exam_score hoặc điểm tổng kết đã tính từ điểm cuối kỳ thực tế vào input.  
- Các feature ứng viên: điểm TX1/TX2 nếu đã công bố, điểm giữa kỳ, tỷ lệ nghỉ học, tỷ lệ LMS, điểm tương tác, đi làm thêm, stress, động lực; ngành học/mã học phần chỉ dùng nếu đủ mẫu và có lý do phù hợp.  
- Ngày sinh/năm sinh và quê quán: có trong dữ liệu/EDA theo yêu cầu, nhưng mặc định không đưa vào model trước khi đánh giá tính cần thiết, độ lệch và quyền riêng tư.  
**4.2. Quy trình huấn luyện**  
1. Validate schema và target.  
2. Tạo tập dữ liệu theo đúng thời điểm dự đoán.  
3. Chia train/validation/test, ưu tiên split theo thời gian/học kỳ khi có đủ lịch sử; tránh để các bản ghi của cùng sinh viên rò rỉ giữa các tập nếu điều đó làm đánh giá quá lạc quan.  
4. Dùng pipeline duy nhất cho imputation/encoding/scaling và model; mọi bước fit chỉ diễn ra trên phần train trong từng fold.  
5. Baseline tối thiểu: dự đoán trung bình/median (DummyRegressor) và LinearRegression.  
6. So sánh ứng viên như Ridge, Lasso, RandomForestRegressor, HistGradientBoostingRegressor; chỉ bổ sung XGBoost khi môi trường và lượng dữ liệu hợp lý.  
7. Chọn model bằng cross-validation trên train, không dùng test để chọn model.  
8. Đánh giá cuối cùng trên test độc lập bằng MAE, RMSE, R²; kiểm tra sai số theo ngành/học phần/nhóm dữ liệu nếu đủ mẫu.  
9. Lưu model bundle, danh sách feature, cấu hình tiền xử lý, dataset fingerprint, seed, version thư viện, metric và thời điểm huấn luyện.  
**4.3. Điều kiện để khẳng định mô hình có giá trị**  
Cần dữ liệu lịch sử bao gồm cả input trước kỳ thi và điểm thi cuối kỳ thực tế tương ứng. Nếu chưa có target thực tế hoặc không có mốc thời gian, vẫn xây dựng được schema và EDA, nhưng không thể kết luận model dự đoán có độ chính xác thực tế đáng tin cậy.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSPBCj7fFRYQwYwEZiywEZJWQZeZ2ao9AAD+4lyruzq+ngAA8Nr1AMTJBeJDClAyAAAAAElFTkSuQmCC)  
**5. Thiết kế dự đoán nguy cơ học vụ**  
**5.1. Tách điều kiện cứng và điểm nguy cơ tổng hợp**  
- **Điều kiện ưu tiên:**absence_percentage > 30 kích hoạt trạng thái RETAKE_RULE_TRIGGERED / “Vượt ngưỡng nghỉ học — cần kiểm tra điều kiện học phần” theo yêu cầu của chủ dự án. Không để các yếu tố khác xóa trạng thái này.  
- Việc gắn nhãn “học lại” chính thức cần đối chiếu quy định áp dụng của học phần trước khi dùng trong hệ thống thực tế; về code nên giữ tên trạng thái cấu hình được.  
- Khi tỷ lệ nghỉ học không có hoặc không hợp lệ, không được kết luận rằng sinh viên không vượt ngưỡng; trả về INSUFFICIENT_DATA.  
- Ngoài điều kiện cứng, hệ thống tính điểm nguy cơ tổng hợp từ các thành phần đã chuẩn hóa về cùng hướng: **100 = tín hiệu rủi ro cao, 0 = tín hiệu rủi ro thấp**.  
**5.2. Trọng số khởi tạo đã thống nhất**  
| | | |  
|-|-|-|  
| **Thành phần** | **Trọng số** | **Cách hiểu ban đầu** |   
| Tỷ lệ nghỉ học | **75%** | Thành phần ưu tiên chính; dùng tỷ lệ nguồn đã cung cấp |   
| Điểm số học tập | 8% | Dùng điểm đã có và/hoặc điểm học phần dự kiến; cần định nghĩa chính xác cách đổi thành risk points |   
| Tỷ lệ thời gian học trên hệ thống | 8% | Dùng tỷ lệ nguồn đã cung cấp; tỷ lệ thấp hơn dự kiến tương ứng rủi ro cao hơn |   
| Điểm tương tác | 4% | Xếp vào nhóm hoạt động học tập dù được thu thập từ khảo sát; cần chuẩn hóa chiều thang đo |   
| Đi làm thêm | 2% | Biến bối cảnh; không mặc định có việc làm thêm luôn là bất lợi nếu dữ liệu chưa chứng minh |   
| Mức độ stress | 1.5% | Điểm stress cao dự kiến là rủi ro cao hơn nếu thang đo đúng chiều |   
| Động lực học tập | 1.5% | Động lực thấp dự kiến là rủi ro cao hơn nếu thang đo đúng chiều |   
| **Tổng** | **100%** | Bộ trọng số khởi tạo để thử nghiệm, chưa được hiệu chỉnh bằng dữ liệu |   
   
**5.3. Phân loại theo tỷ lệ nghỉ học**  
Các khoảng phải được triển khai không chồng lấn:  
| | |  
|-|-|  
| **absence_percentage** | **Trạng thái chuyên cần** |   
| > 30 | Điều kiện ưu tiên: vượt ngưỡng; hiển thị cảnh báo theo quy tắc dự án |   
| 20–30 (bao gồm 20 và 30) | Nguy cơ cao |   
| 10–<20 | Nguy cơ trung bình |   
| 5–<10 | Nguy cơ thấp |   
| 0–<5 | Nguy cơ rất thấp |   
| Ngoài 0–100 / thiếu | Dữ liệu không hợp lệ hoặc chưa đủ; không phân loại |   
   
**5.4. Phân mức từ tỷ lệ LMS**  
Giả định cột LMS là tỷ lệ tham gia/học tập, trong đó tỷ lệ cao hơn là tín hiệu thuận lợi. Dùng cùng ranh giới phần trăm theo chiều đảo ngược như khởi tạo ban đầu:  
| | |  
|-|-|  
| **lms_learning_percentage** | **Mức rủi ro LMS tham khảo** |   
| < 5 | Rất cao |   
| 5–<10 | Cao |   
| 10–<20 | Trung bình |   
| 20–<30 | Thấp |   
| 30–100 | Rất thấp |   
   
Đây là ngưỡng cấu hình khởi tạo, chưa được chứng minh trên dữ liệu HaUI. Phải xác nhận ý nghĩa cột từ nguồn: nếu “thời gian học” thực chất là một chỉ số có chiều nghĩa khác, không áp dụng bảng này một cách máy móc.  
**5.5. Cách kết hợp trọng số**  
Mỗi thành phần chuyển thành risk_points trong khoảng 0–100, với 100 luôn có nghĩa là rủi ro cao. Công thức tổng quát:  
[  
   
 R = 0.75A + 0.08G + 0.08L + 0.04I + 0.02W + 0.015S + 0.015M  
   
 ]  
Trong đó:  
- A: risk points từ tỷ lệ nghỉ học.  
- G: risk points từ điểm số.  
- L: risk points từ tỷ lệ LMS.  
- I: risk points từ điểm tương tác.  
- W: risk points từ đi làm thêm.  
- S: risk points từ stress.  
- M: risk points từ động lực học.  
**Không được lấy trực tiếp các phần trăm/điểm thô nhân với trọng số** nếu chúng chưa cùng thang và cùng chiều. Ví dụ, tương tác cao thường là tín hiệu tốt, do đó phải đổi chiều thành risk points thấp; stress cao có thể là rủi ro cao nếu thang khảo sát được mã hóa đúng chiều.  
Bản mapping khởi tạo để triển khai prototype có thể dùng như sau; tất cả mapping phải nằm trong cấu hình và được hiệu chỉnh bằng dữ liệu:  
| | | |  
|-|-|-|  
| **Thành phần** | **Giá trị nguồn** | **Risk points khởi tạo** |   
| Nghỉ học | <5% | 0 |   
| Nghỉ học | 5–<10% | 30 |   
| Nghỉ học | 10–<20% | 55 |   
| Nghỉ học | 20–30% | 80 |   
| Nghỉ học | >30% | 100 và kích hoạt trạng thái ưu tiên |   
| LMS | <5% | 100 |   
| LMS | 5–<10% | 75 |   
| LMS | 10–<20% | 50 |   
| LMS | 20–<30% | 25 |   
| LMS | 30–100% | 0 |   
| Điểm học tập | Nếu điểm học phần dự kiến đã được xác nhận là thang 0–10 | clip((10 - predicted_course_score) * 10, 0, 100) |   
| Tương tác | Thang đã chuẩn hóa; điểm cao có nghĩa tương tác tốt | Đảo chiều về risk points: điểm tương tác cao → risk points thấp |   
| Stress | Thang đã chuẩn hóa; điểm cao có nghĩa stress nhiều hơn | Chuẩn hóa cùng chiều: điểm stress cao → risk points cao |   
| Động lực | Thang đã chuẩn hóa; điểm cao có nghĩa động lực tốt hơn | Đảo chiều: động lực cao → risk points thấp |   
| Đi làm thêm | Có/không | Chưa gán chiều tốt/xấu trước khi được xác nhận; giữ trung tính hoặc hiệu chỉnh trên dữ liệu |   
   
Bảng điểm mapping nghỉ học/LMS là **đề xuất prototype**, không phải ngưỡng chính thức của HaUI. Mapping điểm học tập chỉ dùng nếu xác nhận thang điểm học phần là 0–10 và chốt dùng điểm tổng kết dự kiến làm tín hiệu nguy cơ.  
Cách đổi điểm cần được đặt trong module mapping riêng và có unit test. Riêng “đi làm thêm” không nên gán tự động là xấu; mức đóng góp phải được xác nhận với chủ dự án hoặc học từ dữ liệu có nhãn. Tương tự, G cần quy tắc rõ ràng dựa trên điểm học phần dự kiến/thực tế và ngưỡng kết quả được phê duyệt.  
**5.6. Nhãn điểm nguy cơ tổng hợp**  
Có thể dùng các ngưỡng ban đầu để triển khai UI, rồi hiệu chỉnh sau:  
| | |  
|-|-|  
| **Điểm nguy cơ tổng hợp ** **R** | **Nhãn dự kiến** |   
| 0–<20 | Rất thấp |   
| 20–<40 | Thấp |   
| 40–<60 | Trung bình |   
| 60–<80 | Cao |   
| 80–100 | Rất cao |   
   
Các ngưỡng tổng hợp này **là ngưỡng prototype đề xuất**, không phải tiêu chuẩn chính thức của HaUI. Chúng phải nằm trong file cấu hình để dễ hiệu chỉnh sau đánh giá. Cảnh báo chuyên cần trên 30% được hiển thị song song và không phụ thuộc vào nhãn tổng hợp.  
Nếu một thành phần phụ bị thiếu, hệ thống không tự gán 0. Có thể tính điểm trên các thành phần hiện có và chuẩn hóa lại tổng trọng số khả dụng, nhưng phải hiển thị tỷ lệ dữ liệu đầy đủ và đánh dấu kết quả có độ tin cậy hạn chế. Nếu thiếu tỷ lệ nghỉ học, trạng thái ưu tiên không thể kết luận.  
**5.7. Có phải mô hình ML phân loại nguy cơ không?**  
Ở MVP, điểm R là **hệ thống tính điểm nguy cơ dựa trên quy tắc/trọng số**, không nên trình bày nó như một mô hình ML đã học từ dữ liệu. Chỉ phát triển bộ phân loại ML khi dataset có nhãn lịch sử đáng tin cậy, ví dụ kết quả học phần thực tế/nhãn nguy cơ được định nghĩa hợp lệ. Nếu triển khai ML sau này, giữ hai kết quả riêng: rule_based_risk và ml_predicted_risk, sau đó đánh giá xem có nên kết hợp hay không.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSPBCUZfEnoYmFDBhAU2QtIq6DIzW7UHAMBfnGt1V8fXEwAAXrse/wcF74lXkIsAAAAASUVORK5CYII=)  
**6. Thiết kế EDA**  
Theo yêu cầu, các trường ngoài MSV và họ tên được đưa vào phạm vi EDA; điều này không có nghĩa mọi trường đều tự động được đưa vào huấn luyện model.  
| | |  
|-|-|  
| **Nhóm** | **Phân tích / biểu đồ** |   
| Chất lượng dữ liệu | Số dòng/cột, kiểu dữ liệu, thiếu dữ liệu, giá trị ngoài phạm vi, duplicates, phân bố missing theo cột |   
| Điểm | Histogram/boxplot TX1, TX2, giữa kỳ, điểm cuối kỳ thực tế; scatter/correlation giữa điểm thành phần và điểm cuối kỳ |   
| Chuyên cần | Histogram tỷ lệ nghỉ, biểu đồ số sinh viên theo từng dải nguy cơ, so sánh điểm theo dải nghỉ học |   
| LMS | Phân bố tỷ lệ LMS; so sánh LMS với điểm cuối kỳ và điểm tương tác |   
| Tương tác | Phân bố điểm tương tác và liên hệ với điểm số/chuyên cần/LMS |   
| Khảo sát | Phân bố stress, động lực và đi làm thêm; so sánh điểm theo nhóm nếu cỡ mẫu cho phép |   
| Nhân khẩu học | Phân tích tuổi/năm sinh, quê quán theo vùng/tỉnh, ngành học; dùng thống kê tổng hợp, tránh hiển thị nhóm quá nhỏ có thể nhận diện cá nhân |   
| Tương quan | Heatmap cho biến số; biểu đồ nhóm/boxplot cho biến phân loại; không tính tương quan Pearson trên MSV/họ tên |   
| Mô hình | Actual vs predicted, residual plot, phân bố sai số, MAE/RMSE theo nhóm đủ mẫu |   
| Nguy cơ | Phân bố điểm tổng hợp, đối chiếu với điểm cuối kỳ/thực tế nếu có nhãn hợp lệ, ma trận nhầm lẫn nếu triển khai classifier |   
   
Báo cáo EDA phải tách **mối tương quan** khỏi kết luận  **nguyên nhân**. Không kết luận stress/đi làm thêm làm điểm giảm chỉ vì biểu đồ thể hiện tương quan; dữ liệu quan sát không tự chứng minh quan hệ nhân quả.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQ2AQBAAsSHhiQI0IWp9ngBsYIEfIWkVdJuZs5oAAPiLe6+O6vp6AgDAa+sBhYwEOqBD7p8AAAAASUVORK5CYII=)  
**7. Những gì dùng lại, viết lại và loại bỏ từ **student_analysis  
Repository cũ mô tả PyQt6, schema/validation/data quality, pipeline ML, PredictionService, model registry, EDA, báo cáo và What-If. Tuy nhiên, kế hoạch refactor của repo cũng ghi nhận một số đường chạy cũ còn tồn tại, gồm UI Diagnosis/What-If dùng engine cũ và đường nhập liệu legacy có nguy cơ phá dữ liệu. Vì vậy, chỉ tái sử dụng sau khi đọc và kiểm thử từng thành phần, không copy toàn bộ thư mục. Tham khảo: README của repo và REFACTOR_MASTER_PLAN.md.  
| | | | |  
|-|-|-|-|  
| **Thành phần cũ** | **Quyết định** | **Cách xử lý trong dự án mới** | **Lý do** |   
| PyQt6 desktop UI | Dùng lại công nghệ/ý tưởng | Dựng UI mới theo module; chỉ lấy widget nhỏ nếu phụ thuộc sạch | Giữ trải nghiệm desktop quen thuộc, tránh kéo theo legacy |   
| Schema, validator, data quality | Dùng lại ý tưởng; viết schema mới | Xây schema HaUI, kiểm tra phần trăm 0–100, điểm đúng thang, trạng thái TX2 | Dataset hoàn toàn thay đổi |   
| Ingestion pipeline | Chỉ giữ mô hình thiết kế, viết lại/an toàn hóa | File → staging → validate → preview → commit; lỗi thì rollback/giữ dataset trước | Tránh dùng đường ingest cũ có drop_all được ghi nhận trong master plan |   
| EDA service/charts | Tái sử dụng có chọn lọc | Viết lại biểu đồ theo biến HaUI; loại đồ thị gắn chặt schema cũ | Khác trường dữ liệu và câu hỏi phân tích |   
| ML preprocessing pipeline | Dùng lại nguyên tắc | Dựng pipeline scikit-learn mới theo feature HaUI, fit transform trong CV folds | Ngăn leakage và đảm bảo tái lập |   
| PredictionService | Tái sử dụng kiến trúc, code chỉ sau audit | Service mới trả dự đoán điểm cuối kỳ và metadata | Tránh UI gọi model trực tiếp |   
| ModelRegistry | Dùng lại ý tưởng/thiết kế | Tạo registry mới lưu dataset version, feature schema, model version, metric, seed, hash | Model cũ không hợp dữ liệu mới |   
| Lasso / Ridge / XGBoost model artifacts | **Không dùng lại weights/artifacts** | Huấn luyện mới từ dữ liệu HaUI; benchmark nhiều baseline | Model artifact học từ dữ liệu cũ không có ý nghĩa với schema/target mới |   
| WhatIfEngine legacy | Không dùng | Nếu vẫn cần What-If, viết service mới dựa trên PredictionService và ràng buộc điểm hợp lệ | Master plan cũ nêu engine legacy/fallback cần loại bỏ |   
| DiagnosisWidget cũ | Không copy nguyên | Viết RiskPredictionView/AcademicRiskView mới theo quy tắc mới | Đổi terminology và thuật toán nguy cơ |   
| MissingDetector cũ | Chỉ tham khảo nếu tổng quát | Viết bộ kiểm tra missing/schema dùng trong data validation; phân biệt thiếu hợp lệ và lỗi | Bài toán TX2 optional khác dataset cũ |   
| Tính năng báo cáo | Dùng lại layout/pattern nếu tốt | Tạo báo cáo EDA, dự đoán điểm, nguy cơ, chất lượng dữ liệu theo schema mới | Báo cáo không được mang metric hard-code từ model cũ |   
| Docker/PostgreSQL | Tùy chọn theo nhu cầu | MVP có thể nhập Excel/CSV; thiết kế repository interface để bổ sung database | Không tăng độ phức tạp trước khi có yêu cầu lưu trữ chung |   
| CI, pytest, lint | Dùng lại cấu hình/ý tưởng | Tạo test suite mới; giữ GitHub Actions cho lint + tests | Bộ test cũ không xác minh logic HaUI |   
| README/docs cũ | Không dùng làm tài liệu chính | Viết tài liệu mới; dùng kinh nghiệm cũ làm tham khảo | Tránh mô tả không khớp implementation mới |   
| Dataset mẫu và metric cũ | **Loại khỏi pipeline mới** | Chỉ giữ trong branch/archive cũ; không đưa vào báo cáo dự án mới | Không đại diện kết quả HaUI |   
| Hard-coded fallback/fake metrics | **Loại bỏ hoàn toàn** | Registry/model chưa có thì UI báo chưa huấn luyện | Tránh tạo kết quả giả |   
| ID/tên trong model features | **Không sử dụng** | Chỉ lưu cho liên kết/hiển thị | Tránh leakage, overfit và lộ thông tin cá nhân |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OMQ2AABAAsSPBCj5fFgpQwYwEZiywEZJWQZeZ2ao9AAD+4lyruzq+ngAA8Nr1AMTRBeEgNK9YAAAAAElFTkSuQmCC)  
**8. Công nghệ đề xuất**  
| | | |  
|-|-|-|  
| **Lớp** | **Công nghệ khởi đầu** | **Ghi chú** |   
| Ngôn ngữ | Python | Phù hợp hệ sinh thái data/ML và app hiện tại |   
| Desktop UI | PyQt6 | Giữ công nghệ nếu tiếp tục định hướng desktop; UI mới không phụ thuộc logic ML |   
| Data | pandas, NumPy, openpyxl | Đọc Excel/CSV, làm sạch và chuẩn hóa |   
| ML | scikit-learn | Baseline, pipeline, CV, hồi quy, đánh giá |   
| Mô hình tăng cường | HistGradientBoosting hoặc XGBoost tùy benchmark | Không bắt buộc trước khi benchmark baseline |   
| Biểu đồ | Matplotlib; Seaborn nếu thuận tiện | Dùng trong EDA/report; bảo đảm hiển thị tiếng Việt |   
| Model artifact | joblib hoặc định dạng được kiểm soát | Chỉ nạp artifact từ nguồn tin cậy; gắn metadata/version |   
| Lưu trữ MVP | CSV/XLSX raw + processed files có version | Không commit dữ liệu nhạy cảm vào Git |   
| Cơ sở dữ liệu (phase sau) | PostgreSQL qua repository/service layer nếu cần | Không dùng lệnh phá hủy schema trong luồng import |   
| Kiểm thử | pytest | Unit, integration, contract tests |   
| Chất lượng code | Ruff / formatter và type checking theo mức cần thiết | CI chạy lint và test |   
| Môi trường | .venv, pyproject.toml làm nguồn dependency chính | Tránh 2 danh sách dependency bị lệch; requirements.txt có thể xuất từ nguồn chuẩn |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSd4EKxgBjP+Asa0hxW8ibAl2DIzR3UFAMBf3Gu1VefXEwAAXtsfSqwDVbgKngwAAAAASUVORK5CYII=)  
**9. Cấu trúc thư mục repository đề xuất**  
haui_student_performance_prediction/  
 ├── README.md  
 ├── pyproject.toml  
 ├── requirements.txt  
 ├── .env.example  
 ├── .gitignore  
 ├── LICENSE  
 ├── main.py  
 ├── data/  
 │   ├── raw/                 # file nguồn, không commit dữ liệu cá nhân  
 │   ├── staging/             # dữ liệu chờ kiểm tra/duyệt  
 │   ├── processed/           # dữ liệu chuẩn hóa, thường nằm trong .gitignore  
 │   └── sample/              # chỉ dùng dữ liệu giả lập/đã ẩn danh  
 ├── docs/  
 │   ├── project_plan.md  
 │   ├── architecture.md  
 │   ├── data_dictionary.md  
 │   ├── data_ingestion.md  
 │   ├── scoring_formula.md  
 │   ├── risk_scoring.md  
 │   ├── ml_methodology.md  
 │   ├── eda_specification.md  
 │   ├── privacy_and_data_governance.md  
 │   ├── model_card_template.md  
 │   ├── testing_strategy.md  
 │   └── user_guide.md  
 ├── models/  
 │   ├── registry.json         # metadata; không lưu thông tin cá nhân  
 │   └── artifacts/            # model binary, thường nằm trong .gitignore  
 ├── outputs/  
 │   ├── eda/                  # biểu đồ/báo cáo được tạo ra, thường ignore  
 │   └── reports/              # báo cáo đầu ra, cần xử lý quyền truy cập  
 ├── scripts/  
 │   ├── validate_dataset.py  
 │   ├── run_eda.py  
 │   ├── train_final_exam_model.py  
 │   ├── evaluate_models.py  
 │   └── export_report.py  
 ├── src/  
 │   └── haui_student_prediction/  
 │       ├── __init__.py  
 │       ├── config.py  
 │       ├── data/  
 │       │   ├── schema.py  
 │       │   ├── column_mapping.py  
 │       │   ├── validator.py  
 │       │   ├── quality_report.py  
 │       │   ├── cleaning.py  
 │       │   └── ingestion_service.py  
 │       ├── scoring/  
 │       │   ├── course_scoring_config.py  
 │       │   ├── final_course_score.py  
 │       │   └── score_validation.py  
 │       ├── features/  
 │       │   ├── feature_builder.py  
 │       │   ├── feature_contract.py  
 │       │   └── temporal_snapshot.py  
 │       ├── ml/  
 │       │   ├── preprocessing.py  
 │       │   ├── train_regressor.py  
 │       │   ├── evaluate_regressor.py  
 │       │   ├── predictor.py  
 │       │   ├── model_registry.py  
 │       │   └── explainability.py  
 │       ├── risk/  
 │       │   ├── attendance_rules.py  
 │       │   ├── component_mapping.py  
 │       │   ├── risk_weights.py  
 │       │   ├── risk_calculator.py  
 │       │   └── risk_labels.py  
 │       ├── eda/  
 │       │   ├── statistics.py  
 │       │   ├── distributions.py  
 │       │   ├── relationships.py  
 │       │   └── charts.py  
 │       ├── services/  
 │       │   ├── dataset_service.py  
 │       │   ├── prediction_service.py  
 │       │   ├── risk_prediction_service.py  
 │       │   └── report_service.py  
 │       ├── storage/  
 │       │   ├── dataset_repository.py  
 │       │   └── artifact_repository.py  
 │       └── ui/  
 │           ├── main_window.py  
 │           ├── views/  
 │           │   ├── dashboard_view.py  
 │           │   ├── data_import_view.py  
 │           │   ├── data_quality_view.py  
 │           │   ├── eda_view.py  
 │           │   ├── score_prediction_view.py  
 │           │   ├── risk_prediction_view.py  
 │           │   ├── model_management_view.py  
 │           │   └── reports_view.py  
 └── tests/  
     ├── fixtures/  
     │   └── synthetic_haui_sample.csv  
     ├── unit/  
     │   ├── test_schema.py  
     │   ├── test_validator.py  
     │   ├── test_course_scoring.py  
     │   ├── test_attendance_rules.py  
     │   ├── test_risk_calculator.py  
     │   ├── test_feature_builder.py  
     │   ├── test_preprocessing.py  
     │   ├── test_model_registry.py  
     │   └── test_prediction_service.py  
     ├── integration/  
     │   ├── test_ingestion_flow.py  
     │   ├── test_training_pipeline.py  
     │   ├── test_prediction_flow.py  
     │   └── test_report_flow.py  
     └── contracts/  
         ├── test_input_output_contract.py  
         └── test_model_feature_contract.py  
   
**9.1. Ranh giới module**  
- data/: đọc/kiểm tra/chuẩn hóa dữ liệu; không huấn luyện model.  
- scoring/: áp dụng công thức điểm theo cấu hình học phần; không tự huấn luyện model.  
- features/: tạo input model từ snapshot ở thời điểm dự đoán; không chứa UI.  
- ml/: train/evaluate/predict và registry; không truy cập widget.  
- risk/: áp dụng điều kiện nghỉ học và trọng số nguy cơ; thành phần có test độc lập.  
- eda/: thống kê và biểu đồ; không dùng dữ liệu định danh làm biến phân tích định lượng.  
- services/: điều phối luồng nghiệp vụ; UI gọi qua đây.  
- storage/: đọc/ghi file và artifact; có thể mở rộng sang database sau.  
- ui/: trình bày kết quả, trạng thái loading/error/insufficient data; không chứa công thức ML.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNBCUpfEJ5YGBDBgAU2QtIq6DIzW7UHAMBfHGt1V+fXEwAAXrseHDYF+yOk59sAAAAASUVORK5CYII=)  
**10. Đặc tả giao diện**  
| | | |  
|-|-|-|  
| **Màn hình** | **Chức năng** | **Đầu ra chính** |   
| Dashboard | Tổng quan dataset và kết quả mới nhất | Số dòng hợp lệ, lỗi dữ liệu, phân bố điểm/nguy cơ, phiên bản model |   
| Nhập dữ liệu | Chọn Excel/CSV, ánh xạ cột và xem trước | Preview số dòng/cột, lỗi, thiếu, duplicate, schema status trước khi chấp nhận |   
| Chất lượng dữ liệu | Xem chi tiết lỗi và thống kê missing | Báo cáo kiểm tra; không tự âm thầm sửa dữ liệu |   
| EDA | Chọn biến/nhóm phân tích và xem biểu đồ | Biểu đồ phân bố, liên hệ, bảng thống kê, xuất báo cáo |   
| Dự đoán điểm cuối kỳ | Chọn sinh viên–học phần và kiểm tra feature đã có | Điểm thi cuối kỳ dự đoán, điểm tổng kết dự kiến, công thức được dùng, model version và cảnh báo thiếu dữ liệu |   
| Dự đoán nguy cơ học vụ | Xem trạng thái chuyên cần, điểm nguy cơ và yếu tố ảnh hưởng | Ngưỡng nghỉ học, risk score, mức nguy cơ, lý do, độ đầy đủ dữ liệu; không ra quyết định học vụ chính thức |   
| Quản lý model | Xem model đã train và metric | Trạng thái chưa có/cũ/lỗi/sẵn sàng, ngày train, dataset hash, metric |   
| Báo cáo | Tổng hợp dữ liệu, EDA, model và nguy cơ | CSV/PDF hoặc định dạng được phê duyệt; không chứa định danh ngoài phạm vi cần thiết |   
   
**UX bắt buộc**  
- Hiển thị rõ model chưa huấn luyện hoặc model cũ so với dataset.  
- Nếu thiếu feature bắt buộc, không tạo kết quả giả.  
- Có thông báo loading và lỗi; không để UI bị treo trong lúc huấn luyện.  
- Không dùng màu sắc là tín hiệu duy nhất; nhãn nguy cơ phải có chữ.  
- Phân biệt điểm thật, điểm dự đoán và điểm tổng kết tính từ dự đoán.  
- Hiển thị nguồn dataset/model version để có thể truy vết.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OQQmAABRAsScYxpi/i2HMYARvRrCCNxG2BFtmZquOAAD4i3Ot7mr/egIAwGvXA3dVBcrz/x+1AAAAAElFTkSuQmCC)  
**11. Kế hoạch thực hiện theo giai đoạn**  
| | | | |  
|-|-|-|-|  
| **Giai đoạn** | **Công việc** | **Sản phẩm bàn giao** | **Điều kiện hoàn thành** |   
| **P0 — Khóa yêu cầu** | Chốt mục tiêu, xác nhận các trường, quy ước TX2, nghĩa của tỷ lệ LMS, ngưỡng nghỉ học, quyền sử dụng dữ liệu | project_plan.md, data_dictionary.md, risk_scoring.md, danh sách câu hỏi còn mở | Mỗi trường có ý nghĩa, kiểu dữ liệu, đơn vị và quy tắc thiếu rõ ràng |   
| **P1 — Khởi tạo repo sạch** | Tạo repository mới, cấu trúc module, môi trường Python, lint/test/CI, .gitignore, logging | Skeleton chạy được; CI chạy test mẫu | Clone sạch, cài đặt theo README và mở ứng dụng cơ bản thành công |   
| **P2 — Nhập dữ liệu và data quality** | Column mapping, schema validation, kiểm tra tỷ lệ/điểm, preview, staging/rollback | Ingestion service và quality report | File lỗi không ghi đè dataset hợp lệ; dữ liệu gốc không bị sửa |   
| **P3 — EDA** | Thống kê dữ liệu, phân bố, tương quan, khảo sát và báo cáo | EDA service/UI, biểu đồ và báo cáo đầu tiên | Biểu đồ tái tạo được, loại ID/name khỏi phân tích định lượng, kết luận không phóng đại nhân quả |   
| **P4 — Công thức điểm** | Cấu hình công thức có TX2/không TX2; validation điểm và thiếu dữ liệu | Scoring service + unit tests | Test đúng hai công thức; không nhầm TX2 chưa nhập với môn không có TX2 |   
| **P5 — ML dự đoán điểm** | Feature contract, split, pipeline, baseline, CV, test, model registry | Model huấn luyện mới và evaluation report | Target là điểm cuối kỳ thực tế; chống leakage; báo MAE/RMSE/R² trên dữ liệu chưa dùng để chọn model |   
| **P6 — Dự đoán nguy cơ** | Quy tắc nghỉ học, mapping từng component, weights, aggregate score, nhãn đầu ra | Risk engine + bảng giải thích điểm | absence_percentage > 30 luôn kích hoạt trạng thái riêng; điểm thành phần cùng thang/chiều; thiếu dữ liệu không bị coi là 0 |   
| **P7 — Tích hợp UI** | Tích hợp Dashboard, Import, EDA, Score Prediction, Risk Prediction, Reports | Ứng dụng MVP | UI chỉ gọi service; xử lý loading/error/model missing/model stale |   
| **P8 — Kiểm thử toàn luồng** | Unit/integration/contract/regression, clean install, kiểm thử dataset nhỏ và dữ liệu thực được phép sử dụng | Test report, hướng dẫn dùng, known limitations | Không có hard-coded metrics, fake fallback hoặc đường import phá dữ liệu |   
| **P9 — Hoàn thiện và phát hành** | Cập nhật tài liệu, phiên bản, model card, hướng dẫn bảo mật, release tag | v1.0.0 nội bộ/học thuật | Có thể tái lập kết quả; không đưa dữ liệu thật/secret vào Git; nêu rõ giới hạn mô hình |   
   
**Thứ tự ưu tiên**  
1. **P0/P1:** đặc tả dữ liệu, repository sạch và cơ chế nhập an toàn.  
2. **P2/P3/P4:** data quality, EDA và công thức điểm có test.  
3. **P5/P6:** model dự đoán và risk engine chỉ khi dữ liệu phù hợp.  
4. **P7/P8/P9:** UI, tích hợp, kiểm thử và phát hành.  
Không nên làm giao diện cầu kỳ trước khi schema, công thức và target đã được xác định.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSdYxZ4/mJjEsxE8W8GbCFuCLTOzVXsAAPzFuVZ3dXw9AQDgtesBxPEF3bv7x0IAAAAASUVORK5CYII=)  
**12. Kiểm thử và tiêu chí nghiệm thu**  
**12.1. Data**  
- Phần trăm âm hoặc trên 100 bị đánh dấu lỗi.  
- Điểm ngoài thang được xác định từ data contract bị đánh dấu lỗi.  
- TX2 optional được phân biệt với TX2 missing.  
- Dòng duplicate được báo cáo trước khi xử lý.  
- Import lỗi không phá dữ liệu trước đó.  
- Dataset version/hash được ghi lại.  
**12.2. Công thức điểm**  
- Môn không có TX2 áp dụng đúng 0.2 TX1 + 0.2 GK + 0.6 predicted CK.  
- Môn có TX2 áp dụng đúng 0.15 TX1 + 0.15 TX2 + 0.2 GK + 0.5 predicted CK.  
- Môn có TX2 nhưng chưa có điểm TX2 không tự chuyển sang công thức A.  
- Điểm dự đoán, điểm thi thực tế và điểm tổng kết dự kiến được thể hiện tách biệt.  
**12.3. Nguy cơ**  
- Nghỉ >30% kích hoạt trạng thái ưu tiên.  
- Ranh giới nghỉ học 5%, 10%, 20%, 30% và LMS 5%, 10%, 20%, 30% không bị chồng lấn hoặc bỏ sót.  
- Tỷ lệ LMS thấp tạo risk points cao hơn nếu ý nghĩa cột đúng như giả định.  
- Điểm tương tác được chuẩn hóa đúng chiều.  
- Trọng số cộng đúng 100%.  
- Thiếu tỷ lệ nghỉ học không bị xem là 0% nghỉ.  
- UI hiển thị riêng cảnh báo điều kiện chuyên cần và điểm nguy cơ tổng hợp.  
**12.4. ML**  
- Không sử dụng MSV, họ tên, điểm cuối kỳ thực tế hay biến hình thành target làm input.  
- Preprocessing được fit bên trong các fold train.  
- Tập test không được dùng chọn model.  
- Model registry chứa feature list, dataset version/hash, metrics, seed, package versions và thời điểm train.  
- Không hiện metric cứng nếu không có model/registry.  
- So sánh với baseline đơn giản trước khi tuyên bố model tốt.  
**12.5. Bảo mật và sử dụng có trách nhiệm**  
- Dữ liệu sinh viên thật không nằm trong repo công khai.  
- Log không ghi họ tên/MSV/đáp án khảo sát nhạy cảm nếu không cần thiết.  
- Export báo cáo có nguyên tắc hạn chế định danh và nhóm nhỏ.  
- Không diễn giải dự đoán như quyết định chính thức hay quan hệ nhân quả.  
- Model chưa được đánh giá ngoài mẫu phải được đánh dấu thử nghiệm.  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSPBCUZfE2IYmVDBhAU2QtIq6DIzW7UHAMBfnGt1V8fXEwAAXrse/xcF7U7sx4wAAAAASUVORK5CYII=)  
**13. Các quyết định còn phải chốt trước khi code**  
| | | |  
|-|-|-|  
| **Câu hỏi** | **Đề xuất mặc định** | **Vì sao cần chốt** |   
| Nguồn dữ liệu là một dòng/sinh viên hay một dòng/sinh viên–học phần? | Chọn một dòng/sinh viên–học phần–học kỳ–snapshot nếu có dữ liệu phù hợp | Công thức điểm áp dụng theo học phần và phải liên kết target cuối kỳ |   
| Thang điểm? | Xác minh từ file nguồn; model/regression contract ghi rõ min/max | Tránh giả định thang điểm sai |   
| TX2 không áp dụng được đánh dấu thế nào? | has_tx2=False, không chỉ dựa vào ô trống | Phân biệt “không có” và “chưa nhập” |   
| Tỷ lệ LMS có ý nghĩa gì? | Xác nhận cột là chỉ số tham gia/học tập, 100 là mức cao | Cần biết chiều đúng trước khi quy đổi nguy cơ |   
| Thang stress/động lực/tương tác? | Lưu thang gốc, min/max, chiều tốt/xấu và mapping | Không thể đảo chiều hoặc chuẩn hóa đúng nếu chưa biết thang |   
| Điểm học phần nào được xem là nguy cơ? | Đặt trong cấu hình sau khi xác minh cách đánh giá/nhãn lịch sử | Chưa có ngưỡng kết quả thấp do chủ dự án cung cấp |   
| Nguy cơ tổng hợp được phân nhãn ở ngưỡng nào? | Tạm dùng 0–<20, 20–<40, 40–<60, 60–<80, 80–100 | Đây là ngưỡng prototype, cần hiệu chỉnh trên lịch sử |   
| “Nghỉ >75% = học lại” áp dụng với mọi môn? | Giữ thành rule cấu hình, xác nhận phạm vi với dữ liệu/quy định áp dụng | Không hard-code giả định cho mọi loại học phần khi chưa xác minh |   
| Có cần PostgreSQL ngay? | MVP nhập Excel/CSV; thiết kế repository để nâng cấp | Tránh tăng độ phức tạp trước khi có nhu cầu |   
| Có đủ target lịch sử để train? | Kiểm tra trước P5 | Không target = không thể đánh giá supervised model một cách đúng đắn |   
   
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSd4NIGhrOTvaQBrWMGbCFuCLTOzV2cAAPzFvVZbdXw9AQDgtesBhYQEO+64Y8AAAAAASUVORK5CYII=)  
**14. Định nghĩa thành công của phiên bản đầu tiên**  
Dự án được xem là có một MVP tốt khi có thể thực hiện trọn luồng:  
**Nhập dữ liệu → kiểm tra chất lượng → EDA → chọn công thức học phần → huấn luyện/evaluate mô hình điểm (nếu có target) → dự đoán điểm cuối kỳ → tính điểm tổng kết dự kiến → tính điểm nguy cơ → giải thích kết quả → xuất báo cáo.**  
MVP phải:  
- Tái lập được từ repository mới sau khi clone sạch.  
- Không sử dụng artifact/metrics của dataset cũ.  
- Không có silent fallback hoặc dữ liệu giả được trình bày như thật.  
- Có test cho công thức điểm, ranh giới chuyên cần, risk weights và luồng import.  
- Có báo cáo EDA tái tạo được.  
- Nêu rõ model được đánh giá trên dữ liệu nào, những giới hạn nào còn tồn tại và trường nào chưa đủ dữ liệu.  
   
