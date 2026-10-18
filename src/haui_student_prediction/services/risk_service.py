from data.data_loader import StudentDataLoader
from risk.risk_calculator import RiskCalculator

class StudentRiskService:
    def __init__(self):
        self.data_loader = StudentDataLoader()
        self.calculator = RiskCalculator()

    def evaluate_student(self, student_id: str):
        # 1. Gọi Data Loader lấy dữ liệu sinh viên từ SQLite
        student_data = self.data_loader.get_student_by_id(student_id)
        
        if not student_data:
            return {"error": f"Không tìm thấy sinh viên với mã {student_id}"}

        # 2. Bóc tách dữ liệu từ Database
        # Giả sử tỷ lệ vắng mặt = 100 - tỷ lệ đi học (attendance_rate)
        attendance_rate = student_data.get('attendance_rate', 100.0)
        absence_pct = 100.0 - attendance_rate 
        
        gpa_score = student_data.get('gpa', 0.0)
        # Chuyển đổi GPA hệ 4 sang hệ 10 để đưa vào mô hình nếu cần
        predicted_course_score = gpa_score * 2.5 

        # 3. Truyền vào RiskCalculator để tính toán logic thuần túy
        try:
            # Các thông số lms, interaction, stress... có thể được lấy thêm từ các bảng khác
            # Ở đây đang để giá trị mẫu (mock) để bạn thấy cách ráp code
            risk_result = self.calculator.calculate_risk(
                absence_pct=absence_pct,
                lms_pct=85.0,                   # Lấy từ bảng điểm danh LMS
                predicted_course_score=predicted_course_score,
                interaction=45.0,               # Lấy từ bảng tương tác
                interaction_max=100.0,
                stress=30.0,                    # Điểm khảo sát tâm lý
                stress_max=100.0,
                motivation=75.0,
                motivation_max=100.0
            )
            
            # Trả về kết quả đánh giá rủi ro (RiskResult dataclass)
            return risk_result
            
        except ValueError as ve:
            return {"error": f"Lỗi tính toán: {str(ve)}"}

# --- TEST THỬ LUỒNG HOẠT ĐỘNG ---
if __name__ == "__main__":
    service = StudentRiskService()
    # Thử chạy với 1 mã sinh viên có trong file csv giả lập
    result = service.evaluate_student("SV001") 
    print(result)