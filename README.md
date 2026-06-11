# 🛡️ Ứng dụng Học Máy Phát Hiện Giao Dịch Gian Lận & Rủi Ro Mặc Định

Ứng dụng web được xây dựng dựa trên framework **Streamlit** mạnh mẽ kết hợp cùng thư viện Học máy chuyên sâu **Scikit-Learn**. Hệ thống cho phép chuyển đổi mô hình nghiên cứu từ dạng Notebook (`.ipynb`) thành giao diện trực quan hỗ trợ đắc lực cho các nhà phân tích rủi ro tài chính kiểm định và sử dụng mô hình AI.

## ✨ Tính năng chính

- **Cấu hình động**: Tùy chỉnh trực tiếp siêu tham số mô hình `RandomForestClassifier` (Số lượng cây, độ sâu tối đa) ngay trên giao diện Sidebar.
- **Tổng quan Thống kê**: Tự động phân tích quy mô dữ liệu và mô tả các biến đặc trưng đầu vào mẫu.
- **Trực quan hóa Phân Phối**: Biểu đồ phân bổ dữ liệu trực quan dựa trên thư viện tương tác cao `Plotly`.
- **Đánh giá Chuyên Sâu**: Hiển thị tường minh các chỉ số kiểm tra mô hình bao gồm: *Accuracy, Precision, Recall, F1-Score, AUC*, cùng biểu đồ Ma trận nhầm lẫn (Confusion Matrix) và đường cong ROC.
- **Sử Dụng Linh Hoạt**:
  - Chế độ nhập liệu thủ công để kiểm tra rủi ro tức thì cho một giao dịch đơn lẻ.
  - Chế độ chấm điểm hàng loạt qua file danh sách Excel/CSV giúp tự động hóa vận hành.

## 📁 Cấu trúc dữ liệu yêu cầu

Mẫu dữ liệu huấn luyện hoặc chấm điểm cần tuân thủ cấu trúc định dạng cột sau:
- **Biến đặc trưng (X)**: Gồm chính xác 14 cột số mang tên từ `X_1` đến `X_14`.
- **Biến mục tiêu (y)**: Cột nhãn `default` chứa giá trị nhị phân (`0`: Giao dịch bình thường, `1`: Giao dịch rủi ro/gian lận). *Lưu ý: Đối với tệp chấm điểm hàng loạt (X_new), không bắt buộc cần kèm theo cột mục tiêu này.*

## ⚙️ Hướng dẫn Cài đặt & Khởi chạy

Để vận hành ứng dụng trên môi trường máy cục bộ của bạn, hãy thực hiện các bước sau:

### Bước 1: Khởi tạo và kích hoạt môi trường ảo (Khuyến nghị)
```bash
# Đối với Windows:
python -m venv venv
.\venv\Scripts\activate

# Đối với macOS/Linux:
python3 -m venv venv
source venv/bin/activate
