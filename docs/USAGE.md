# Hướng dẫn sử dụng hệ thống chấm điểm karaoke

## 🎯 Mô tả hệ thống

Hệ thống chấm điểm giọng hát karaoke bằng AI được thiết kế để:
- Tách giọng hát từ file karaoke
- Phát hiện phím âm nhạc của beat và giọng hát
- Chấm điểm dựa trên nhiều tiêu chí karaoke

## 📋 Yêu cầu hệ thống

- Python 3.8 trở lên
- RAM: Tối thiểu 4GB (khuyến nghị 8GB)
- GPU: Không bắt buộc nhưng sẽ tăng tốc độ xử lý
- Dung lượng: Khoảng 2GB cho các model AI

## 🚀 Cài đặt

1. **Cài đặt Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy ứng dụng:**
   ```bash
   python main.py
   ```

## 📁 Cấu trúc dự án

```
singing scoring AI/
├── main.py                 # File chính để chạy ứng dụng
├── gui.py                  # Giao diện người dùng
├── audio_processor.py      # Xử lý âm thanh và tách giọng
├── key_detector.py         # Phát hiện phím âm nhạc
├── scoring_system.py      # Hệ thống chấm điểm
├── requirements.txt        # Danh sách dependencies
└── README.md              # Tài liệu dự án
```

## 🎵 Cách sử dụng

### Bước 1: Chuẩn bị file âm thanh
- **File karaoke**: File ghi âm của người hát (có thể có nhạc nền)
- **File beat**: File nhạc beat gốc (không có giọng hát)

### Bước 2: Chạy ứng dụng
1. Mở terminal/command prompt
2. Chạy lệnh: `python main.py`
3. Giao diện GUI sẽ hiển thị

### Bước 3: Phân tích và chấm điểm
1. Nhấn "Chọn file" để chọn file karaoke
2. Nhấn "Chọn file" để chọn file beat nhạc
3. Nhấn "Bắt đầu phân tích và chấm điểm"
4. Chờ hệ thống xử lý (có thể mất vài phút)
5. Xem kết quả chấm điểm

## 📊 Các tiêu chí chấm điểm

### 1. Độ chính xác phím (25%)
- So sánh phím âm nhạc giữa beat và giọng hát
- Sử dụng AI model để phát hiện phím

### 2. Độ chính xác cao độ (20%)
- Phân tích pitch của giọng hát
- So sánh với pitch của beat nhạc

### 3. Độ chính xác nhịp điệu (15%)
- Phân tích tempo và beat tracking
- Đánh giá độ đồng bộ nhịp điệu

### 4. Độ chính xác thời gian (15%)
- So sánh độ dài của các file
- Đánh giá timing tổng thể

### 5. Chất lượng giọng hát (10%)
- Phân tích năng lượng âm thanh
- Đánh giá độ rõ ràng và sáng của giọng

### 6. Tính nhất quán năng lượng (10%)
- Phân tích sự ổn định của năng lượng
- Đánh giá độ nhất quán trong suốt bài hát

### 7. Phát âm (5%)
- Phân tích đặc trưng MFCC
- Đánh giá độ rõ ràng của phát âm

## 🎯 Thang điểm

- **90-100**: A+ (Xuất sắc)
- **80-89**: A (Giỏi)
- **70-79**: B+ (Khá)
- **60-69**: B (Trung bình khá)
- **50-59**: C (Trung bình)
- **40-49**: D (Yếu)
- **0-39**: F (Kém)

## ⚠️ Lưu ý quan trọng

1. **Chất lượng file âm thanh**: File chất lượng cao sẽ cho kết quả chính xác hơn
2. **Thời gian xử lý**: Quá trình phân tích có thể mất 2-5 phút tùy thuộc vào độ dài file
3. **Tương thích**: Hỗ trợ các định dạng WAV, MP3, FLAC, M4A
4. **Kết nối internet**: Cần internet để tải các AI model lần đầu

## 🔧 Xử lý sự cố

### Lỗi "Module not found"
```bash
pip install -r requirements.txt
```

### Lỗi "CUDA out of memory"
- Hệ thống sẽ tự động chuyển sang CPU
- Giảm kích thước file âm thanh nếu cần

### Lỗi "File not found"
- Kiểm tra đường dẫn file
- Đảm bảo file có định dạng được hỗ trợ

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra lại các bước cài đặt
2. Đảm bảo file âm thanh có chất lượng tốt
3. Kiểm tra log lỗi trong terminal

## 🔄 Cập nhật

Để cập nhật hệ thống:
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

