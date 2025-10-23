# 🎤 Hệ thống chấm điểm giọng hát karaoke bằng AI 🎵

## 📋 Mô tả

Hệ thống chấm điểm giọng hát karaoke sử dụng trí tuệ nhân tạo để đánh giá chất lượng giọng hát với các tính năng:

- 🎵 **Tách giọng hát**: Sử dụng AI Audio Separator để tách giọng từ file karaoke
- 🎹 **Phát hiện phím âm nhạc**: Sử dụng jcarbonnell/key_class_detection để phân tích phím
- 📊 **Chấm điểm đa tiêu chí**: Đánh giá dựa trên 7 tiêu chí karaoke khác nhau
- 🖥️ **Giao diện thân thiện**: GUI đẹp mắt với tkinter

## 🚀 Cài đặt nhanh

### Windows:
```bash
run.bat
```

### Linux/Mac:
```bash
chmod +x run.sh
./run.sh
```

### Cài đặt thủ công:
```bash
pip install -r requirements.txt
python main.py
```

## 📁 Cấu trúc dự án

```
singing scoring AI/
├── main.py                 # 🚀 File chính để chạy ứng dụng
├── gui.py                  # 🖥️ Giao diện người dùng
├── audio_processor.py      # 🎵 Xử lý âm thanh và tách giọng
├── key_detector.py         # 🎹 Phát hiện phím âm nhạc
├── scoring_system.py       # 📊 Hệ thống chấm điểm
├── config.py               # ⚙️ Cấu hình hệ thống
├── demo.py                 # 🎯 Script demo và test
├── requirements.txt        # 📦 Danh sách dependencies
├── run.bat                 # 🪟 Script chạy cho Windows
├── run.sh                  # 🐧 Script chạy cho Linux/Mac
├── README.md               # 📖 Tài liệu dự án
└── USAGE.md                # 📚 Hướng dẫn sử dụng chi tiết
```

## 🎯 Cách sử dụng

1. **Chuẩn bị file âm thanh:**
   - File karaoke: Ghi âm của người hát (có thể có nhạc nền)
   - File beat: Nhạc beat gốc (không có giọng hát)

2. **Chạy ứng dụng:**
   ```bash
   python main.py
   ```

3. **Phân tích và chấm điểm:**
   - Chọn file karaoke và beat nhạc
   - Nhấn "Bắt đầu phân tích và chấm điểm"
   - Chờ hệ thống xử lý (2-5 phút)
   - Xem kết quả chấm điểm

## 📊 Các tiêu chí chấm điểm

| Tiêu chí | Trọng số | Mô tả |
|----------|----------|-------|
| 🎹 Độ chính xác phím | 25% | So sánh phím âm nhạc giữa beat và giọng hát |
| 🎵 Độ chính xác cao độ | 20% | Phân tích pitch và độ lệch tone |
| 🥁 Độ chính xác nhịp điệu | 15% | Tempo và beat tracking |
| ⏰ Độ chính xác thời gian | 15% | Timing và độ dài bài hát |
| 🎤 Chất lượng giọng hát | 10% | Năng lượng và độ rõ ràng |
| ⚡ Tính nhất quán năng lượng | 10% | Sự ổn định trong suốt bài hát |
| 🗣️ Phát âm | 5% | Độ rõ ràng của phát âm |

## 🏆 Thang điểm

- **90-100**: A+ (Xuất sắc) 🌟
- **80-89**: A (Giỏi) 🥇
- **70-79**: B+ (Khá) 🥈
- **60-69**: B (Trung bình khá) 🥉
- **50-59**: C (Trung bình) ✅
- **40-49**: D (Yếu) ⚠️
- **0-39**: F (Kém) ❌

## 🧪 Demo và Test

Chạy script demo để test hệ thống:
```bash
python demo.py
```

Script này sẽ:
- Tạo file âm thanh demo
- Test tất cả các components
- Hiển thị kết quả chấm điểm

## ⚙️ Yêu cầu hệ thống

- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **GPU**: Không bắt buộc (sẽ tăng tốc độ)
- **Dung lượng**: ~2GB cho AI models
- **Internet**: Cần để tải models lần đầu

## 🔧 Xử lý sự cố

### Lỗi "Module not found"
```bash
pip install -r requirements.txt
```

### Lỗi "CUDA out of memory"
- Hệ thống tự động chuyển sang CPU
- Giảm kích thước file nếu cần

### File không được hỗ trợ
- Hỗ trợ: WAV, MP3, FLAC, M4A
- Đảm bảo file có chất lượng tốt

## 📚 Tài liệu

- [Hướng dẫn sử dụng chi tiết](USAGE.md)
- [Cấu hình hệ thống](config.py)

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:
1. Fork dự án
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📄 License

Dự án này được phát hành dưới MIT License.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) cho các AI models
- [Librosa](https://librosa.org/) cho xử lý âm thanh
- [PyTorch](https://pytorch.org/) cho deep learning
