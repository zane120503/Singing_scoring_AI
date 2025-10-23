# 🎤 Hệ thống chấm điểm giọng hát karaoke bằng AI

## 📁 Cấu trúc thư mục

```
singing scoring AI/
├── src/                    # Source code chính
│   ├── core/              # Core functionality
│   │   ├── __init__.py
│   │   ├── scoring_system.py    # Hệ thống chấm điểm
│   │   └── config.py            # Cấu hình
│   ├── ai/                # AI models và processors
│   │   ├── __init__.py
│   │   ├── advanced_audio_processor.py    # Xử lý âm thanh nâng cao
│   │   ├── advanced_key_detector.py       # Phát hiện phím âm nhạc
│   │   ├── audio_processor.py            # Xử lý âm thanh cơ bản
│   │   ├── audio_separator_integration.py # Tích hợp Audio Separator
│   │   ├── ai_audio_separator.py         # AI Audio Separator wrapper
│   │   ├── key_detector.py               # Phát hiện phím cơ bản
│   │   └── real_audio_processor.py       # Xử lý âm thanh thực tế
│   ├── gui/               # GUI components
│   │   ├── __init__.py
│   │   └── gui.py                        # Giao diện chính
│   └── utils/             # Utilities
│       ├── __init__.py
│       └── essentia_docker_wrapper.py    # Docker wrapper cho Essentia
├── scripts/               # Scripts tiện ích
│   ├── build_docker.bat
│   ├── run_docker.bat
│   ├── run.bat
│   ├── run.sh
│   ├── install_essentia_*.py
│   ├── setup_essentia_*.py
│   └── essentia_*.py
├── tests/                 # Test files
│   ├── test_*.py
│   ├── demo*.py
│   └── simple_test.py
├── docs/                  # Documentation
│   ├── README.md
│   ├── USAGE.md
│   └── DOCKER_GUIDE.md
├── config/                # Configuration files
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/                  # Data files
│   └── temp_output/       # File output tạm thời
├── models/                # AI models (trống)
├── Audio_separator_ui/    # External AI separator
│   ├── app.py
│   ├── mdx_models/
│   └── ...
├── main.py               # Entry point
└── requirements.txt      # Dependencies
```

## 🚀 Cách sử dụng

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Chạy hệ thống
```bash
python main.py
```

### 3. Sử dụng GUI
- Chọn file karaoke (file ghi âm)
- Chọn file beat nhạc
- Nhấn "Bắt đầu phân tích và chấm điểm"
- Xem kết quả chấm điểm

## ⚡ Fast Mode

Hệ thống mặc định sử dụng **Fast Mode** để tăng tốc độ:
- Tách giọng nhanh (5-15 giây)
- Chất lượng chấp nhận được
- Phù hợp cho chấm điểm nhanh

## 🎯 Tính năng

- ✅ AI Audio Separator (MDX-Net)
- ✅ Essentia AI Key Detection
- ✅ Fast Mode cho tốc độ cao
- ✅ GUI thân thiện
- ✅ Docker support
- ✅ Logging chi tiết

## 📊 Trạng thái AI Models

- **AI Audio Separator:** ⚡ Fast Mode (Tốc độ cao)
- **Essentia AI:** 🐳 Docker (Fallback)
- **Device:** CPU/GPU

## 🔧 Troubleshooting

1. **Lỗi import:** Kiểm tra Python path
2. **Lỗi Docker:** Chạy `docker-compose up`
3. **Lỗi Essentia:** Sử dụng fallback method
4. **Lỗi Audio Separator:** Sử dụng Fast Mode

## 📝 Ghi chú

- Fast Mode được khuyến nghị cho tốc độ
- AI Mode cho chất lượng cao nhất
- Hệ thống tự động fallback khi có lỗi
