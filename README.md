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

## 🚀 Cách cài đặt nhanh (Clone và chạy ngay)

### 0. Yêu cầu hệ thống
- **Python**: 3.10 hoặc 3.11 (khuyến nghị 3.11)
- **Git**: để clone repo
- **ffmpeg**: để xử lý audio (Windows: cài từ `https://ffmpeg.org` và thêm vào PATH)
- (Tùy chọn) **GPU + CUDA 12.1** nếu muốn tăng tốc với GPU
- (Tùy chọn) **Docker Desktop** nếu muốn dùng Essentia AI qua Docker (Windows cần WSL2)

### 1. Clone repo và tạo môi trường ảo
```bash
git clone https://github.com/your-org/singing-scoring-AI.git
cd "singing scoring AI"

# Tạo venv
python -m venv .venv

# Kích hoạt venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

### 2. Cài đặt dependencies cơ bản
```bash
pip install -r requirements.txt
```

### 3. Cài đặt PyTorch (giải quyết triệt để lỗi "Import \"torch\" could not be resolved")
- CPU (ổn định, không cần CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
- GPU (CUDA 12.1 – phổ biến trên Windows hiện tại):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Kiểm tra nhanh:
```bash
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
```

### 4. (Tùy chọn) Bật Essentia AI qua Docker
Essentia AI cho key detection có thể chạy trong Docker. Cài Docker Desktop trước (Windows: bật WSL2 backend).

Chạy bằng docker compose (khuyến nghị):
```bash
docker compose -f config/docker-compose.yml up -d --build
```
Hoặc dùng script có sẵn (Windows):
```bat
scripts\run_docker.bat
```
Kiểm tra Docker Essentia hoạt động:
```bash
python scripts/check_docker_essentia.py
```

Lưu ý Windows/WSL: nếu Docker báo "WSL is unresponsive", chạy script:
```bash
python scripts/fix_wsl_docker.py
```

### 5. Chạy hệ thống
```bash
python main.py
```

### 6. (Tùy chọn) Demo GPU song song và kiểm tra bộ nhớ GPU
```bash
python gpu_parallel_demo.py
```

## 🚀 Cách sử dụng

Đã gộp trong phần Cách cài đặt nhanh ở trên.

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

1. **Import "torch" could not be resolved**: Cài đúng PyTorch (mục 3), reload IDE, đảm bảo bạn đang ở trong venv khi chạy.
2. **Docker không chạy/không thấy container Essentia**: Mở Docker Desktop, bật WSL2 backend (Windows), chạy `docker compose -f config/docker-compose.yml up -d --build`, sau đó `python scripts/check_docker_essentia.py`.
3. **Essentia lỗi trong container**: Dùng `scripts/check_docker_essentia.py` để chẩn đoán; nếu WSL treo, chạy `python scripts/fix_wsl_docker.py` rồi mở lại Docker Desktop.
4. **Audio Separator chậm**: Dùng Fast Mode (mặc định) hoặc đảm bảo GPU đã được nhận (`torch.cuda.is_available()` là True) nếu dùng GPU.

## 📝 Ghi chú

- Fast Mode được khuyến nghị cho tốc độ
- AI Mode cho chất lượng cao nhất
- Hệ thống tự động fallback khi có lỗi
