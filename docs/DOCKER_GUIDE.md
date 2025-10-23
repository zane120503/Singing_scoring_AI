# 🐳 Hướng dẫn sử dụng Docker với Essentia

## 📋 Yêu cầu hệ thống

- **Docker Desktop**: Cài đặt từ https://www.docker.com/products/docker-desktop/
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB)
- **Dung lượng**: ~3GB cho Docker image

## 🚀 Cài đặt và sử dụng

### Bước 1: Cài đặt Docker Desktop

1. **Tải Docker Desktop** từ: https://www.docker.com/products/docker-desktop/
2. **Chạy installer** và làm theo hướng dẫn
3. **Khởi động lại** máy tính
4. **Mở Docker Desktop** và đợi nó khởi động hoàn toàn

### Bước 2: Build Docker Image

```bash
# Cách 1: Sử dụng batch file (Windows)
build_docker.bat

# Cách 2: Sử dụng command line
docker-compose build
```

### Bước 3: Chạy ứng dụng

```bash
# Cách 1: Sử dụng batch file (Windows)
run_docker.bat

# Cách 2: Sử dụng command line
docker-compose up
```

## 🎯 Các tính năng trong Docker

### ✅ **Essentia Integration**
- Key detection chuyên nghiệp
- Music analysis algorithms
- High accuracy key estimation

### ✅ **Audio Separator UI**
- Tích hợp sẵn Audio Separator
- Vocal separation với chất lượng cao
- Support nhiều format audio

### ✅ **Complete System**
- GUI interface
- Advanced scoring system
- All dependencies included

## 🔧 Troubleshooting

### Lỗi "Docker not running"
```bash
# Khởi động Docker Desktop
# Đợi cho đến khi Docker icon xanh trong system tray
```

### Lỗi "Port already in use"
```bash
# Thay đổi port trong docker-compose.yml
ports:
  - "8001:8000"  # Thay vì 8000:8000
```

### Lỗi "Out of memory"
```bash
# Tăng RAM cho Docker trong Settings
# Docker Desktop > Settings > Resources > Memory
```

## 📁 File Structure trong Docker

```
/app/
├── main.py                    # Main application
├── gui.py                     # GUI interface
├── advanced_audio_processor.py # Audio processing
├── advanced_key_detector.py   # Key detection with Essentia
├── scoring_system.py          # Scoring system
├── Audio_separator_ui/        # Audio Separator
└── temp_output/               # Temporary files
```

## 🎵 Sử dụng Essentia

```python
# Trong Docker container
import essentia.standard as es

# Key detection
loader = es.MonoLoader(filename='audio.wav')
audio = loader()
key_extractor = es.KeyExtractor()
key, scale, strength = key_extractor(audio)

print(f"Key: {key} {scale}, Strength: {strength}")
```

## 🔄 Development Mode

```bash
# Chạy với volume mounting để development
docker-compose -f docker-compose.dev.yml up
```

## 📊 Performance

- **Build time**: ~10-15 phút (lần đầu)
- **Startup time**: ~30 giây
- **Memory usage**: ~2GB
- **CPU usage**: Tùy thuộc vào audio processing

## 🆘 Support

Nếu gặp vấn đề:
1. Kiểm tra Docker Desktop đang chạy
2. Kiểm tra RAM và CPU usage
3. Restart Docker Desktop
4. Rebuild image: `docker-compose build --no-cache`

