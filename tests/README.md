# Test Structure

Thư mục `tests` được tổ chức theo chức năng và loại test:

## 📁 Cấu trúc thư mục

### 🎤 `voice_detection/`
Test các thuật toán phát hiện giọng hát:
- `test_accurate_voice_detector.py` - Detector chính xác
- `test_advanced_voice_detector.py` - Detector nâng cao
- `test_correct_voice_detector.py` - Detector đã sửa
- `test_final_voice_detector.py` - Detector cuối cùng
- `test_smart_voice_detector*.py` - Detector thông minh
- `test_pyannote_voice_detector.py` - PyAnnote detector
- `test_*_voice_detector.py` - Các detector khác

### 🔄 `workflow/`
Test các workflow xử lý:
- `test_optimized_workflow.py` - Workflow tối ưu
- `test_correct_workflow.py` - Workflow đã sửa
- `test_final_workflow.py` - Workflow cuối cùng
- `test_smart_optimized_workflow.py` - Workflow thông minh
- `test_gpu_workflow.py` - Workflow GPU

### 🎹 `key_detection/`
Test phát hiện phím âm nhạc:
- `test_vocals_key_detection.py` - Key detection cho vocals
- `test_vocals_key_detection_new.py` - Key detection mới

### 🎵 `audio_processing/`
Test xử lý âm thanh:
- `test_audio_slice_analysis.py` - Phân tích audio slice
- `test_middle_audio_slicer.py` - Audio slicer
- `test_smart_audio_processor.py` - Audio processor thông minh
- `test_pyannote_audio.py` - PyAnnote audio

### 🎵 `audio/`
Test audio separator và GPU:
- `test_audio_separator_*.py` - Audio separator
- `test_gpu_*.py` - GPU processing
- `test_hybrid_key_detector.py` - Hybrid key detector
- `test_beat_key_detection.py` - Beat key detection

### 🔧 `integration/`
Test tích hợp hệ thống:
- `test_full_workflow.py` - Workflow đầy đủ
- `test_docker_essentia.py` - Docker Essentia
- `test_ai_audio_separator_system.py` - AI Audio Separator

### ⚡ `performance/`
Test hiệu suất:
- `test_speed*.py` - Test tốc độ
- `test_gpu*.py` - Test GPU performance

### 🧪 `unit/`
Test đơn vị:
- `test_core.py` - Core functionality
- `test_libraries.py` - Library tests
- `test_structure.py` - Structure tests

### 📦 `fixtures/`
Demo và test data:
- `demo*.py` - Demo files
- Test fixtures và sample data

### 🗂️ `legacy/`
Test cũ và deprecated:
- `test_analyze*.py` - Analysis tests cũ
- `test_find*.py` - Find tests cũ
- `test_waiting*.py` - Waiting tests cũ

## 🚀 Cách chạy test

```bash
# Chạy tất cả test
python -m pytest tests/

# Chạy test theo thư mục
python -m pytest tests/voice_detection/
python -m pytest tests/workflow/
python -m pytest tests/key_detection/

# Chạy test cụ thể
python -m pytest tests/workflow/test_optimized_workflow.py
```

## 📝 Ghi chú

- **Voice Detection**: Tập trung vào các thuật toán phát hiện giọng hát
- **Workflow**: Test các pipeline xử lý hoàn chỉnh
- **Key Detection**: Test phát hiện phím âm nhạc
- **Audio Processing**: Test xử lý âm thanh cơ bản
- **Integration**: Test tích hợp các component
- **Performance**: Test hiệu suất và tốc độ
- **Legacy**: Test cũ, có thể deprecated trong tương lai
