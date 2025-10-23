# Singing Scoring AI - Project Structure

## 📁 Cấu trúc thư mục

```
singing scoring AI/
├── src/                    # Source code chính
│   ├── ai/                # AI processing modules
│   ├── core/              # Core functionality
│   ├── gui/               # GUI components
│   └── utils/             # Utility functions
├── tests/                 # Test files được tổ chức
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests  
│   ├── performance/       # Performance tests
│   ├── audio/             # Audio processing tests
│   └── fixtures/          # Test data và demo files
├── assets/                # Tất cả assets
│   ├── models/            # AI models (.onnx files)
│   ├── audio/             # Sample audio files
│   └── data/              # Data files
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── docs/                  # Documentation
└── output/                # Output files
```

## 🧪 Chạy tests

```bash
# Chạy tất cả tests
python -m pytest tests/

# Chạy theo loại
python -m pytest tests/unit/        # Unit tests
python -m pytest tests/integration/ # Integration tests
python -m pytest tests/audio/       # Audio tests
python -m pytest tests/performance/ # Performance tests
```

## 📝 Ghi chú

- Tất cả test files đã được tổ chức theo chức năng
- Assets được tập trung trong thư mục `assets/`
- Output files được lưu trong `output/`
- Scripts tiện ích trong `scripts/`
