#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tổ chức lại cấu trúc project cho gọn gàng
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Tạo cấu trúc thư mục mới"""
    base_path = Path(__file__).parent.parent
    
    # Tạo các thư mục mới
    new_dirs = [
        "tests/unit",
        "tests/integration", 
        "tests/performance",
        "tests/audio",
        "tests/fixtures",
        "assets/models",
        "assets/audio",
        "assets/data",
        "output"
    ]
    
    for dir_path in new_dirs:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {full_path}")

def categorize_test_files():
    """Phân loại và di chuyển các file test"""
    base_path = Path(__file__).parent.parent
    
    # Mapping file test -> thư mục đích
    test_mapping = {
        # Unit tests
        "test_core.py": "tests/unit/",
        "test_structure.py": "tests/unit/",
        "test_libraries.py": "tests/unit/",
        
        # Integration tests
        "test_ai_audio_separator_system.py": "tests/integration/",
        "test_audio_separator_integration.py": "tests/integration/",
        "test_full_workflow.py": "tests/integration/",
        "test_custom_wrapper.py": "tests/integration/",
        
        # Performance tests
        "test_speed_optimization.py": "tests/performance/",
        "test_speed.py": "tests/performance/",
        "test_gpu_usage.py": "tests/performance/",
        "test_gpu.py": "tests/performance/",
        
        # Audio processing tests
        "test_audio_gpu.py": "tests/audio/",
        "test_audio_separator_real.py": "tests/audio/",
        "test_audio_separator_simple.py": "tests/audio/",
        "test_beat_key_detection.py": "tests/audio/",
        "test_hybrid_key_detector.py": "tests/audio/",
        "test_single_mp3.py": "tests/audio/",
        "test_single_mp3_only.py": "tests/audio/",
        "test_vocals_path.py": "tests/audio/",
        "test_waiting_for_you_beat.py": "tests/audio/",
        "test_gpu_audio_separator.py": "tests/audio/",
        "test_gpu_simple.py": "tests/audio/",
        
        # Docker tests
        "test_docker_essentia.py": "tests/integration/",
        "test_enhanced_docker_essentia.py": "tests/integration/",
    }
    
    # Di chuyển các file test
    for filename, target_dir in test_mapping.items():
        source_path = base_path / filename
        target_path = base_path / target_dir / filename
        
        if source_path.exists():
            shutil.move(str(source_path), str(target_path))
            print(f"Moved: {filename} -> {target_dir}")
        else:
            print(f"Not found: {filename}")

def organize_assets():
    """Tổ chức lại các assets"""
    base_path = Path(__file__).parent.parent
    
    # Di chuyển models
    models_source = base_path / "models"
    models_target = base_path / "assets" / "models"
    
    if models_source.exists():
        if models_target.exists():
            shutil.rmtree(models_target)
        shutil.move(str(models_source), str(models_target))
        print(f"Moved: models -> assets/models")
    
    # Di chuyển data
    data_source = base_path / "data"
    data_target = base_path / "assets" / "data"
    
    if data_source.exists():
        if data_target.exists():
            # Merge contents
            for item in data_source.iterdir():
                shutil.move(str(item), str(data_target / item.name))
            data_source.rmdir()
        else:
            shutil.move(str(data_source), str(data_target))
        print(f"Moved: data -> assets/data")
    
    # Di chuyển input_files
    input_source = base_path / "input_files"
    input_target = base_path / "assets" / "audio"
    
    if input_source.exists():
        for item in input_source.iterdir():
            if item.is_file():
                shutil.move(str(item), str(input_target / item.name))
        input_source.rmdir()
        print(f"Moved: input_files -> assets/audio")

def organize_audio_separator():
    """Tổ chức Audio_separator_ui"""
    base_path = Path(__file__).parent.parent
    audio_sep_path = base_path / "Audio_separator_ui"
    
    if audio_sep_path.exists():
        # Di chuyển models
        models_source = audio_sep_path / "mdx_models"
        models_target = base_path / "assets" / "models" / "mdx_models"
        
        if models_source.exists():
            shutil.move(str(models_source), str(models_target))
            print(f"Moved: Audio_separator_ui/mdx_models -> assets/models/mdx_models")
        
        # Di chuyển test audio files
        test_files = ["test.mp3", "test_stereo.wav"]
        for test_file in test_files:
            source_file = audio_sep_path / test_file
            if source_file.exists():
                target_file = base_path / "assets" / "audio" / test_file
                shutil.move(str(source_file), str(target_file))
                print(f"Moved: {test_file} -> assets/audio/")
        
        # Di chuyển output
        output_source = audio_sep_path / "clean_song_output"
        output_target = base_path / "output" / "clean_song_output"
        
        if output_source.exists():
            shutil.move(str(output_source), str(output_target))
            print(f"Moved: clean_song_output -> output/")

def consolidate_tests_directory():
    """Gộp thư mục tests cũ vào cấu trúc mới"""
    base_path = Path(__file__).parent.parent
    old_tests = base_path / "tests"
    
    if old_tests.exists():
        # Di chuyển các file từ tests cũ
        for test_file in old_tests.iterdir():
            if test_file.is_file() and test_file.suffix == '.py':
                # Phân loại dựa trên tên file
                if "demo" in test_file.name.lower():
                    target_dir = "tests/fixtures"
                elif "integration" in test_file.name.lower() or "workflow" in test_file.name.lower():
                    target_dir = "tests/integration"
                elif "performance" in test_file.name.lower() or "speed" in test_file.name.lower():
                    target_dir = "tests/performance"
                else:
                    target_dir = "tests/unit"
                
                target_path = base_path / target_dir / test_file.name
                shutil.move(str(test_file), str(target_path))
                print(f"Moved: tests/{test_file.name} -> {target_dir}/")
        
        # Xóa thư mục tests cũ nếu trống
        try:
            old_tests.rmdir()
            print("Removed old tests directory")
        except OSError:
            print("Old tests directory not empty, keeping it")

def create_readme():
    """Tạo README cho cấu trúc mới"""
    base_path = Path(__file__).parent.parent
    readme_content = """# Singing Scoring AI - Project Structure

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
"""
    
    readme_path = base_path / "PROJECT_STRUCTURE.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"Created: {readme_path}")

def main():
    """Ham chinh"""
    print("To chuc lai cau truc project...")
    print("=" * 50)
    
    # Tao cau truc thu muc moi
    print("\n1. Tao cau truc thu muc moi...")
    create_directory_structure()
    
    # Phan loai va di chuyen test files
    print("\n2. Phan loai va di chuyen test files...")
    categorize_test_files()
    
    # To chuc assets
    print("\n3. To chuc assets...")
    organize_assets()
    
    # To chuc Audio_separator_ui
    print("\n4. To chuc Audio_separator_ui...")
    organize_audio_separator()
    
    # Gop thu muc tests cu
    print("\n5. Gop thu muc tests cu...")
    consolidate_tests_directory()
    
    # Tao README
    print("\n6. Tao documentation...")
    create_readme()
    
    print("\nHoan thanh to chuc lai project!")
    print("\nCau truc moi da duoc tao. Vui long kiem tra va dieu chinh neu can.")

if __name__ == "__main__":
    main()
