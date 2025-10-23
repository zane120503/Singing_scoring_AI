#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để sửa lỗi file lock trong audio processing
"""

import os
import sys
import time
import shutil
from pathlib import Path

def check_and_clean_temp_files():
    """Kiểm tra và dọn dẹp các file temp"""
    base_path = Path(__file__).parent.parent
    
    # Các thư mục cần kiểm tra
    temp_dirs = [
        base_path / "Audio_separator_ui" / "clean_song_output",
        base_path / "output" / "clean_song_output",
        base_path / "data" / "temp_output"
    ]
    
    print("Kiem tra va don dep cac file temp...")
    
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            print(f"\nKiểm tra thư mục: {temp_dir}")
            
            # Liệt kê tất cả file trong thư mục
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        # Thử mở file để kiểm tra lock
                        with open(file_path, 'rb') as f:
                            f.read(1)
                        print(f"  ✅ {file_path.name} - OK")
                    except Exception as e:
                        print(f"  ❌ {file_path.name} - Lỗi: {e}")
                        
                        # Thử xóa file bị lock
                        try:
                            file_path.unlink()
                            print(f"  🗑️ Đã xóa file bị lock: {file_path.name}")
                        except Exception as e2:
                            print(f"  ⚠️ Không thể xóa: {e2}")
        else:
            print(f"\nThư mục không tồn tại: {temp_dir}")

def create_fresh_output_dirs():
    """Tạo lại các thư mục output"""
    base_path = Path(__file__).parent.parent
    
    output_dirs = [
        base_path / "output" / "clean_song_output",
        base_path / "data" / "temp_output"
    ]
    
    print("\nTạo lại các thư mục output...")
    
    for output_dir in output_dirs:
        try:
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"🗑️ Đã xóa thư mục cũ: {output_dir}")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Đã tạo thư mục: {output_dir}")
        except Exception as e:
            print(f"❌ Lỗi khi tạo thư mục {output_dir}: {e}")

def check_audio_separator_paths():
    """Kiểm tra đường dẫn Audio Separator"""
    base_path = Path(__file__).parent.parent
    
    print("\nKiểm tra đường dẫn Audio Separator...")
    
    # Kiểm tra Audio_separator_ui
    audio_sep_path = base_path / "Audio_separator_ui"
    if audio_sep_path.exists():
        print(f"✅ Audio_separator_ui tồn tại: {audio_sep_path}")
        
        # Kiểm tra models
        models_path = audio_sep_path / "mdx_models"
        if models_path.exists():
            print(f"✅ Models tồn tại: {models_path}")
        else:
            print(f"❌ Models không tồn tại: {models_path}")
    else:
        print(f"❌ Audio_separator_ui không tồn tại: {audio_sep_path}")
    
    # Kiểm tra assets/models
    assets_models_path = base_path / "assets" / "models" / "mdx_models"
    if assets_models_path.exists():
        print(f"✅ Assets models tồn tại: {assets_models_path}")
    else:
        print(f"❌ Assets models không tồn tại: {assets_models_path}")

def main():
    """Hàm chính"""
    print("=== SUA LOI FILE LOCK TRONG AUDIO PROCESSING ===")
    
    # Kiểm tra và dọn dẹp file temp
    check_and_clean_temp_files()
    
    # Tạo lại thư mục output
    create_fresh_output_dirs()
    
    # Kiểm tra đường dẫn
    check_audio_separator_paths()
    
    print("\n✅ Hoàn thành sửa lỗi file lock!")

if __name__ == "__main__":
    main()
