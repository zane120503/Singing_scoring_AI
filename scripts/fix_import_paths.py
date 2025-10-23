#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để sửa các import paths sau khi tổ chức lại cấu trúc
"""

import os
import re
from pathlib import Path

def fix_import_paths_in_file(file_path):
    """Sửa import paths trong một file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Sửa sys.path.insert để trỏ đúng vị trí src
        # Từ: sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        # Thành: sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
        
        old_pattern = r"sys\.path\.insert\(0,\s*os\.path\.join\(os\.path\.dirname\(__file__\),\s*'src'\)\)"
        new_path = "sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))"
        
        content = re.sub(old_pattern, new_path, content)
        
        # Sửa các đường dẫn test file
        # Từ: "D:\\singing scoring AI\\Audio_separator_ui\\test.mp3"
        # Thành: "D:\\singing scoring AI\\assets\\audio\\test.mp3"
        
        old_test_path = r"D:\\\\singing scoring AI\\\\Audio_separator_ui\\\\test\.mp3"
        new_test_path = "D:\\\\singing scoring AI\\\\assets\\\\audio\\\\test.mp3"
        content = re.sub(old_test_path, new_test_path, content)
        
        # Sửa đường dẫn test_stereo.wav
        old_stereo_path = r"D:\\\\singing scoring AI\\\\Audio_separator_ui\\\\test_stereo\.wav"
        new_stereo_path = "D:\\\\singing scoring AI\\\\assets\\\\audio\\\\test_stereo.wav"
        content = re.sub(old_stereo_path, new_stereo_path, content)
        
        # Nếu có thay đổi, ghi lại file
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed imports in: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def fix_all_test_files():
    """Sửa tất cả file test"""
    base_path = Path(__file__).parent.parent
    test_dirs = [
        base_path / "tests" / "unit",
        base_path / "tests" / "integration", 
        base_path / "tests" / "performance",
        base_path / "tests" / "audio",
        base_path / "tests" / "fixtures"
    ]
    
    fixed_count = 0
    total_count = 0
    
    for test_dir in test_dirs:
        if test_dir.exists():
            for py_file in test_dir.glob("*.py"):
                total_count += 1
                if fix_import_paths_in_file(py_file):
                    fixed_count += 1
    
    print(f"\nFixed {fixed_count} out of {total_count} test files")

def main():
    """Ham chinh"""
    print("Sua cac import paths trong test files...")
    fix_all_test_files()
    print("Hoan thanh!")

if __name__ == "__main__":
    main()
