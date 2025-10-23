#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để chạy tests theo loại
"""

import sys
import os
import subprocess
from pathlib import Path

def run_tests(test_type=None):
    """Chạy tests theo loại"""
    base_path = Path(__file__).parent.parent
    tests_dir = base_path / "tests"
    
    if test_type:
        # Chạy tests theo loại cụ thể
        test_path = tests_dir / test_type
        if test_path.exists():
            print(f"Chay {test_type} tests...")
            cmd = [sys.executable, "-m", "pytest", str(test_path), "-v"]
        else:
            print(f"Khong tim thay thu muc tests: {test_path}")
            return False
    else:
        # Chạy tất cả tests
        print("Chay tat ca tests...")
        cmd = [sys.executable, "-m", "pytest", str(tests_dir), "-v"]
    
    try:
        result = subprocess.run(cmd, cwd=str(base_path))
        return result.returncode == 0
    except Exception as e:
        print(f"Loi khi chay tests: {e}")
        return False

def list_test_categories():
    """Liệt kê các loại test có sẵn"""
    base_path = Path(__file__).parent.parent
    tests_dir = base_path / "tests"
    
    print("Cac loai test co san:")
    if tests_dir.exists():
        for subdir in tests_dir.iterdir():
            if subdir.is_dir():
                test_count = len(list(subdir.glob("*.py")))
                print(f"  - {subdir.name}: {test_count} files")

def main():
    """Ham chinh"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1]
        
        if test_type == "list":
            list_test_categories()
            return
        
        valid_types = ["unit", "integration", "performance", "audio", "fixtures"]
        if test_type in valid_types:
            success = run_tests(test_type)
            if success:
                print(f"\n{test_type.upper()} TESTS THANH CONG!")
            else:
                print(f"\n{test_type.upper()} TESTS THAT BAI!")
        else:
            print(f"Loai test khong hop le: {test_type}")
            print("Cac loai hop le:", ", ".join(valid_types))
    else:
        # Chạy tất cả tests
        success = run_tests()
        if success:
            print("\nTAT CA TESTS THANH CONG!")
        else:
            print("\nMOT SO TESTS THAT BAI!")

if __name__ == "__main__":
    main()
