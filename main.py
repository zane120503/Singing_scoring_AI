#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hệ thống chấm điểm giọng hát karaoke bằng AI
Karaoke Singing Score AI System

Tác giả: AI Assistant
Ngày tạo: 2024
"""

import sys
import os
import warnings

# Tắt warnings không cần thiết
warnings.filterwarnings("ignore")

# Thêm src vào Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    required_packages = [
        'torch', 'torchaudio', 'librosa', 'soundfile', 
        'numpy', 'matplotlib', 'tkinter', 'transformers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Thieu cac package sau:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nVui long cai dat bang lenh:")
        print("   pip install -r requirements.txt")
        return False
    
    print("Tat ca dependencies da duoc cai dat!")
    return True

def main():
    """Hàm main chính"""
    print("He thong cham diem giong hat karaoke bang AI")
    print("=" * 50)
    
    # Kiểm tra dependencies
    if not check_dependencies():
        input("\nNhấn Enter để thoát...")
        return
    
    try:
        # Import và chạy GUI
        from src.gui.gui import main as run_gui
        print("Dang khoi dong giao dien...")
        run_gui()
        
    except ImportError as e:
        print(f"Loi import: {e}")
        print("Vui long kiem tra lai cac file trong du an.")
        input("\nNhan Enter de thoat...")
        
    except Exception as e:
        print(f"Loi khong mong muon: {e}")
        input("\nNhan Enter de thoat...")

if __name__ == "__main__":
    main()

