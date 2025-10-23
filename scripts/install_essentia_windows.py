#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script để cài đặt Essentia trên Windows với các phương pháp khác nhau
"""

import subprocess
import sys
import os

def run_command(command):
    """Chạy command và hiển thị output"""
    print(f"🔄 Chạy: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Thành công!")
            if result.stdout:
                print(f"Output: {result.stdout}")
        else:
            print(f"❌ Lỗi: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def install_essentia_windows():
    """Cài đặt Essentia trên Windows"""
    print("🎵 CÀI ĐẶT ESSENTIA TRÊN WINDOWS")
    print("=" * 40)
    
    # Phương pháp 1: Cập nhật pip và setuptools
    print("\n📦 Bước 1: Cập nhật pip và setuptools...")
    run_command("python -m pip install --upgrade pip")
    run_command("python -m pip install --upgrade setuptools wheel")
    
    # Phương pháp 2: Thử cài đặt từ conda-forge
    print("\n📦 Bước 2: Thử cài đặt từ conda-forge...")
    if run_command("conda install -c conda-forge essentia"):
        print("✅ Cài đặt Essentia thành công từ conda-forge!")
        return True
    
    # Phương pháp 3: Thử cài đặt pre-built wheel
    print("\n📦 Bước 3: Thử cài đặt pre-built wheel...")
    if run_command("pip install essentia-tensorflow"):
        print("✅ Cài đặt essentia-tensorflow thành công!")
        return True
    
    # Phương pháp 4: Thử cài đặt từ wheel file
    print("\n📦 Bước 4: Thử cài đặt từ wheel file...")
    wheel_urls = [
        "https://github.com/MTG/essentia/releases/download/v2.1b6.dev1030/essentia-2.1b6.dev1030-cp39-cp39-win_amd64.whl",
        "https://github.com/MTG/essentia/releases/download/v2.1b6.dev1030/essentia-2.1b6.dev1030-cp38-cp38-win_amd64.whl",
        "https://github.com/MTG/essentia/releases/download/v2.1b6.dev1030/essentia-2.1b6.dev1030-cp37-cp37-win_amd64.whl"
    ]
    
    for url in wheel_urls:
        print(f"🔄 Thử wheel: {url}")
        if run_command(f"pip install {url}"):
            print("✅ Cài đặt Essentia thành công từ wheel!")
            return True
    
    # Phương pháp 5: Fallback - sử dụng Docker
    print("\n📦 Bước 5: Fallback - sử dụng Docker...")
    print("⚠️ Nếu tất cả phương pháp trên thất bại, bạn có thể:")
    print("   1. Sử dụng Docker: docker run -it mtgupf/essentia")
    print("   2. Sử dụng WSL2 với Ubuntu")
    print("   3. Sử dụng hệ thống fallback (không cần Essentia)")
    
    return False

def test_essentia_installation():
    """Test Essentia installation"""
    print("\n🧪 Test Essentia installation...")
    try:
        import essentia.standard as es
        print("✅ Essentia đã được cài đặt thành công!")
        
        # Test KeyExtractor
        key_extractor = es.KeyExtractor()
        print("✅ Essentia KeyExtractor sẵn sàng!")
        
        return True
    except ImportError as e:
        print(f"❌ Essentia chưa được cài đặt: {e}")
        return False

def main():
    """Hàm main"""
    print("🎤 ESSENTIA WINDOWS INSTALLATION HELPER")
    print("=" * 50)
    
    # Kiểm tra Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("⚠️ Cần Python 3.7 trở lên!")
        return
    
    # Cài đặt Essentia
    success = install_essentia_windows()
    
    if success:
        # Test installation
        test_essentia_installation()
        print("\n🎉 Cài đặt Essentia thành công!")
    else:
        print("\n⚠️ Không thể cài đặt Essentia.")
        print("💡 Hệ thống sẽ sử dụng fallback method (không cần Essentia)")

if __name__ == "__main__":
    main()
