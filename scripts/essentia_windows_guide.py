#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual installation guide cho Essentia trên Windows
"""

def print_manual_installation_guide():
    """In hướng dẫn cài đặt thủ công"""
    print("🛠️ HƯỚNG DẪN CÀI ĐẶT ESSENTIA THỦ CÔNG TRÊN WINDOWS")
    print("=" * 60)
    
    print("\n📋 BƯỚC 1: Cài đặt Visual Studio Build Tools")
    print("   1. Tải Visual Studio Build Tools từ:")
    print("      https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("   2. Chọn 'Desktop development with C++'")
    print("   3. Đảm bảo chọn:")
    print("      - MSVC v143 - VS 2022 C++ x64/x86 build tools")
    print("      - Windows 11 SDK (latest)")
    print("      - CMake tools for Visual Studio")
    
    print("\n📋 BƯỚC 2: Cài đặt CMake")
    print("   1. Tải CMake từ: https://cmake.org/download/")
    print("   2. Chọn 'Add CMake to system PATH'")
    
    print("\n📋 BƯỚC 3: Cài đặt Git")
    print("   1. Tải Git từ: https://git-scm.com/download/win")
    print("   2. Đảm bảo Git được thêm vào PATH")
    
    print("\n📋 BƯỚC 4: Cài đặt Essentia từ source")
    print("   Mở Command Prompt as Administrator và chạy:")
    print("   ```")
    print("   git clone https://github.com/MTG/essentia.git")
    print("   cd essentia")
    print("   python waf configure --build-static --with-example=streaming_extractor_music")
    print("   python waf")
    print("   python waf install")
    print("   ```")
    
    print("\n📋 BƯỚC 5: Test installation")
    print("   ```")
    print("   python -c \"import essentia.standard as es; print('Success!')\"")
    print("   ```")

def print_alternative_solutions():
    """In các giải pháp thay thế"""
    print("\n🔄 CÁC GIẢI PHÁP THAY THẾ")
    print("=" * 30)
    
    print("\n🐳 1. SỬ DỤNG DOCKER")
    print("   ```")
    print("   docker pull mtgupf/essentia")
    print("   docker run -it mtgupf/essentia")
    print("   ```")
    
    print("\n🐧 2. SỬ DỤNG WSL2")
    print("   ```")
    print("   wsl --install")
    print("   # Trong WSL2 Ubuntu:")
    print("   sudo apt update")
    print("   sudo apt install python3-pip")
    print("   pip install essentia")
    print("   ```")
    
    print("\n☁️ 3. SỬ DỤNG CLOUD")
    print("   - Google Colab")
    print("   - Kaggle Notebooks")
    print("   - AWS/GCP instances")
    
    print("\n🔄 4. SỬ DỤNG FALLBACK")
    print("   Hệ thống sẽ tự động sử dụng phương pháp truyền thống")
    print("   khi Essentia không khả dụng")

def print_quick_fixes():
    """In các fix nhanh"""
    print("\n⚡ CÁC FIX NHANH")
    print("=" * 20)
    
    print("\n🔧 Fix 1: Cập nhật pip")
    print("   python -m pip install --upgrade pip")
    
    print("\n🔧 Fix 2: Cài đặt Visual C++ Redistributable")
    print("   Tải từ: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    
    print("\n🔧 Fix 3: Sử dụng conda")
    print("   conda install -c conda-forge essentia")
    
    print("\n🔧 Fix 4: Sử dụng pre-built wheel")
    print("   pip install essentia-tensorflow")
    
    print("\n🔧 Fix 5: Sử dụng wheel từ GitHub")
    print("   pip install https://github.com/MTG/essentia/releases/download/v2.1b6.dev1030/essentia-2.1b6.dev1030-cp39-cp39-win_amd64.whl")

def main():
    """Hàm main"""
    print_manual_installation_guide()
    print_alternative_solutions()
    print_quick_fixes()
    
    print("\n💡 KHUYẾN NGHỊ:")
    print("   1. Thử các fix nhanh trước")
    print("   2. Nếu không được, sử dụng Docker hoặc WSL2")
    print("   3. Cuối cùng, hệ thống sẽ hoạt động với fallback method")

if __name__ == "__main__":
    main()
