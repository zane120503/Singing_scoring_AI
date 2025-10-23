#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script cài đặt dependencies cho Voice Detection
"""

import subprocess
import sys
import os

def install_package(package):
    """Cài đặt package"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Cài đặt các dependencies cần thiết"""
    print("=== INSTALLING VOICE DETECTION DEPENDENCIES ===")
    
    # Danh sách packages cần cài đặt
    packages = [
        "pyannote.audio",
        "webrtcvad",
        "torch",
        "torchaudio",
        "librosa",
        "soundfile",
        "numpy",
        "scipy"
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n=== INSTALLATION SUMMARY ===")
    print(f"Successfully installed: {success_count}/{total_count} packages")
    
    if success_count == total_count:
        print("🎉 All packages installed successfully!")
    else:
        print("⚠️ Some packages failed to install")
    
    # Test imports
    print("\n=== TESTING IMPORTS ===")
    
    try:
        import pyannote.audio
        print("✅ pyannote.audio imported successfully")
    except ImportError:
        print("❌ pyannote.audio import failed")
    
    try:
        import webrtcvad
        print("✅ webrtcvad imported successfully")
    except ImportError:
        print("❌ webrtcvad import failed")
    
    try:
        import torch
        print("✅ torch imported successfully")
    except ImportError:
        print("❌ torch import failed")

if __name__ == "__main__":
    main()
