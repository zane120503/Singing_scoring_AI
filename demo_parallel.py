#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Parallel Key Detection - Chạy thử song song
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_parallel_demo():
    """Chạy demo song song với file có sẵn"""
    
    print("🚀 DEMO SONG SONG KEY DETECTION")
    print("=" * 50)
    
    # Import optimized workflow
    try:
        from optimized_middle_workflow import run_workflow
        print("✅ Import optimized_middle_workflow thành công")
    except ImportError as e:
        print(f"❌ Lỗi import: {e}")
        return False
    
    # Tìm file test có sẵn
    test_files = [
        "D:\\singing scoring AI\\assets\\audio\\test_stereo.wav",
        "D:\\singing scoring AI\\assets\\audio\\test.mp3",
        "assets\\audio\\test_stereo.wav",
        "assets\\audio\\test.mp3"
    ]
    
    karaoke_file = None
    beat_file = None
    
    for file_path in test_files:
        if os.path.exists(file_path):
            if file_path.endswith('.wav'):
                karaoke_file = file_path
            elif file_path.endswith('.mp3'):
                beat_file = file_path
    
    if not karaoke_file or not beat_file:
        print("❌ Không tìm thấy file test")
        print("Vui lòng đặt file audio vào thư mục assets/audio/")
        return False
    
    print(f"📁 Karaoke file: {karaoke_file}")
    print(f"📁 Beat file: {beat_file}")
    
    # Chạy workflow song song
    print("\n⚡ Chạy workflow song song...")
    try:
        result = run_workflow(karaoke_file, beat_file, duration=30.0)
        
        if result and result.get("success"):
            print("\n🎉 THÀNH CÔNG!")
            print(f"✅ Vocals key: {result['vocals_key']['key']}")
            print(f"✅ Beat key: {result['beat_key']['key']}")
            print(f"✅ Key match: {result['key_compare']['match']}")
            print(f"✅ Score: {result['key_compare']['score']}")
            return True
        else:
            print(f"❌ Thất bại: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_parallel_demo()
    
    if success:
        print("\n🎊 Demo hoàn tất! Song song hoạt động tốt!")
    else:
        print("\n🔧 Cần kiểm tra lại cấu hình.")
    
    input("\nNhấn Enter để thoát...")
