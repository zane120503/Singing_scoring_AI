"""
Test tốc độ Fast Mode vs AI Mode
"""

import time
import os
from advanced_audio_processor import AdvancedAudioProcessor

def test_speed():
    """Test tốc độ giữa Fast Mode và AI Mode"""
    
    # File test (thay đổi đường dẫn này)
    test_file = "test.mp3"  # Thay đổi thành file của bạn
    
    if not os.path.exists(test_file):
        print(f"Khong tim thay file test: {test_file}")
        print("Vui long dat file test.mp3 trong thu muc hien tai")
        return
    
    print("Testing toc do tach giong...")
    print("=" * 50)
    
    # Test Fast Mode
    print("Testing Fast Mode...")
    fast_processor = AdvancedAudioProcessor(fast_mode=True)
    
    start_time = time.time()
    try:
        fast_result = fast_processor.separate_vocals(test_file)
        fast_time = time.time() - start_time
        print(f"Fast Mode: {fast_time:.2f} giay")
        print(f"   Output: {fast_result}")
    except Exception as e:
        print(f"Fast Mode failed: {e}")
        fast_time = None
    
    print()
    
    # Test AI Mode
    print("Testing AI Mode...")
    ai_processor = AdvancedAudioProcessor(fast_mode=False)
    
    start_time = time.time()
    try:
        ai_result = ai_processor.separate_vocals(test_file)
        ai_time = time.time() - start_time
        print(f"AI Mode: {ai_time:.2f} giay")
        print(f"   Output: {ai_result}")
    except Exception as e:
        print(f"AI Mode failed: {e}")
        ai_time = None
    
    print()
    print("=" * 50)
    print("KET QUA SO SANH:")
    
    if fast_time and ai_time:
        speedup = ai_time / fast_time
        print(f"Fast Mode: {fast_time:.2f} giay")
        print(f"AI Mode: {ai_time:.2f} giay")
        print(f"Fast Mode nhanh hon {speedup:.1f}x")
        
        if speedup > 5:
            print("Fast Mode dang ke nhanh hon!")
        elif speedup > 2:
            print("Fast Mode nhanh hon dang ke")
        else:
            print("Chenh lech khong nhieu")
    else:
        print("Khong the so sanh do loi")

if __name__ == "__main__":
    test_speed()
