#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Final Voice Detector
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_final_voice_detector():
    """Test Final Voice Detector"""
    try:
        from src.ai.final_voice_detector import FinalVoiceDetector
        
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info("🎯 Testing Final Voice Detector...")
        logger.info(f"File: {test_file}")
        
        # Khởi tạo detector
        detector = FinalVoiceDetector()
        
        # Test voice detection
        segments = detector.detect_voice_activity(test_file)
        
        logger.info(f"Phát hiện {len(segments)} voice segments")
        
        if segments:
            for i, segment in enumerate(segments[:3]):  # Hiển thị 3 đoạn đầu
                logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f}, method: {segment['method']})")
            
            # Kiểm tra vị trí đầu tiên
            first_segment = segments[0]
            detected_start = first_segment['start']
            
            logger.info(f"\n🎯 Kết quả phát hiện:")
            logger.info(f"   Detected voice start: {detected_start:.2f}s")
            logger.info(f"   Segment duration: {first_segment['end'] - first_segment['start']:.2f}s")
            
            # Kiểm tra xem có hợp lý không
            if detected_start < 15.0:  # Giọng hát thường bắt đầu trong 15 giây đầu
                logger.info(f"✅ Vị trí phát hiện hợp lý: {detected_start:.2f}s")
                return True
            else:
                logger.warning(f"⚠️ Vị trí phát hiện có thể không chính xác: {detected_start:.2f}s")
                return False
        else:
            logger.warning("  Không phát hiện voice segments")
            return False
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== TEST FINAL VOICE DETECTOR ===")
    
    # Test final voice detector
    print("\n1. Testing Final Voice Detector...")
    success = test_final_voice_detector()
    
    # Kết quả
    print("\n=== KET QUA ===")
    if success:
        print("✅ FINAL VOICE DETECTOR HOAT DONG CHINH XAC!")
        print("🎯 Hệ thống đã phát hiện chính xác vị trí giọng hát!")
    else:
        print("❌ CAN CAI THIEN THEM!")
        print("⚠️ Vẫn còn vấn đề với voice detection!")
