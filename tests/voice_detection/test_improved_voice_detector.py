#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Improved Smart Voice Detector
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_improved_voice_detector():
    """Test Improved Smart Voice Detector"""
    try:
        from src.ai.improved_smart_voice_detector import ImprovedSmartVoiceDetector
        
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info("🧠 Testing Improved Smart Voice Detector...")
        logger.info(f"File: {test_file}")
        
        # Khởi tạo detector
        detector = ImprovedSmartVoiceDetector()
        
        # Test voice detection
        segments = detector.detect_voice_activity(test_file)
        
        logger.info(f"Phát hiện {len(segments)} voice segments")
        
        if segments:
            for i, segment in enumerate(segments[:5]):  # Hiển thị 5 đoạn đầu
                logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f}, method: {segment['method']})")
            
            # Kiểm tra vị trí đầu tiên
            first_segment = segments[0]
            
            # Kiểm tra độ chính xác
            real_voice_start = 7.82  # Vị trí thực sự từ phân tích trước
            detected_start = first_segment['start']
            error = abs(detected_start - real_voice_start)
            
            logger.info(f"\n🎯 Đánh giá độ chính xác:")
            logger.info(f"   Real voice start: {real_voice_start:.2f}s")
            logger.info(f"   Detected start: {detected_start:.2f}s")
            logger.info(f"   Error: {error:.2f}s")
            
            if error < 2.0:
                logger.info(f"✅ CHÍNH XÁC (sai số: {error:.2f}s)")
                return True
            else:
                logger.warning(f"❌ KHÔNG CHÍNH XÁC (sai số: {error:.2f}s)")
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
    print("=== TEST IMPROVED SMART VOICE DETECTOR ===")
    
    # Test improved voice detector
    print("\n1. Testing Improved Smart Voice Detector...")
    success = test_improved_voice_detector()
    
    # Kết quả
    print("\n=== KET QUA ===")
    if success:
        print("✅ IMPROVED SMART VOICE DETECTOR HOAT DONG CHINH XAC!")
        print("🎯 Hệ thống đã phát hiện chính xác vị trí giọng hát!")
    else:
        print("❌ CAN CAI THIEN THEM!")
        print("⚠️ Vẫn còn vấn đề với voice detection!")
