#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Improved Advanced Voice Detector
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_improved_advanced_detector():
    """Test Improved Advanced Voice Detector"""
    try:
        from src.ai.advanced_voice_detector import AdvancedVoiceDetector
        
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🔧 Testing Improved Advanced Voice Detector...")
        logger.info(f"File: {test_file}")
        
        # Khởi tạo detector
        detector = AdvancedVoiceDetector()
        
        # Test với method "auto" (sẽ ưu tiên Smart Voice Detector)
        segments = detector.detect_voice_activity_advanced(test_file, method="auto")
        
        logger.info(f"Phát hiện {len(segments)} voice segments")
        
        if segments:
            for i, segment in enumerate(segments[:5]):  # Hiển thị 5 đoạn đầu
                logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f}, method: {segment['method']})")
            
            # Kiểm tra vị trí đầu tiên
            first_segment = segments[0]
            if first_segment['start'] < 5.0:
                logger.warning(f"⚠️ Vẫn phát hiện giọng hát từ {first_segment['start']:.2f}s - có thể sai!")
            else:
                logger.info(f"✅ Phát hiện giọng hát từ {first_segment['start']:.2f}s - có vẻ chính xác!")
                
            # Kiểm tra độ chính xác
            real_voice_start = 12.77
            detected_start = first_segment['start']
            error = abs(detected_start - real_voice_start)
            
            if error < 2.0:
                logger.info(f"🎯 Độ chính xác: CHÍNH XÁC (sai số: {error:.2f}s)")
                return True
            else:
                logger.warning(f"🎯 Độ chính xác: KHÔNG CHÍNH XÁC (sai số: {error:.2f}s)")
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
    print("=== TEST IMPROVED ADVANCED VOICE DETECTOR ===")
    
    # Test improved advanced voice detector
    print("\n1. Testing Improved Advanced Voice Detector...")
    success = test_improved_advanced_detector()
    
    # Kết quả
    print("\n=== KET QUA ===")
    if success:
        print("✅ IMPROVED ADVANCED VOICE DETECTOR HOAT DONG CHINH XAC!")
        print("🎯 Hệ thống đã phát hiện chính xác vị trí giọng hát!")
    else:
        print("❌ CAN CAI THIEN THEM!")
        print("⚠️ Vẫn còn vấn đề với voice detection!")
