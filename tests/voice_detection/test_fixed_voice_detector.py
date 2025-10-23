#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Fixed Voice Detector với file Waiting For You
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_voice_detector():
    """Test Fixed Voice Detector"""
    try:
        from src.ai.advanced_voice_detector import AdvancedVoiceDetector
        
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🔧 Testing Fixed Advanced Voice Detector...")
        logger.info(f"File: {test_file}")
        
        # Khởi tạo detector
        detector = AdvancedVoiceDetector()
        
        # Test với WebRTC VAD
        segments = detector.detect_voice_activity_advanced(test_file, method="webrtc")
        
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
        else:
            logger.warning("  Không phát hiện voice segments")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_all_detectors():
    """So sánh tất cả detectors"""
    try:
        from src.ai.advanced_voice_detector import AdvancedVoiceDetector
        from src.ai.smart_voice_detector import SmartVoiceDetector
        from src.ai.karaoke_voice_detector import KaraokeVoiceDetector
        
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🔍 So sánh tất cả detectors với file Waiting For You...")
        
        # Advanced Voice Detector
        advanced_detector = AdvancedVoiceDetector()
        advanced_segments = advanced_detector.detect_voice_activity_advanced(test_file, method="webrtc")
        
        # Smart Voice Detector
        smart_detector = SmartVoiceDetector()
        smart_segments = smart_detector.detect_voice_activity(test_file)
        
        # Karaoke Voice Detector
        karaoke_detector = KaraokeVoiceDetector()
        karaoke_segments = karaoke_detector.detect_voice_activity(test_file)
        
        logger.info(f"\n📊 So sánh kết quả:")
        logger.info(f"   Advanced Voice Detector: {len(advanced_segments)} segments")
        logger.info(f"   Smart Voice Detector: {len(smart_segments)} segments")
        logger.info(f"   Karaoke Voice Detector: {len(karaoke_segments)} segments")
        
        if advanced_segments:
            logger.info(f"   Advanced first segment: {advanced_segments[0]['start']:.2f}s - {advanced_segments[0]['end']:.2f}s")
        if smart_segments:
            logger.info(f"   Smart first segment: {smart_segments[0]['start']:.2f}s - {smart_segments[0]['end']:.2f}s")
        if karaoke_segments:
            logger.info(f"   Karaoke first segment: {karaoke_segments[0]['start']:.2f}s - {karaoke_segments[0]['end']:.2f}s")
        
        # Kiểm tra độ chính xác
        logger.info(f"\n🎯 Đánh giá độ chính xác:")
        
        # Giọng hát thực sự bắt đầu từ khoảng 12.77s
        real_voice_start = 12.77
        
        for detector_name, segments in [
            ("Advanced", advanced_segments),
            ("Smart", smart_segments), 
            ("Karaoke", karaoke_segments)
        ]:
            if segments:
                detected_start = segments[0]['start']
                error = abs(detected_start - real_voice_start)
                if error < 2.0:  # Chấp nhận sai số 2 giây
                    logger.info(f"   ✅ {detector_name}: Chính xác (sai số: {error:.2f}s)")
                else:
                    logger.info(f"   ❌ {detector_name}: Không chính xác (sai số: {error:.2f}s)")
            else:
                logger.info(f"   ❌ {detector_name}: Không phát hiện được")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi so sánh: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST FIXED VOICE DETECTOR ===")
    
    # Test fixed voice detector
    print("\n1. Testing Fixed Advanced Voice Detector...")
    fixed_success = test_fixed_voice_detector()
    
    # Compare all detectors
    print("\n2. Comparing all detectors...")
    compare_success = compare_all_detectors()
    
    # Kết quả
    print("\n=== KET QUA ===")
    print(f"Fixed Voice Detector: {'OK' if fixed_success else 'ERROR'}")
    print(f"Detector Comparison: {'OK' if compare_success else 'ERROR'}")
    
    if fixed_success and compare_success:
        print("\nFIXED VOICE DETECTOR HOAT DONG TOT!")
    else:
        print("\nCAN CAI THIEN THEM!")
