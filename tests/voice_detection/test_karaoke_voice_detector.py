#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Karaoke Voice Detector
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_karaoke_voice_detector():
    """Test Karaoke Voice Detector"""
    try:
        from src.ai.karaoke_voice_detector import KaraokeVoiceDetector
        
        # Test file mới
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\62b6c4008acb2bcafc_mdx\\input.wav"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🎤 Testing Karaoke Voice Detector...")
        logger.info(f"File: {test_file}")
        
        # Khởi tạo detector
        detector = KaraokeVoiceDetector()
        
        # Test voice detection
        segments = detector.detect_voice_activity(test_file)
        logger.info(f"Phát hiện {len(segments)} voice segments")
        
        if segments:
            for i, segment in enumerate(segments[:5]):  # Hiển thị 5 đoạn đầu
                logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f}, method: {segment['method']})")
        else:
            logger.warning("  Không phát hiện voice segments")
        
        # Tìm đoạn voice đầu tiên phù hợp
        first_voice = detector.find_first_voice_segment(test_file, min_duration=1.0)
        if first_voice['start'] > 0:
            logger.info(f"\n🎯 First suitable karaoke voice segment: {first_voice['start']:.2f}s - {first_voice['end']:.2f}s")
        else:
            logger.warning("⚠️ Không tìm thấy voice segment phù hợp")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_voice_detectors():
    """So sánh các voice detectors"""
    try:
        from src.ai.karaoke_voice_detector import KaraokeVoiceDetector
        from src.ai.advanced_voice_detector import AdvancedVoiceDetector
        
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\62b6c4008acb2bcafc_mdx\\input.wav"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🔍 So sánh các voice detectors...")
        
        # Karaoke Voice Detector
        karaoke_detector = KaraokeVoiceDetector()
        karaoke_segments = karaoke_detector.detect_voice_activity(test_file)
        
        # Advanced Voice Detector
        advanced_detector = AdvancedVoiceDetector()
        advanced_segments = advanced_detector.detect_voice_activity_advanced(test_file, method="webrtc")
        
        logger.info(f"\n📊 So sánh kết quả:")
        logger.info(f"   Karaoke Voice Detector: {len(karaoke_segments)} segments")
        logger.info(f"   Advanced Voice Detector: {len(advanced_segments)} segments")
        
        if karaoke_segments:
            logger.info(f"   Karaoke first segment: {karaoke_segments[0]['start']:.2f}s - {karaoke_segments[0]['end']:.2f}s")
        if advanced_segments:
            logger.info(f"   Advanced first segment: {advanced_segments[0]['start']:.2f}s - {advanced_segments[0]['end']:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi so sánh: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST KARAOKE VOICE DETECTOR actividades")
    
    # Test karaoke voice detector
    print("\n1. Testing Karaoke Voice Detector...")
    karaoke_success = test_karaoke_voice_detector()
    
    # Compare detectors
    print("\n2. Comparing voice detectors...")
    compare_success = compare_voice_detectors()
    
    # Kết quả
    print("\n=== KET QUA ===")
    print(f"Karaoke Voice Detector: {'OK' if karaoke_success else 'ERROR'}")
    print(f"Detector Comparison: {'OK' if compare_success else 'ERROR'}")
    
    if karaoke_success and compare_success:
        print("\nKARAOKE VOICE DETECTOR HOAT DONG TOT!")
    else:
        print("\nCAN CAI THIEN VOICE DETECTION!")
