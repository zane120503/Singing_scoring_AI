#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test New Voice Detector
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_new_voice_detector():
    """Test New Voice Detector"""
    try:
        from src.ai.new_voice_detector import NewVoiceDetector
        
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file khong ton tai: {test_file}")
            return False
        
        logger.info("Testing New Voice Detector...")
        logger.info(f"File: {test_file}")
        
        # Khoi tao detector
        detector = NewVoiceDetector()
        
        # Test voice detection
        segments = detector.detect_voice_activity(test_file)
        
        logger.info(f"Phat hien {len(segments)} voice segments")
        
        if segments:
            for i, segment in enumerate(segments[:3]):  # Hien thi 3 doan dau
                logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f}, method: {segment['method']})")
            
            # Kiem tra vi tri dau tien
            first_segment = segments[0]
            detected_start = first_segment['start']
            
            logger.info(f"\nKet qua phat hien:")
            logger.info(f"   Detected voice start: {detected_start:.2f}s")
            logger.info(f"   Segment duration: {first_segment['end'] - first_segment['start']:.2f}s")
            
            # Kiem tra xem co hop ly khong
            if 10.0 < detected_start < 60.0:  # Giong hat thuong bat dau trong khoang 10-60 giay
                logger.info(f"Vi tri phat hien hop ly: {detected_start:.2f}s")
                return True
            else:
                logger.warning(f"Vi tri phat hien co the khong chinh xac: {detected_start:.2f}s")
                return False
        else:
            logger.warning("  Khong phat hien voice segments")
            return False
        
    except Exception as e:
        logger.error(f"Loi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== TEST NEW VOICE DETECTOR ===")
    
    # Test new voice detector
    print("\n1. Testing New Voice Detector...")
    success = test_new_voice_detector()
    
    # Ket qua
    print("\n=== KET QUA ===")
    if success:
        print("NEW VOICE DETECTOR HOAT DONG CHINH XAC!")
        print("He thong da phat hien chinh xac vi tri giong hat!")
    else:
        print("CAN CAI THIEN THEM!")
        print("Van con van de voi voice detection!")
