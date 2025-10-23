#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Middle Audio Slicer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_middle_audio_slicer():
    """Test Middle Audio Slicer"""
    try:
        from src.ai.middle_audio_slicer import MiddleAudioSlicer
        
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file khong ton tai: {test_file}")
            return False
        
        logger.info("Testing Middle Audio Slicer...")
        logger.info(f"File: {test_file}")
        
        # Khoi tao slicer
        slicer = MiddleAudioSlicer()
        
        # Test cắt 20 giây ở giữa
        result = slicer.slice_middle_audio_with_info(test_file, duration=20.0)
        
        if result["success"]:
            logger.info(f"\nKet qua cắt audio:")
            logger.info(f"   Input file: {result['input_file']}")
            logger.info(f"   Output file: {result['output_file']}")
            logger.info(f"   Duration: {result['duration']}s")
            logger.info(f"   Start time: {result['start_time']:.2f}s")
            logger.info(f"   End time: {result['end_time']:.2f}s")
            logger.info(f"   Total duration: {result['total_duration']:.2f}s")
            
            # Audio features
            features = result["audio_features"]
            logger.info(f"\nAudio features của đoạn đã cắt:")
            logger.info(f"   RMS: {features['rms']:.4f}")
            logger.info(f"   Spectral Centroid: {features['spectral_centroid']:.1f} Hz")
            logger.info(f"   Spectral Rolloff: {features['spectral_rolloff']:.1f} Hz")
            logger.info(f"   Zero Crossing Rate: {features['zero_crossing_rate']:.4f}")
            
            return True
        else:
            logger.error(f"Loi khi cat audio: {result['error']}")
            return False
        
    except Exception as e:
        logger.error(f"Loi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== TEST MIDDLE AUDIO SLICER ===")
    
    # Test middle audio slicer
    print("\n1. Testing Middle Audio Slicer...")
    success = test_middle_audio_slicer()
    
    # Ket qua
    print("\n=== KET QUA ===")
    if success:
        print("MIDDLE AUDIO SLICER HOAT DONG CHINH XAC!")
        print("He thong da cat thanh cong 20 giay o giua file!")
    else:
        print("CAN CAI THIEN THEM!")
        print("Van con van de voi audio slicing!")

