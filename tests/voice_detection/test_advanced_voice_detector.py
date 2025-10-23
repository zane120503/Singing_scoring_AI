#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Advanced Voice Detector
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_advanced_voice_detector():
    """Test Advanced Voice Detector"""
    try:
        from src.ai.advanced_voice_detector import AdvancedVoiceDetector
        
        # Test file
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\f26e80366d7c184105_mdx\\input.wav"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🎤 Testing Advanced Voice Detector...")
        logger.info(f"File: {test_file}")
        
        # Khởi tạo detector
        detector = AdvancedVoiceDetector()
        
        # Test các phương pháp khác nhau
        methods = ["auto", "pyannote", "silero", "webrtc", "fallback"]
        
        for method in methods:
            logger.info(f"\n--- Testing method: {method} ---")
            segments = detector.detect_voice_activity_advanced(test_file, method=method)
            logger.info(f"Phát hiện {len(segments)} voice segments")
            
            if segments:
                for i, segment in enumerate(segments[:3]):  # Hiển thị 3 đoạn đầu
                    logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f}, method: {segment['method']})")
            else:
                logger.warning("  Không phát hiện voice segments")
        
        # Test voice analysis
        logger.info(f"\n--- Testing voice analysis ---")
        analysis = detector.get_voice_analysis(test_file)
        
        if "error" not in analysis:
            logger.info(f"📊 Voice Analysis:")
            logger.info(f"   Total duration: {analysis['total_duration']:.2f}s")
            logger.info(f"   Voice duration: {analysis['voice_duration']:.2f}s")
            logger.info(f"   Voice ratio: {analysis['voice_ratio']:.1%}")
            logger.info(f"   Voice segments: {analysis['voice_count']}")
            
            if analysis['first_voice']['start'] > 0:
                logger.info(f"   First voice: {analysis['first_voice']['start']:.2f}s - {analysis['first_voice']['end']:.2f}s")
        else:
            logger.error(f"Voice analysis failed: {analysis['error']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_availability():
    """Test availability của các models"""
    try:
        logger.info("🔍 Testing model availability...")
        
        # Test pyannote.audio
        try:
            from pyannote.audio import Pipeline
            logger.info("✅ pyannote.audio available")
        except ImportError:
            logger.warning("⚠️ pyannote.audio not available")
        
        # Test webrtcvad
        try:
            import webrtcvad
            logger.info("✅ webrtcvad available")
        except ImportError:
            logger.warning("⚠️ webrtcvad not available")
        
        # Test torch
        try:
            import torch
            logger.info("✅ torch available")
        except ImportError:
            logger.warning("⚠️ torch not available")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi test model availability: {e}")
        return False

def test_voice_detection_accuracy():
    """Test độ chính xác của voice detection"""
    try:
        logger.info("🎯 Testing voice detection accuracy...")
        
        from src.ai.advanced_voice_detector import AdvancedVoiceDetector
        
        # Test file
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\f26e80366d7c184105_mdx\\input.wav"
        
        detector = AdvancedVoiceDetector()
        
        # Test với method auto (tốt nhất)
        segments = detector.detect_voice_activity_advanced(test_file, method="auto")
        
        if segments:
            logger.info(f"✅ Voice detection successful: {len(segments)} segments")
            
            # Kiểm tra chất lượng segments
            for i, segment in enumerate(segments):
                duration = segment['end'] - segment['start']
                confidence = segment['confidence']
                method = segment['method']
                
                logger.info(f"  Segment {i+1}: {duration:.2f}s, confidence: {confidence:.2f}, method: {method}")
            
            # Tìm đoạn voice đầu tiên
            first_voice = detector.find_first_voice_segment(test_file, min_duration=1.0)
            if first_voice['start'] > 0:
                logger.info(f"🎯 First voice segment: {first_voice['start']:.2f}s - {first_voice['end']:.2f}s")
                return True
            else:
                logger.warning("⚠️ No suitable voice segment found")
                return False
        else:
            logger.error("❌ No voice segments detected")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi test accuracy: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST ADVANCED VOICE DETECTOR ===")
    
    # Test model availability
    print("\n1. Testing model availability...")
    model_availability = test_model_availability()
    
    # Test advanced voice detector
    print("\n2. Testing Advanced Voice Detector...")
    detector_success = test_advanced_voice_detector()
    
    # Test accuracy
    print("\n3. Testing voice detection accuracy...")
    accuracy_success = test_voice_detection_accuracy()
    
    # Kết quả
    print("\n=== KET QUA ===")
    print(f"Model Availability: {'OK' if model_availability else 'ERROR'}")
    print(f"Advanced Detector: {'OK' if detector_success else 'ERROR'}")
    print(f"Detection Accuracy: {'OK' if accuracy_success else 'ERROR'}")
    
    if all([model_availability, detector_success, accuracy_success]):
        print("\nADVANCED VOICE DETECTOR HOAT DONG TOT!")
    else:
        print("\nCAN CAI THIEN VOICE DETECTION!")
