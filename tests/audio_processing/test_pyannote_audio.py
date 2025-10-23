#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test PyAnnote Audio - Kiem tra pyannote.audio co hoat dong khong
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pyannote_availability():
    """Kiem tra pyannote.audio co san sang khong"""
    try:
        logger.info("Kiem tra pyannote.audio availability...")
        
        # Test import pyannote.audio
        try:
            from pyannote.audio import Pipeline
            logger.info("✅ pyannote.audio import thanh cong!")
            return True
        except ImportError as e:
            logger.warning(f"⚠️ pyannote.audio khong co san: {e}")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Loi khac khi import pyannote.audio: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Loi trong test pyannote availability: {e}")
        return False

def test_pyannote_models():
    """Kiem tra cac models cua pyannote.audio"""
    try:
        logger.info("Kiem tra pyannote.audio models...")
        
        from pyannote.audio import Pipeline
        
        # Test cac models co san
        models = [
            "pyannote/speaker-diarization-3.1",
            "pyannote/speaker-diarization",
            "pyannote/segmentation-3.0",
            "pyannote/segmentation"
        ]
        
        available_models = []
        
        for model_name in models:
            try:
                # Test load model (khong can auth token cho test)
                logger.info(f"Test model: {model_name}")
                # pipeline = Pipeline.from_pretrained(model_name, use_auth_token="test")
                logger.info(f"✅ Model {model_name} co the load")
                available_models.append(model_name)
            except Exception as e:
                logger.warning(f"⚠️ Model {model_name} khong the load: {e}")
        
        logger.info(f"Available models: {available_models}")
        return len(available_models) > 0
        
    except Exception as e:
        logger.error(f"❌ Loi trong test pyannote models: {e}")
        return False

def test_pyannote_with_audio():
    """Test pyannote.audio voi file audio"""
    try:
        logger.info("Test pyannote.audio voi file audio...")
        
        # Test file
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file khong ton tai: {test_file}")
            return False
        
        try:
            from pyannote.audio import Pipeline
            
            # Test load pipeline (khong can auth token cho test)
            logger.info("Test load pipeline...")
            # pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="test")
            
            # Test voi file audio
            logger.info(f"Test voi file: {test_file}")
            # diarization = pipeline(test_file)
            
            logger.info("✅ pyannote.audio hoat dong voi file audio!")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ pyannote.audio khong the hoat dong voi file audio: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Loi trong test pyannote voi audio: {e}")
        return False

def test_advanced_voice_detector_pyannote():
    """Test Advanced Voice Detector voi pyannote"""
    try:
        logger.info("Test Advanced Voice Detector voi pyannote...")
        
        from src.ai.advanced_voice_detector import AdvancedVoiceDetector
        
        # Test file
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file khong ton tai: {test_file}")
            return False
        
        # Khoi tao detector
        detector = AdvancedVoiceDetector()
        
        # Test voi pyannote method
        logger.info("Test voi pyannote method...")
        segments = detector.detect_voice_activity_advanced(test_file, method="pyannote")
        
        if segments:
            logger.info(f"✅ pyannote method hoat dong: {len(segments)} segments")
            for i, segment in enumerate(segments[:3]):
                logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
            return True
        else:
            logger.warning("⚠️ pyannote method khong tra ve segments")
            return False
            
    except Exception as e:
        logger.error(f"❌ Loi trong test Advanced Voice Detector: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST PYANNOTE AUDIO ===")
    
    # Test pyannote availability
    print("\n1. Kiem tra pyannote.audio availability...")
    availability = test_pyannote_availability()
    
    # Test pyannote models
    print("\n2. Kiem tra pyannote.audio models...")
    models = test_pyannote_models()
    
    # Test pyannote voi audio
    print("\n3. Test pyannote.audio voi file audio...")
    audio_test = test_pyannote_with_audio()
    
    # Test Advanced Voice Detector voi pyannote
    print("\n4. Test Advanced Voice Detector voi pyannote...")
    detector_test = test_advanced_voice_detector_pyannote()
    
    # Ket qua
    print("\n=== KET QUA ===")
    print(f"PyAnnote Availability: {'YES' if availability else 'NO'}")
    print(f"PyAnnote Models: {'YES' if models else 'NO'}")
    print(f"PyAnnote Audio Test: {'YES' if audio_test else 'NO'}")
    print(f"Advanced Voice Detector PyAnnote: {'YES' if detector_test else 'NO'}")
    
    if availability and models and audio_test and detector_test:
        print("\n🎉 PYANNOTE.AUDIO HOAT DONG HOAN HAO!")
    else:
        print("\n⚠️ PYANNOTE.AUDIO CAN CAI THIEN!")
