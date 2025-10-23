#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Voice Detection với file Waiting For You
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import numpy as np
import librosa
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_waiting_for_you():
    """Test Voice Detection với file Waiting For You"""
    try:
        from src.ai.smart_voice_detector import SmartVoiceDetector
        from src.ai.karaoke_voice_detector import KaraokeVoiceDetector
        from src.ai.advanced_voice_detector import AdvancedVoiceDetector
        
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You"
        
        # Tìm file audio trong thư mục
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']
        audio_file = None
        
        for ext in audio_extensions:
            potential_file = test_file + ext
            if os.path.exists(potential_file):
                audio_file = potential_file
                break
        
        if not audio_file:
            # Tìm trong thư mục
            if os.path.exists(test_file):
                for file in os.listdir(test_file):
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        audio_file = os.path.join(test_file, file)
                        break
        
        if not audio_file:
            logger.error(f"Không tìm thấy file audio trong: {test_file}")
            return False
        
        logger.info(f"🎤 Testing Voice Detection với file: {audio_file}")
        
        # Load audio để phân tích
        audio, sr = librosa.load(audio_file, sr=22050)
        duration = len(audio) / sr
        
        logger.info(f"Audio info: {duration:.2f}s, {sr} Hz, {len(audio)} samples")
        
        # Test với các detectors khác nhau
        detectors = {
            "Smart Voice Detector": SmartVoiceDetector(),
            "Karaoke Voice Detector": KaraokeVoiceDetector(),
            "Advanced Voice Detector": AdvancedVoiceDetector()
        }
        
        results = {}
        
        for detector_name, detector in detectors.items():
            logger.info(f"\n--- Testing {detector_name} ---")
            
            try:
                if detector_name == "Advanced Voice Detector":
                    segments = detector.detect_voice_activity_advanced(audio_file, method="webrtc")
                else:
                    segments = detector.detect_voice_activity(audio_file)
                
                logger.info(f"Phát hiện {len(segments)} voice segments")
                
                if segments:
                    for i, segment in enumerate(segments[:5]):  # Hiển thị 5 đoạn đầu
                        logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f})")
                
                results[detector_name] = segments
                
            except Exception as e:
                logger.error(f"Lỗi với {detector_name}: {e}")
                results[detector_name] = []
        
        # So sánh kết quả
        logger.info(f"\n📊 So sánh kết quả:")
        for detector_name, segments in results.items():
            if segments:
                first_segment = segments[0]
                logger.info(f"   {detector_name}: First segment {first_segment['start']:.2f}s - {first_segment['end']:.2f}s")
            else:
                logger.info(f"   {detector_name}: No segments detected")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_audio_manually():
    """Phân tích audio thủ công để tìm vị trí giọng hát thực sự"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You"
        
        # Tìm file audio
        audio_extensions = ['.mp3', '.wav', '.m4a', '.flac']
        audio_file = None
        
        for ext in audio_extensions:
            potential_file = test_file + ext
            if os.path.exists(potential_file):
                audio_file = potential_file
                break
        
        if not audio_file:
            if os.path.exists(test_file):
                for file in os.listdir(test_file):
                    if any(file.lower().endswith(ext) for ext in audio_extensions):
                        audio_file = os.path.join(test_file, file)
                        break
        
        if not audio_file:
            logger.error(f"Không tìm thấy file audio trong: {test_file}")
            return False
        
        logger.info(f"🔍 Phân tích audio thủ công: {audio_file}")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=22050)
        
        # Tính toán các features
        rms = librosa.feature.rms(y=audio)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Tìm các đoạn có energy cao
        energy_threshold = np.percentile(rms, 20)
        voice_frames = np.where(rms > energy_threshold)[0]
        
        if len(voice_frames) > 0:
            # Tìm đoạn liên tục đầu tiên
            segments = []
            current_start = voice_frames[0]
            current_end = voice_frames[0]
            
            for i in range(1, len(voice_frames)):
                if voice_frames[i] - voice_frames[i-1] <= 2:  # Liên tục
                    current_end = voice_frames[i]
                else:
                    segments.append((current_start, current_end))
                    current_start = voice_frames[i]
                    current_end = voice_frames[i]
            
            segments.append((current_start, current_end))
            
            logger.info(f"📊 Phân tích chi tiết:")
            logger.info(f"   Total voice frames: {len(voice_frames)}")
            logger.info(f"   Continuous segments: {len(segments)}")
            
            for i, (start, end) in enumerate(segments[:10]):
                start_time = start * 512 / sr  # hop_length = 512
                end_time = end * 512 / sr
                duration = end_time - start_time
                logger.info(f"     Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích audio: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST WAITING FOR YOU ===")
    
    # Test voice detection
    print("\n1. Testing Voice Detection...")
    detection_success = test_waiting_for_you()
    
    # Analyze audio manually
    print("\n2. Analyzing audio manually...")
    analysis_success = analyze_audio_manually()
    
    # Kết quả
    print("\n=== KET QUA ===")
    print(f"Voice Detection: {'OK' if detection_success else 'ERROR'}")
    print(f"Audio Analysis: {'OK' if analysis_success else 'ERROR'}")
    
    if detection_success and analysis_success:
        print("\nDA KIEM TRA FILE WAITING FOR YOU!")
    else:
        print("\nCO VAN DE VOI FILE WAITING FOR YOU!")
