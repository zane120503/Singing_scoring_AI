#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test cải thiện Voice Activity Detection
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

def test_voice_detection_improved():
    """Test cải thiện Voice Activity Detection"""
    try:
        from src.ai.voice_activity_detector import VoiceActivityDetector
        
        # Test file
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\f26e80366d7c184105_mdx\\input.wav"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🎤 Testing improved Voice Activity Detection...")
        logger.info(f"File: {test_file}")
        
        # Load audio để phân tích
        audio, sr = librosa.load(test_file, sr=22050)
        duration = len(audio) / sr
        
        logger.info(f"Audio info: {duration:.2f}s, {sr} Hz, {len(audio)} samples")
        
        # Khởi tạo detector
        detector = VoiceActivityDetector(sr)
        
        # Test các phương pháp khác nhau
        methods = ["spectral", "energy", "zero_crossing", "combined"]
        
        for method in methods:
            logger.info(f"\n--- Testing method: {method} ---")
            segments = detector.detect_voice_activity(test_file, method=method)
            logger.info(f"Phát hiện {len(segments)} voice segments")
            
            if segments:
                for i, segment in enumerate(segments[:5]):  # Hiển thị 5 đoạn đầu
                    logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f})")
            else:
                logger.warning("  Không phát hiện voice segments")
        
        # Tìm đoạn voice đầu tiên phù hợp
        first_voice = detector.find_first_voice_segment(test_file, min_duration=1.0)
        if first_voice['start'] > 0:
            logger.info(f"\n🎯 First suitable voice segment: {first_voice['start']:.2f}s - {first_voice['end']:.2f}s")
        else:
            logger.warning("⚠️ Không tìm thấy voice segment phù hợp")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_audio_manually():
    """Phân tích audio thủ công để hiểu tại sao không phát hiện voice"""
    try:
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\f26e80366d7c184105_mdx\\input.wav"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🔍 Phân tích audio thủ công...")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        
        # Tính toán các features
        rms = librosa.feature.rms(y=audio)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Tính toán thống kê
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_threshold = rms_mean + 0.5 * rms_std
        
        centroid_mean = np.mean(spectral_centroids)
        centroid_std = np.std(spectral_centroids)
        
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        logger.info(f"📊 Audio Analysis:")
        logger.info(f"   RMS Energy: mean={rms_mean:.4f}, std={rms_std:.4f}, threshold={rms_threshold:.4f}")
        logger.info(f"   Spectral Centroid: mean={centroid_mean:.1f} Hz, std={centroid_std:.1f} Hz")
        logger.info(f"   Zero Crossing Rate: mean={zcr_mean:.4f}, std={zcr_std:.4f}")
        
        # Tìm các frame có energy cao
        high_energy_frames = np.where(rms > rms_threshold)[0]
        if len(high_energy_frames) > 0:
            logger.info(f"   High energy frames: {len(high_energy_frames)} frames")
            logger.info(f"   First high energy frame: {high_energy_frames[0]}")
            logger.info(f"   Last high energy frame: {high_energy_frames[-1]}")
        else:
            logger.warning("   Không tìm thấy high energy frames")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích audio: {e}")
        return False

def create_improved_voice_detector():
    """Tạo Voice Activity Detector cải thiện"""
    try:
        logger.info("🔧 Tạo Voice Activity Detector cải thiện...")
        
        improved_detector_code = '''
def _detect_voice_activity_improved(self, audio: np.ndarray, sr: int) -> List[Dict]:
    """Phát hiện voice activity cải thiện"""
    try:
        # 1. RMS Energy Analysis
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # Adaptive threshold - thấp hơn để phát hiện voice nhẹ
        rms_threshold = np.percentile(rms, 20)  # Sử dụng percentile thấp hơn
        
        # 2. Spectral Centroid Analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Voice frequency range (80-3000 Hz)
        voice_low = 80
        voice_high = 3000
        
        # 3. Zero Crossing Rate Analysis
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Voice có ZCR trung bình
        zcr_low = np.percentile(zcr, 10)
        zcr_high = np.percentile(zcr, 90)
        
        # 4. Spectral Rolloff Analysis
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        rolloff_threshold = np.percentile(spectral_rolloff, 30)
        
        # 5. MFCC Analysis (cho voice characteristics)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)[0]
        
        # Detect voice frames
        voice_frames = []
        for i in range(len(rms)):
            # Điều kiện phát hiện voice
            rms_ok = rms[i] > rms_threshold
            centroid_ok = voice_low < spectral_centroids[i] < voice_high
            zcr_ok = zcr_low < zcr[i] < zcr_high
            rolloff_ok = spectral_rolloff[i] > rolloff_threshold
            
            # Cần ít nhất 2/4 điều kiện
            if sum([rms_ok, centroid_ok, zcr_ok, rolloff_ok]) >= 2:
                voice_frames.append(i)
        
        return self._frames_to_segments(voice_frames, sr)
        
    except Exception as e:
        logger.warning(f"Improved voice detection failed: {e}")
        return []
'''
        
        # Ghi vào file
        with open("D:\\singing scoring AI\\improved_voice_detector.py", "w", encoding="utf-8") as f:
            f.write(improved_detector_code)
        
        logger.info("✅ Improved Voice Detector code created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi tạo improved detector: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST VOICE DETECTION IMPROVED ===")
    
    # Test voice detection
    print("\n1. Testing Voice Activity Detection...")
    detection_success = test_voice_detection_improved()
    
    # Analyze audio manually
    print("\n2. Analyzing audio manually...")
    analysis_success = analyze_audio_manually()
    
    # Create improved detector
    print("\n3. Creating improved detector...")
    improved_success = create_improved_voice_detector()
    
    # Kết quả
    print("\n=== KET QUA ===")
    print(f"Voice Detection: {'OK' if detection_success else 'ERROR'}")
    print(f"Audio Analysis: {'OK' if analysis_success else 'ERROR'}")
    print(f"Improved Detector: {'OK' if improved_success else 'ERROR'}")
    
    if detection_success and analysis_success:
        print("\nCAN CAI THIEN VOICE DETECTION!")
    else:
        print("\nCO VAN DE VOI VOICE DETECTION!")
