#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Voice Detection với file mới
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

def test_voice_detection_new_file():
    """Test Voice Detection với file mới"""
    try:
        from src.ai.advanced_voice_detector import AdvancedVoiceDetector
        
        # Test file mới
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\62b6c4008acb2bcafc_mdx\\input.wav"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🎤 Testing Voice Detection với file mới...")
        logger.info(f"File: {test_file}")
        
        # Load audio để phân tích
        audio, sr = librosa.load(test_file, sr=22050)
        duration = len(audio) / sr
        
        logger.info(f"Audio info: {duration:.2f}s, {sr} Hz, {len(audio)} samples")
        
        # Khởi tạo detector
        detector = AdvancedVoiceDetector(sr)
        
        # Test các phương pháp khác nhau
        methods = ["webrtc", "auto", "fallback"]
        
        for method in methods:
            logger.info(f"\n--- Testing method: {method} ---")
            segments = detector.detect_voice_activity_advanced(test_file, method=method)
            logger.info(f"Phát hiện {len(segments)} voice segments")
            
            if segments:
                for i, segment in enumerate(segments[:5]):  # Hiển thị 5 đoạn đầu
                    logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f}, method: {segment['method']})")
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

def analyze_audio_manually_new_file():
    """Phân tích audio thủ công để hiểu tại sao không phát hiện voice"""
    try:
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\62b6c4008acb2bcafc_mdx\\input.wav"
        
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
        
        # Phân tích chi tiết hơn
        logger.info(f"\n🔍 Detailed Analysis:")
        
        # Tìm các đoạn có energy cao liên tục
        energy_threshold = np.percentile(rms, 30)
        voice_frames = np.where(rms > energy_threshold)[0]
        
        if len(voice_frames) > 0:
            logger.info(f"   Voice frames (30th percentile): {len(voice_frames)} frames")
            
            # Tìm các đoạn liên tục
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
            
            logger.info(f"   Continuous voice segments: {len(segments)}")
            for i, (start, end) in enumerate(segments[:5]):
                start_time = start * 512 / sr  # hop_length = 512
                end_time = end * 512 / sr
                duration = end_time - start_time
                logger.info(f"     Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích audio: {e}")
        return False

def create_improved_voice_detector():
    """Tạo Voice Activity Detector cải thiện cho file mới"""
    try:
        logger.info("🔧 Tạo Voice Activity Detector cải thiện...")
        
        improved_detector_code = '''
def _detect_voice_activity_improved_v2(self, audio: np.ndarray, sr: int) -> List[Dict]:
    """Phát hiện voice activity cải thiện v2 - dành cho file karaoke"""
    try:
        # 1. RMS Energy Analysis với threshold thấp hơn
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # Sử dụng percentile thấp hơn để phát hiện voice nhẹ
        rms_threshold = np.percentile(rms, 15)  # Thấp hơn nữa
        
        # 2. Spectral Centroid Analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Voice frequency range (100-3500 Hz) - mở rộng hơn
        voice_low = 100
        voice_high = 3500
        
        # 3. Zero Crossing Rate Analysis
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        # Voice có ZCR trung bình
        zcr_low = np.percentile(zcr, 5)   # Thấp hơn
        zcr_high = np.percentile(zcr, 95) # Cao hơn
        
        # 4. Spectral Rolloff Analysis
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        rolloff_threshold = np.percentile(spectral_rolloff, 20)  # Thấp hơn
        
        # 5. MFCC Analysis (cho voice characteristics)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)[0]
        
        # 6. Spectral Bandwidth Analysis
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        bandwidth_threshold = np.percentile(spectral_bandwidth, 25)
        
        # Detect voice frames
        voice_frames = []
        for i in range(len(rms)):
            # Điều kiện phát hiện voice - cần ít nhất 2/6 điều kiện
            rms_ok = rms[i] > rms_threshold
            centroid_ok = voice_low < spectral_centroids[i] < voice_high
            zcr_ok = zcr_low < zcr[i] < zcr_high
            rolloff_ok = spectral_rolloff[i] > rolloff_threshold
            bandwidth_ok = spectral_bandwidth[i] > bandwidth_threshold
            
            # Cần ít nhất 2/5 điều kiện
            if sum([rms_ok, centroid_ok, zcr_ok, rolloff_ok, bandwidth_ok]) >= 2:
                voice_frames.append(i)
        
        return self._frames_to_segments(voice_frames, sr)
        
    except Exception as e:
        logger.warning(f"Improved voice detection v2 failed: {e}")
        return []
'''
        
        # Ghi vào file
        with open("D:\\singing scoring AI\\improved_voice_detector_v2.py", "w", encoding="utf-8") as f:
            f.write(improved_detector_code)
        
        logger.info("✅ Improved Voice Detector v2 code created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi tạo improved detector v2: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST VOICE DETECTION NEW FILE ===")
    
    # Test voice detection
    print("\n1. Testing Voice Detection...")
    detection_success = test_voice_detection_new_file()
    
    # Analyze audio manually
    print("\n2. Analyzing audio manually...")
    analysis_success = analyze_audio_manually_new_file()
    
    # Create improved detector
    print("\n3. Creating improved detector v2...")
    improved_success = create_improved_voice_detector()
    
    # Kết quả
    print("\n=== KET QUA ===")
    print(f"Voice Detection: {'OK' if detection_success else 'ERROR'}")
    print(f"Audio Analysis: {'OK' if analysis_success else 'ERROR'}")
    print(f"Improved Detector v2: {'OK' if improved_success else 'ERROR'}")
    
    if detection_success and analysis_success:
        print("\nCAN CAI THIEN VOICE DETECTION CHO FILE MOI!")
    else:
        print("\nCO VAN DE VOI VOICE DETECTION!")
