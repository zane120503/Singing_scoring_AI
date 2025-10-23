#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Analyze Waiting For You - Phân tích chi tiết vị trí giọng hát thực sự
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_waiting_for_you_detailed():
    """Phân tích chi tiết file Waiting For You để tìm vị trí giọng hát thực sự"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info("🔍 Phân tích chi tiết file Waiting For You...")
        logger.info(f"File: {test_file}")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        duration = len(audio) / sr
        
        logger.info(f"Audio info: {duration:.2f}s, {sr} Hz, {len(audio)} samples")
        
        # Phân tích chi tiết từng giây
        hop_length = 512
        frame_length = 2048
        
        # Tính các features
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Chuyển frames thành thời gian
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Phân tích từng giây để tìm vị trí giọng hát thực sự
        logger.info(f"\n📊 Phân tích từng giây để tìm giọng hát thực sự:")
        
        # Phân tích 60 giây đầu (đủ để tìm giọng hát)
        for i in range(min(60, len(rms))):
            start_time = i * hop_length / sr
            end_time = min((i + 1) * hop_length / sr, duration)
            
            # Lấy features của giây này
            segment_rms = rms[i]
            segment_centroid = spectral_centroids[i]
            segment_rolloff = spectral_rolloff[i]
            segment_zcr = zcr[i]
            
            # Phân tích harmonic content
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Harmonic-percussive separation
            try:
                harmonic, percussive = librosa.effects.hpss(segment_audio)
                harmonic_energy = np.mean(np.abs(harmonic))
                percussive_energy = np.mean(np.abs(percussive))
                harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-8)
            except:
                harmonic_ratio = 0.5
            
            # MFCC để phát hiện voice characteristics
            try:
                mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfccs[0])
                mfcc_std = np.std(mfccs[0])
            except:
                mfcc_mean = 0
                mfcc_std = 0
            
            # Điều kiện để xác định có giọng hát (thresholds thấp hơn)
            has_voice = (
                segment_rms > 0.05 and  # Energy đủ cao (giảm threshold)
                segment_centroid > 800 and segment_centroid < 5000 and  # Voice frequency range rộng hơn
                segment_rolloff > 800 and  # Có high frequency content (giảm threshold)
                harmonic_ratio > 0.3 and  # Harmonic content (giảm threshold)
                mfcc_std > 10  # MFCC variation (giảm threshold)
            )
            
            # Log chi tiết cho 30 giây đầu
            if start_time < 30:
                logger.info(f"   {start_time:.1f}s: RMS={segment_rms:.4f}, Centroid={segment_centroid:.1f}Hz, Rolloff={segment_rolloff:.1f}Hz, Harmonic={harmonic_ratio:.3f}, MFCC_std={mfcc_std:.1f}, Voice={'YES' if has_voice else 'NO'}")
            
            # Tìm vị trí giọng hát đầu tiên
            if has_voice and start_time > 5.0:  # Bỏ qua 5 giây đầu (có thể là intro)
                logger.info(f"\n🎯 Tìm thấy giọng hát thực sự tại: {start_time:.2f}s")
                logger.info(f"   RMS: {segment_rms:.4f}")
                logger.info(f"   Centroid: {segment_centroid:.1f}Hz")
                logger.info(f"   Harmonic ratio: {harmonic_ratio:.3f}")
                return start_time
        
        logger.warning("⚠️ Không tìm thấy giọng hát trong 60 giây đầu!")
        return None
        
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_manual_voice_detection():
    """Test voice detection thủ công với thresholds thấp"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info("🎯 Test voice detection thủ công...")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        
        # Phân tích từng giây
        hop_length = 512
        frame_length = 2048
        
        # Tính RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Tính baseline từ 5 giây đầu
        baseline_length = min(5, len(rms))
        baseline_rms = np.mean(rms[:baseline_length])
        
        logger.info(f"📊 Baseline RMS: {baseline_rms:.4f}")
        
        # Tìm vị trí giọng hát với thresholds thấp
        voice_start_candidates = []
        
        # Phân tích 60 giây đầu
        for i in range(min(60, len(rms))):
            start_time = i * hop_length / sr
            segment_rms = rms[i]
            
            # Điều kiện đơn giản: RMS cao hơn baseline + 5%
            threshold = baseline_rms * 1.05  # Chỉ cần cao hơn 5%
            has_voice = segment_rms > threshold
            
            if has_voice:
                voice_start_candidates.append({
                    'time': start_time,
                    'rms': segment_rms,
                    'confidence': segment_rms / baseline_rms
                })
            
            # Log chi tiết cho 30 giây đầu
            if start_time < 30:
                logger.info(f"   {start_time:.1f}s: RMS={segment_rms:.4f}, Threshold={threshold:.4f}, Voice={'YES' if has_voice else 'NO'}")
        
        # Tìm vị trí giọng hát bắt đầu thực sự
        if voice_start_candidates:
            # Lấy vị trí đầu tiên có voice sau 5 giây
            for candidate in voice_start_candidates:
                if candidate['time'] > 5.0:
                    logger.info(f"\n🎯 Vị trí giọng hát thực sự: {candidate['time']:.2f}s")
                    logger.info(f"   RMS: {candidate['rms']:.4f}")
                    logger.info(f"   Confidence: {candidate['confidence']:.3f}")
                    return candidate['time']
        
        logger.warning("⚠️ Không tìm thấy vị trí giọng hát!")
        return None
        
    except Exception as e:
        logger.error(f"❌ Lỗi test: {e}")
        return None

if __name__ == "__main__":
    print("=== TEST ANALYZE WAITING FOR YOU ===")
    
    # Phân tích chi tiết
    print("\n1. Phân tích chi tiết file Waiting For You...")
    detailed_analysis = analyze_waiting_for_you_detailed()
    
    # Test voice detection thủ công
    print("\n2. Test voice detection thủ công...")
    manual_detection = test_manual_voice_detection()
    
    # Kết quả
    print("\n=== KET QUA ===")
    print(f"Detailed Analysis: {'OK' if detailed_analysis is not None else 'ERROR'}")
    print(f"Manual Detection: {'OK' if manual_detection is not None else 'ERROR'}")
    
    if detailed_analysis is not None:
        print(f"\n🎯 Giọng hát thực sự bắt đầu tại: {detailed_analysis:.2f}s")
    elif manual_detection is not None:
        print(f"\n🎯 Giọng hát thực sự bắt đầu tại: {manual_detection:.2f}s")
    else:
        print("\n⚠️ Cần phân tích thêm để tìm vị trí giọng hát!")
