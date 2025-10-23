#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Voice Detection Analysis - Phân tích chi tiết vị trí giọng hát
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

def analyze_voice_precisely():
    """Phân tích chính xác vị trí giọng hát trong file Waiting For You"""
    try:
        # File gốc
        audio_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(audio_file):
            logger.error(f"File không tồn tại: {audio_file}")
            return False
        
        logger.info(f"🔍 Phân tích chính xác vị trí giọng hát...")
        logger.info(f"File: {audio_file}")
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=22050)
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
        logger.info(f"\n📊 Phân tích từng giây để tìm giọng hát:")
        
        voice_start_candidates = []
        
        # Phân tích 30 giây đầu
        for i in range(min(30, len(rms))):
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
            
            # Điều kiện để xác định có giọng hát
            has_voice = (
                segment_rms > 0.08 and  # Energy đủ cao
                segment_centroid > 1000 and segment_centroid < 4000 and  # Voice frequency range
                segment_rolloff > 1000 and  # Có high frequency content
                harmonic_ratio > 0.6 and  # Harmonic content cao (voice có harmonic)
                mfcc_std > 50  # MFCC variation (voice có variation)
            )
            
            if has_voice:
                voice_start_candidates.append({
                    'time': start_time,
                    'rms': segment_rms,
                    'centroid': segment_centroid,
                    'rolloff': segment_rolloff,
                    'harmonic_ratio': harmonic_ratio,
                    'mfcc_std': mfcc_std
                })
            
            # Log chi tiết cho 20 giây đầu
            if start_time < 20:
                logger.info(f"   {start_time:.1f}s: RMS={segment_rms:.4f}, Centroid={segment_centroid:.1f}Hz, Rolloff={segment_rolloff:.1f}Hz, Harmonic={harmonic_ratio:.3f}, MFCC_std={mfcc_std:.1f}, Voice={'YES' if has_voice else 'NO'}")
        
        # Tìm vị trí giọng hát bắt đầu thực sự
        if voice_start_candidates:
            # Lấy vị trí đầu tiên có voice
            first_voice = voice_start_candidates[0]
            logger.info(f"\n🎯 Vị trí giọng hát thực sự:")
            logger.info(f"   First voice detected at: {first_voice['time']:.2f}s")
            logger.info(f"   RMS: {first_voice['rms']:.4f}")
            logger.info(f"   Centroid: {first_voice['centroid']:.1f}Hz")
            logger.info(f"   Harmonic ratio: {first_voice['harmonic_ratio']:.3f}")
            
            return first_voice['time']
        else:
            logger.warning("⚠️ Không tìm thấy vị trí giọng hát!")
            return None
        
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_precise_voice_detector():
    """Tạo Precise Voice Detector dựa trên phân tích"""
    try:
        from src.ai.improved_smart_voice_detector import ImprovedSmartVoiceDetector
        
        # Phân tích để tìm vị trí chính xác
        precise_voice_start = analyze_voice_precisely()
        
        if precise_voice_start is None:
            logger.error("❌ Không thể xác định vị trí giọng hát chính xác")
            return False
        
        logger.info(f"🎯 Tạo Precise Voice Detector với voice start: {precise_voice_start:.2f}s")
        
        # Test với Improved Smart Voice Detector
        detector = ImprovedSmartVoiceDetector()
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        segments = detector.detect_voice_activity(test_file)
        
        if segments:
            detected_start = segments[0]['start']
            error = abs(detected_start - precise_voice_start)
            
            logger.info(f"\n📊 So sánh kết quả:")
            logger.info(f"   Precise voice start: {precise_voice_start:.2f}s")
            logger.info(f"   Detected voice start: {detected_start:.2f}s")
            logger.info(f"   Error: {error:.2f}s")
            
            if error < 1.0:
                logger.info(f"✅ CHÍNH XÁC (sai số: {error:.2f}s)")
                return True
            else:
                logger.warning(f"❌ KHÔNG CHÍNH XÁC (sai số: {error:.2f}s)")
                return False
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Lỗi tạo detector: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST VOICE DETECTION ANALYSIS ===")
    
    # Phân tích chính xác vị trí giọng hát
    print("\n1. Phân tích chính xác vị trí giọng hát...")
    precise_analysis = analyze_voice_precisely()
    
    # Tạo precise voice detector
    print("\n2. Tạo Precise Voice Detector...")
    detector_success = create_precise_voice_detector()
    
    # Kết quả
    print("\n=== KET QUA ===")
    print(f"Precise Analysis: {'OK' if precise_analysis is not None else 'ERROR'}")
    print(f"Detector Creation: {'OK' if detector_success else 'ERROR'}")
    
    if precise_analysis is not None and detector_success:
        print("\nDA PHAN TICH CHINH XAC VI TRI GIONG HAT!")
    else:
        print("\nCAN CAI THIEN THEM!")
