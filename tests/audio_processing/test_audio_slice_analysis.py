#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Audio Slice Analysis - Kiểm tra file được cắt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import librosa
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_sliced_audio():
    """Phân tích file audio đã được cắt"""
    try:
        # File được cắt từ workflow
        sliced_file = "D:\\singing scoring AI\\src\\ai\\..\\..\\output\\optimized_processing\\karaoke_slice_Waiting For You_10.8s_20.0s.wav"
        
        if not os.path.exists(sliced_file):
            logger.error(f"File cắt không tồn tại: {sliced_file}")
            return False
        
        logger.info(f"🔍 Phân tích file đã cắt: {os.path.basename(sliced_file)}")
        
        # Load audio
        audio, sr = librosa.load(sliced_file, sr=22050)
        duration = len(audio) / sr
        
        logger.info(f"Audio info: {duration:.2f}s, {sr} Hz, {len(audio)} samples")
        
        # Phân tích energy theo thời gian
        hop_length = 512
        frame_length = 2048
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Tìm vị trí có voice
        voice_threshold = np.percentile(rms, 30)
        voice_frames = np.where(rms > voice_threshold)[0]
        
        if len(voice_frames) > 0:
            # Chuyển frames thành thời gian
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            voice_times = times[voice_frames]
            
            logger.info(f"📊 Phân tích voice trong file cắt:")
            logger.info(f"   Total frames: {len(rms)}")
            logger.info(f"   Voice frames: {len(voice_frames)}")
            logger.info(f"   Voice threshold: {voice_threshold:.4f}")
            
            if len(voice_times) > 0:
                first_voice_time = voice_times[0]
                last_voice_time = voice_times[-1]
                
                logger.info(f"   First voice at: {first_voice_time:.2f}s")
                logger.info(f"   Last voice at: {last_voice_time:.2f}s")
                
                # Phân tích chi tiết từng đoạn
                logger.info(f"\n📈 Phân tích chi tiết:")
                for i in range(0, len(rms), len(rms)//10):  # 10 đoạn
                    start_time = i * hop_length / sr
                    end_time = min((i + len(rms)//10) * hop_length / sr, duration)
                    segment_rms = rms[i:i+len(rms)//10]
                    segment_centroid = spectral_centroids[i:i+len(rms)//10]
                    
                    avg_rms = np.mean(segment_rms)
                    avg_centroid = np.mean(segment_centroid)
                    
                    has_voice = avg_rms > voice_threshold
                    
                    logger.info(f"   {start_time:.1f}s - {end_time:.1f}s: RMS={avg_rms:.4f}, Centroid={avg_centroid:.1f}Hz, Voice={'YES' if has_voice else 'NO'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_original_voice_position():
    """Phân tích vị trí voice trong file gốc"""
    try:
        from src.ai.smart_voice_detector import SmartVoiceDetector
        
        # File gốc
        original_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(original_file):
            logger.error(f"File gốc không tồn tại: {original_file}")
            return False
        
        logger.info(f"🔍 Phân tích vị trí voice trong file gốc...")
        
        # Smart Voice Detector
        detector = SmartVoiceDetector()
        segments = detector.detect_voice_activity(original_file)
        
        if segments:
            first_segment = segments[0]
            logger.info(f"📊 Voice segments trong file gốc:")
            logger.info(f"   First segment: {first_segment['start']:.2f}s - {first_segment['end']:.2f}s")
            
            # Phân tích chi tiết vùng xung quanh
            start_time = first_segment['start']
            end_time = first_segment['end']
            
            # Load audio và phân tích vùng xung quanh
            audio, sr = librosa.load(original_file, sr=22050)
            
            # Phân tích 5s trước và sau vị trí detected
            analysis_start = max(0, start_time - 5)
            analysis_end = min(len(audio)/sr, start_time + 10)
            
            logger.info(f"\n📈 Phân tích vùng xung quanh {start_time:.2f}s:")
            logger.info(f"   Analysis range: {analysis_start:.2f}s - {analysis_end:.2f}s")
            
            # Cắt audio để phân tích
            start_sample = int(analysis_start * sr)
            end_sample = int(analysis_end * sr)
            analysis_audio = audio[start_sample:end_sample]
            
            # Phân tích energy
            hop_length = 512
            rms = librosa.feature.rms(y=analysis_audio, frame_length=2048, hop_length=hop_length)[0]
            
            # Tìm vị trí voice thực sự
            voice_threshold = np.percentile(rms, 25)
            voice_frames = np.where(rms > voice_threshold)[0]
            
            if len(voice_frames) > 0:
                times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
                voice_times = times[voice_frames] + analysis_start
                
                logger.info(f"   Voice threshold: {voice_threshold:.4f}")
                logger.info(f"   Voice frames: {len(voice_frames)}")
                logger.info(f"   First voice at: {voice_times[0]:.2f}s")
                logger.info(f"   Last voice at: {voice_times[-1]:.2f}s")
                
                # So sánh với vị trí detected
                detected_start = start_time
                actual_start = voice_times[0]
                error = abs(detected_start - actual_start)
                
                logger.info(f"\n🎯 So sánh:")
                logger.info(f"   Detected start: {detected_start:.2f}s")
                logger.info(f"   Actual start: {actual_start:.2f}s")
                logger.info(f"   Error: {error:.2f}s")
                
                if error > 2.0:
                    logger.warning(f"⚠️ Voice detection có sai số lớn: {error:.2f}s")
                    return False
                else:
                    logger.info(f"✅ Voice detection chính xác: {error:.2f}s")
                    return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST AUDIO SLICE ANALYSIS ===")
    
    # Phân tích file đã cắt
    print("\n1. Phan tich file da cat...")
    slice_success = analyze_sliced_audio()
    
    # Phan tich vi tri voice goc
    print("\n2. Phan tich vi tri voice goc...")
    original_success = analyze_original_voice_position()
    
    # Ket qua
    print("\n=== KET QUA ===")
    print(f"Slice Analysis: {'OK' if slice_success else 'ERROR'}")
    print(f"Original Analysis: {'OK' if original_success else 'ERROR'}")
    
    if slice_success and original_success:
        print("\nDA PHAN TICH FILE AUDIO!")
    else:
        print("\nCO VAN DE VOI FILE AUDIO!")
