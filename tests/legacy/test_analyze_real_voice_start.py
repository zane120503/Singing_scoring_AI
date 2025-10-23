#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Analyze Real Voice Start - Phan tich vi tri giong hat thuc su
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

def analyze_real_voice_start():
    """Phan tich vi tri giong hat thuc su"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file khong ton tai: {test_file}")
            return False
        
        logger.info("Phan tich vi tri giong hat thuc su...")
        logger.info(f"File: {test_file}")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        duration = len(audio) / sr
        
        logger.info(f"Audio info: {duration:.2f}s, {sr} Hz, {len(audio)} samples")
        
        # Phan tich chi tiet
        hop_length = 512
        frame_length = 2048
        
        # Tinh cac features
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Chuyen frames thanh thoi gian
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Phan tich tung giay de tim vi tri giong hat thuc su
        logger.info(f"\nPhan tich tung giay de tim giong hat thuc su:")
        
        # Phan tich 120 giay dau (du de tim giong hat)
        for i in range(min(120, len(rms))):
            start_time = i * hop_length / sr
            end_time = min((i + 1) * hop_length / sr, duration)
            
            # Lay features cua giay nay
            segment_rms = rms[i]
            segment_centroid = spectral_centroids[i]
            segment_rolloff = spectral_rolloff[i]
            segment_zcr = zcr[i]
            
            # Phan tich harmonic content
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
            
            # MFCC de phat hien voice characteristics
            try:
                mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfccs[0])
                mfcc_std = np.std(mfccs[0])
            except:
                mfcc_mean = 0
                mfcc_std = 0
            
            # Dieu kien de xac dinh co giong hat (thresholds thap hon)
            has_voice = (
                segment_rms > 0.03 and  # Energy du cao (giam threshold)
                segment_centroid > 600 and segment_centroid < 6000 and  # Voice frequency range rong hon
                segment_rolloff > 600 and  # Co high frequency content (giam threshold)
                harmonic_ratio > 0.2 and  # Harmonic content (giam threshold)
                mfcc_std > 5  # MFCC variation (giam threshold)
            )
            
            # Log chi tiet cho 60 giay dau
            if start_time < 60:
                logger.info(f"   {start_time:.1f}s: RMS={segment_rms:.4f}, Centroid={segment_centroid:.1f}Hz, Rolloff={segment_rolloff:.1f}Hz, Harmonic={harmonic_ratio:.3f}, MFCC_std={mfcc_std:.1f}, Voice={'YES' if has_voice else 'NO'}")
            
            # Tim vi tri giong hat dau tien
            if has_voice and start_time > 10.0:  # Bo qua 10 giay dau (co the la intro)
                logger.info(f"\nTim thay giong hat thuc su tai: {start_time:.2f}s")
                logger.info(f"   RMS: {segment_rms:.4f}")
                logger.info(f"   Centroid: {segment_centroid:.1f}Hz")
                logger.info(f"   Harmonic ratio: {harmonic_ratio:.3f}")
                return start_time
        
        logger.warning("Khong tim thay giong hat trong 120 giay dau!")
        return None
        
    except Exception as e:
        logger.error(f"Loi phan tich: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_energy_pattern():
    """Phan tich energy pattern"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        logger.info("Phan tich energy pattern...")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        
        # Tinh RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # Tim cac energy spike cao
        energy_spikes = []
        
        for i in range(len(rms)):
            start_time = i * 512 / sr
            segment_rms = rms[i]
            
            # Tim cac spike cao hon 0.06
            if segment_rms > 0.06:
                energy_spikes.append({
                    'time': start_time,
                    'rms': segment_rms
                })
        
        # Hien thi cac spike
        logger.info(f"Tim thay {len(energy_spikes)} energy spikes cao:")
        for spike in energy_spikes[:20]:  # Hien thi 20 spike dau tien
            logger.info(f"   {spike['time']:.2f}s: RMS={spike['rms']:.4f}")
        
        # Tim spike dau tien sau 10 giay
        for spike in energy_spikes:
            if spike['time'] > 10.0:
                logger.info(f"\nEnergy spike dau tien sau 10s: {spike['time']:.2f}s")
                return spike['time']
        
        return None
        
    except Exception as e:
        logger.error(f"Loi phan tich energy pattern: {e}")
        return None

if __name__ == "__main__":
    print("=== TEST ANALYZE REAL VOICE START ===")
    
    # Phan tich vi tri giong hat thuc su
    print("\n1. Phan tich vi tri giong hat thuc su...")
    result1 = analyze_real_voice_start()
    
    # Phan tich energy pattern
    print("\n2. Phan tich energy pattern...")
    result2 = analyze_energy_pattern()
    
    # Ket qua
    print("\n=== KET QUA ===")
    if result1 is not None:
        print(f"Giong hat thuc su bat dau tai: {result1:.2f}s")
    elif result2 is not None:
        print(f"Energy spike dau tien tai: {result2:.2f}s")
    else:
        print("Can phan tich them de tim vi tri giong hat!")
