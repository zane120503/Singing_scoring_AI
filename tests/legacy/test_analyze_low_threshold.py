#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Analyze Low Threshold - Phan tich voi threshold thap
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

def analyze_low_threshold():
    """Phan tich voi threshold thap"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file khong ton tai: {test_file}")
            return False
        
        logger.info("Phan tich voi threshold thap...")
        logger.info(f"File: {test_file}")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        duration = len(audio) / sr
        
        logger.info(f"Audio info: {duration:.2f}s, {sr} Hz, {len(audio)} samples")
        
        # DINH NGHIA THRESHOLD THAP
        hop_length = 512
        frame_length = 2048
        
        # Tinh RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Tinh baseline tu 5 giay dau
        baseline_length = min(5, len(rms))
        baseline_rms = np.mean(rms[:baseline_length])
        
        logger.info(f"Baseline RMS: {baseline_rms:.4f}")
        
        # Tim vi tri giong hat voi threshold RAT THAP
        voice_start_candidates = []
        
        # Phan tich 60 giay dau
        for i in range(min(60, len(rms))):
            start_time = i * hop_length / sr
            segment_rms = rms[i]
            
            # THRESHOLD RAT THAP: RMS cao hon baseline + 1%
            threshold = baseline_rms * 1.01  # Chi can cao hon 1%
            has_voice = segment_rms > threshold
            
            if has_voice:
                voice_start_candidates.append({
                    'time': start_time,
                    'rms': segment_rms,
                    'confidence': segment_rms / baseline_rms
                })
            
            # Log chi tiet cho 30 giay dau
            if start_time < 30:
                logger.info(f"   {start_time:.1f}s: RMS={segment_rms:.4f}, Threshold={threshold:.4f}, Voice={'YES' if has_voice else 'NO'}")
        
        # Tim vi tri giong hat bat dau thuc su
        if voice_start_candidates:
            # Lay vi tri dau tien co voice sau 5 giay
            for candidate in voice_start_candidates:
                if candidate['time'] > 5.0:
                    logger.info(f"\nVi tri giong hat thuc su: {candidate['time']:.2f}s")
                    logger.info(f"   RMS: {candidate['rms']:.4f}")
                    logger.info(f"   Confidence: {candidate['confidence']:.3f}")
                    return candidate['time']
        
        logger.warning("Khong tim thay vi tri giong hat!")
        return None
        
    except Exception as e:
        logger.error(f"Loi phan tich: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_energy_spike():
    """Phan tich energy spike"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        logger.info("Phan tich energy spike...")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        
        # Tinh RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # Tim cac energy spike
        energy_spikes = []
        
        for i in range(len(rms)):
            start_time = i * 512 / sr
            segment_rms = rms[i]
            
            # Tim cac spike cao hon 0.08
            if segment_rms > 0.08:
                energy_spikes.append({
                    'time': start_time,
                    'rms': segment_rms
                })
        
        # Hien thi cac spike
        logger.info(f"Tim thay {len(energy_spikes)} energy spikes:")
        for spike in energy_spikes[:10]:  # Hien thi 10 spike dau tien
            logger.info(f"   {spike['time']:.2f}s: RMS={spike['rms']:.4f}")
        
        # Tim spike dau tien sau 5 giay
        for spike in energy_spikes:
            if spike['time'] > 5.0:
                logger.info(f"\nEnergy spike dau tien sau 5s: {spike['time']:.2f}s")
                return spike['time']
        
        return None
        
    except Exception as e:
        logger.error(f"Loi phan tich energy spike: {e}")
        return None

if __name__ == "__main__":
    print("=== TEST ANALYZE LOW THRESHOLD ===")
    
    # Phan tich voi threshold thap
    print("\n1. Phan tich voi threshold thap...")
    result1 = analyze_low_threshold()
    
    # Phan tich energy spike
    print("\n2. Phan tich energy spike...")
    result2 = analyze_energy_spike()
    
    # Ket qua
    print("\n=== KET QUA ===")
    if result1 is not None:
        print(f"Giong hat thuc su bat dau tai: {result1:.2f}s")
    elif result2 is not None:
        print(f"Energy spike dau tien tai: {result2:.2f}s")
    else:
        print("Can phan tich them de tim vi tri giong hat!")
