#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Analyze Simple - Phan tich don gian vi tri giong hat
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

def analyze_simple():
    """Phan tich don gian file Waiting For You"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file khong ton tai: {test_file}")
            return False
        
        logger.info("Phan tich don gian file Waiting For You...")
        logger.info(f"File: {test_file}")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        duration = len(audio) / sr
        
        logger.info(f"Audio info: {duration:.2f}s, {sr} Hz, {len(audio)} samples")
        
        # Phan tich don gian
        hop_length = 512
        frame_length = 2048
        
        # Tinh RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Tinh baseline tu 5 giay dau
        baseline_length = min(5, len(rms))
        baseline_rms = np.mean(rms[:baseline_length])
        
        logger.info(f"Baseline RMS: {baseline_rms:.4f}")
        
        # Tim vi tri giong hat voi thresholds thap
        voice_start_candidates = []
        
        # Phan tich 60 giay dau
        for i in range(min(60, len(rms))):
            start_time = i * hop_length / sr
            segment_rms = rms[i]
            
            # Dieu kien don gian: RMS cao hon baseline + 5%
            threshold = baseline_rms * 1.05
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

if __name__ == "__main__":
    print("=== TEST ANALYZE SIMPLE ===")
    
    # Phan tich don gian
    print("\n1. Phan tich don gian file Waiting For You...")
    result = analyze_simple()
    
    # Ket qua
    print("\n=== KET QUA ===")
    if result is not None:
        print(f"Giong hat thuc su bat dau tai: {result:.2f}s")
    else:
        print("Can phan tich them de tim vi tri giong hat!")
