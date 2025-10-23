#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Find Voice Start Final - Tim vi tri giong hat cuoi cung
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

def find_voice_start_final():
    """Tim vi tri giong hat cuoi cung"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file khong ton tai: {test_file}")
            return False
        
        logger.info("Tim vi tri giong hat cuoi cung...")
        logger.info(f"File: {test_file}")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        duration = len(audio) / sr
        
        logger.info(f"Audio info: {duration:.2f}s, {sr} Hz, {len(audio)} samples")
        
        # Phan tich chi tiet
        hop_length = 512
        frame_length = 2048
        
        # Tinh RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Tinh baseline tu 20 giay dau (intro)
        baseline_length = min(20, len(rms))
        baseline_rms = np.mean(rms[:baseline_length])
        baseline_std = np.std(rms[:baseline_length])
        
        logger.info(f"Baseline RMS: {baseline_rms:.4f} ± {baseline_std:.4f}")
        
        # Tim su thay doi dot ngot trong energy
        energy_changes = []
        
        # Phan tich 120 giay dau
        for i in range(min(120, len(rms))):
            start_time = i * hop_length / sr
            segment_rms = rms[i]
            
            # Tinh do thay doi so voi baseline
            energy_change = (segment_rms - baseline_rms) / baseline_std
            
            # Tim cac thay doi lon (threshold thap hon)
            if energy_change > 1.5:  # Thay doi lon gap 1.5 lan standard deviation
                energy_changes.append({
                    'time': start_time,
                    'rms': segment_rms,
                    'change': energy_change
                })
        
        # Hien thi cac thay doi lon
        logger.info(f"Tim thay {len(energy_changes)} energy changes lon:")
        for change in energy_changes[:20]:  # Hien thi 20 thay doi dau tien
            logger.info(f"   {change['time']:.2f}s: RMS={change['rms']:.4f}, Change={change['change']:.2f}")
        
        # Tim thay doi dau tien sau 25 giay (bo qua intro)
        for change in energy_changes:
            if change['time'] > 25.0:
                logger.info(f"\nEnergy change dau tien sau 25s: {change['time']:.2f}s")
                logger.info(f"   RMS: {change['rms']:.4f}")
                logger.info(f"   Change: {change['change']:.2f}")
                return change['time']
        
        # Fallback: tim thay doi dau tien sau 20 giay
        for change in energy_changes:
            if change['time'] > 20.0:
                logger.info(f"\nFallback - Energy change dau tien sau 20s: {change['time']:.2f}s")
                logger.info(f"   RMS: {change['rms']:.4f}")
                logger.info(f"   Change: {change['change']:.2f}")
                return change['time']
        
        return None
        
    except Exception as e:
        logger.error(f"Loi phan tich: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_voice_pattern():
    """Phan tich pattern giong hat"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        logger.info("Phan tich pattern giong hat...")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        
        # Tinh MFCC features (dac trung giong hat)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=512)
        
        # Tinh baseline tu 20 giay dau
        baseline_length = min(20, mfccs.shape[1])
        baseline_mfcc = np.mean(mfccs[:, :baseline_length], axis=1)
        
        logger.info(f"Baseline MFCC: {baseline_mfcc[0]:.2f}")
        
        # Tim su thay doi trong MFCC features
        mfcc_changes = []
        
        for i in range(mfccs.shape[1]):
            start_time = i * 512 / sr
            mfcc = mfccs[:, i]
            
            # Tinh do thay doi trong MFCC
            mfcc_change = np.mean(np.abs(mfcc - baseline_mfcc))
            
            if mfcc_change > 20.0:  # Thay doi lon trong MFCC (threshold thap hon)
                mfcc_changes.append({
                    'time': start_time,
                    'mfcc_change': mfcc_change
                })
        
        # Hien thi cac thay doi MFCC
        logger.info(f"Tim thay {len(mfcc_changes)} MFCC changes:")
        for change in mfcc_changes[:20]:  # Hien thi 20 thay doi dau tien
            logger.info(f"   {change['time']:.2f}s: MFCC Change={change['mfcc_change']:.2f}")
        
        # Tim thay doi dau tien sau 25 giay
        for change in mfcc_changes:
            if change['time'] > 25.0:
                logger.info(f"\nMFCC change dau tien sau 25s: {change['time']:.2f}s")
                return change['time']
        
        # Fallback: tim thay doi dau tien sau 20 giay
        for change in mfcc_changes:
            if change['time'] > 20.0:
                logger.info(f"\nFallback - MFCC change dau tien sau 20s: {change['time']:.2f}s")
                return change['time']
        
        return None
        
    except Exception as e:
        logger.error(f"Loi phan tich MFCC: {e}")
        return None

if __name__ == "__main__":
    print("=== TEST FIND VOICE START FINAL ===")
    
    # Tim vi tri giong hat cuoi cung
    print("\n1. Tim vi tri giong hat cuoi cung...")
    result1 = find_voice_start_final()
    
    # Phan tich pattern giong hat
    print("\n2. Phan tich pattern giong hat...")
    result2 = analyze_voice_pattern()
    
    # Ket qua
    print("\n=== KET QUA ===")
    if result1 is not None:
        print(f"Giong hat thuc su bat dau tai: {result1:.2f}s")
    elif result2 is not None:
        print(f"MFCC change dau tien tai: {result2:.2f}s")
    else:
        print("Can phan tich them de tim vi tri giong hat!")
