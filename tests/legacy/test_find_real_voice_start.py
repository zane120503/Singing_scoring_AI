#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Find Real Voice Start - Tim vi tri giong hat thuc su
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

def find_real_voice_start():
    """Tim vi tri giong hat thuc su"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file khong ton tai: {test_file}")
            return False
        
        logger.info("Tim vi tri giong hat thuc su...")
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
        
        # Tinh baseline tu 10 giay dau (intro)
        baseline_length = min(10, len(rms))
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
            
            # Tim cac thay doi lon
            if energy_change > 2.0:  # Thay doi lon gap 2 lan standard deviation
                energy_changes.append({
                    'time': start_time,
                    'rms': segment_rms,
                    'change': energy_change
                })
        
        # Hien thi cac thay doi lon
        logger.info(f"Tim thay {len(energy_changes)} energy changes lon:")
        for change in energy_changes[:20]:  # Hien thi 20 thay doi dau tien
            logger.info(f"   {change['time']:.2f}s: RMS={change['rms']:.4f}, Change={change['change']:.2f}")
        
        # Tim thay doi dau tien sau 15 giay (bo qua intro)
        for change in energy_changes:
            if change['time'] > 15.0:
                logger.info(f"\nEnergy change dau tien sau 15s: {change['time']:.2f}s")
                logger.info(f"   RMS: {change['rms']:.4f}")
                logger.info(f"   Change: {change['change']:.2f}")
                return change['time']
        
        # Fallback: tim thay doi dau tien sau 10 giay
        for change in energy_changes:
            if change['time'] > 10.0:
                logger.info(f"\nFallback - Energy change dau tien sau 10s: {change['time']:.2f}s")
                logger.info(f"   RMS: {change['rms']:.4f}")
                logger.info(f"   Change: {change['change']:.2f}")
                return change['time']
        
        return None
        
    except Exception as e:
        logger.error(f"Loi phan tich: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_spectral_changes():
    """Phan tich su thay doi spectral"""
    try:
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        logger.info("Phan tich su thay doi spectral...")
        
        # Load audio
        audio, sr = librosa.load(test_file, sr=22050)
        
        # Tinh spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=512)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=512)[0]
        
        # Tinh baseline tu 10 giay dau
        baseline_length = min(10, len(spectral_centroids))
        baseline_centroid = np.mean(spectral_centroids[:baseline_length])
        baseline_rolloff = np.mean(spectral_rolloff[:baseline_length])
        
        logger.info(f"Baseline Centroid: {baseline_centroid:.1f} Hz")
        logger.info(f"Baseline Rolloff: {baseline_rolloff:.1f} Hz")
        
        # Tim su thay doi trong spectral features
        spectral_changes = []
        
        for i in range(len(spectral_centroids)):
            start_time = i * 512 / sr
            centroid = spectral_centroids[i]
            rolloff = spectral_rolloff[i]
            
            # Tim su thay doi lon trong spectral features
            centroid_change = abs(centroid - baseline_centroid) / baseline_centroid
            rolloff_change = abs(rolloff - baseline_rolloff) / baseline_rolloff
            
            if centroid_change > 0.3 or rolloff_change > 0.3:  # Thay doi lon hon 30%
                spectral_changes.append({
                    'time': start_time,
                    'centroid': centroid,
                    'rolloff': rolloff,
                    'centroid_change': centroid_change,
                    'rolloff_change': rolloff_change
                })
        
        # Hien thi cac thay doi spectral
        logger.info(f"Tim thay {len(spectral_changes)} spectral changes:")
        for change in spectral_changes[:20]:  # Hien thi 20 thay doi dau tien
            logger.info(f"   {change['time']:.2f}s: Centroid={change['centroid']:.1f}Hz, Rolloff={change['rolloff']:.1f}Hz")
        
        # Tim thay doi dau tien sau 15 giay
        for change in spectral_changes:
            if change['time'] > 15.0:
                logger.info(f"\nSpectral change dau tien sau 15s: {change['time']:.2f}s")
                return change['time']
        
        return None
        
    except Exception as e:
        logger.error(f"Loi phan tich spectral: {e}")
        return None

if __name__ == "__main__":
    print("=== TEST FIND REAL VOICE START ===")
    
    # Tim vi tri giong hat thuc su
    print("\n1. Tim vi tri giong hat thuc su...")
    result1 = find_real_voice_start()
    
    # Phan tich su thay doi spectral
    print("\n2. Phan tich su thay doi spectral...")
    result2 = analyze_spectral_changes()
    
    # Ket qua
    print("\n=== KET QUA ===")
    if result1 is not None:
        print(f"Giong hat thuc su bat dau tai: {result1:.2f}s")
    elif result2 is not None:
        print(f"Spectral change dau tien tai: {result2:.2f}s")
    else:
        print("Can phan tich them de tim vi tri giong hat!")
