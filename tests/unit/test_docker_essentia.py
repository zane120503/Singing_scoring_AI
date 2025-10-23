#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test hệ thống với Docker Essentia
"""

import numpy as np
import soundfile as sf
from advanced_key_detector import AdvancedKeyDetector

def create_test_audio():
    """Tao file audio test"""
    print("Tao file audio test...")
    
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Beat nhac (C major chord)
    beat = (0.3 * np.sin(2 * np.pi * 261.63 * t) +  # C4
            0.2 * np.sin(2 * np.pi * 329.63 * t) +  # E4
            0.2 * np.sin(2 * np.pi * 392.00 * t))    # G4
    
    # Giong hat (C major scale)
    vocals = np.zeros_like(t)
    scale_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C-D-E-F-G-A-B
    for i, freq in enumerate(scale_notes):
        start_time = i * duration / len(scale_notes)
        end_time = (i + 1) * duration / len(scale_notes)
        mask = (t >= start_time) & (t < end_time)
        vocals[mask] = 0.3 * np.sin(2 * np.pi * freq * t[mask])
    
    # Karaoke (beat + vocals)
    karaoke = beat + vocals
    karaoke = karaoke / np.max(np.abs(karaoke))
    
    # Luu file
    sf.write('test_beat.wav', beat, sr)
    sf.write('test_karaoke.wav', karaoke, sr)
    print("File audio test da duoc tao!")

def test_key_detection():
    """Test key detection với Docker Essentia"""
    print("\nTEST KEY DETECTION VOI DOCKER ESSENTIA")
    print("=" * 50)
    
    # Tạo file test
    create_test_audio()
    
    # Khởi tạo detector
    detector = AdvancedKeyDetector()
    
    # Test beat detection
    print("\nTest beat key detection...")
    beat_result = detector.detect_key('test_beat.wav')
    if beat_result:
        print(f"Beat key: {beat_result['key']} {beat_result['scale']} (confidence: {beat_result['confidence']:.3f})")
        print(f"   Method: {beat_result['method']}")
    else:
        print("Failed to detect beat key")
    
    # Test karaoke detection
    print("\nTest karaoke key detection...")
    karaoke_result = detector.detect_key('test_karaoke.wav')
    if karaoke_result:
        print(f"Karaoke key: {karaoke_result['key']} {karaoke_result['scale']} (confidence: {karaoke_result['confidence']:.3f})")
        print(f"   Method: {karaoke_result['method']}")
    else:
        print("Failed to detect karaoke key")
    
    # Cleanup
    import os
    for file in ['test_beat.wav', 'test_karaoke.wav']:
        if os.path.exists(file):
            os.remove(file)
    
    print("\nTest hoan thanh!")

if __name__ == "__main__":
    test_key_detection()
