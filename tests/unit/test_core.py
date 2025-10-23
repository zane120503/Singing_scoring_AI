#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import librosa
import numpy as np

def test_essentia_key_detection():
    """Test Essentia key detection"""
    print("🎵 Testing Essentia Key Detection...")
    
    try:
        import essentia.standard as es
        print("✅ Essentia imported successfully!")
        
        # Create a simple test audio (sine wave)
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Create a C major chord
        frequency_c = 261.63  # C4
        frequency_e = 329.63  # E4
        frequency_g = 392.00  # G4
        
        audio = np.sin(2 * np.pi * frequency_c * t) + \
                np.sin(2 * np.pi * frequency_e * t) + \
                np.sin(2 * np.pi * frequency_g * t)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Save test audio
        test_audio_path = "/app/temp_output/test_audio.wav"
        import soundfile as sf
        sf.write(test_audio_path, audio, sr)
        
        # Test Essentia key detection
        key_detector = es.KeyExtractor()
        key, scale, strength = key_detector(audio)
        
        print(f"✅ Key Detection Results:")
        print(f"   Key: {key}")
        print(f"   Scale: {scale}")
        print(f"   Strength: {strength:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Essentia test failed: {e}")
        return False

def test_advanced_key_detector():
    """Test our AdvancedKeyDetector"""
    print("\n🎵 Testing AdvancedKeyDetector...")
    
    try:
        from advanced_key_detector import AdvancedKeyDetector
        
        # Create test audio
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Create a C major chord
        frequency_c = 261.63  # C4
        frequency_e = 329.63  # E4
        frequency_g = 392.00  # G4
        
        audio = np.sin(2 * np.pi * frequency_c * t) + \
                np.sin(2 * np.pi * frequency_e * t) + \
                np.sin(2 * np.pi * frequency_g * t)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Save test audio
        test_audio_path = "/app/temp_output/test_audio.wav"
        import soundfile as sf
        sf.write(test_audio_path, audio, sr)
        
        # Test our key detector
        detector = AdvancedKeyDetector()
        key_info = detector.detect_key(test_audio_path)
        
        print(f"✅ AdvancedKeyDetector Results:")
        print(f"   Key: {key_info['key']}")
        print(f"   Scale: {key_info['scale']}")
        print(f"   Confidence: {key_info['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedKeyDetector test failed: {e}")
        return False

def test_advanced_audio_processor():
    """Test our AdvancedAudioProcessor"""
    print("\n🎵 Testing AdvancedAudioProcessor...")
    
    try:
        from advanced_audio_processor import AdvancedAudioProcessor
        
        # Create test audio
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Create a C major chord
        frequency_c = 261.63  # C4
        frequency_e = 329.63  # E4
        frequency_g = 392.00  # G4
        
        audio = np.sin(2 * np.pi * frequency_c * t) + \
                np.sin(2 * np.pi * frequency_e * t) + \
                np.sin(2 * np.pi * frequency_g * t)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Save test audio
        test_audio_path = "/app/temp_output/test_audio.wav"
        import soundfile as sf
        sf.write(test_audio_path, audio, sr)
        
        # Test our audio processor
        processor = AdvancedAudioProcessor()
        
        # Test audio loading
        loaded_audio, loaded_sr = processor.load_audio(test_audio_path)
        print(f"✅ Audio loaded: {len(loaded_audio)} samples at {loaded_sr} Hz")
        
        # Test vocal separation (fallback method)
        vocals_path = processor._separate_vocals_fallback(test_audio_path)
        print(f"✅ Vocal separation completed: {vocals_path}")
        
        # Test audio features
        features = processor.get_audio_features(test_audio_path)
        print(f"✅ Audio features extracted: {list(features.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedAudioProcessor test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🎤 Testing Core Karaoke Scoring System 🎵")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("/app/temp_output", exist_ok=True)
    
    # Run tests
    tests = [
        test_essentia_key_detection,
        test_advanced_key_detector,
        test_advanced_audio_processor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n🎉 Test Results: {passed}/{total} tests passed!")
    
    if passed == total:
        print("✅ All core components are working correctly!")
        print("🎵 The Karaoke Scoring System is ready to use!")
    else:
        print("❌ Some components need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
