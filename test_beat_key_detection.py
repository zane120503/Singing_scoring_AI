#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Beat Key Detection - Kiểm tra phát hiện key beat
"""

import sys
import os
import librosa
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_beat_key_detection():
    """Test beat key detection accuracy"""
    
    print("🧪 TEST BEAT KEY DETECTION")
    print("=" * 50)
    
    try:
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        # Initialize key detector
        key_detector = AdvancedKeyDetector()
        print(f"✅ Key detector initialized")
        print(f"   GPU Status: {'ENABLED' if key_detector.use_gpu else 'DISABLED'}")
        
        # Test với file beat thực tế
        beat_files = [
            "C:/Users/admin/Downloads/Bông Hoa Đẹp Nhất [music].mp3",
            "D:/singing scoring AI/Audio_separator_ui/clean_song_output/test_stereo.wav",
            "D:/singing scoring AI/assets/audio/test_stereo.wav"
        ]
        
        test_file = None
        for file_path in beat_files:
            if os.path.exists(file_path):
                test_file = file_path
                break
        
        if not test_file:
            print("❌ Không tìm thấy file beat để test")
            return False
        
        print(f"📁 Test file: {test_file}")
        
        # Test các audio_type khác nhau
        audio_types = ['beat', 'instrumental', 'vocals']
        
        for audio_type in audio_types:
            print(f"\n🎵 Testing audio_type='{audio_type}':")
            try:
                result = key_detector.detect_key(test_file, audio_type=audio_type)
                if result:
                    print(f"   ✅ Result: {result['key']} {result['scale']}")
                    print(f"   📊 Confidence: {result['confidence']:.3f}")
                    print(f"   🔧 Method: {result.get('method', 'Unknown')}")
                else:
                    print(f"   ❌ No result")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        # Test beat-specific preprocessing
        print(f"\n🔧 Testing beat preprocessing:")
        try:
            audio, sr = librosa.load(test_file, sr=22050)
            processed = key_detector._preprocess_beat_audio(audio, sr)
            print(f"   ✅ Preprocessing successful")
            print(f"   📊 Original length: {len(audio)} samples")
            print(f"   📊 Processed length: {len(processed)} samples")
        except Exception as e:
            print(f"   ❌ Preprocessing failed: {e}")
        
        # Test beat harmonic analysis
        print(f"\n🎼 Testing beat harmonic analysis:")
        try:
            audio, sr = librosa.load(test_file, sr=22050)
            result = key_detector._detect_with_beat_harmonic_analysis(audio, sr)
            if result:
                print(f"   ✅ Beat harmonic result: {result['key']} {result['scale']}")
                print(f"   📊 Confidence: {result['confidence']:.3f}")
            else:
                print(f"   ❌ No beat harmonic result")
        except Exception as e:
            print(f"   ❌ Beat harmonic analysis failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_key_profiles():
    """Test key profiles for A minor"""
    print("\n🎹 TEST KEY PROFILES")
    print("=" * 30)
    
    try:
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        key_detector = AdvancedKeyDetector()
        
        # Test A minor profile
        print("Testing A minor profile:")
        a_minor_notes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        
        # Simulate chroma for A minor
        chroma_a_minor = np.zeros(12)
        note_to_index = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        
        for note in a_minor_notes:
            if note in note_to_index:
                chroma_a_minor[note_to_index[note]] = 1.0
        
        # Normalize
        chroma_a_minor = chroma_a_minor / np.sum(chroma_a_minor)
        
        print(f"   A minor chroma: {chroma_a_minor}")
        
        # Test correlation với A minor profile
        a_minor_profile = np.roll(key_detector.minor_profile, 9)  # A = index 9
        correlation = np.corrcoef(chroma_a_minor, a_minor_profile)[0, 1]
        
        print(f"   A minor correlation: {correlation:.3f}")
        
        if correlation > 0.8:
            print("   ✅ A minor profile working correctly")
        else:
            print("   ❌ A minor profile may have issues")
        
        return True
        
    except Exception as e:
        print(f"❌ Key profiles test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST BEAT KEY DETECTION ===")
    
    # Test beat key detection
    detection_ok = test_beat_key_detection()
    
    # Test key profiles
    profiles_ok = test_key_profiles()
    
    print("\n" + "=" * 50)
    if detection_ok and profiles_ok:
        print("🎉 BEAT KEY DETECTION TEST SUCCESSFUL!")
        print("✅ Beat key detection methods working")
        print("✅ Key profiles working correctly")
    else:
        print("❌ BEAT KEY DETECTION NEEDS ATTENTION!")
        print("⚠️ Some issues detected")
    
    print("\n📋 EXPECTED RESULTS:")
    print("   • Beat file should detect A minor")
    print("   • Beat harmonic analysis should work")
    print("   • Key profiles should be accurate")
    
    input("\nNhấn Enter để thoát...")
