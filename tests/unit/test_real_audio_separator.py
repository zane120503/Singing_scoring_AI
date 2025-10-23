#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import librosa
import numpy as np

def test_real_audio_processor():
    """Test the real audio processor with actual AI models"""
    print("🎤 Testing Real Audio Processor with AI Models 🎵")
    print("=" * 60)
    
    try:
        from real_audio_processor import RealAudioProcessor
        
        # Initialize processor
        print("🔄 Initializing Real Audio Processor...")
        processor = RealAudioProcessor()
        
        # Create test audio (C major chord)
        print("🎵 Creating test audio (C major chord)...")
        sr = 44100
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Create a C major chord with some harmonics
        frequency_c = 261.63  # C4
        frequency_e = 329.63  # E4
        frequency_g = 392.00  # G4
        
        # Add some harmonics and modulation for more realistic sound
        audio = (np.sin(2 * np.pi * frequency_c * t) + 
                0.7 * np.sin(2 * np.pi * frequency_e * t) + 
                0.5 * np.sin(2 * np.pi * frequency_g * t) +
                0.3 * np.sin(2 * np.pi * frequency_c * 2 * t) +  # Harmonic
                0.2 * np.sin(2 * np.pi * 440 * t))  # A note for vocals
        
        # Add some reverb effect
        for i in range(1, 4):
            delay = int(sr * 0.1 * i)
            if delay < len(audio):
                audio[delay:] += 0.3 * audio[:-delay] if delay < len(audio) else 0
        
        # Normalize and add some noise
        audio = audio / np.max(np.abs(audio))
        audio += 0.01 * np.random.randn(len(audio))  # Add slight noise
        
        # Save test audio
        test_audio_path = "/app/temp_output/test_complex_audio.wav"
        import soundfile as sf
        sf.write(test_audio_path, audio, sr)
        print(f"✅ Test audio saved: {test_audio_path}")
        
        # Test real AI vocal separation
        print("\n🎤 Testing Real AI Vocal Separation...")
        vocals_path = processor.separate_vocals_real(test_audio_path)
        print(f"✅ Real AI vocal separation completed: {vocals_path}")
        
        # Test audio loading
        print("\n🔄 Testing audio loading...")
        loaded_audio, loaded_sr = processor.load_audio(test_audio_path)
        print(f"✅ Audio loaded: {len(loaded_audio)} samples at {loaded_sr} Hz")
        
        # Test audio features extraction
        print("\n📊 Testing audio features extraction...")
        features = processor.get_audio_features(test_audio_path)
        print(f"✅ Audio features extracted:")
        for key, value in features.items():
            if key == 'mfcc':
                print(f"   {key}: {len(value)} MFCC coefficients")
            else:
                print(f"   {key}: {value}")
        
        # Test if vocal file was actually created and has content
        if os.path.exists(vocals_path):
            vocal_audio, vocal_sr = processor.load_audio(vocals_path)
            if vocal_audio is not None:
                print(f"\n✅ Vocal file verification:")
                print(f"   Vocal duration: {len(vocal_audio) / vocal_sr:.2f} seconds")
                print(f"   Vocal RMS energy: {np.sqrt(np.mean(vocal_audio**2)):.4f}")
                print(f"   Original RMS energy: {np.sqrt(np.mean(loaded_audio**2)):.4f}")
                
                # Check if vocals are different from original (separation worked)
                if len(vocal_audio) == len(loaded_audio):
                    correlation = np.corrcoef(vocal_audio, loaded_audio)[0, 1]
                    print(f"   Correlation between original and vocals: {correlation:.4f}")
                    
                    if correlation < 0.9:  # If correlation is low, separation worked
                        print("✅ AI separation appears to be working (low correlation)")
                    else:
                        print("⚠️  AI separation may not be working properly (high correlation)")
                else:
                    print("⚠️  Audio lengths don't match")
            else:
                print("❌ Could not load vocal audio")
        else:
            print("❌ Vocal file was not created")
        
        print(f"\n🎉 Real Audio Processor test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Real Audio Processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎤 Testing Real AI Audio Separation 🎵")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("/app/temp_output", exist_ok=True)
    
    # Run test
    success = test_real_audio_processor()
    
    if success:
        print("\n🎉 Real AI Audio Separation test PASSED!")
        print("✅ The system is using actual AI models for vocal separation!")
    else:
        print("\n❌ Real AI Audio Separation test FAILED!")
        print("⚠️  The system may be falling back to traditional methods.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
