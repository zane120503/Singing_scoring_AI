#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import librosa
import numpy as np
import soundfile as sf

def create_test_audio_files():
    """Tạo các file audio test để mô phỏng beat nhạc và ghi âm karaoke"""
    print("🎵 Creating test audio files...")
    
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # 1. Tạo beat nhạc (C major chord + drums)
    print("   📀 Creating beat track (C major + drums)...")
    
    # C major chord
    frequency_c = 261.63  # C4
    frequency_e = 329.63  # E4  
    frequency_g = 392.00  # G4
    
    beat_chord = (0.3 * np.sin(2 * np.pi * frequency_c * t) + 
                  0.2 * np.sin(2 * np.pi * frequency_e * t) + 
                  0.2 * np.sin(2 * np.pi * frequency_g * t))
    
    # Add drums (simple kick and snare pattern)
    drums = np.zeros_like(t)
    for i in range(int(duration * 2)):  # 2 beats per second
        beat_time = i * 0.5
        if beat_time < duration:
            beat_idx = int(beat_time * sr)
            if beat_idx < len(drums):
                # Kick drum
                drums[beat_idx:beat_idx+int(0.1*sr)] += 0.5 * np.exp(-10 * np.linspace(0, 1, int(0.1*sr)))
                # Snare drum (every other beat)
                if i % 2 == 1 and beat_idx + int(0.05*sr) < len(drums):
                    drums[beat_idx + int(0.05*sr):beat_idx + int(0.15*sr)] += 0.3 * np.exp(-15 * np.linspace(0, 1, int(0.1*sr)))
    
    beat_audio = beat_chord + drums
    beat_audio = beat_audio / np.max(np.abs(beat_audio)) * 0.8
    
    beat_path = "/app/temp_output/beat_track.wav"
    sf.write(beat_path, beat_audio, sr)
    print(f"   ✅ Beat track saved: {beat_path}")
    
    # 2. Tạo ghi âm karaoke (vocals + beat + noise)
    print("   🎤 Creating karaoke recording...")
    
    # Vocals (slightly off-key to test scoring)
    vocals = (0.4 * np.sin(2 * np.pi * 264 * t) +  # Slightly higher than C
              0.3 * np.sin(2 * np.pi * 334 * t) +  # Slightly higher than E
              0.2 * np.sin(2 * np.pi * 396 * t))   # Slightly higher than G
    
    # Add some vibrato and dynamics
    vibrato = 0.02 * np.sin(2 * np.pi * 5 * t)  # 5 Hz vibrato
    vocals_with_vibrato = np.zeros_like(vocals)
    for i, sample in enumerate(vocals):
        freq_shift = 1 + vibrato[i]
        vocals_with_vibrato[i] = sample * freq_shift
    
    # Combine vocals with beat (karaoke recording)
    karaoke_audio = vocals_with_vibrato + 0.3 * beat_audio  # Vocals + quieter beat
    karaoke_audio += 0.02 * np.random.randn(len(karaoke_audio))  # Add noise
    
    # Normalize
    karaoke_audio = karaoke_audio / np.max(np.abs(karaoke_audio)) * 0.9
    
    karaoke_path = "/app/temp_output/karaoke_recording.wav"
    sf.write(karaoke_path, karaoke_audio, sr)
    print(f"   ✅ Karaoke recording saved: {karaoke_path}")
    
    return beat_path, karaoke_path

def test_full_workflow():
    """Test toàn bộ quy trình chấm điểm karaoke"""
    print("🎤 Testing Full Karaoke Scoring Workflow 🎵")
    print("=" * 60)
    
    try:
        # Import các components
        from advanced_audio_processor import AdvancedAudioProcessor
        from advanced_key_detector import AdvancedKeyDetector  
        from scoring_system import KaraokeScoringSystem
        
        # Tạo file test
        beat_path, karaoke_path = create_test_audio_files()
        
        # Khởi tạo các components
        print("\n🔄 Initializing components...")
        audio_processor = AdvancedAudioProcessor()
        key_detector = AdvancedKeyDetector()
        scoring_system = KaraokeScoringSystem()
        
        # === BƯỚC 1: TÁCH GIỌNG HÁT ===
        print("\n🎤 STEP 1: Vocal Separation using AI Audio Separator...")
        vocals_path = audio_processor.separate_vocals(karaoke_path)
        print(f"✅ Vocals separated: {vocals_path}")
        
        # Kiểm tra file vocals có tồn tại không
        if not os.path.exists(vocals_path):
            print("❌ Vocal file was not created!")
            return False
        
        # === BƯỚC 2: PHÁT HIỆN PHÍM ÂM NHẠC ===
        print("\n🎵 STEP 2: Key Detection using Essentia AI...")
        
        # Detect key của beat nhạc
        print("   🔍 Detecting key of beat track...")
        beat_key = key_detector.detect_key(beat_path)
        print(f"   ✅ Beat key: {beat_key['key']} {beat_key['scale']} (confidence: {beat_key['confidence']:.3f})")
        
        # Detect key của giọng hát
        print("   🔍 Detecting key of vocals...")
        vocals_key = key_detector.detect_key(vocals_path)
        print(f"   ✅ Vocals key: {vocals_key['key']} {vocals_key['scale']} (confidence: {vocals_key['confidence']:.3f})")
        
        # === BƯỚC 3: SO SÁNH PHÍM ===
        print("\n🎯 STEP 3: Key Comparison...")
        key_comparison = key_detector.compare_keys(beat_key, vocals_key)
        print(f"   ✅ Key similarity score: {key_comparison['score']}/100")
        print(f"   📊 Key match: {key_comparison['key_match']}")
        print(f"   📊 Mode match: {key_comparison['mode_match']}")
        
        # === BƯỚC 4: CHẤM ĐIỂM TỔNG THỂ ===
        print("\n📊 STEP 4: Overall Scoring...")
        scoring_result = scoring_system.calculate_overall_score(
            karaoke_path, beat_path, vocals_path
        )
        
        print(f"   ✅ Overall score: {scoring_result['overall_score']}/100")
        print(f"   🏆 Grade: {scoring_result['grade']}")
        
        # Hiển thị điểm chi tiết
        print("\n📋 Detailed Scores:")
        criterion_names = {
            'key_accuracy': 'Độ chính xác phím',
            'pitch_accuracy': 'Độ chính xác cao độ', 
            'rhythm_accuracy': 'Độ chính xác nhịp điệu',
            'timing_accuracy': 'Độ chính xác thời gian',
            'vocal_quality': 'Chất lượng giọng hát',
            'energy_consistency': 'Tính nhất quán năng lượng',
            'pronunciation': 'Phát âm'
        }
        
        for criterion, score in scoring_result['detailed_scores'].items():
            weight = scoring_result['weights'][criterion] * 100
            criterion_name = criterion_names.get(criterion, criterion)
            print(f"   📊 {criterion_name}: {score:.1f}/100 (weight: {weight:.1f}%)")
        
        # === KIỂM TRA QUY TRÌNH ===
        print("\n🔍 WORKFLOW VERIFICATION:")
        
        # Kiểm tra input files
        print(f"   ✅ Input 1 - Beat track: {beat_path}")
        print(f"   ✅ Input 2 - Karaoke recording: {karaoke_path}")
        
        # Kiểm tra vocal separation
        if os.path.exists(vocals_path):
            print(f"   ✅ Vocal separation: SUCCESS")
            vocals_audio, vocals_sr = audio_processor.load_audio(vocals_path)
            print(f"      📊 Vocals duration: {len(vocals_audio) / vocals_sr:.2f}s")
        else:
            print(f"   ❌ Vocal separation: FAILED")
            return False
        
        # Kiểm tra key detection
        if beat_key['key'] and vocals_key['key']:
            print(f"   ✅ Key detection: SUCCESS")
            print(f"      📊 Beat key: {beat_key['key']} {beat_key['scale']}")
            print(f"      📊 Vocals key: {vocals_key['key']} {vocals_key['scale']}")
        else:
            print(f"   ❌ Key detection: FAILED")
            return False
        
        # Kiểm tra scoring
        if scoring_result['overall_score'] > 0:
            print(f"   ✅ Scoring system: SUCCESS")
            print(f"      📊 Overall score: {scoring_result['overall_score']}/100")
        else:
            print(f"   ❌ Scoring system: FAILED")
            return False
        
        print(f"\n🎉 FULL WORKFLOW TEST COMPLETED SUCCESSFULLY!")
        print(f"✅ The system is working exactly as you described:")
        print(f"   1. ✅ Takes 2 inputs: beat track + karaoke recording")
        print(f"   2. ✅ Uses AI Audio Separator to extract vocals")
        print(f"   3. ✅ Uses Essentia AI to detect keys of both files")
        print(f"   4. ✅ Uses scoring algorithm to calculate final score")
        
        return True
        
    except Exception as e:
        print(f"❌ Full workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🎤 Testing Complete Karaoke Scoring Workflow 🎵")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("/app/temp_output", exist_ok=True)
    
    # Run test
    success = test_full_workflow()
    
    if success:
        print("\n🎉 WORKFLOW TEST PASSED!")
        print("✅ Your system is working exactly as intended!")
    else:
        print("\n❌ WORKFLOW TEST FAILED!")
        print("⚠️  The system needs some adjustments.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
