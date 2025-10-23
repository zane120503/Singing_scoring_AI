#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script để test hệ thống chấm điểm karaoke
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from audio_processor import AudioProcessor
from key_detector import KeyDetector
from scoring_system import KaraokeScoringSystem

def create_demo_audio():
    """Tạo file âm thanh demo để test"""
    print("🎵 Đang tạo file âm thanh demo...")
    
    # Tạo beat nhạc đơn giản (C major scale)
    sr = 22050
    duration = 10  # 10 giây
    
    # Tạo beat với C major chord progression
    t = np.linspace(0, duration, int(sr * duration))
    
    # C major chord: C-E-G
    beat_freqs = [261.63, 329.63, 392.00]  # C4, E4, G4
    beat_audio = np.zeros_like(t)
    
    for freq in beat_freqs:
        beat_audio += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Thêm một chút percussion
    beat_audio += 0.1 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 0.5)
    
    # Normalize
    beat_audio = beat_audio / np.max(np.abs(beat_audio))
    
    # Tạo giọng hát đơn giản (hát theo beat)
    vocals_freqs = [261.63, 293.66, 329.63, 349.23, 392.00]  # C-D-E-F-G
    vocals_audio = np.zeros_like(t)
    
    for i, freq in enumerate(vocals_freqs):
        start_time = i * duration / len(vocals_freqs)
        end_time = (i + 1) * duration / len(vocals_freqs)
        mask = (t >= start_time) & (t < end_time)
        vocals_audio[mask] = 0.2 * np.sin(2 * np.pi * freq * t[mask])
    
    # Tạo karaoke (beat + vocals)
    karaoke_audio = beat_audio + vocals_audio
    karaoke_audio = karaoke_audio / np.max(np.abs(karaoke_audio))
    
    # Lưu các file
    sf.write('demo_beat.wav', beat_audio, sr)
    sf.write('demo_vocals.wav', vocals_audio, sr)
    sf.write('demo_karaoke.wav', karaoke_audio, sr)
    
    print("✅ Đã tạo các file demo:")
    print("   - demo_beat.wav (beat nhạc)")
    print("   - demo_vocals.wav (giọng hát)")
    print("   - demo_karaoke.wav (karaoke)")
    
    return 'demo_karaoke.wav', 'demo_beat.wav', 'demo_vocals.wav'

def test_audio_processor():
    """Test AudioProcessor"""
    print("\n🔧 Testing AudioProcessor...")
    
    processor = AudioProcessor()
    
    # Test với file demo
    if os.path.exists('demo_karaoke.wav'):
        try:
            # Test tách giọng
            vocals_path = processor.separate_vocals('demo_karaoke.wav', 'test_vocals.wav')
            print(f"✅ Tách giọng thành công: {vocals_path}")
            
            # Test trích xuất đặc trưng
            features = processor.get_audio_features('demo_karaoke.wav')
            print(f"✅ Đặc trưng âm thanh: {len(features)} features")
            
        except Exception as e:
            print(f"❌ Lỗi AudioProcessor: {e}")
    else:
        print("⚠️ File demo không tồn tại")

def test_key_detector():
    """Test KeyDetector"""
    print("\n🎹 Testing KeyDetector...")
    
    detector = KeyDetector()
    
    # Test với file demo
    if os.path.exists('demo_beat.wav') and os.path.exists('demo_vocals.wav'):
        try:
            # Phát hiện phím beat
            beat_key = detector.detect_key('demo_beat.wav')
            print(f"✅ Phím beat: {beat_key['key']} {beat_key['mode']} (độ tin cậy: {beat_key['confidence']:.2f})")
            
            # Phát hiện phím vocals
            vocals_key = detector.detect_key('demo_vocals.wav')
            print(f"✅ Phím vocals: {vocals_key['key']} {vocals_key['mode']} (độ tin cậy: {vocals_key['confidence']:.2f})")
            
            # So sánh phím
            comparison = detector.compare_keys(beat_key, vocals_key)
            print(f"✅ Điểm tương đồng phím: {comparison['score']}/100")
            
        except Exception as e:
            print(f"❌ Lỗi KeyDetector: {e}")
    else:
        print("⚠️ File demo không tồn tại")

def test_scoring_system():
    """Test KaraokeScoringSystem"""
    print("\n📊 Testing KaraokeScoringSystem...")
    
    scoring = KaraokeScoringSystem()
    
    # Test với file demo
    if all(os.path.exists(f) for f in ['demo_karaoke.wav', 'demo_beat.wav', 'demo_vocals.wav']):
        try:
            # Tính điểm tổng thể
            result = scoring.calculate_overall_score(
                'demo_karaoke.wav', 
                'demo_beat.wav', 
                'demo_vocals.wav'
            )
            
            print(f"✅ Điểm tổng thể: {result['overall_score']}/100")
            print(f"✅ Xếp loại: {result['grade']}")
            
            print("\n📋 Chi tiết điểm số:")
            for criterion, score in result['detailed_scores'].items():
                print(f"   - {criterion}: {score:.1f}/100")
            
            print("\n💬 Phản hồi:")
            for feedback in result['feedback']:
                print(f"   {feedback}")
                
        except Exception as e:
            print(f"❌ Lỗi KaraokeScoringSystem: {e}")
    else:
        print("⚠️ File demo không tồn tại")

def cleanup_demo_files():
    """Dọn dẹp file demo"""
    demo_files = [
        'demo_beat.wav', 'demo_vocals.wav', 'demo_karaoke.wav',
        'test_vocals.wav'
    ]
    
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ Đã xóa {file}")

def main():
    """Hàm main cho demo"""
    print("🎤 Demo hệ thống chấm điểm karaoke")
    print("=" * 40)
    
    try:
        # Tạo file demo
        karaoke_file, beat_file, vocals_file = create_demo_audio()
        
        # Test các components
        test_audio_processor()
        test_key_detector()
        test_scoring_system()
        
        print("\n🎉 Demo hoàn thành!")
        print("\n💡 Để sử dụng với file thật:")
        print("   1. Chạy: python main.py")
        print("   2. Chọn file karaoke và beat nhạc của bạn")
        print("   3. Nhấn 'Bắt đầu phân tích và chấm điểm'")
        
    except Exception as e:
        print(f"❌ Lỗi trong demo: {e}")
    
    finally:
        # Hỏi có muốn dọn dẹp không
        try:
            cleanup = input("\n🗑️ Có muốn xóa file demo? (y/n): ").lower().strip()
            if cleanup in ['y', 'yes', 'có']:
                cleanup_demo_files()
                print("✅ Đã dọn dẹp xong!")
        except:
            pass

if __name__ == "__main__":
    main()

