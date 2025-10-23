#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra tích hợp Audio Separator
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf

def test_audio_separator_integration():
    """Test tích hợp Audio Separator"""
    print("Testing Audio Separator Integration")
    print("=" * 50)
    
    try:
        # Import Advanced Audio Processor
        from advanced_audio_processor import AdvancedAudioProcessor
        
        # Khởi tạo processor
        processor = AdvancedAudioProcessor()
        print("Advanced Audio Processor đã được khởi tạo")
        
        # Tạo file âm thanh demo
        print("\nTạo file âm thanh demo...")
        demo_audio = create_demo_audio()
        
        # Test tách giọng
        print("\nTesting vocal separation...")
        vocals_path = processor.separate_vocals(demo_audio)
        
        if vocals_path and os.path.exists(vocals_path):
            print(f"Tách giọng thành công: {vocals_path}")
            
            # Test trích xuất đặc trưng
            print("\nTesting feature extraction...")
            features = processor.get_audio_features(vocals_path)
            print(f"Đặc trưng: {len(features)} features")
            
            # Test cleanup
            print("\nTesting cleanup...")
            processor.cleanup_temp_files()
            print("Cleanup thành công")
            
        else:
            print("Tách giọng thất bại")
            
    except Exception as e:
        print(f"Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()

def create_demo_audio():
    """Tạo file âm thanh demo"""
    try:
        # Tạo beat nhạc đơn giản
        sr = 44100
        duration = 5  # 5 giây
        
        t = np.linspace(0, duration, int(sr * duration))
        
        # C major chord: C-E-G
        beat_freqs = [261.63, 329.63, 392.00]  # C4, E4, G4
        beat_audio = np.zeros_like(t)
        
        for freq in beat_freqs:
            beat_audio += 0.3 * np.sin(2 * np.pi * freq * t)
        
        # Tạo giọng hát đơn giản
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
        
        # Lưu file
        demo_path = 'test_karaoke_demo.wav'
        sf.write(demo_path, karaoke_audio, sr)
        
        print(f"Đã tạo file demo: {demo_path}")
        return demo_path
        
    except Exception as e:
        print(f"Lỗi khi tạo demo: {e}")
        return None

def test_audio_separator_ui():
    """Test Audio Separator UI trực tiếp"""
    print("\nTesting Audio Separator UI...")
    
    try:
        # Kiểm tra xem Audio Separator UI có sẵn không
        separator_path = os.path.join('Audio_separator_ui', 'app.py')
        if os.path.exists(separator_path):
            print("Audio Separator UI đã có sẵn")
            
            # Kiểm tra models
            models_path = os.path.join('..', '..', 'assets', 'models', 'mdx_models')
            if os.path.exists(models_path):
                print("Models directory đã có sẵn")
                
                # Liệt kê các file trong models
                model_files = os.listdir(models_path)
                print(f"Model files: {len(model_files)} files")
                for file in model_files[:5]:  # Hiển thị 5 file đầu
                    print(f"   - {file}")
                    
            else:
                print("Models directory chưa có")
                
        else:
            print("Audio Separator UI không tìm thấy")
            
    except Exception as e:
        print(f"Lỗi khi test Audio Separator UI: {e}")

def main():
    """Hàm main"""
    print("Test Audio Separator Integration")
    print("=" * 50)
    
    # Test Audio Separator UI
    test_audio_separator_ui()
    
    # Test tích hợp
    test_audio_separator_integration()
    
    print("\nTest hoàn thành!")
    
    # Dọn dẹp
    try:
        if os.path.exists('test_karaoke_demo.wav'):
            os.remove('test_karaoke_demo.wav')
            print("Đã dọn dẹp file demo")
    except:
        pass

if __name__ == "__main__":
    main()
