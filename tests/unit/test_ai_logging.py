#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script test để kiểm tra logging AI models và xuất file MP3
"""

import os
import sys
import logging
import numpy as np
import librosa
import soundfile as sf
from audio_processor import AudioProcessor
from key_detector import KeyDetector

# Thiết lập logging chi tiết
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_models_log.txt', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def create_test_audio():
    """Tạo file audio test"""
    logger.info("🎵 Tạo file audio test...")
    
    sr = 44100
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Tạo beat nhạc (C major chord)
    beat_freqs = [261.63, 329.63, 392.00]  # C-E-G
    beat_audio = np.zeros_like(t)
    
    for freq in beat_freqs:
        beat_audio += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Tạo giọng hát (hát theo beat)
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
    beat_path = 'test_beat.wav'
    karaoke_path = 'test_karaoke.wav'
    
    sf.write(beat_path, beat_audio, sr)
    sf.write(karaoke_path, karaoke_audio, sr)
    
    logger.info(f"✅ Đã tạo file test: {beat_path}, {karaoke_path}")
    return beat_path, karaoke_path

def test_audio_processor_with_logging():
    """Test AudioProcessor với logging chi tiết"""
    logger.info("🔧 Testing AudioProcessor với logging...")
    
    try:
        # Khởi tạo processor
        logger.info("🔄 Khởi tạo AudioProcessor...")
        processor = AudioProcessor()
        
        # Tạo file test
        beat_path, karaoke_path = create_test_audio()
        
        # Test tách giọng với xuất MP3
        logger.info("🎤 Test tách giọng và xuất MP3...")
        vocals_path = processor.separate_vocals(karaoke_path, 'test_vocals.mp3')
        
        # Kiểm tra file đã tạo
        if os.path.exists(vocals_path):
            file_size = os.path.getsize(vocals_path)
            logger.info(f"✅ File vocals MP3 đã tạo: {vocals_path}")
            logger.info(f"📊 Kích thước file: {file_size / 1024:.2f} KB")
        else:
            logger.error(f"❌ File vocals không được tạo!")
        
        return vocals_path
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong AudioProcessor: {e}")
        return None

def test_key_detector_with_logging():
    """Test KeyDetector với logging chi tiết"""
    logger.info("🎹 Testing KeyDetector với logging...")
    
    try:
        # Khởi tạo detector
        logger.info("🔄 Khởi tạo KeyDetector...")
        detector = KeyDetector()
        
        # Test với file beat
        logger.info("🎵 Test phát hiện phím beat...")
        beat_key = detector.detect_key('test_beat.wav')
        logger.info(f"✅ Beat key result: {beat_key}")
        
        # Test với file vocals
        if os.path.exists('test_vocals.mp3'):
            logger.info("🎤 Test phát hiện phím vocals...")
            vocals_key = detector.detect_key('test_vocals.mp3')
            logger.info(f"✅ Vocals key result: {vocals_key}")
            
            # So sánh phím
            logger.info("🔄 Test so sánh phím...")
            comparison = detector.compare_keys(beat_key, vocals_key)
            logger.info(f"✅ Key comparison result: {comparison}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong KeyDetector: {e}")
        return False

def cleanup_test_files():
    """Dọn dẹp file test"""
    test_files = ['test_beat.wav', 'test_karaoke.wav', 'test_vocals.mp3']
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            logger.info(f"🗑️ Đã xóa {file}")

def main():
    """Hàm main"""
    logger.info("🎤 BẮT ĐẦU TEST AI MODELS VỚI LOGGING")
    logger.info("=" * 50)
    
    try:
        # Test AudioProcessor
        vocals_path = test_audio_processor_with_logging()
        
        # Test KeyDetector
        key_success = test_key_detector_with_logging()
        
        # Tổng kết
        logger.info("📋 TỔNG KẾT TEST:")
        logger.info(f"✅ AudioProcessor: {'THÀNH CÔNG' if vocals_path else 'THẤT BẠI'}")
        logger.info(f"✅ KeyDetector: {'THÀNH CÔNG' if key_success else 'THẤT BẠI'}")
        
        if vocals_path and os.path.exists(vocals_path):
            logger.info(f"🎉 File giọng đã tách (MP3): {vocals_path}")
            logger.info("📁 Bạn có thể kiểm tra file này!")
        
        logger.info("📝 Log chi tiết đã được lưu vào file: ai_models_log.txt")
        
    except Exception as e:
        logger.error(f"❌ Lỗi tổng thể: {e}")
    
    finally:
        # Hỏi có muốn dọn dẹp không
        try:
            cleanup = input("\n🗑️ Có muốn xóa file test? (y/n): ").lower().strip()
            if cleanup in ['y', 'yes', 'có']:
                cleanup_test_files()
                logger.info("✅ Đã dọn dẹp xong!")
        except:
            pass

if __name__ == "__main__":
    main()
