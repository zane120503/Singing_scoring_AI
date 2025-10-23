#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script để test AI models với logging và xuất MP3
"""

import os
import sys
import logging
import numpy as np
import librosa
import soundfile as sf

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quick_demo():
    """Demo nhanh AI models"""
    logger.info("🎤 DEMO AI MODELS VỚI LOGGING")
    logger.info("=" * 40)
    
    try:
        # Import các components
        from audio_processor import AudioProcessor
        from key_detector import KeyDetector
        
        # Tạo file audio demo đơn giản
        logger.info("🎵 Tạo file audio demo...")
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Beat nhạc (C major)
        beat = 0.3 * np.sin(2 * np.pi * 261.63 * t)  # C4
        beat += 0.2 * np.sin(2 * np.pi * 329.63 * t)  # E4
        beat += 0.2 * np.sin(2 * np.pi * 392.00 * t)  # G4
        
        # Giọng hát
        vocals = 0.4 * np.sin(2 * np.pi * 261.63 * t)  # C4
        
        # Karaoke (beat + vocals)
        karaoke = beat + vocals
        karaoke = karaoke / np.max(np.abs(karaoke))
        
        # Lưu file
        sf.write('demo_beat.wav', beat, sr)
        sf.write('demo_karaoke.wav', karaoke, sr)
        logger.info("✅ Đã tạo file demo")
        
        # Test AudioProcessor
        logger.info("🔧 Test AudioProcessor...")
        processor = AudioProcessor()
        vocals_path = processor.separate_vocals('demo_karaoke.wav', 'demo_vocals.mp3')
        logger.info(f"✅ Vocals MP3: {vocals_path}")
        
        # Test KeyDetector
        logger.info("🎹 Test KeyDetector...")
        detector = KeyDetector()
        beat_key = detector.detect_key('demo_beat.wav')
        vocals_key = detector.detect_key(vocals_path)
        logger.info(f"✅ Beat key: {beat_key['key']} {beat_key['mode']}")
        logger.info(f"✅ Vocals key: {vocals_key['key']} {vocals_key['mode']}")
        
        # So sánh
        comparison = detector.compare_keys(beat_key, vocals_key)
        logger.info(f"✅ Key similarity: {comparison['score']}/100")
        
        logger.info("🎉 DEMO HOÀN THÀNH!")
        logger.info(f"📁 File giọng đã tách (MP3): {vocals_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}")
        return False
    
    finally:
        # Dọn dẹp
        for file in ['demo_beat.wav', 'demo_karaoke.wav']:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    success = quick_demo()
    if success:
        print("\n✅ Demo thành công! Kiểm tra file demo_vocals.mp3")
    else:
        print("\n❌ Demo thất bại!")
