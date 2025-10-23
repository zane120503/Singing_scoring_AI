#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script để test Essentia AI với logging chi tiết
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

def test_essentia_ai():
    """Test Essentia AI với logging chi tiết"""
    logger.info("🎵 DEMO ESSENTIA AI KEY DETECTION")
    logger.info("=" * 50)
    
    try:
        # Import AdvancedKeyDetector với Essentia
        from advanced_key_detector import AdvancedKeyDetector
        
        # Tạo file audio demo
        logger.info("🎵 Tạo file audio demo...")
        sr = 22050
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), False)
        
        # Beat nhạc (C major chord)
        beat = (0.3 * np.sin(2 * np.pi * 261.63 * t) +  # C4
                0.2 * np.sin(2 * np.pi * 329.63 * t) +  # E4
                0.2 * np.sin(2 * np.pi * 392.00 * t))    # G4
        
        # Giọng hát (C major scale)
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
        
        # Lưu file
        sf.write('demo_beat.wav', beat, sr)
        sf.write('demo_karaoke.wav', karaoke, sr)
        logger.info("✅ Đã tạo file demo")
        
        # Test AdvancedKeyDetector với Essentia
        logger.info("🎹 Test AdvancedKeyDetector với Essentia AI...")
        detector = AdvancedKeyDetector()
        
        # Kiểm tra trạng thái Essentia
        if detector.essentia_available:
            logger.info("✅ Essentia AI đã sẵn sàng!")
        else:
            logger.warning("⚠️ Essentia AI không khả dụng, sẽ sử dụng fallback")
        
        # Test phát hiện phím beat
        logger.info("🎵 Test phát hiện phím beat...")
        beat_key = detector.detect_key('demo_beat.wav')
        logger.info(f"✅ Beat key: {beat_key['key']} {beat_key['scale']} (method: {beat_key['method']})")
        
        # Test phát hiện phím karaoke
        logger.info("🎤 Test phát hiện phím karaoke...")
        karaoke_key = detector.detect_key('demo_karaoke.wav')
        logger.info(f"✅ Karaoke key: {karaoke_key['key']} {karaoke_key['scale']} (method: {karaoke_key['method']})")
        
        # So sánh phím
        logger.info("🔄 Test so sánh phím...")
        comparison = detector.compare_keys(beat_key, karaoke_key)
        logger.info(f"✅ Key similarity: {comparison['score']}/100")
        logger.info(f"📊 Key match: {comparison['key_match']}")
        logger.info(f"📊 Mode match: {comparison['mode_match']}")
        
        logger.info("🎉 DEMO ESSENTIA AI HOÀN THÀNH!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}")
        return False
    
    finally:
        # Dọn dẹp
        for file in ['demo_beat.wav', 'demo_karaoke.wav']:
            if os.path.exists(file):
                os.remove(file)

def check_essentia_installation():
    """Kiểm tra Essentia có được cài đặt không"""
    logger.info("🔍 Kiểm tra Essentia installation...")
    
    try:
        import essentia.standard as es
        logger.info("✅ Essentia đã được cài đặt!")
        
        # Test KeyExtractor
        key_extractor = es.KeyExtractor()
        logger.info("✅ Essentia KeyExtractor sẵn sàng!")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Essentia chưa được cài đặt: {e}")
        logger.info("📦 Để cài đặt Essentia:")
        logger.info("   pip install essentia")
        logger.info("   hoặc")
        logger.info("   pip install essentia-tensorflow")
        return False

def main():
    """Hàm main"""
    logger.info("🎤 ESSENTIA AI DEMO")
    logger.info("=" * 30)
    
    # Kiểm tra Essentia
    essentia_ok = check_essentia_installation()
    
    if essentia_ok:
        # Test Essentia AI
        success = test_essentia_ai()
        if success:
            print("\n✅ Demo Essentia AI thành công!")
            print("🎵 Hệ thống đã được cập nhật để sử dụng Essentia AI!")
        else:
            print("\n❌ Demo Essentia AI thất bại!")
    else:
        print("\n⚠️ Vui lòng cài đặt Essentia trước:")
        print("   pip install essentia")

if __name__ == "__main__":
    main()
