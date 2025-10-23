#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test hệ thống karaoke scoring không cần Essentia
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

def test_system_without_essentia():
    """Test hệ thống không cần Essentia"""
    logger.info("🎤 TEST HỆ THỐNG KHÔNG CẦN ESSENTIA")
    logger.info("=" * 50)
    
    try:
        # Import các components
        from audio_processor import AudioProcessor
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
        sf.write('test_beat.wav', beat, sr)
        sf.write('test_karaoke.wav', karaoke, sr)
        logger.info("✅ Đã tạo file demo")
        
        # Test AudioProcessor
        logger.info("🔧 Test AudioProcessor...")
        processor = AudioProcessor()
        vocals_path = processor.separate_vocals('test_karaoke.wav', 'test_vocals.mp3')
        logger.info(f"✅ Vocals MP3: {vocals_path}")
        
        # Test AdvancedKeyDetector (sẽ sử dụng fallback)
        logger.info("🎹 Test AdvancedKeyDetector...")
        detector = AdvancedKeyDetector()
        
        # Kiểm tra trạng thái Essentia
        if detector.essentia_available:
            logger.info("✅ Essentia AI đã sẵn sàng!")
        else:
            logger.info("⚠️ Essentia AI không khả dụng, sử dụng fallback method")
        
        # Test phát hiện phím beat
        logger.info("🎵 Test phát hiện phím beat...")
        beat_key = detector.detect_key('test_beat.wav')
        logger.info(f"✅ Beat key: {beat_key['key']} {beat_key['scale']} (method: {beat_key['method']})")
        
        # Test phát hiện phím vocals
        logger.info("🎤 Test phát hiện phím vocals...")
        vocals_key = detector.detect_key(vocals_path)
        logger.info(f"✅ Vocals key: {vocals_key['key']} {vocals_key['scale']} (method: {vocals_key['method']})")
        
        # So sánh phím
        logger.info("🔄 Test so sánh phím...")
        comparison = detector.compare_keys(beat_key, vocals_key)
        logger.info(f"✅ Key similarity: {comparison['score']}/100")
        
        # Test Scoring System
        logger.info("📊 Test Scoring System...")
        from scoring_system import KaraokeScoringSystem
        scoring = KaraokeScoringSystem()
        result = scoring.calculate_overall_score('test_karaoke.wav', 'test_beat.wav', vocals_path)
        logger.info(f"✅ Overall score: {result['overall_score']}/100")
        logger.info(f"🏆 Grade: {result['grade']}")
        
        logger.info("🎉 TEST HOÀN THÀNH!")
        logger.info("✅ Hệ thống hoạt động tốt mà không cần Essentia!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}")
        return False
    
    finally:
        # Dọn dẹp
        for file in ['test_beat.wav', 'test_karaoke.wav', 'test_vocals.mp3']:
            if os.path.exists(file):
                os.remove(file)

def check_system_status():
    """Kiểm tra trạng thái hệ thống"""
    logger.info("🔍 KIỂM TRA TRẠNG THÁI HỆ THỐNG")
    logger.info("=" * 40)
    
    # Kiểm tra các thư viện cần thiết
    required_libs = ['torch', 'librosa', 'soundfile', 'numpy', 'scipy']
    missing_libs = []
    
    for lib in required_libs:
        try:
            __import__(lib)
            logger.info(f"✅ {lib}: OK")
        except ImportError:
            logger.error(f"❌ {lib}: MISSING")
            missing_libs.append(lib)
    
    # Kiểm tra Essentia
    try:
        import essentia.standard as es
        logger.info("✅ Essentia: AVAILABLE")
        return True
    except ImportError:
        logger.warning("⚠️ Essentia: NOT AVAILABLE (sẽ sử dụng fallback)")
        return len(missing_libs) == 0

def main():
    """Hàm main"""
    logger.info("🎤 KARAOKE SCORING SYSTEM - NO ESSENTIA TEST")
    logger.info("=" * 60)
    
    # Kiểm tra trạng thái hệ thống
    system_ok = check_system_status()
    
    if system_ok:
        # Test hệ thống
        success = test_system_without_essentia()
        if success:
            print("\n🎉 HỆ THỐNG HOẠT ĐỘNG TỐT!")
            print("✅ Bạn có thể sử dụng hệ thống mà không cần Essentia")
            print("💡 Để cài đặt Essentia sau:")
            print("   python install_essentia_windows.py")
        else:
            print("\n❌ Hệ thống có vấn đề!")
    else:
        print("\n⚠️ Thiếu một số thư viện cần thiết!")
        print("📦 Chạy: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
