#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Parallel Key Detection - Kiểm tra song song hoạt động
"""

import sys
import os
import time
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ai.advanced_key_detector import AdvancedKeyDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_parallel_key_detection():
    """Test song song key detection với 2 file audio"""
    
    # Đường dẫn test files (cần cập nhật theo hệ thống của bạn)
    vocals_file = "D:\\singing scoring AI\\assets\\audio\\test_stereo.wav"
    beat_file = "D:\\singing scoring AI\\assets\\audio\\test.mp3"
    
    # Kiểm tra files tồn tại
    if not os.path.exists(vocals_file):
        logger.warning(f"Vocals file không tồn tại: {vocals_file}")
        vocals_file = None
    if not os.path.exists(beat_file):
        logger.warning(f"Beat file không tồn tại: {beat_file}")
        beat_file = None
    
    if not vocals_file or not beat_file:
        logger.error("❌ Không có file test để chạy song song")
        return False
    
    logger.info("🧪 Bắt đầu test song song key detection...")
    logger.info(f"Vocals file: {vocals_file}")
    logger.info(f"Beat file: {beat_file}")
    
    key_detector = AdvancedKeyDetector()
    
    # Test 1: Sequential (tuần tự) - để so sánh thời gian
    logger.info("\n📊 Test 1: Sequential Key Detection")
    start_time = time.time()
    
    try:
        vocals_key_seq = key_detector.detect_key(vocals_file, audio_type='vocals')
        beat_key_seq = key_detector.detect_key(beat_file, audio_type='beat')
        sequential_time = time.time() - start_time
        
        logger.info(f"✅ Sequential completed in {sequential_time:.2f}s")
        logger.info(f"   Vocals key: {vocals_key_seq.get('key', 'Unknown')}")
        logger.info(f"   Beat key: {beat_key_seq.get('key', 'Unknown')}")
    except Exception as e:
        logger.error(f"❌ Sequential test failed: {e}")
        return False
    
    # Test 2: Parallel (song song)
    logger.info("\n⚡ Test 2: Parallel Key Detection")
    start_time = time.time()
    
    import concurrent.futures
    
    def detect_vocals():
        try:
            logger.info("🎤 Detecting vocals key...")
            result = key_detector.detect_key(vocals_file, audio_type='vocals')
            logger.info(f"✅ Vocals key: {result.get('key', 'Unknown')}")
            return result
        except Exception as e:
            logger.error(f"Vocals detection failed: {e}")
            return None
    
    def detect_beat():
        try:
            logger.info("🎵 Detecting beat key...")
            result = key_detector.detect_key(beat_file, audio_type='beat')
            logger.info(f"✅ Beat key: {result.get('key', 'Unknown')}")
            return result
        except Exception as e:
            logger.error(f"Beat detection failed: {e}")
            return None
    
    # Chạy song song
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        vocals_future = executor.submit(detect_vocals)
        beat_future = executor.submit(detect_beat)
        
        vocals_key_par = vocals_future.result()
        beat_key_par = beat_future.result()
    
    parallel_time = time.time() - start_time
    
    logger.info(f"✅ Parallel completed in {parallel_time:.2f}s")
    
    # So sánh kết quả
    logger.info("\n📈 Kết quả so sánh:")
    logger.info(f"   Sequential time: {sequential_time:.2f}s")
    logger.info(f"   Parallel time: {parallel_time:.2f}s")
    
    if parallel_time < sequential_time:
        speedup = sequential_time / parallel_time
        logger.info(f"   🚀 Speedup: {speedup:.2f}x faster!")
    else:
        logger.info("   ⚠️ Parallel không nhanh hơn (có thể do overhead)")
    
    # Kiểm tra kết quả có giống nhau không
    vocals_match = vocals_key_seq.get('key') == vocals_key_par.get('key')
    beat_match = beat_key_seq.get('key') == beat_key_par.get('key')
    
    logger.info(f"   Vocals key match: {vocals_match}")
    logger.info(f"   Beat key match: {beat_match}")
    
    if vocals_match and beat_match:
        logger.info("✅ Song song hoạt động chính xác!")
        return True
    else:
        logger.warning("⚠️ Kết quả song song khác với tuần tự")
        return False

if __name__ == "__main__":
    print("=== TEST PARALLEL KEY DETECTION ===")
    success = test_parallel_key_detection()
    
    if success:
        print("\n🎉 Test thành công! Song song hoạt động tốt.")
    else:
        print("\n❌ Test thất bại hoặc cần cải thiện.")
