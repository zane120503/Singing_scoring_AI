#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Parallel Key Detection Demo - Demo song song với GPU acceleration
"""

import sys
import os
import time
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ai.advanced_key_detector import AdvancedKeyDetector
from src.core.gpu_config import CUDA_AVAILABLE, get_device

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_key_detection():
    """Test GPU key detection với 2 file audio"""
    
    print("🚀 GPU PARALLEL KEY DETECTION DEMO")
    print("=" * 60)
    
    # Kiểm tra GPU availability
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"GPU Device: {get_device()}")
    else:
        print("⚠️ CUDA not available, will use CPU")
    
    # Đường dẫn test files
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
        logger.error("❌ Không có file test để chạy GPU song song")
        return False
    
    logger.info("🧪 Bắt đầu test GPU song song key detection...")
    logger.info(f"Vocals file: {vocals_file}")
    logger.info(f"Beat file: {beat_file}")
    
    # Test 1: Sequential (tuần tự) - để so sánh thời gian
    logger.info("\n📊 Test 1: Sequential Key Detection")
    start_time = time.time()
    
    try:
        key_detector = AdvancedKeyDetector()
        logger.info(f"Key Detector GPU: {'ENABLED' if key_detector.use_gpu else 'DISABLED'}")
        
        vocals_key_seq = key_detector.detect_key(vocals_file, audio_type='vocals')
        beat_key_seq = key_detector.detect_key(beat_file, audio_type='beat')
        sequential_time = time.time() - start_time
        
        logger.info(f"✅ Sequential completed in {sequential_time:.2f}s")
        logger.info(f"   Vocals key: {vocals_key_seq.get('key', 'Unknown')} ({vocals_key_seq.get('method', 'Unknown')})")
        logger.info(f"   Beat key: {beat_key_seq.get('key', 'Unknown')} ({beat_key_seq.get('method', 'Unknown')})")
    except Exception as e:
        logger.error(f"❌ Sequential test failed: {e}")
        return False
    
    # Test 2: Parallel (song song) với GPU
    logger.info("\n⚡ Test 2: GPU Parallel Key Detection")
    start_time = time.time()
    
    import concurrent.futures
    
    def detect_vocals():
        try:
            logger.info("🎤 Detecting vocals key with GPU...")
            result = key_detector.detect_key(vocals_file, audio_type='vocals')
            logger.info(f"✅ Vocals key: {result.get('key', 'Unknown')} ({result.get('method', 'Unknown')})")
            return result
        except Exception as e:
            logger.error(f"Vocals detection failed: {e}")
            return None
    
    def detect_beat():
        try:
            logger.info("🎵 Detecting beat key with GPU...")
            result = key_detector.detect_key(beat_file, audio_type='beat')
            logger.info(f"✅ Beat key: {result.get('key', 'Unknown')} ({result.get('method', 'Unknown')})")
            return result
        except Exception as e:
            logger.error(f"Beat detection failed: {e}")
            return None
    
    # Chạy song song với GPU
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        vocals_future = executor.submit(detect_vocals)
        beat_future = executor.submit(detect_beat)
        
        vocals_key_par = vocals_future.result()
        beat_key_par = beat_future.result()
    
    parallel_time = time.time() - start_time
    
    logger.info(f"✅ GPU Parallel completed in {parallel_time:.2f}s")
    
    # So sánh kết quả
    logger.info("\n📈 Kết quả so sánh:")
    logger.info(f"   Sequential time: {sequential_time:.2f}s")
    logger.info(f"   GPU Parallel time: {parallel_time:.2f}s")
    
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
    
    # Kiểm tra GPU usage
    if CUDA_AVAILABLE:
        try:
            import torch
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            cached_memory = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"   GPU Memory Allocated: {allocated_memory:.2f} GB")
            logger.info(f"   GPU Memory Cached: {cached_memory:.2f} GB")
        except:
            pass
    
    if vocals_match and beat_match:
        logger.info("✅ GPU song song hoạt động chính xác!")
        return True
    else:
        logger.warning("⚠️ Kết quả song song khác với tuần tự")
        return False

def test_gpu_memory_usage():
    """Test GPU memory usage"""
    if not CUDA_AVAILABLE:
        print("❌ CUDA not available, cannot test GPU memory")
        return False
    
    try:
        import torch
        
        print("\n🧠 GPU Memory Usage Test")
        print("-" * 30)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Test memory allocation
        device = get_device()
        print(f"Using device: {device}")
        
        # Allocate some memory
        x = torch.randn(1000, 1000, device=device)
        allocated = torch.cuda.memory_allocated() / (1024**2)
        print(f"Memory after allocation: {allocated:.2f} MB")
        
        # Test key detector memory usage
        key_detector = AdvancedKeyDetector()
        detector_memory = torch.cuda.memory_allocated() / (1024**2)
        print(f"Memory after key detector: {detector_memory:.2f} MB")
        
        # Clean up
        del x
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / (1024**2)
        print(f"Memory after cleanup: {final_memory:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU memory test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== GPU PARALLEL KEY DETECTION DEMO ===")
    
    # Test GPU memory
    memory_test = test_gpu_memory_usage()
    
    # Test GPU key detection
    success = test_gpu_key_detection()
    
    print("\n" + "=" * 60)
    if success and memory_test:
        print("🎉 GPU Demo thành công! Song song với GPU hoạt động tốt!")
    else:
        print("❌ GPU Demo cần cải thiện hoặc GPU không khả dụng.")
    
    input("\nNhấn Enter để thoát...")
