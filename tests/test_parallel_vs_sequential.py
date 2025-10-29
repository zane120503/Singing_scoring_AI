#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Parallel vs Sequential - So sánh song song vs tuần tự
"""

import sys
import os
import time
import logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sequential_vs_parallel():
    """Test so sánh sequential vs parallel"""
    
    print("🧪 TEST SEQUENTIAL vs PARALLEL KEY DETECTION")
    print("=" * 60)
    
    # Import workflows
    from optimized_middle_workflow import run_workflow
    from src.ai.advanced_key_detector import AdvancedKeyDetector
    
    # Test files
    karaoke_file = "D:\\singing scoring AI\\assets\\audio\\test_stereo.wav"
    beat_file = "D:\\singing scoring AI\\assets\\audio\\test.mp3"
    
    if not os.path.exists(karaoke_file) or not os.path.exists(beat_file):
        print("❌ Test files không tồn tại")
        return False
    
    print(f"📁 Karaoke: {os.path.basename(karaoke_file)}")
    print(f"📁 Beat: {os.path.basename(beat_file)}")
    
    # Test 1: Sequential (manual)
    print("\n📊 Test 1: Sequential Key Detection")
    start_time = time.time()
    
    try:
        key_detector = AdvancedKeyDetector()
        print(f"GPU Status: {'ENABLED' if key_detector.use_gpu else 'DISABLED'}")
        
        # Sequential detection
        vocals_key_seq = key_detector.detect_key(karaoke_file, audio_type='vocals')
        beat_key_seq = key_detector.detect_key(beat_file, audio_type='beat')
        
        sequential_time = time.time() - start_time
        print(f"✅ Sequential completed in {sequential_time:.2f}s")
        print(f"   Vocals: {vocals_key_seq.get('key', 'Unknown')} {vocals_key_seq.get('scale', 'Unknown')}")
        print(f"   Beat: {beat_key_seq.get('key', 'Unknown')} {beat_key_seq.get('scale', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Sequential test failed: {e}")
        return False
    
    # Test 2: Parallel (workflow)
    print("\n⚡ Test 2: Parallel Key Detection")
    start_time = time.time()
    
    try:
        result = run_workflow(karaoke_file, beat_file, duration=30.0)
        
        if result["success"]:
            parallel_time = time.time() - start_time
            print(f"✅ Parallel completed in {parallel_time:.2f}s")
            print(f"   Vocals: {result['vocals_key']['key']} {result['vocals_key']['scale']}")
            print(f"   Beat: {result['beat_key']['key']} {result['beat_key']['scale']}")
            print(f"   Score: {result['key_compare']['score']:.1f}/100")
            
            # So sánh performance
            print("\n📈 Performance Comparison:")
            print(f"   Sequential time: {sequential_time:.2f}s")
            print(f"   Parallel time: {parallel_time:.2f}s")
            
            if parallel_time < sequential_time:
                speedup = sequential_time / parallel_time
                print(f"   🚀 Speedup: {speedup:.2f}x faster!")
            else:
                print("   ⚠️ Parallel không nhanh hơn")
            
            # Kiểm tra kết quả
            vocals_match = vocals_key_seq.get('key') == result['vocals_key']['key']
            beat_match = beat_key_seq.get('key') == result['beat_key']['key']
            
            print(f"   Vocals key match: {vocals_match}")
            print(f"   Beat key match: {beat_match}")
            
            if vocals_match and beat_match:
                print("✅ Parallel workflow hoạt động chính xác!")
                return True
            else:
                print("⚠️ Kết quả parallel khác với sequential")
                return False
        else:
            print(f"❌ Parallel test failed: {result.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"❌ Parallel test failed: {e}")
        return False

def test_gpu_status():
    """Test GPU status"""
    print("\n🚀 GPU STATUS CHECK")
    print("-" * 30)
    
    try:
        from src.core.gpu_config import CUDA_AVAILABLE, get_device
        import torch
        
        print(f"CUDA Available: {CUDA_AVAILABLE}")
        if CUDA_AVAILABLE:
            print(f"Device: {get_device()}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            
            # Test GPU memory
            allocated = torch.cuda.memory_allocated() / (1024**2)
            cached = torch.cuda.memory_reserved() / (1024**2)
            print(f"GPU Memory Allocated: {allocated:.2f} MB")
            print(f"GPU Memory Cached: {cached:.2f} MB")
            
            return True
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ GPU status check failed: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST PARALLEL vs SEQUENTIAL ===")
    
    # Test GPU status
    gpu_ok = test_gpu_status()
    
    # Test sequential vs parallel
    success = test_sequential_vs_parallel()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST THÀNH CÔNG!")
        print("✅ Parallel workflow hoạt động tốt hơn sequential!")
    else:
        print("❌ TEST CẦN CẢI THIỆN!")
        print("⚠️ Parallel workflow cần được kiểm tra lại.")
    
    input("\nNhấn Enter để thoát...")
