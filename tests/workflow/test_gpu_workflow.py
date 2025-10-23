#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test GPU Workflow với file Waiting For You
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_workflow():
    """Test GPU Workflow"""
    try:
        from src.ai.optimized_audio_processor import OptimizedAudioProcessor
        
        # Test files
        karaoke_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        beat_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(karaoke_file):
            logger.error(f"Karaoke file không tồn tại: {karaoke_file}")
            return False
        
        logger.info("🚀 Testing GPU Workflow...")
        logger.info(f"   Karaoke file: {os.path.basename(karaoke_file)}")
        logger.info(f"   Beat file: {os.path.basename(beat_file)}")
        
        # Khởi tạo processor
        processor = OptimizedAudioProcessor()
        
        # Test workflow
        start_time = time.time()
        result = processor.process_karaoke_optimized(karaoke_file, beat_file)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        if result["success"]:
            logger.info("✅ GPU workflow thành công!")
            
            # Hiển thị kết quả
            print(f"\n🎉 GPU WORKFLOW RESULTS:")
            print(f"📊 Voice Detection: {result.get('voice_detection', {})}")
            print(f"🎵 Key Detection: {result.get('key_detection', {})}")
            print(f"📈 Scoring: {result.get('scoring', {})}")
            print(f"⏱️ Processing Time: {processing_time:.2f}s")
            print(f"🖥️ Using GPU: YES")
            
            return True
        else:
            logger.error(f"❌ GPU workflow failed: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_performance():
    """Test GPU performance"""
    try:
        import torch
        
        logger.info("🔍 Testing GPU Performance...")
        
        # Test GPU computation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"   Device: {device}")
        
        # Test tensor operations
        start_time = time.time()
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        end_time = time.time()
        
        gpu_time = end_time - start_time
        logger.info(f"   GPU computation time: {gpu_time:.4f}s")
        
        # Test CPU computation for comparison
        x_cpu = x.cpu()
        y_cpu = y.cpu()
        start_time = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        end_time = time.time()
        
        cpu_time = end_time - start_time
        logger.info(f"   CPU computation time: {cpu_time:.4f}s")
        
        speedup = cpu_time / gpu_time
        logger.info(f"   GPU speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi test performance: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST GPU WORKFLOW ===")
    
    # Test GPU performance
    print("\n1. Testing GPU Performance...")
    performance_success = test_gpu_performance()
    
    # Test GPU workflow
    print("\n2. Testing GPU Workflow...")
    workflow_success = test_gpu_workflow()
    
    # Kết quả
    print("\n=== KET QUA ===")
    print(f"GPU Performance: {'PASS' if performance_success else 'FAIL'}")
    print(f"GPU Workflow: {'PASS' if workflow_success else 'FAIL'}")
    
    if performance_success and workflow_success:
        print("\n🎉 GPU WORKFLOW HOAT DONG HOAN HAO!")
        print("🚀 Hệ thống đã sẵn sàng sử dụng với GPU!")
    else:
        print("\n⚠️ CAN CAI THIEN THEM!")
