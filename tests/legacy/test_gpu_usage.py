#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test GPU usage cho hệ thống
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_config():
    """Test GPU configuration"""
    try:
        from src.core.gpu_config import get_device, force_cuda, CUDA_AVAILABLE
        
        print("=== TEST GPU CONFIG ===")
        print(f"CUDA Available: {CUDA_AVAILABLE}")
        
        device = get_device()
        print(f"Device: {device}")
        
        force_cuda()
        print("GPU forced successfully")
        
        return True
        
    except Exception as e:
        print(f"Error testing GPU config: {e}")
        return False

def test_audio_processor_gpu():
    """Test Audio Processor GPU usage"""
    try:
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        
        print("\n=== TEST AUDIO PROCESSOR GPU ===")
        
        processor = AdvancedAudioProcessor()
        print(f"Audio Processor device: {processor.device}")
        
        # Test GPU computation
        if processor.device.type == 'cuda':
            x = torch.randn(100, 100).to(processor.device)
            y = torch.mm(x, x.t())
            print("GPU computation test: SUCCESS")
            return True
        else:
            print("Using CPU")
            return False
            
    except Exception as e:
        print(f"Error testing Audio Processor: {e}")
        return False

def test_ai_audio_separator_gpu():
    """Test AI Audio Separator GPU usage"""
    try:
        from src.ai.ai_audio_separator import AIAudioSeparator
        
        print("\n=== TEST AI AUDIO SEPARATOR GPU ===")
        
        separator = AIAudioSeparator()
        print(f"AI Audio Separator device: {separator.device}")
        
        if separator.device.type == 'cuda':
            print("AI Audio Separator using GPU: SUCCESS")
            return True
        else:
            print("AI Audio Separator using CPU")
            return False
            
    except Exception as e:
        print(f"Error testing AI Audio Separator: {e}")
        return False

def test_gpu_memory():
    """Test GPU memory usage"""
    try:
        print("\n=== TEST GPU MEMORY ===")
        
        if torch.cuda.is_available():
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            cached_memory = torch.cuda.memory_reserved() / (1024**3)
            
            print(f"Total GPU Memory: {total_memory:.2f} GB")
            print(f"Allocated Memory: {allocated_memory:.2f} GB")
            print(f"Cached Memory: {cached_memory:.2f} GB")
            print(f"Free Memory: {total_memory - cached_memory:.2f} GB")
            
            # Test memory allocation
            try:
                x = torch.randn(1000, 1000).cuda()
                print("GPU memory allocation test: SUCCESS")
                del x
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                print(f"GPU memory allocation test failed: {e}")
                return False
        else:
            print("CUDA not available")
            return False
            
    except Exception as e:
        print(f"Error testing GPU memory: {e}")
        return False

def test_optimized_processor_gpu():
    """Test Optimized Audio Processor GPU usage"""
    try:
        from src.ai.optimized_audio_processor import OptimizedAudioProcessor
        
        print("\n=== TEST OPTIMIZED PROCESSOR GPU ===")
        
        processor = OptimizedAudioProcessor()
        print("Optimized Audio Processor initialized")
        
        # Check if components use GPU
        if hasattr(processor.audio_processor, 'device'):
            print(f"Audio Processor device: {processor.audio_processor.device}")
        
        return True
        
    except Exception as e:
        print(f"Error testing Optimized Processor: {e}")
        return False

def main():
    """Hàm chính"""
    print("=== TEST GPU USAGE CHO HE THONG ===")
    
    # Test GPU config
    gpu_config_ok = test_gpu_config()
    
    # Test Audio Processor GPU
    audio_processor_gpu = test_audio_processor_gpu()
    
    # Test AI Audio Separator GPU
    ai_separator_gpu = test_ai_audio_separator_gpu()
    
    # Test GPU memory
    gpu_memory_ok = test_gpu_memory()
    
    # Test Optimized Processor GPU
    optimized_processor_gpu = test_optimized_processor_gpu()
    
    # Tóm tắt
    print("\n=== TOM TAT ===")
    print(f"GPU Config: {'OK' if gpu_config_ok else 'ERROR'}")
    print(f"Audio Processor GPU: {'YES' if audio_processor_gpu else 'NO'}")
    print(f"AI Audio Separator GPU: {'YES' if ai_separator_gpu else 'NO'}")
    print(f"GPU Memory: {'OK' if gpu_memory_ok else 'ERROR'}")
    print(f"Optimized Processor GPU: {'OK' if optimized_processor_gpu else 'ERROR'}")
    
    if all([gpu_config_ok, audio_processor_gpu, ai_separator_gpu, gpu_memory_ok]):
        print("\nHE THONG DA DUOC CAU HINH DE SU DUNG GPU THANH CONG!")
    else:
        print("\nCO VAN DE VOI CAU HINH GPU!")

if __name__ == "__main__":
    main()
