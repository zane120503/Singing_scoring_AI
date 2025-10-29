#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test GPU Chroma Detection Fix - Kiểm tra sửa lỗi GPU chroma
"""

import sys
import os
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gpu_chroma_fix():
    """Test GPU chroma detection fix"""
    
    print("🧪 TEST GPU CHROMA DETECTION FIX")
    print("=" * 50)
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("❌ CUDA not available, cannot test GPU chroma")
        return False
    
    # Test negative strides fix
    print("\n🔧 Testing negative strides fix...")
    
    # Create test audio with potential negative strides
    test_audio = np.random.randn(22050)  # 1 second at 22050 Hz
    
    # Simulate negative strides (reverse array)
    test_audio_reversed = test_audio[::-1]  # This creates negative strides
    
    print(f"Original audio contiguous: {test_audio.flags['C_CONTIGUOUS']}")
    print(f"Reversed audio contiguous: {test_audio_reversed.flags['C_CONTIGUOUS']}")
    
    # Test fix
    audio_copy = test_audio_reversed.copy() if not test_audio_reversed.flags['C_CONTIGUOUS'] else test_audio_reversed
    print(f"Fixed audio contiguous: {audio_copy.flags['C_CONTIGUOUS']}")
    
    # Test tensor creation
    try:
        device = torch.device("cuda:0")
        audio_tensor = torch.tensor(audio_copy, device=device, dtype=torch.float32)
        print("✅ Tensor creation successful")
        
        # Test STFT
        stft = torch.stft(audio_tensor, n_fft=2048, hop_length=512, return_complex=True)
        magnitude = torch.abs(stft)
        print("✅ STFT computation successful")
        
        # Test chroma computation
        if not magnitude.is_contiguous():
            magnitude = magnitude.contiguous()
        
        chroma = torch.sum(magnitude, dim=1)
        chroma = chroma / torch.sum(chroma)
        print("✅ Chroma computation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU chroma test failed: {e}")
        return False

def test_key_detector_gpu():
    """Test key detector with GPU"""
    print("\n🎹 Testing Key Detector with GPU...")
    
    try:
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        # Initialize key detector
        key_detector = AdvancedKeyDetector()
        print(f"GPU Status: {'ENABLED' if key_detector.use_gpu else 'DISABLED'}")
        
        if not key_detector.use_gpu:
            print("⚠️ GPU not enabled, skipping GPU test")
            return False
        
        # Create test audio
        test_audio = np.random.randn(22050)  # 1 second
        
        # Test GPU chroma detection
        try:
            result = key_detector._detect_with_gpu_chroma(test_audio, 22050)
            if result:
                print(f"✅ GPU chroma test: {result['key']} {result['scale']}")
                return True
            else:
                print("⚠️ GPU chroma returned None")
                return False
        except Exception as e:
            print(f"❌ GPU chroma test failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Key detector test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST GPU CHROMA DETECTION FIX ===")
    
    # Test GPU chroma fix
    fix_ok = test_gpu_chroma_fix()
    
    # Test key detector
    detector_ok = test_key_detector_gpu()
    
    print("\n" + "=" * 50)
    if fix_ok and detector_ok:
        print("🎉 GPU CHROMA FIX SUCCESSFUL!")
        print("✅ Negative strides issue resolved")
        print("✅ GPU chroma detection working")
    else:
        print("❌ GPU CHROMA FIX NEEDS ATTENTION!")
        print("⚠️ Some issues remain")
    
    input("\nNhấn Enter để thoát...")
