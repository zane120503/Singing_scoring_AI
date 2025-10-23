#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test GPU usage for Karaoke Scoring System
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import onnxruntime as ort

def test_gpu_setup():
    """Test GPU configuration"""
    print("=== GPU CONFIGURATION TEST ===")
    
    # PyTorch GPU test
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # ONNX Runtime GPU test
    print(f"\nONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"Current device: {ort.get_device()}")
    
    # Test Audio Separator device detection
    print(f"\n=== AUDIO SEPARATOR DEVICE DETECTION ===")
    device_base = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device base: {device_base}")
    
    # Test model loading
    print(f"\n=== MODEL LOADING TEST ===")
    try:
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        processor = AdvancedAudioProcessor(fast_mode=False)
        print(f"Audio processor initialized successfully")
        print(f"Model type: {processor.model}")
    except Exception as e:
        print(f"Error initializing audio processor: {e}")
    
    try:
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        detector = AdvancedKeyDetector()
        print(f"Key detector initialized successfully")
    except Exception as e:
        print(f"Error initializing key detector: {e}")

if __name__ == "__main__":
    test_gpu_setup()
