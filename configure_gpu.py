#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Configuration Script - Cấu hình GPU cho hệ thống
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_setup():
    """Kiểm tra và cấu hình GPU"""
    print("🔧 GPU CONFIGURATION SCRIPT")
    print("=" * 50)
    
    # Kiểm tra CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test GPU computation
        try:
            device = torch.device("cuda:0")
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.mm(x, y)
            print("✅ GPU computation test: SUCCESS")
            
            # Test memory
            allocated = torch.cuda.memory_allocated() / (1024**2)
            print(f"GPU Memory Allocated: {allocated:.2f} MB")
            
            return True
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
            return False
    else:
        print("❌ CUDA not available")
        return False

def configure_gpu_settings():
    """Cấu hình GPU settings"""
    print("\n⚙️ CONFIGURING GPU SETTINGS")
    print("-" * 30)
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Configure PyTorch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        print("✅ GPU settings configured")
        return True
    else:
        print("❌ Cannot configure GPU settings - CUDA not available")
        return False

def test_key_detector_gpu():
    """Test key detector với GPU"""
    print("\n🎹 TESTING KEY DETECTOR WITH GPU")
    print("-" * 40)
    
    try:
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        # Initialize key detector
        key_detector = AdvancedKeyDetector()
        
        print(f"Key Detector GPU: {'ENABLED' if key_detector.use_gpu else 'DISABLED'}")
        if key_detector.use_gpu:
            print(f"Device: {key_detector.device}")
        
        # Test with dummy audio
        import numpy as np
        
        # Create dummy audio signal
        dummy_audio = np.random.randn(22050)  # 1 second at 22050 Hz
        
        # Test GPU chroma analysis
        if key_detector.use_gpu:
            try:
                result = key_detector._detect_with_gpu_chroma(dummy_audio, 22050)
                if result:
                    print(f"✅ GPU chroma test: {result['key']} {result['scale']}")
                else:
                    print("⚠️ GPU chroma test returned None")
            except Exception as e:
                print(f"❌ GPU chroma test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Key detector test failed: {e}")
        return False

def create_gpu_config_file():
    """Tạo file cấu hình GPU"""
    print("\n📝 CREATING GPU CONFIG FILE")
    print("-" * 30)
    
    config_content = '''import torch
import os
import logging

logger = logging.getLogger(__name__)

# GPU Configuration
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

def get_device():
    """Returns the currently configured device (cuda or cpu)."""
    return DEVICE

def force_cuda():
    """Forces the system to use CUDA if available, setting environment variables."""
    if CUDA_AVAILABLE:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
        torch.backends.cudnn.benchmark = True
        logger.info(f"CUDA forced. Using device: {DEVICE}")
    else:
        logger.warning("CUDA not available. Forcing CPU usage.")

# Initialize device on import
force_cuda()

# Additional GPU settings
FORCE_GPU = True
GPU_DEVICE = "cuda:0" if CUDA_AVAILABLE else "cpu"
CUDA_VISIBLE_DEVICES = "0"
TORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

# Model Configuration
USE_GPU_FOR_AUDIO_SEPARATOR = True
USE_GPU_FOR_KEY_DETECTION = True
USE_GPU_FOR_SCORING = True

# Performance Settings
GPU_MEMORY_FRACTION = 0.8
BATCH_SIZE = 1
NUM_WORKERS = 0
'''
    
    config_file = os.path.join(os.path.dirname(__file__), 'src', 'core', 'gpu_config.py')
    
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"✅ GPU config created: {config_file}")
        return True
    except Exception as e:
        print(f"❌ Error creating GPU config: {e}")
        return False

def main():
    """Hàm chính"""
    print("=== GPU CONFIGURATION FOR SINGING SCORING AI ===")
    
    # Kiểm tra GPU setup
    gpu_ok = check_gpu_setup()
    
    # Cấu hình GPU settings
    if gpu_ok:
        config_ok = configure_gpu_settings()
    else:
        config_ok = False
    
    # Test key detector
    detector_ok = test_key_detector_gpu()
    
    # Tạo config file
    file_ok = create_gpu_config_file()
    
    # Tóm tắt
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    print(f"   GPU Available: {'✅ YES' if gpu_ok else '❌ NO'}")
    print(f"   GPU Configured: {'✅ YES' if config_ok else '❌ NO'}")
    print(f"   Key Detector GPU: {'✅ YES' if detector_ok else '❌ NO'}")
    print(f"   Config File: {'✅ YES' if file_ok else '❌ NO'}")
    
    if gpu_ok and config_ok and detector_ok:
        print("\n🎉 GPU CONFIGURATION COMPLETE!")
        print("🚀 Hệ thống đã sẵn sàng sử dụng GPU acceleration!")
    else:
        print("\n⚠️ GPU CONFIGURATION NEEDS ATTENTION!")
        print("🔧 Một số cấu hình cần được kiểm tra lại.")

if __name__ == "__main__":
    main()
    input("\nNhấn Enter để thoát...")
