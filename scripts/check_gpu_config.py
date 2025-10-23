#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script kiểm tra cấu hình GPU cho hệ thống
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Kiểm tra khả năng sử dụng GPU"""
    print("=== KIEM TRA GPU AVAILABILITY ===")
    
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
        
        # Kiểm tra GPU memory
        current_device = torch.cuda.current_device()
        allocated_memory = torch.cuda.memory_allocated() / (1024**3)
        cached_memory = torch.cuda.memory_reserved() / (1024**3)
        print(f"Current Device: {current_device}")
        print(f"Allocated Memory: {allocated_memory:.2f} GB")
        print(f"Cached Memory: {cached_memory:.2f} GB")
        
        return True
    else:
        print("CUDA not available. Will use CPU.")
        return False

def check_system_components():
    """Kiểm tra các components của hệ thống"""
    print("\n=== KIEM TRA SYSTEM COMPONENTS ===")
    
    try:
        # Kiểm tra Audio Separator
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        audio_processor = AdvancedAudioProcessor()
        print(f"Advanced Audio Processor device: {audio_processor.device}")
        
        # Kiểm tra Key Detector
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        key_detector = AdvancedKeyDetector()
        print(f"Advanced Key Detector: Initialized")
        
        # Kiểm tra AI Audio Separator
        from src.ai.ai_audio_separator import AIAudioSeparator
        ai_separator = AIAudioSeparator()
        print(f"AI Audio Separator device: {ai_separator.device}")
        
        return True
        
    except Exception as e:
        print(f"Error checking components: {e}")
        return False

def force_gpu_usage():
    """Cấu hình để force sử dụng GPU"""
    print("\n=== FORCE GPU USAGE ===")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Force CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        print(f"Forced device: {device}")
        
        # Test GPU computation
        try:
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print("GPU computation test: SUCCESS")
            return True
        except Exception as e:
            print(f"GPU computation test failed: {e}")
            return False
    else:
        print("CUDA not available, cannot force GPU usage")
        return False

def check_docker_essentia_gpu():
    """Kiểm tra Docker Essentia có sử dụng GPU không"""
    print("\n=== KIEM TRA DOCKER ESSENTIA GPU ===")
    
    try:
        import docker
        client = docker.from_env()
        
        # Kiểm tra containers đang chạy
        containers = client.containers.list()
        essentia_containers = [c for c in containers if 'essentia' in c.name.lower()]
        
        if essentia_containers:
            for container in essentia_containers:
                print(f"Essentia container: {container.name}")
                print(f"Status: {container.status}")
                print(f"Image: {container.image.tags}")
                
                # Kiểm tra GPU access
                try:
                    inspect = container.attrs
                    if 'HostConfig' in inspect and 'DeviceRequests' in inspect['HostConfig']:
                        device_requests = inspect['HostConfig']['DeviceRequests']
                        if device_requests:
                            print("GPU access: ENABLED")
                        else:
                            print("GPU access: NOT ENABLED")
                    else:
                        print("GPU access: NOT ENABLED")
                except:
                    print("Could not check GPU access")
        else:
            print("No Essentia containers found")
            
    except ImportError:
        print("Docker Python library not installed")
    except Exception as e:
        print(f"Error checking Docker Essentia: {e}")

def create_gpu_config():
    """Tạo file cấu hình GPU"""
    print("\n=== TAO GPU CONFIG ===")
    
    config_content = """
# GPU Configuration
CUDA_AVAILABLE = True
FORCE_GPU = True
GPU_DEVICE = "cuda:0"
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
"""
    
    config_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'gpu_config.py')
    
    try:
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"GPU config created: {config_file}")
        return True
    except Exception as e:
        print(f"Error creating GPU config: {e}")
        return False

def main():
    """Hàm chính"""
    print("=== KIEM TRA VA CAU HINH GPU CHO HE THONG ===")
    
    # Kiểm tra GPU availability
    gpu_available = check_gpu_availability()
    
    # Kiểm tra system components
    components_ok = check_system_components()
    
    # Force GPU usage
    if gpu_available:
        gpu_forced = force_gpu_usage()
    else:
        gpu_forced = False
    
    # Kiểm tra Docker Essentia
    check_docker_essentia_gpu()
    
    # Tạo GPU config
    config_created = create_gpu_config()
    
    # Tóm tắt
    print("\n=== TOM TAT ===")
    print(f"GPU Available: {'YES' if gpu_available else 'NO'}")
    print(f"System Components: {'OK' if components_ok else 'ERROR'}")
    print(f"GPU Forced: {'YES' if gpu_forced else 'NO'}")
    print(f"GPU Config Created: {'YES' if config_created else 'NO'}")
    
    if gpu_available and gpu_forced:
        print("\nHE THONG DA DUOC CAU HINH DE SU DUNG GPU!")
    else:
        print("\nCAN KIEM TRA CAU HINH GPU!")

if __name__ == "__main__":
    main()
