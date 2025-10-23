import torch
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