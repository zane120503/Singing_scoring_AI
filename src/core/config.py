# Cấu hình hệ thống chấm điểm karaoke

# Cấu hình Audio Processing
AUDIO_CONFIG = {
    'sample_rate': 22050,
    'hop_length': 512,
    'n_fft': 2048,
    'n_mfcc': 13,
    'max_duration': 300,  # 5 phút tối đa
}

# Cấu hình AI Models
MODEL_CONFIG = {
    'audio_separator': {
        'model_name': 'facebook/demucs',
        'fallback_enabled': True,
        'device': 'auto'  # auto, cpu, cuda
    },
    'key_detection': {
        'model_name': 'jcarbonnell/key_class_detection',
        'fallback_enabled': True,
        'confidence_threshold': 0.5
    }
}

# Cấu hình Scoring System
SCORING_CONFIG = {
    'weights': {
        'key_accuracy': 0.25,
        'pitch_accuracy': 0.20,
        'rhythm_accuracy': 0.15,
        'timing_accuracy': 0.15,
        'vocal_quality': 0.10,
        'energy_consistency': 0.10,
        'pronunciation': 0.05
    },
    'grade_thresholds': {
        'A+': 90,
        'A': 80,
        'B+': 70,
        'B': 60,
        'C': 50,
        'D': 40,
        'F': 0
    }
}

# Cấu hình GUI
GUI_CONFIG = {
    'window_size': '1000x700',
    'theme': {
        'bg_color': '#f0f0f0',
        'primary_color': '#3498db',
        'success_color': '#27ae60',
        'warning_color': '#f39c12',
        'error_color': '#e74c3c'
    },
    'fonts': {
        'title': ('Arial', 16, 'bold'),
        'heading': ('Arial', 12, 'bold'),
        'body': ('Arial', 10),
        'small': ('Arial', 9)
    }
}

# Cấu hình File Formats
FILE_CONFIG = {
    'supported_formats': ['.wav', '.mp3', '.flac', '.m4a'],
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'temp_dir': './temp',
    'output_dir': './output'
}

# Cấu hình Logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'karaoke_scoring.log'
}

