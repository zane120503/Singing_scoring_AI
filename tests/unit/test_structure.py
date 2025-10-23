"""
Test hệ thống với cấu trúc mới
"""

import sys
import os

# Thêm src vào Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_imports():
    """Test các import chính"""
    print("Testing imports...")
    
    try:
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        print("AdvancedAudioProcessor imported")
    except Exception as e:
        print(f"AdvancedAudioProcessor failed: {e}")
    
    try:
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        print("AdvancedKeyDetector imported")
    except Exception as e:
        print(f"AdvancedKeyDetector failed: {e}")
    
    try:
        from src.core.scoring_system import KaraokeScoringSystem
        print("KaraokeScoringSystem imported")
    except Exception as e:
        print(f"KaraokeScoringSystem failed: {e}")
    
    try:
        from src.gui.gui import KaraokeScoringGUI
        print("KaraokeScoringGUI imported")
    except Exception as e:
        print(f"KaraokeScoringGUI failed: {e}")

def test_audio_processor():
    """Test Audio Processor"""
    print("\nTesting Audio Processor...")
    
    try:
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        
        # Test Fast Mode
        processor = AdvancedAudioProcessor(fast_mode=True)
        print(f"Fast Mode: {processor.model}")
        
        # Test AI Mode
        processor_ai = AdvancedAudioProcessor(fast_mode=False)
        print(f"AI Mode: {processor_ai.model}")
        
    except Exception as e:
        print(f"Audio Processor test failed: {e}")

def test_key_detector():
    """Test Key Detector"""
    print("\nTesting Key Detector...")
    
    try:
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        detector = AdvancedKeyDetector()
        print(f"Key Detector initialized")
        print(f"   Essentia available: {detector.essentia_available}")
        print(f"   Docker available: {detector.docker_available}")
        
    except Exception as e:
        print(f"Key Detector test failed: {e}")

if __name__ == "__main__":
    print("Testing he thong voi cau truc moi...")
    print("=" * 50)
    
    test_imports()
    test_audio_processor()
    test_key_detector()
    
    print("\n" + "=" * 50)
    print("Test hoan thanh!")
