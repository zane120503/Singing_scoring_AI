#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Docker Essentia AI - Key Detection
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_docker_essentia():
    """Test Docker Essentia AI key detection"""
    print("=== TEST DOCKER ESSENTIA AI ===")
    
    try:
        # Import Advanced Key Detector
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        print("1. Initializing Advanced Key Detector...")
        
        detector = AdvancedKeyDetector()
        print(f"   Essentia available: {detector.essentia_available}")
        print(f"   Docker Essentia available: {detector.docker_available}")
        
        # Test với file âm thanh
        print("\n2. Testing Key Detection...")
        test_file = "D:\singing scoring AI\assets\audio\test.mp3"
        
        if os.path.exists(test_file):
            print(f"   Test file: {test_file}")
            
            try:
                # Test key detection
                print("   Testing key detection...")
                key_result = detector.detect_key(test_file)
                
                print(f"   SUCCESS Key detection result:")
                print(f"     Key: {key_result['key']}")
                print(f"     Scale: {key_result['scale']}")
                print(f"     Confidence: {key_result['confidence']}")
                print(f"     Method: {key_result['method']}")
                
                # Test với file vocals
                print("\n3. Testing Key Detection với vocals...")
                vocals_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\9b772f1d226920b229_mdx\\input_Vocals_DeReverb_converted.mp3"
                
                if os.path.exists(vocals_file):
                    print(f"   Vocals file: {vocals_file}")
                    
                    try:
                        vocals_key_result = detector.detect_key(vocals_file)
                        
                        print(f"   SUCCESS Vocals key detection result:")
                        print(f"     Key: {vocals_key_result['key']}")
                        print(f"     Scale: {vocals_key_result['scale']}")
                        print(f"     Confidence: {vocals_key_result['confidence']}")
                        print(f"     Method: {vocals_key_result['method']}")
                        
                    except Exception as e:
                        print(f"   ERROR Vocals key detection error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"   Vocals file not found: {vocals_file}")
                
            except Exception as e:
                print(f"   ERROR Key detection error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   Test file not found: {test_file}")
        
    except Exception as e:
        print(f"ERROR Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_docker_essentia()
