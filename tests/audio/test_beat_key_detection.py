#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Beat Key Detection - Check if beat key detection is accurate
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_beat_key_detection():
    """Test beat key detection accuracy"""
    print("=== TEST BEAT KEY DETECTION ===")
    
    try:
        # Import Advanced Key Detector
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        print("1. Initializing Advanced Key Detector...")
        
        detector = AdvancedKeyDetector()
        print(f"   Essentia available: {detector.essentia_available}")
        print(f"   Docker Essentia available: {detector.docker_available}")
        
        # Test với beat file (nếu có)
        print("\n2. Testing Beat Key Detection...")
        
        # Tìm beat file trong thư mục
        beat_files = []
        search_dirs = [
            "D:\\singing scoring AI\\data",
            "D:\\singing scoring AI\\Audio_separator_ui",
            "D:\\singing scoring AI"
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file.endswith(('.mp3', '.wav', '.m4a')) and ('beat' in file.lower() or 'instrumental' in file.lower()):
                            beat_files.append(os.path.join(root, file))
        
        if beat_files:
            print(f"   Found {len(beat_files)} potential beat files:")
            for i, beat_file in enumerate(beat_files, 1):
                print(f"     {i}. {os.path.basename(beat_file)}")
            
            # Test với file đầu tiên
            test_file = beat_files[0]
            print(f"\n3. Testing Beat Key Detection voi: {os.path.basename(test_file)}")
            
            try:
                # Test beat key detection
                print("   Testing beat key detection...")
                key_result = detector.detect_key(test_file)
                
                print(f"\n   SUCCESS Beat Key detection result:")
                print(f"     Key: {key_result['key']}")
                print(f"     Scale: {key_result['scale']}")
                print(f"     Confidence: {key_result['confidence']:.3f}")
                print(f"     Method: {key_result['method']}")
                
                # Test với file khác nếu có
                if len(beat_files) > 1:
                    print(f"\n4. Testing voi beat file thu 2: {os.path.basename(beat_files[1])}")
                    try:
                        key_result2 = detector.detect_key(beat_files[1])
                        print(f"   SUCCESS Beat Key detection result 2:")
                        print(f"     Key: {key_result2['key']}")
                        print(f"     Scale: {key_result2['scale']}")
                        print(f"     Confidence: {key_result2['confidence']:.3f}")
                        print(f"     Method: {key_result2['method']}")
                    except Exception as e:
                        print(f"   ERROR Beat key detection error 2: {e}")
                
            except Exception as e:
                print(f"   ERROR Beat key detection error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("   No beat files found")
            print("   Please provide a beat file path to test")
            
            # Test với file vocals để so sánh
            vocals_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\d01f4ec899c813f91b_mdx\\input_Vocals_DeReverb_converted.mp3"
            if os.path.exists(vocals_file):
                print(f"\n3. Testing voi vocals file để so sánh: {os.path.basename(vocals_file)}")
                try:
                    vocals_result = detector.detect_key(vocals_file)
                    print(f"   SUCCESS Vocals Key detection result:")
                    print(f"     Key: {vocals_result['key']}")
                    print(f"     Scale: {vocals_result['scale']}")
                    print(f"     Confidence: {vocals_result['confidence']:.3f}")
                    print(f"     Method: {vocals_result['method']}")
                except Exception as e:
                    print(f"   ERROR Vocals key detection error: {e}")
        
    except Exception as e:
        print(f"ERROR Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_beat_key_detection()

