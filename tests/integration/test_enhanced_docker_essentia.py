#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Enhanced Docker Essentia AI - Key Detection Accuracy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_docker_essentia():
    """Test Enhanced Docker Essentia AI với file MP3 trong clean_song_output"""
    print("=== TEST ENHANCED DOCKER ESSENTIA AI ===")
    
    try:
        # Import Advanced Key Detector
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        print("1. Initializing Enhanced Advanced Key Detector...")
        
        detector = AdvancedKeyDetector()
        print(f"   Essentia available: {detector.essentia_available}")
        print(f"   Docker Essentia available: {detector.docker_available}")
        
        # Test với file MP3 trong clean_song_output
        print("\n2. Testing Enhanced Key Detection...")
        
        # Tìm file MP3 trong clean_song_output
        clean_song_dir = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output"
        
        if os.path.exists(clean_song_dir):
            print(f"   Clean song directory: {clean_song_dir}")
            
            # Tìm file MP3
            mp3_files = []
            for root, dirs, files in os.walk(clean_song_dir):
                for file in files:
                    if file.endswith('.mp3'):
                        mp3_files.append(os.path.join(root, file))
            
            if mp3_files:
                print(f"   Found {len(mp3_files)} MP3 files:")
                for i, mp3_file in enumerate(mp3_files, 1):
                    print(f"     {i}. {os.path.basename(mp3_file)}")
                
                # Test với file đầu tiên
                test_file = mp3_files[0]
                print(f"\n3. Testing Enhanced Key Detection voi: {os.path.basename(test_file)}")
                
                try:
                    # Test enhanced key detection
                    print("   Testing enhanced key detection...")
                    key_result = detector.detect_key(test_file)
                    
                    print(f"   SUCCESS Enhanced Key detection result:")
                    print(f"     Key: {key_result['key']}")
                    print(f"     Scale: {key_result['scale']}")
                    print(f"     Confidence: {key_result['confidence']:.3f}")
                    print(f"     Method: {key_result['method']}")
                    
                    # Test với file khác nếu có
                    if len(mp3_files) > 1:
                        print(f"\n4. Testing voi file thu 2: {os.path.basename(mp3_files[1])}")
                        try:
                            key_result2 = detector.detect_key(mp3_files[1])
                            print(f"   SUCCESS Key detection result 2:")
                            print(f"     Key: {key_result2['key']}")
                            print(f"     Scale: {key_result2['scale']}")
                            print(f"     Confidence: {key_result2['confidence']:.3f}")
                            print(f"     Method: {key_result2['method']}")
                        except Exception as e:
                            print(f"   ERROR Key detection error 2: {e}")
                    
                except Exception as e:
                    print(f"   ERROR Enhanced key detection error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   No MP3 files found in clean_song_output")
        else:
            print(f"   Clean song directory not found: {clean_song_dir}")
        
    except Exception as e:
        print(f"ERROR Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_docker_essentia()
