#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Beat Key Detection - Waiting For You [music].mp3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_beat_key_detection():
    """Test beat key detection với file Waiting For You"""
    print("=== TEST BEAT KEY DETECTION - WAITING FOR YOU ===")
    
    try:
        # Import Advanced Key Detector
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        print("1. Initializing Advanced Key Detector...")
        
        detector = AdvancedKeyDetector()
        print(f"   Essentia available: {detector.essentia_available}")
        print(f"   Docker Essentia available: {detector.docker_available}")
        
        # Test với beat file
        beat_file = r"C:\Users\admin\Downloads\Waiting For You [music].mp3"
        
        if os.path.exists(beat_file):
            print(f"\n2. Testing Beat Key Detection...")
            print(f"   Beat file: {os.path.basename(beat_file)}")
            
            try:
                # Test 1: General detection (old method)
                print("\n3. Testing General Key Detection (old method)...")
                general_result = detector.detect_key(beat_file, "general")
                
                print(f"   SUCCESS General Key detection result:")
                print(f"     Key: {general_result['key']}")
                print(f"     Scale: {general_result['scale']}")
                print(f"     Confidence: {general_result['confidence']:.3f}")
                print(f"     Method: {general_result['method']}")
                
                # Test 2: Beat-specific detection (new method)
                print("\n4. Testing Beat-Specific Key Detection (new method)...")
                beat_result = detector.detect_key(beat_file, "beat")
                
                print(f"   SUCCESS Beat-Specific Key detection result:")
                print(f"     Key: {beat_result['key']}")
                print(f"     Scale: {beat_result['scale']}")
                print(f"     Confidence: {beat_result['confidence']:.3f}")
                print(f"     Method: {beat_result['method']}")
                
                # Compare results
                print(f"\n5. Comparison:")
                print(f"   General: {general_result['key']} {general_result['scale']} (conf: {general_result['confidence']:.3f})")
                print(f"   Beat-specific: {beat_result['key']} {beat_result['scale']} (conf: {beat_result['confidence']:.3f})")
                
                if general_result['key'] != beat_result['key'] or general_result['scale'] != beat_result['scale']:
                    print("   DIFFERENT results between methods!")
                else:
                    print("   SAME results between methods")
                
            except Exception as e:
                print(f"   ERROR Beat key detection error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   Beat file not found: {beat_file}")
            print("   Please check the file path")
        
    except Exception as e:
        print(f"ERROR Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_beat_key_detection()

