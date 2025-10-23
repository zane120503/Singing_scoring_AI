#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Hybrid Key Detector - Multiple Methods Voting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_key_detector():
    """Test Hybrid Key Detector với file MP3 trong clean_song_output"""
    print("=== TEST HYBRID KEY DETECTOR ===")
    
    try:
        # Import Advanced Key Detector
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        print("1. Initializing Hybrid Advanced Key Detector...")
        
        detector = AdvancedKeyDetector()
        print(f"   Essentia available: {detector.essentia_available}")
        print(f"   Docker Essentia available: {detector.docker_available}")
        
        # Test với file MP3 trong clean_song_output
        print("\n2. Testing Hybrid Key Detection...")
        
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
                print(f"\n3. Testing Hybrid Key Detection voi: {os.path.basename(test_file)}")
                print("   Expected: A minor (from user verification)")
                
                try:
                    # Test hybrid key detection
                    print("   Testing hybrid key detection...")
                    key_result = detector.detect_key(test_file)
                    
                    print(f"\n   SUCCESS Hybrid Key detection result:")
                    print(f"     Key: {key_result['key']}")
                    print(f"     Scale: {key_result['scale']}")
                    print(f"     Confidence: {key_result['confidence']:.3f}")
                    print(f"     Method: {key_result['method']}")
                    
                    # Check if result matches expected
                    if key_result['key'] == 'A' and key_result['scale'] == 'minor':
                        print("   ✅ CORRECT! Matches expected A minor")
                    else:
                        print(f"   ❌ INCORRECT! Expected A minor, got {key_result['key']} {key_result['scale']}")
                    
                except Exception as e:
                    print(f"   ERROR Hybrid key detection error: {e}")
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
    test_hybrid_key_detector()

