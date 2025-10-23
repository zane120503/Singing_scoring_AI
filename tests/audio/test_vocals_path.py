#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test duong dan file vocals sau khi AI Audio Separator
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vocals_file_path():
    """Test duong dan file vocals"""
    print("=== TEST DUONG DAN FILE VOCALS ===")
    
    try:
        # Import Audio Processor
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        
        print("1. Initializing Audio Processor...")
        
        # Test voi AI Mode
        processor = AdvancedAudioProcessor(fast_mode=False)
        print(f"   Model: {processor.model}")
        print(f"   Device: {processor.device}")
        
        # Test vocal separation voi file thuc te
        print("\n2. Testing AI Vocal Separation...")
        test_file = "D:\singing scoring AI\assets\audio\test.mp3"
        
        if os.path.exists(test_file):
            print(f"   Test file: {test_file}")
            
            # Test AI Mode
            if processor.audio_separator and processor.audio_separator.available:
                print("   Testing AI Mode vocal separation...")
                try:
                    vocals_path = processor.separate_vocals(test_file)
                    print(f"   SUCCESS AI Mode vocals saved: {vocals_path}")
                    
                    # Check if file exists
                    if os.path.exists(vocals_path):
                        print(f"   SUCCESS Vocals file exists: {os.path.getsize(vocals_path)} bytes")
                        print(f"   File path: {vocals_path}")
                        
                        # Test key detection voi vocals
                        print("\n3. Testing Key Detection voi vocals...")
                        try:
                            from src.ai.advanced_key_detector import AdvancedKeyDetector
                            detector = AdvancedKeyDetector()
                            
                            key_result = detector.detect_key(vocals_path)
                            print(f"   SUCCESS Key detection result: {key_result}")
                            
                            # Test complete scoring
                            print("\n4. Testing Complete Scoring...")
                            try:
                                from src.core.scoring_system import KaraokeScoringSystem
                                scoring_system = KaraokeScoringSystem()
                                
                                # Use same file as beat for test
                                beat_file = test_file
                                
                                score_result = scoring_system.calculate_overall_score(
                                    test_file, beat_file, vocals_path
                                )
                                
                                print(f"   SUCCESS Scoring completed!")
                                print(f"   Overall Score: {score_result['overall_score']}")
                                print(f"   Grade: {score_result['grade']}")
                                print(f"   Detailed Scores:")
                                for criterion, score in score_result['detailed_scores'].items():
                                    print(f"     {criterion}: {score}")
                            except Exception as e:
                                print(f"   ERROR Scoring error: {e}")
                                import traceback
                                traceback.print_exc()
                        except Exception as e:
                            print(f"   ERROR Key detection error: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"   ERROR Vocals file not created: {vocals_path}")
                        
                        # Check alternative paths
                        print("   Checking alternative paths...")
                        audio_separator_dir = "D:\\singing scoring AI\\Audio_separator_ui"
                        clean_song_dir = os.path.join(audio_separator_dir, "clean_song_output")
                        
                        if os.path.exists(clean_song_dir):
                            print(f"   Clean song directory exists: {clean_song_dir}")
                            # List all files in clean_song_output
                            for root, dirs, files in os.walk(clean_song_dir):
                                for file in files:
                                    if "Vocals_DeReverb" in file:
                                        full_path = os.path.join(root, file)
                                        print(f"   Found vocals file: {full_path}")
                                        print(f"   File size: {os.path.getsize(full_path)} bytes")
                except Exception as e:
                    print(f"   ERROR AI Mode error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   AI Mode not available")
        else:
            print(f"   Test file not found: {test_file}")
        
    except Exception as e:
        print(f"ERROR Error in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vocals_file_path()
