#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Audio Separator AI voi file thuc te
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_separator_with_file():
    """Test Audio Separator AI voi file thuc te"""
    print("=== TEST AUDIO SEPARATOR AI VOI FILE THUC TE ===")
    
    # Check GPU status
    print(f"1. GPU Status:")
    print(f"   PyTorch CUDA: {torch.cuda.is_available()}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    try:
        # Import Audio Processor
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        
        print("2. Initializing Audio Processor...")
        
        # Test voi AI Mode
        processor = AdvancedAudioProcessor(fast_mode=False)
        print(f"   Model: {processor.model}")
        print(f"   Device: {processor.device}")
        print(f"   Audio Separator Available: {processor.audio_separator.available if processor.audio_separator else False}")
        
        # Test vocal separation voi file thuc te
        print("\n3. Testing Vocal Separation voi file thuc te...")
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
                        print(f"   Vocals file exists: {os.path.getsize(vocals_path)} bytes")
                        
                        # Test key detection voi vocals
                        print("\n4. Testing Key Detection voi vocals...")
                        try:
                            from src.ai.advanced_key_detector import AdvancedKeyDetector
                            detector = AdvancedKeyDetector()
                            
                            key_result = detector.detect_key(vocals_path)
                            print(f"   Key detection result: {key_result}")
                        except Exception as e:
                            print(f"   ERROR Key detection error: {e}")
                        
                        # Test complete scoring
                        print("\n5. Testing Complete Scoring...")
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
                    else:
                        print(f"   ERROR Vocals file not created: {vocals_path}")
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
    test_audio_separator_with_file()
