#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Audio Separator AI trong he thong karaoke scoring
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audio_separator_integration():
    """Test Audio Separator AI integration"""
    print("=== TEST AUDIO SEPARATOR AI INTEGRATION ===")
    
    # Check GPU status
    print(f"1. GPU Status:")
    print(f"   PyTorch CUDA: {torch.cuda.is_available()}")
    print(f"   GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    try:
        # Import Audio Processor
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        
        print("2. Initializing Audio Processor...")
        
        # Test voi AI Mode (khong phai Fast Mode)
        processor = AdvancedAudioProcessor(fast_mode=False)
        print(f"   Model: {processor.model}")
        print(f"   Device: {processor.device}")
        print(f"   Audio Separator Available: {processor.audio_separator.available if processor.audio_separator else False}")
        
        # Test voi Fast Mode
        print("\n3. Testing Fast Mode...")
        fast_processor = AdvancedAudioProcessor(fast_mode=True)
        print(f"   Model: {fast_processor.model}")
        print(f"   Device: {fast_processor.device}")
        
        # Test vocal separation voi file mau
        print("\n4. Testing Vocal Separation...")
        test_file = "../../assets/audio/test.mp3"
        
        if os.path.exists(test_file):
            print(f"   Test file: {test_file}")
            
            # Test AI Mode
            if processor.audio_separator and processor.audio_separator.available:
                print("   Testing AI Mode vocal separation...")
                try:
                    vocals_path = processor.separate_vocals(test_file)
                    print(f"   SUCCESS AI Mode vocals saved: {vocals_path}")
                except Exception as e:
                    print(f"   ERROR AI Mode error: {e}")
            else:
                print("   AI Mode not available, using Fast Mode...")
                try:
                    vocals_path = fast_processor.separate_vocals(test_file)
                    print(f"   SUCCESS Fast Mode vocals saved: {vocals_path}")
                except Exception as e:
                    print(f"   ERROR Fast Mode error: {e}")
        else:
            print(f"   Test file not found: {test_file}")
        
        print("\n5. Testing Key Detection...")
        try:
            from src.ai.advanced_key_detector import AdvancedKeyDetector
            detector = AdvancedKeyDetector()
            print(f"   SUCCESS Key Detector initialized")
            
            # Test key detection voi file vocals
            if os.path.exists(test_file):
                key_result = detector.detect_key(test_file)
                print(f"   Key detection result: {key_result}")
        except Exception as e:
            print(f"   ERROR Key detection error: {e}")
        
        print("\n6. Testing Complete Scoring System...")
        try:
            from src.core.scoring_system import KaraokeScoringSystem
            scoring_system = KaraokeScoringSystem()
            print(f"   SUCCESS Scoring System initialized")
            
            # Test scoring voi file mau
            if os.path.exists(test_file):
                # Tao file beat gia (cung file de test)
                beat_file = test_file
                
                # Separate vocals
                vocals_path = fast_processor.separate_vocals(test_file)
                
                if os.path.exists(vocals_path):
                    print(f"   Testing scoring with:")
                    print(f"     Karaoke: {test_file}")
                    print(f"     Beat: {beat_file}")
                    print(f"     Vocals: {vocals_path}")
                    
                    # Calculate score
                    score_result = scoring_system.calculate_overall_score(
                        test_file, beat_file, vocals_path
                    )
                    
                    print(f"   SUCCESS Scoring completed!")
                    print(f"   Overall Score: {score_result['overall_score']}")
                    print(f"   Grade: {score_result['grade']}")
                    print(f"   Detailed Scores: {score_result['detailed_scores']}")
                else:
                    print(f"   ERROR Vocals file not found: {vocals_path}")
        except Exception as e:
            print(f"   ERROR Scoring system error: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"ERROR Error in integration test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_audio_separator_integration()
