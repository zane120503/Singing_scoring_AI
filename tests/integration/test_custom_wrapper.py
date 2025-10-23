#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test AI Audio Separator - wrapper tùy chỉnh chỉ tạo 1 file
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_custom_wrapper():
    """Test AI Audio Separator wrapper tùy chỉnh"""
    print("=== TEST AI AUDIO SEPARATOR - CUSTOM WRAPPER ===")
    
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
                        
                        # Check output directory - should only have 1 file
                        output_dir = os.path.dirname(vocals_path)
                        print(f"\n3. Checking output directory: {output_dir}")
                        
                        if os.path.exists(output_dir):
                            files = os.listdir(output_dir)
                            print(f"   Files in output directory: {len(files)}")
                            
                            for file in files:
                                file_path = os.path.join(output_dir, file)
                                file_size = os.path.getsize(file_path)
                                print(f"     - {file} ({file_size} bytes)")
                            
                            # Should only have 1 file (the MP3 vocals)
                            if len(files) == 1:
                                print("   SUCCESS: Only 1 file in output directory!")
                            else:
                                print(f"   WARNING: {len(files)} files in output directory (expected 1)")
                        else:
                            print("   ERROR: Output directory not found")
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
    test_custom_wrapper()

