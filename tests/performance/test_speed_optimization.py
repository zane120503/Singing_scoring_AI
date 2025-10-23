#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test AI Audio Separator - tối ưu hóa tốc độ
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_speed_optimization():
    """Test AI Audio Separator với tối ưu hóa tốc độ"""
    print("=== TEST AI AUDIO SEPARATOR - SPEED OPTIMIZATION ===")
    
    try:
        # Import Audio Processor
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        
        print("1. Initializing Audio Processor...")
        
        # Test voi AI Mode (fast_mode=False)
        processor = AdvancedAudioProcessor(fast_mode=False)
        print(f"   Model: {processor.model}")
        print(f"   Device: {processor.device}")
        
        # Test vocal separation voi file thuc te
        print("\n2. Testing AI Vocal Separation voi toi uu hoa toc do...")
        test_file = "D:\singing scoring AI\assets\audio\test.mp3"
        
        if os.path.exists(test_file):
            print(f"   Test file: {test_file}")
            
            # Test AI Mode
            if processor.audio_separator and processor.audio_separator.available:
                print("   Testing AI Mode vocal separation...")
                try:
                    # Đo thời gian
                    start_time = time.time()
                    vocals_path = processor.separate_vocals(test_file)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    print(f"   SUCCESS AI Mode vocals saved: {vocals_path}")
                    print(f"   Processing time: {processing_time:.2f} seconds")
                    
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
    test_speed_optimization()
