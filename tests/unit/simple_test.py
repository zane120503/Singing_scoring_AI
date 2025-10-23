#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for Audio Separator integration
"""

import os
import sys

def test_audio_separator_ui():
    """Test Audio Separator UI directly"""
    print("\nTesting Audio Separator UI...")
    
    try:
        # Check if Audio Separator UI is available
        separator_path = os.path.join('Audio_separator_ui', 'app.py')
        if os.path.exists(separator_path):
            print("Audio Separator UI is available")
            
            # Check models
            models_path = os.path.join('..', '..', 'assets', 'models', 'mdx_models')
            if os.path.exists(models_path):
                print("Models directory is available")
                
                # List files in models
                model_files = os.listdir(models_path)
                print(f"Model files: {len(model_files)} files")
                for file in model_files[:5]:  # Show first 5 files
                    print(f"   - {file}")
                    
            else:
                print("Models directory not found")
                
        else:
            print("Audio Separator UI not found")
            
    except Exception as e:
        print(f"Error testing Audio Separator UI: {e}")

def test_advanced_processor():
    """Test Advanced Audio Processor"""
    print("\nTesting Advanced Audio Processor...")
    
    try:
        # Import Advanced Audio Processor
        from advanced_audio_processor import AdvancedAudioProcessor
        
        # Initialize processor
        processor = AdvancedAudioProcessor()
        print("Advanced Audio Processor initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"Error initializing Advanced Audio Processor: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Test Audio Separator Integration")
    print("=" * 50)
    
    # Test Audio Separator UI
    test_audio_separator_ui()
    
    # Test Advanced Processor
    success = test_advanced_processor()
    
    if success:
        print("\nIntegration test completed successfully!")
    else:
        print("\nIntegration test failed!")

if __name__ == "__main__":
    main()

