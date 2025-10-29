"""
Test vocals key detection with improved weights
"""

import sys
sys.path.insert(0, 'src')

from src.ai.advanced_key_detector import AdvancedKeyDetector
import os

print('=' * 60)
print('TESTING VOCALS KEY DETECTION WITH IMPROVED WEIGHTS')
print('=' * 60)

detector = AdvancedKeyDetector()

# Test với file vocals thực tế
test_file = 'output/voice_samples/Bông Hoa Đẹp Nhất_slice_vocals.mp3'
if os.path.exists(test_file):
    print(f'Loading file: {test_file}')
    result = detector.detect_key(test_file, audio_type='vocals')
    
    print()
    print('RESULT:')
    print(f'Key: {result["key"]} {result["scale"]}')
    print(f'Confidence: {result["confidence"]:.3f}')
    print(f'Method: {result["method"]}')
else:
    print(f'Test file not found: {test_file}')
    print('Trying alternative path...')
    
    # Try different path
    alt_file = '../output/voice_samples/Bông Hoa Đẹp Nhất_slice_vocals.mp3'
    if os.path.exists(alt_file):
        print(f'Loading file: {alt_file}')
        result = detector.detect_key(alt_file, audio_type='vocals')
        
        print()
        print('RESULT:')
        print(f'Key: {result["key"]} {result["scale"]}')
        print(f'Confidence: {result["confidence"]:.3f}')
        print(f'Method: {result["method"]}')
    else:
        print('File not found in any location')

print('=' * 60)

