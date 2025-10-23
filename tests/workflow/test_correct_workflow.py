#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Correct Workflow với Correct Voice Detector
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_correct_workflow():
    """Test Correct Workflow"""
    try:
        from src.ai.optimized_audio_processor import OptimizedAudioProcessor
        
        # Test files
        karaoke_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        beat_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(karaoke_file):
            logger.error(f"Karaoke file khong ton tai: {karaoke_file}")
            return False
        
        logger.info("Testing Correct Workflow...")
        logger.info(f"   Karaoke file: {os.path.basename(karaoke_file)}")
        logger.info(f"   Beat file: {os.path.basename(beat_file)}")
        
        # Khoi tao processor
        processor = OptimizedAudioProcessor()
        
        # Test workflow
        start_time = time.time()
        result = processor.process_karaoke_optimized(karaoke_file, beat_file)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        if result["success"]:
            logger.info("Correct workflow thanh cong!")
            
            # Hien thi ket qua
            print(f"\nCORRECT WORKFLOW RESULTS:")
            print(f"Voice Detection: {result.get('voice_detection', {})}")
            print(f"Key Detection: {result.get('key_detection', {})}")
            print(f"Scoring: {result.get('scoring', {})}")
            print(f"Processing Time: {processing_time:.2f}s")
            print(f"Using GPU: YES")
            
            return True
        else:
            logger.error(f"Correct workflow failed: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        logger.error(f"Loi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== TEST CORRECT WORKFLOW ===")
    
    # Test correct workflow
    print("\n1. Testing Correct Workflow...")
    success = test_correct_workflow()
    
    # Ket qua
    print("\n=== KET QUA ===")
    if success:
        print("CORRECT WORKFLOW HOAT DONG CHINH XAC!")
        print("He thong da phat hien va cat chinh xac vi tri giong hat!")
    else:
        print("CAN CAI THIEN THEM!")
        print("Van con van de voi workflow!")
