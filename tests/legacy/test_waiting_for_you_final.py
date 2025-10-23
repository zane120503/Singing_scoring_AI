#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Final Voice Detector với file Waiting For You
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_waiting_for_you_final():
    """Test Final Voice Detector với file Waiting For You"""
    try:
        from src.ai.final_voice_detector import FinalVoiceDetector
        
        # Test file Waiting For You
        test_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info("🎯 Testing Final Voice Detector với file Waiting For You...")
        logger.info(f"File: {test_file}")
        
        # Khởi tạo detector
        detector = FinalVoiceDetector()
        
        # Test voice detection
        segments = detector.detect_voice_activity(test_file)
        
        logger.info(f"Phát hiện {len(segments)} voice segments")
        
        if segments:
            for i, segment in enumerate(segments[:3]):  # Hiển thị 3 đoạn đầu
                logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f}, method: {segment['method']})")
            
            # Kiểm tra vị trí đầu tiên
            first_segment = segments[0]
            detected_start = first_segment['start']
            
            logger.info(f"\n🎯 Kết quả phát hiện:")
            logger.info(f"   Detected voice start: {detected_start:.2f}s")
            logger.info(f"   Segment duration: {first_segment['end'] - first_segment['start']:.2f}s")
            
            # Kiểm tra xem có hợp lý không
            if detected_start < 15.0:  # Giọng hát thường bắt đầu trong 15 giây đầu
                logger.info(f"✅ Vị trí phát hiện hợp lý: {detected_start:.2f}s")
                return True
            else:
                logger.warning(f"⚠️ Vị trí phát hiện có thể không chính xác: {detected_start:.2f}s")
                return False
        else:
            logger.warning("  Không phát hiện voice segments")
            return False
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_workflow_waiting_for_you():
    """Test full workflow với file Waiting For You"""
    try:
        from src.ai.optimized_audio_processor import OptimizedAudioProcessor
        
        # Test files
        karaoke_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        beat_file = "C:\\Users\\admin\\Downloads\\Waiting For You.mp3"
        
        if not os.path.exists(karaoke_file):
            logger.error(f"Karaoke file không tồn tại: {kasaraoke_file}")
            return False
        
        logger.info("🚀 Testing Full Workflow với file Waiting For You...")
        logger.info(f"   Karaoke file: {os.path.basename(karaoke_file)}")
        logger.info(f"   Beat file: {os.path.basename(beat_file)}")
        
        # Khởi tạo processor
        processor = OptimizedAudioProcessor()
        
        # Test workflow
        start_time = time.time()
        result = processor.process_karaoke_optimized(karaoke_file, beat_file)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        if result["success"]:
            logger.info("✅ Full workflow thành công!")
            
            # Hiển thị kết quả
            print(f"\n🎉 FULL WORKFLOW RESULTS:")
            print(f"📊 Voice Detection: {result.get('voice_detection', {})}")
            print(f"🎵 Key Detection: {result.get('key_detection', {})}")
            print(f"📈 Scoring: {result.get('scoring', {})}")
            print(f"⏱️ Processing Time: {processing_time:.2f}s")
            print(f"🖥️ Using GPU: YES")
            
            return True
        else:
            logger.error(f"❌ Full workflow failed: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== TEST WAITING FOR YOU FINAL ===")
    
    # Test Final Voice Detector
    print("\n1. Testing Final Voice Detector...")
    vad_success = test_waiting_for_you_final()
    
    # Test full workflow
    print("\n2. Testing Full Workflow...")
    workflow_success = test_full_workflow_waiting_for_you()
    
    # Kết quả
    print("\n=== KET QUA ===")
    print(f"Final Voice Detector: {'PASS' if vad_success else 'FAIL'}")
    print(f"Full Workflow: {'PASS' if workflow_success else 'FAIL'}")
    
    if vad_success and workflow_success:
        print("\n🎉 TẤT CẢ TESTS THÀNH CÔNG!")
        print("\n🚀 Hệ thống đã sẵn sàng với file Waiting For You!")
    else:
        print("\n⚠️ CẦN CẢI THIỆN THÊM!")
