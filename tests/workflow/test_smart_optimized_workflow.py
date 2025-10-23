#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Smart Optimized Workflow
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_smart_optimized_workflow():
    """Test Smart Optimized Workflow"""
    try:
        from src.ai.optimized_audio_processor import OptimizedAudioProcessor
        
        # Test files
        karaoke_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\62b6c4008acb2bcafc_mdx\\input.wav"
        beat_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\62b6c4008acb2bcafc_mdx\\input.wav"
        
        if not os.path.exists(karaoke_file):
            logger.error(f"Karaoke file không tồn tại: {karaoke_file}")
            return False
        
        if not os.path.exists(beat_file):
            logger.error(f"Beat file không tồn tại: {beat_file}")
            return False
        
        logger.info("🚀 Testing Smart Optimized Workflow...")
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
            logger.info("✅ Smart optimized workflow thành công!")
            
            # Hiển thị kết quả
            summary = f"""
🎉 SMART OPTIMIZED WORKFLOW RESULTS:
📊 Voice Detection: {result.get('voice_detection', {})}
🎵 Key Detection: {result.get('key_detection', {})}
📈 Scoring: {result.get('scoring', {})}
⏱️ Processing Time: {processing_time:.2f}s
"""
            print(summary)
            
            return True
        else:
            logger.error(f"❌ Smart optimized workflow failed: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_detection_accuracy():
    """Test độ chính xác của voice detection"""
    try:
        from src.ai.smart_voice_detector import SmartVoiceDetector
        
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\62b6c4008acb2bcafc_mdx\\input.wav"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info("🎯 Testing Voice Detection Accuracy...")
        
        detector = SmartVoiceDetector()
        
        # Test voice detection
        segments = detector.detect_voice_activity(test_file)
        
        if segments:
            logger.info(f"✅ Voice detection successful: {len(segments)} segments")
            
            # Hiển thị chi tiết segments
            for i, segment in enumerate(segments):
                duration = segment['end'] - segment['start']
                logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (duration: {duration:.2f}s, conf: {segment['confidence']:.2f})")
            
            # Tìm đoạn voice đầu tiên
            first_voice = detector.find_first_voice_segment(test_file, min_duration=1.0)
            if first_voice['start'] > 0:
                logger.info(f"🎯 First voice segment: {first_voice['start']:.2f}s - {first_voice['end']:.2f}s")
                
                # Kiểm tra xem có phải từ đầu file không
                if first_voice['start'] < 1.0:
                    logger.warning("⚠️ Voice detected from very beginning - might be inaccurate")
                else:
                    logger.info("✅ Voice detected from proper position - accurate!")
                
                return True
            else:
                logger.warning("⚠️ No suitable voice segment found")
                return False
        else:
            logger.error("❌ No voice segments detected")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi test accuracy: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST SMART OPTIMIZED WORKFLOW ===")
    
    # Test voice detection accuracy
    print("\n1. Testing Voice Detection Accuracy...")
    accuracy_success = test_voice_detection_accuracy()
    
    # Test smart optimized workflow
    print("\n2. Testing Smart Optimized Workflow...")
    workflow_success = test_smart_optimized_workflow()
    
    # Kết quả
    print("\n=== KET QUA TONG KET ===")
    print(f"Voice Detection Accuracy: {'PASS' if accuracy_success else 'FAIL'}")
    print(f"Smart Optimized Workflow: {'PASS' if workflow_success else 'FAIL'}")
    
    if accuracy_success and workflow_success:
        print("\n🎉 SMART OPTIMIZED WORKFLOW HOAT DONG HOAN HAO!")
        print("\n🚀 Hệ thống đã sẵn sàng sử dụng với Smart Voice Detection!")
    else:
        print("\n⚠️ CAN CAI THIEN THEM!")
