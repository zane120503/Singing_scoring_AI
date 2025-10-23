#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script cho Smart Audio Processor
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_smart_audio_processor():
    """Test Smart Audio Processor"""
    try:
        from src.ai.smart_audio_processor import SmartAudioProcessor
        
        # Khởi tạo processor
        processor = SmartAudioProcessor()
        
        # Test file (thay đổi đường dẫn phù hợp)
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\1478205f445c36e54c_mdx\\input_Vocals_DeReverb_converted.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🎤 Testing Smart Audio Processor với file: {test_file}")
        
        # Bước 1: Phân tích voice activity
        logger.info("\n=== BƯỚC 1: PHÂN TÍCH VOICE ACTIVITY ===")
        analysis = processor.get_voice_analysis(test_file)
        
        if "error" in analysis:
            logger.error(f"Lỗi phân tích: {analysis['error']}")
            return False
        
        logger.info(f"📊 Kết quả phân tích:")
        logger.info(f"   Total duration: {analysis['total_duration']:.2f}s")
        logger.info(f"   Voice duration: {analysis['voice_duration']:.2f}s")
        logger.info(f"   Voice ratio: {analysis['voice_ratio']:.1%}")
        logger.info(f"   Voice segments: {analysis['voice_count']}")
        
        # Hiển thị các đoạn voice
        for i, segment in enumerate(analysis['voice_segments'][:5]):  # Chỉ hiển thị 5 đoạn đầu
            logger.info(f"   Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s (conf: {segment['confidence']:.2f})")
        
        # Bước 2: Xử lý file để tạo voice sample
        logger.info("\n=== BƯỚC 2: TẠO VOICE SAMPLE ===")
        output_dir = "D:\\singing scoring AI\\output\\voice_samples"
        
        result = processor.process_karaoke_file(test_file, output_dir, slice_duration=20.0)
        
        if result["success"]:
            logger.info(f"✅ Thành công tạo voice sample:")
            logger.info(f"   Output file: {result['output_file']}")
            logger.info(f"   Voice start: {result['selected_voice']['start']:.2f}s")
            logger.info(f"   Slice duration: {result['slice_duration']}s")
            
            return True
        else:
            logger.error(f"❌ Thất bại: {result['error']}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi trong test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_activity_detector():
    """Test riêng Voice Activity Detector"""
    try:
        from src.ai.voice_activity_detector import VoiceActivityDetector
        
        detector = VoiceActivityDetector()
        
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\1478205f445c36e54c_mdx\\input_Vocals_DeReverb_converted.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"🎤 Testing Voice Activity Detector...")
        
        # Test các phương pháp khác nhau
        methods = ["spectral", "energy", "zero_crossing", "combined"]
        
        for method in methods:
            logger.info(f"\n--- Testing method: {method} ---")
            segments = detector.detect_voice_activity(test_file, method=method)
            logger.info(f"Phát hiện {len(segments)} voice segments")
            
            for i, segment in enumerate(segments[:3]):  # Chỉ hiển thị 3 đoạn đầu
                logger.info(f"  Segment {i+1}: {segment['start']:.2f}s - {segment['end']:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Lỗi test VAD: {e}")
        return False

def test_audio_slicer():
    """Test riêng Audio Slicer"""
    try:
        from src.ai.audio_slicer import AudioSlicer
        
        slicer = AudioSlicer()
        
        test_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\1478205f445c36e54c_mdx\\input_Vocals_DeReverb_converted.mp3"
        
        if not os.path.exists(test_file):
            logger.error(f"Test file không tồn tại: {test_file}")
            return False
        
        logger.info(f"✂️ Testing Audio Slicer...")
        
        # Test slice audio
        output_file = "D:\\singing scoring AI\\output\\test_slice.wav"
        success = slicer.slice_audio(test_file, output_file, start_time=10.0, duration=5.0)
        
        if success:
            logger.info(f"✅ Thành công cắt audio: {output_file}")
            
            # Test preview
            preview_success = slicer.preview_voice_segment(output_file, 0, 5.0)
            if preview_success:
                logger.info("✅ Preview thành công")
            
            return True
        else:
            logger.error("❌ Thất bại cắt audio")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi test Audio Slicer: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST SMART AUDIO PROCESSOR ===")
    
    # Test các components riêng lẻ
    print("\n1. Testing Voice Activity Detector...")
    vad_success = test_voice_activity_detector()
    
    print("\n2. Testing Audio Slicer...")
    slicer_success = test_audio_slicer()
    
    # Test Smart Audio Processor
    print("\n3. Testing Smart Audio Processor...")
    processor_success = test_smart_audio_processor()
    
    # Kết quả tổng kết
    print("\n=== KẾT QUẢ TỔNG KẾT ===")
    print(f"Voice Activity Detector: {'✅ PASS' if vad_success else '❌ FAIL'}")
    print(f"Audio Slicer: {'✅ PASS' if slicer_success else '❌ FAIL'}")
    print(f"Smart Audio Processor: {'✅ PASS' if processor_success else '❌ FAIL'}")
    
    if all([vad_success, slicer_success, processor_success]):
        print("\n🎉 TẤT CẢ TESTS THÀNH CÔNG!")
    else:
        print("\n⚠️ MỘT SỐ TESTS THẤT BẠI!")
