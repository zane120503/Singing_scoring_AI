#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra key detection cho file vocals MP3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vocals_key_detection():
    """Test key detection cho file vocals"""
    logger.info("=== TEST VOCALS KEY DETECTION ===")
    
    try:
        # Import key detector
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        # Khởi tạo key detector
        key_detector = AdvancedKeyDetector()
        
        # Đường dẫn file vocals
        vocals_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\a782fca021e952ec38_mdx\\input_Vocals_DeReverb_converted.mp3"
        
        if not os.path.exists(vocals_file):
            logger.error(f"File vocals không tồn tại: {vocals_file}")
            return False
        
        logger.info(f"Testing vocals file: {vocals_file}")
        
        # Test key detection
        logger.info("Đang thực hiện key detection...")
        result = key_detector.detect_key(vocals_file, "vocals")
        
        if result:
            logger.info(f"✅ Kết quả key detection:")
            logger.info(f"   Key: {result['key']} {result['scale']}")
            logger.info(f"   Confidence: {result['confidence']:.3f}")
            logger.info(f"   Method: {result['method']}")
            
            # So sánh với key của beat nhạc (C major)
            beat_key = "C major"
            vocals_key = f"{result['key']} {result['scale']}"
            
            logger.info(f"\n📊 So sánh:")
            logger.info(f"   Beat nhạc: {beat_key}")
            logger.info(f"   Giọng hát: {vocals_key}")
            
            if result['key'] == 'C' and result['scale'] == 'major':
                logger.info("✅ CHÍNH XÁC: Key detection cho vocals đã đúng!")
                return True
            else:
                logger.warning(f"❌ CHƯA CHÍNH XÁC: Key detection cho vocals vẫn sai (mong đợi C major)")
                return False
        else:
            logger.error("❌ Key detection thất bại")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vocals_key_detection()
    if success:
        print("\n🎉 TEST THÀNH CÔNG!")
    else:
        print("\n⚠️ TEST CẦN CẢI THIỆN!")
