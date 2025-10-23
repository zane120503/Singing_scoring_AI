#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra key detection cho file vocals MP3 mới
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vocals_key_detection_new():
    """Test key detection cho file vocals mới"""
    logger.info("=== TEST VOCALS KEY DETECTION - FILE MỚI ===")
    
    try:
        # Import key detector
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        # Khởi tạo key detector
        key_detector = AdvancedKeyDetector()
        
        # Đường dẫn file vocals mới
        vocals_file = "D:\\singing scoring AI\\Audio_separator_ui\\clean_song_output\\d01f4ec899c813f91b_mdx\\input_Vocals_DeReverb_converted.mp3"
        
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
            
            # Phân tích kết quả
            logger.info(f"\n📊 Phân tích kết quả:")
            logger.info(f"   Key được phát hiện: {result['key']} {result['scale']}")
            logger.info(f"   Độ tin cậy: {result['confidence']:.1%}")
            logger.info(f"   Phương pháp: {result['method']}")
            
            # Đánh giá chất lượng
            if result['confidence'] > 0.8:
                logger.info("✅ Độ tin cậy CAO - Key detection có thể chính xác")
            elif result['confidence'] > 0.6:
                logger.info("⚠️ Độ tin cậy TRUNG BÌNH - Cần xem xét thêm")
            else:
                logger.info("❌ Độ tin cậy THẤP - Key detection có thể không chính xác")
            
            return True
        else:
            logger.error("❌ Key detection thất bại")
            return False
            
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vocals_key_detection_new()
    if success:
        print("\nHoan thanh test key detection!")
    else:
        print("\nTest that bai!")
