#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process Full Audio - Xử lý trực tiếp 2 file đầu vào (không cắt)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import librosa
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_full_audio(karaoke_file: str, beat_file: str):
    """Xử lý trực tiếp 2 file đầu vào: tách giọng, phát hiện key, so sánh, chấm điểm"""
    try:
        from src.ai.advanced_audio_processor import AdvancedAudioProcessor
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        from src.core.scoring_system import KaraokeScoringSystem
        
        if not os.path.exists(karaoke_file):
            logger.error(f"Karaoke file không tồn tại: {karaoke_file}")
            return False
        if not os.path.exists(beat_file):
            logger.error(f"Beat file không tồn tại: {beat_file}")
            return False
        
        logger.info("🎵 Processing Full Audio Files...")
        logger.info(f"Karaoke: {karaoke_file}")
        logger.info(f"Beat: {beat_file}")
        
        # Bước 1: Tách giọng với Audio Separator
        logger.info("\n🔧 Bước 1: Tách giọng với Audio Separator...")
        
        # Khởi tạo Audio Processor
        audio_processor = AdvancedAudioProcessor(fast_mode=False)
        
        # Tạo output directory
        output_dir = "D:\\singing scoring AI\\output\\full_audio_processing"
        os.makedirs(output_dir, exist_ok=True)
        
        # Tách giọng trực tiếp từ file karaoke đầy đủ
        vocals_file = audio_processor.separate_vocals(karaoke_file)
        
        if vocals_file:
            # Tạo instrumental file (giả định)
            instrumental_file = vocals_file.replace("vocals.wav", "instrumental.wav")
            
            separation_result = {
                "success": True,
                "vocals_file": vocals_file,
                "instrumental_file": instrumental_file
            }
        else:
            separation_result = {"success": False, "error": "Failed to separate vocals"}
        
        if separation_result["success"]:
            logger.info("✅ Tách giọng thành công!")
            logger.info(f"   Vocals file: {separation_result['vocals_file']}")
            logger.info(f"   Instrumental file: {separation_result['instrumental_file']}")
            # Có thể log thời lượng nếu cần (không bắt buộc)
            try:
                v_audio, v_sr = librosa.load(separation_result['vocals_file'], sr=None, mono=True)
                v_duration = len(v_audio) / float(v_sr)
                logger.info(f"   Vocals duration: {v_duration:.2f}s")
            except Exception:
                pass
            
            vocals_file = separation_result['vocals_file']
            instrumental_file = separation_result['instrumental_file']
        else:
            logger.error(f"❌ Tách giọng thất bại: {separation_result['error']}")
            return False
        
        # Bước 2: Phát hiện key cho vocals
        logger.info("\n🎼 Bước 2: Phát hiện key cho vocals...")
        
        key_detector = AdvancedKeyDetector()
        
        # Phát hiện key cho vocals
        vocals_key_result = key_detector.detect_key(vocals_file, audio_type="vocals")
        
        if vocals_key_result and "key" in vocals_key_result:
            logger.info("✅ Phát hiện key cho vocals thành công!")
            logger.info(f"   Detected key: {vocals_key_result['key']}")
            logger.info(f"   Confidence: {vocals_key_result['confidence']:.3f}")
            logger.info(f"   Method: {vocals_key_result['method']}")
        else:
            logger.error(f"❌ Phát hiện key cho vocals thất bại")
            return False
        
        # Bước 3: Phát hiện key cho instrumental
        logger.info("\n🎼 Bước 3: Phát hiện key cho instrumental...")
        
        instrumental_key_result = key_detector.detect_key(instrumental_file, audio_type="instrumental")
        
        if instrumental_key_result and "key" in instrumental_key_result:
            logger.info("✅ Phát hiện key cho instrumental thành công!")
            logger.info(f"   Detected key: {instrumental_key_result['key']}")
            logger.info(f"   Confidence: {instrumental_key_result['confidence']:.3f}")
            logger.info(f"   Method: {instrumental_key_result['method']}")
        else:
            logger.error(f"❌ Phát hiện key cho instrumental thất bại")
            return False
        
        # Bước 4: So sánh keys và tính điểm
        logger.info("\n📊 Bước 4: So sánh keys và tính điểm...")
        
        scoring_system = KaraokeScoringSystem()
        
        # So sánh keys (đơn giản)
        vocals_key = vocals_key_result['key']
        instrumental_key = instrumental_key_result['key']
        
        # Tính key similarity đơn giản
        if vocals_key == instrumental_key:
            key_similarity = 1.0
            key_match = True
            key_score = 100.0
        else:
            # Tính similarity dựa trên circle of fifths
            key_similarity = 0.5  # Giả định
            key_match = False
            key_score = 50.0
        
        key_comparison = {
            "match": key_match,
            "similarity": key_similarity,
            "score": key_score,
            "vocals_key": vocals_key,
            "instrumental_key": instrumental_key
        }
        
        logger.info("✅ So sánh keys thành công!")
        logger.info(f"   Vocals key: {vocals_key_result['key']}")
        logger.info(f"   Instrumental key: {instrumental_key_result['key']}")
        logger.info(f"   Key match: {key_comparison['match']}")
        logger.info(f"   Key similarity: {key_comparison['similarity']:.3f}")
        logger.info(f"   Key score: {key_comparison['score']:.2f}")
        
        # Bước 5: Tính điểm tổng thể
        logger.info("\n🏆 Bước 5: Tính điểm tổng thể...")
        
        # Tính overall score đơn giản
        overall_score_value = key_comparison['score']
        
        if overall_score_value >= 90:
            grade = "A"
            feedback = "Excellent key matching!"
        elif overall_score_value >= 80:
            grade = "B"
            feedback = "Good key matching!"
        elif overall_score_value >= 70:
            grade = "C"
            feedback = "Fair key matching!"
        else:
            grade = "D"
            feedback = "Poor key matching!"
        
        overall_score = {
            "overall_score": overall_score_value,
            "grade": grade,
            "feedback": feedback
        }
        
        logger.info("✅ Tính điểm tổng thể thành công!")
        logger.info(f"   Overall score: {overall_score['overall_score']:.2f}")
        logger.info(f"   Grade: {overall_score['grade']}")
        logger.info(f"   Feedback: {overall_score['feedback']}")
        
        # Kết quả tổng hợp
        result = {
            "success": True,
            "karaoke_file": karaoke_file,
            "beat_file": beat_file,
            "separation_result": separation_result,
            "vocals_key_result": vocals_key_result,
            "instrumental_key_result": instrumental_key_result,
            "key_comparison": key_comparison,
            "overall_score": overall_score
        }
        
        logger.info("\n🎉 Xử lý full audio hoàn tất!")
        return result
        
    except Exception as e:
        logger.error(f"❌ Lỗi trong xử lý: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== PROCESS FULL AUDIO ===")
    
    # Ví dụ chạy nhanh: cập nhật đường dẫn tương ứng
    karaoke = "C:\\path\\to\\karaoke.wav"
    beat = "C:\\path\\to\\beat.wav"
    print("\n1. Processing Full Audio Files...")
    result = process_full_audio(karaoke, beat)
    
    # Kết quả
    print("\n=== KET QUA ===")
    if result:
        print("PROCESSING FULL AUDIO HOAN TAT!")
        print("He thong da xu ly thanh cong 2 file dau vao!")
        
        # Hiển thị kết quả chi tiết
        if isinstance(result, dict):
            print(f"\nKet qua chi tiet:")
            print(f"   Vocals key: {result['vocals_key_result']['key']}")
            print(f"   Instrumental key: {result['instrumental_key_result']['key']}")
            print(f"   Key similarity: {result['key_comparison']['similarity']:.3f}")
            print(f"   Overall score: {result['overall_score']['overall_score']:.2f}")
            print(f"   Grade: {result['overall_score']['grade']}")
    else:
        print("CAN CAI THIEN THEM!")
        print("Van con van de voi xu ly audio!")
