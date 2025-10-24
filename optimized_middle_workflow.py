#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Middle Workflow (20s Slice Processing)
- Input: karaoke_file, beat_file
- Steps:
  1) Cắt file karaoke từ 15s đến 35s (20s)
  2) Tách giọng trên đoạn 20s đã cắt bằng Audio Separator
  3) Xuất file vocals 20s đã tách
  4) Detect key (ưu tiên Docker Essentia) cho vocals 20s và beat
  5) So sánh key và tính điểm
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging
import librosa
import soundfile as sf
import concurrent.futures
import threading

from typing import Dict
import shutil

from src.ai.advanced_audio_processor import AdvancedAudioProcessor
from src.ai.advanced_key_detector import AdvancedKeyDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def run_workflow(karaoke_file: str, beat_file: str, duration: float = 30.0, output_dir: str = None) -> Dict:
    """Chạy workflow cắt 30s (15-45s), tách giọng, detect key, so sánh & chấm điểm."""
    # 0) Chuẩn bị thư mục xuất
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'Audio_separator_ui', 'clean_song_output')
    os.makedirs(output_dir, exist_ok=True)

    base_stem = os.path.splitext(os.path.basename(karaoke_file))[0]

    # 1) Cắt audio thông minh dựa trên độ dài file
    try:
        audio, sr = librosa.load(karaoke_file, sr=None, mono=True)
        total_duration = len(audio) / sr
        
        logger.info(f"📊 File duration: {total_duration:.2f}s")
        
        # Logic cắt thông minh
        if total_duration <= duration:
            # File ngắn: sử dụng toàn bộ file
            logger.info(f"📁 File ngắn ({total_duration:.2f}s ≤ {duration}s), sử dụng toàn bộ file")
            start_t = 0.0
            end_t = total_duration
            slice_audio = audio
        elif total_duration <= 60.0:
            # File trung bình: cắt từ giữa
            logger.info(f"📁 File trung bình ({total_duration:.2f}s), cắt từ giữa")
            start_t = max(0, (total_duration - duration) / 2)
            end_t = start_t + duration
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            slice_audio = audio[start_sample:end_sample]
        else:
            # File dài: cắt từ 15s như cũ
            logger.info(f"📁 File dài ({total_duration:.2f}s), cắt từ 15s")
            start_t = 15.0
            end_t = start_t + duration
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            slice_audio = audio[start_sample:end_sample]
        
        # Lưu file đã cắt
        actual_duration = len(slice_audio) / sr
        sliced_path = os.path.join(output_dir, f"{base_stem}_slice_{int(start_t)}s_{int(start_t + actual_duration)}s.wav")
        sf.write(sliced_path, slice_audio, sr)
        
        logger.info(f"✅ Đã cắt audio: {actual_duration:.2f}s từ {start_t:.1f}s")
        
    except Exception as e:
        return {"success": False, "error": f"Lỗi cắt audio: {e}"}

    # 2) Khởi tạo Key Detector và bắt đầu Beat Key Detection ngay lập tức
    logger.info("🎼 Khởi tạo Key Detector và bắt đầu Beat Key Detection...")
    keydet = AdvancedKeyDetector()
    
    # Log GPU status
    if keydet.use_gpu:
        logger.info(f"🚀 GPU acceleration ENABLED on device: {keydet.device}")
    else:
        logger.info("💻 GPU acceleration DISABLED, using CPU")
    
    def detect_beat_key():
        """Detect key cho beat với focus vào accuracy"""
        try:
            logger.info(f"🎵 Đang phát hiện key cho beat...")
            # Sử dụng audio_type='beat' để trigger beat-specific analysis
            result = keydet.detect_key(beat_file, audio_type='beat')
            if result and 'key' in result:
                logger.info(f"✅ Beat key detected: {result['key']}")
                return result
            else:
                # Fallback: thử với instrumental
                logger.info("🔄 Fallback: thử với audio_type='instrumental'...")
                result = keydet.detect_key(beat_file, audio_type='instrumental')
                if result and 'key' in result:
                    logger.info(f"✅ Beat key detected (fallback): {result['key']}")
                    return result
                else:
                    # Final fallback: vocals method
                    logger.info("🔄 Final fallback: thử với audio_type='vocals'...")
                    result = keydet.detect_key(beat_file, audio_type='vocals')
                    if result and 'key' in result:
                        logger.info(f"✅ Beat key detected (final fallback): {result['key']}")
                        return result
        except Exception as e:
            logger.warning(f"Beat key detection failed: {e}")
        return None
    
    def separate_vocals():
        """Tách giọng từ đoạn audio đã cắt"""
        try:
            logger.info("🎤 Bắt đầu tách giọng hát...")
            audio_proc = AdvancedAudioProcessor(fast_mode=False)
            vocals_path = audio_proc.separate_vocals(sliced_path)
            if not vocals_path or not os.path.exists(vocals_path):
                return None, None
            
            # Xuất/copy vocals đã tách ra output_dir
            vocals_ext = os.path.splitext(vocals_path)[1]
            vocals_export = os.path.join(output_dir, f"{base_stem}_slice_vocals{vocals_ext}")
            try:
                shutil.copy2(vocals_path, vocals_export)
            except Exception:
                # fallback: nếu copy fail vẫn dùng vocals_path gốc
                vocals_export = vocals_path
            
            logger.info("✅ Tách giọng hoàn thành!")
            return vocals_path, vocals_export
        except Exception as e:
            logger.warning(f"Vocal separation failed: {e}")
            return None, None
    
    def detect_vocals_key(vocals_export):
        """Detect key cho vocals"""
        try:
            logger.info("🎤 Đang phát hiện key cho vocals...")
            result = keydet.detect_key(vocals_export, audio_type='vocals')
            logger.info(f"✅ Vocals key detected: {result.get('key', 'Unknown')}")
            return result
        except Exception as e:
            logger.warning(f"Vocals key detection failed: {e}")
            return None
    
    # 3) Chạy Beat Key Detection và Vocal Separation SONG SONG
    logger.info("⚡ Chạy Beat Key Detection và Vocal Separation song song...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit beat key detection ngay lập tức
        beat_future = executor.submit(detect_beat_key)
        # Submit vocal separation song song
        vocals_sep_future = executor.submit(separate_vocals)
        
        # Chờ beat key detection hoàn thành trước
        beat_key = beat_future.result()
        logger.info("🎉 Beat key detection hoàn thành!")
        
        # Chờ vocal separation hoàn thành
        vocals_path, vocals_export = vocals_sep_future.result()
        if not vocals_export:
            return {"success": False, "error": "Tách giọng thất bại"}
        
        # Detect vocals key sau khi separation hoàn thành
        vocals_key = detect_vocals_key(vocals_export)
    
    logger.info("🎉 Hoàn thành tất cả key detection!")
    if not (vocals_key and 'key' in vocals_key and beat_key and 'key' in beat_key):
        return {"success": False, "error": "Phát hiện key thất bại"}

    # 4) So sánh key và tính điểm đơn giản
    v_key = vocals_key['key']
    b_key = beat_key['key']
    if v_key == b_key:
        similarity = 1.0
        score = 100.0
        match = True
    else:
        similarity = 0.5
        score = 50.0
        match = False

    return {
        "success": True,
        "inputs": {"karaoke_file": karaoke_file, "beat_file": beat_file},
        "sliced_karaoke": sliced_path,
        "vocals_src": vocals_path,
        "vocals_export": vocals_export,
        "vocals_key": vocals_key,
        "beat_key": beat_key,
        "key_compare": {"match": match, "similarity": similarity, "score": score}
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Optimized middle workflow: slice 20s, separate vocals, detect keys, compare & score")
    parser.add_argument("karaoke", help="Đường dẫn file ghi âm karaoke")
    parser.add_argument("beat", help="Đường dẫn file beat nhạc")
    parser.add_argument("--output", "-o", help="Thư mục output (mặc định: Audio_separator_ui/clean_song_output)")
    parser.add_argument("--duration", "-d", type=float, default=20.0, help="Thời lượng cắt (mặc định 20s)")
    args = parser.parse_args()

    result = run_workflow(args.karaoke, args.beat, duration=args.duration, output_dir=args.output)
    if not isinstance(result, dict) or not result.get("success"):
        print("❌ Lỗi:", result.get("error") if isinstance(result, dict) else "Không rõ")
        raise SystemExit(1)

    print("✅ Hoàn tất!")
    print("- Karaoke slice:", result["sliced_karaoke"]) 
    print("- Vocals 20s:", result["vocals_export"]) 
    print("- Vocals key:", result["vocals_key"]) 
    print("- Beat key:", result["beat_key"]) 
    print("- So sánh key:", result["key_compare"]) 


