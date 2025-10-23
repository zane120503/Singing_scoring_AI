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

from typing import Dict
import shutil

from src.ai.advanced_audio_processor import AdvancedAudioProcessor
from src.ai.advanced_key_detector import AdvancedKeyDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def run_workflow(karaoke_file: str, beat_file: str, duration: float = 20.0, output_dir: str = None) -> Dict:
    """Chạy workflow cắt 20s (15-35s), tách giọng, detect key, so sánh & chấm điểm."""
    # 0) Chuẩn bị thư mục xuất
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'Audio_separator_ui', 'clean_song_output')
    os.makedirs(output_dir, exist_ok=True)

    base_stem = os.path.splitext(os.path.basename(karaoke_file))[0]

    # 1) Cắt 20s từ 25s đến 45s của file karaoke
    try:
        audio, sr = librosa.load(karaoke_file, sr=None, mono=True)
        start_t = 25.0
        end_t = start_t + duration
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        if start_sample >= len(audio):
            return {"success": False, "error": "Karaoke ngắn hơn 25s"}
        slice_audio = audio[start_sample:min(end_sample, len(audio))]
        sliced_path = os.path.join(output_dir, f"{base_stem}_slice_{int(start_t)}s_{int(end_t)}s.wav")
        sf.write(sliced_path, slice_audio, sr)
    except Exception as e:
        return {"success": False, "error": f"Lỗi cắt audio: {e}"}

    # 2) Tách giọng từ đoạn 20s đã cắt
    audio_proc = AdvancedAudioProcessor(fast_mode=False)
    vocals_path = audio_proc.separate_vocals(sliced_path)
    if not vocals_path or not os.path.exists(vocals_path):
        return {"success": False, "error": "Tách giọng thất bại"}

    # Xuất/copy vocals 20s đã tách ra output_dir
    vocals_ext = os.path.splitext(vocals_path)[1]
    vocals_export = os.path.join(output_dir, f"{base_stem}_slice_vocals{vocals_ext}")
    try:
        shutil.copy2(vocals_path, vocals_export)
    except Exception:
        # fallback: nếu copy fail vẫn dùng vocals_path gốc
        vocals_export = vocals_path

    # 3) Detect key bằng multiple methods cho beat (file gốc) để tăng độ chính xác
    keydet = AdvancedKeyDetector()
    try:
        vocals_key = keydet.detect_key(vocals_export, audio_type='vocals')
    except Exception:
        vocals_key = None
    
    # Thử nhiều phương pháp detect key cho beat (file gốc)
    beat_key = None
    beat_methods = ['beat', 'instrumental', 'vocals']  # Thử các audio_type khác nhau
    
    for method in beat_methods:
        try:
            temp_beat_key = keydet.detect_key(beat_file, audio_type=method)
            if temp_beat_key and 'key' in temp_beat_key:
                beat_key = temp_beat_key
                logger.info(f"✅ Beat key detected với method '{method}': {beat_key['key']}")
                break
        except Exception as e:
            logger.warning(f"Method '{method}' failed: {e}")
            continue
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


