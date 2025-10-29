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
import numpy as np
import time
from datetime import datetime, timedelta

from typing import Dict
import shutil

from src.ai.advanced_audio_processor import AdvancedAudioProcessor
from src.ai.advanced_key_detector import AdvancedKeyDetector

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def check_audio_content(audio_path: str) -> Dict:
    """Kiểm tra file có nội dung audio hay không (không phải file rỗng)"""
    try:
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # Kiểm tra độ dài
        duration = len(audio) / sr
        if duration < 0.1:  # File quá ngắn (< 0.1s)
            return {"has_audio": False, "reason": "File quá ngắn (< 0.1s)"}
        
        # Kiểm tra RMS energy (để phát hiện file không có âm thanh)
        rms = librosa.feature.rms(y=audio)[0]
        mean_rms = np.mean(rms)
        max_rms = np.max(rms)
        
        # Threshold rất thấp để phát hiện file gần như im lặng
        if mean_rms < 1e-6 and max_rms < 1e-5:
            return {"has_audio": False, "reason": "File không có nội dung âm thanh (RMS quá thấp)"}
        
        return {
            "has_audio": True,
            "duration": duration,
            "mean_rms": float(mean_rms),
            "max_rms": float(max_rms)
        }
    except Exception as e:
        logger.error(f"❌ Lỗi kiểm tra nội dung audio: {e}")
        return {"has_audio": False, "reason": f"Lỗi khi đọc file: {e}"}

def clean_old_output_files(output_dir: str):
    """Xóa các file và thư mục cũ trong thư mục output (xóa tất cả để thay thế hoàn toàn)"""
    try:
        if not os.path.exists(output_dir):
            return
        
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac']
        deleted_files_count = 0
        deleted_dirs_count = 0
        deleted_items = []
        
        # Duyệt qua tất cả các item trong thư mục
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            
            try:
                if os.path.isfile(item_path):
                    # Xóa file audio
                    _, ext = os.path.splitext(item)
                    if ext.lower() in audio_extensions:
                        os.remove(item_path)
                        deleted_files_count += 1
                        deleted_items.append(item)
                        logger.info(f"   🗑️ Đã xóa file: {item}")
                elif os.path.isdir(item_path):
                    # Xóa tất cả thư mục con (đặc biệt là các thư mục *_mdx)
                    shutil.rmtree(item_path)
                    deleted_dirs_count += 1
                    deleted_items.append(item + "/")
                    logger.info(f"   🗑️ Đã xóa thư mục: {item}/")
            except Exception as e:
                logger.warning(f"   ⚠️ Không thể xóa {item}: {e}")
        
        total_deleted = deleted_files_count + deleted_dirs_count
        if total_deleted > 0:
            logger.info(f"✅ Đã xóa {total_deleted} item cũ ({deleted_files_count} file, {deleted_dirs_count} thư mục) trong {output_dir}")
            if len(deleted_items) <= 5:
                logger.info(f"   Các item đã xóa: {', '.join(deleted_items)}")
            else:
                logger.info(f"   Các item đã xóa (5 đầu tiên): {', '.join(deleted_items[:5])}...")
        else:
            logger.info(f"   ℹ️ Không có file/thư mục cũ cần xóa trong {output_dir}")
            
    except Exception as e:
        logger.warning(f"⚠️ Lỗi khi xóa file/thư mục cũ: {e}")
        # Không throw exception, chỉ log warning để không làm gián đoạn workflow

def format_duration(seconds: float) -> str:
    """Định dạng thời gian từ giây sang chuỗi dễ đọc"""
    if seconds < 60:
        return f"{seconds:.2f} giây"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes} phút {secs:.2f} giây"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours} giờ {minutes} phút {secs:.2f} giây"

def run_workflow(karaoke_file: str, beat_file: str, duration: float = 30.0, output_dir: str = None) -> Dict:
    """Chạy workflow cắt 30s (15-45s), tách giọng, detect key, so sánh & chấm điểm."""
    # Ghi nhận thời gian bắt đầu
    start_time = time.time()
    start_datetime = datetime.now()
    
    logger.info("\n" + "="*70)
    logger.info("🚀 BẮT ĐẦU PHÂN TÍCH")
    logger.info("="*70)
    logger.info(f"   ⏰ Thời gian bắt đầu: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70 + "\n")
    
    # 0) Chuẩn bị thư mục xuất
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'Audio_separator_ui', 'clean_song_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 0.1) Xóa các file cũ trong thư mục output trước khi xuất file mới
    logger.info("\n" + "="*70)
    logger.info("🗑️ XÓA FILE CŨ TRONG THƯ MỤC OUTPUT")
    logger.info("="*70)
    clean_old_output_files(output_dir)
    logger.info("="*70 + "\n")

    base_stem = os.path.splitext(os.path.basename(karaoke_file))[0]

    # 0.5) VALIDATION: Kiểm tra file có nội dung audio hay không
    logger.info("\n" + "="*70)
    logger.info("🔍 BƯỚC 0: KIỂM TRA FILE ĐẦU VÀO")
    logger.info("="*70)
    
    # Kiểm tra karaoke file
    karaoke_check = check_audio_content(karaoke_file)
    if not karaoke_check.get('has_audio'):
        elapsed = time.time() - start_time
        end_datetime = datetime.now()
        logger.error(f"❌ File karaoke không có nội dung: {karaoke_check.get('reason', 'Unknown')}")
        logger.info("\n" + "="*70)
        logger.info("⏹️ KẾT THÚC PHÂN TÍCH (LỖI)")
        logger.info("="*70)
        logger.info(f"   ⏰ Thời gian kết thúc: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   ⏱️ Thời gian phân tích: {format_duration(elapsed)} ({elapsed:.2f}s)")
        logger.info("="*70 + "\n")
        return {
            "success": False,
            "error": f"File karaoke không có nội dung: {karaoke_check.get('reason', 'Unknown')}",
            "score": 0.0,
            "message": "File karaoke không có nội dung âm thanh",
            "duration_seconds": elapsed,
            "start_time": start_datetime.isoformat(),
            "end_time": end_datetime.isoformat()
        }
    logger.info(f"✅ File karaoke hợp lệ: {karaoke_check.get('duration', 0):.2f}s")
    
    # Kiểm tra beat file
    beat_check = check_audio_content(beat_file)
    if not beat_check.get('has_audio'):
        elapsed = time.time() - start_time
        end_datetime = datetime.now()
        logger.error(f"❌ File beat không có nội dung: {beat_check.get('reason', 'Unknown')}")
        logger.info("\n" + "="*70)
        logger.info("⏹️ KẾT THÚC PHÂN TÍCH (LỖI)")
        logger.info("="*70)
        logger.info(f"   ⏰ Thời gian kết thúc: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   ⏱️ Thời gian phân tích: {format_duration(elapsed)} ({elapsed:.2f}s)")
        logger.info("="*70 + "\n")
        return {
            "success": False,
            "error": f"File beat không có nội dung: {beat_check.get('reason', 'Unknown')}",
            "score": 0.0,
            "message": "File beat không có nội dung âm thanh",
            "duration_seconds": elapsed,
            "start_time": start_datetime.isoformat(),
            "end_time": end_datetime.isoformat()
        }
    logger.info(f"✅ File beat hợp lệ: {beat_check.get('duration', 0):.2f}s")
    
    logger.info("="*70 + "\n")

    # 1) Phát hiện vocal và cắt audio theo vị trí vocal
    try:
        logger.info("\n" + "="*70)
        logger.info("🔍 BƯỚC 1: PHÁT HIỆN VOCAL TRONG FILE KARAOKE")
        logger.info("="*70)
        
        # Phát hiện vocal với VocalsPresenceChecker
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'ai'))
        from vocals_presence_checker import VocalsPresenceChecker
        
        vocals_checker = VocalsPresenceChecker()
        vocals_result = vocals_checker.check_vocals_presence(karaoke_file)
        
        # Kiểm tra nghiêm ngặt hơn: cả has_vocals, confidence và segments
        has_vocals = vocals_result.get('has_vocals', False)
        confidence = vocals_result.get('confidence', 0.0)
        voice_segments = vocals_result.get('voice_segments', [])
        
        # Lọc bỏ các segments quá ngắn (< 1.5s)
        meaningful_segments = [seg for seg in voice_segments if (seg['end'] - seg['start']) >= 1.5]
        
        logger.info(f"   Confidence: {confidence:.2%}")
        logger.info(f"   Số segments: {len(voice_segments)}")
        logger.info(f"   Số segments có ý nghĩa (≥1.5s): {len(meaningful_segments)}")
        
        # Điều kiện nghiêm ngặt: phải có vocal, confidence > 0.4, và có ít nhất 1 segment có ý nghĩa
        if not has_vocals or confidence < 0.4 or len(meaningful_segments) == 0:
            logger.error("❌ KHÔNG PHÁT HIỆN GIỌNG HÁT trong file karaoke!")
            logger.error(f"   Confidence: {confidence:.2%} (yêu cầu: ≥40%)")
            logger.error(f"   Segments có ý nghĩa: {len(meaningful_segments)} (yêu cầu: ≥1)")
            logger.error("   File này có thể chỉ có beat/instrumental, không có giọng hát")
            logger.error("="*70 + "\n")
            elapsed = time.time() - start_time
            end_datetime = datetime.now()
            logger.info("\n" + "="*70)
            logger.info("⏹️ KẾT THÚC PHÂN TÍCH (LỖI)")
            logger.info("="*70)
            logger.info(f"   ⏰ Thời gian kết thúc: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   ⏱️ Thời gian phân tích: {format_duration(elapsed)} ({elapsed:.2f}s)")
            logger.info("="*70 + "\n")
            return {
                "success": False,
                "error": "File karaoke không có giọng hát",
                "score": 0.0,
                "key_compare": {"match": False, "similarity": 0.0, "score": 0.0},
                "message": f"File karaoke không chứa giọng hát (confidence: {confidence:.1%}, segments: {len(meaningful_segments)}). File có thể chỉ có beat/instrumental.",
                "duration_seconds": elapsed,
                "start_time": start_datetime.isoformat(),
                "end_time": end_datetime.isoformat()
            }
        
        # Có vocal hợp lệ: sử dụng meaningful_segments (đã được lọc ≥1.5s)
        # meaningful_segments đã được kiểm tra ở trên, nên chắc chắn có ít nhất 1 segment
        # Merge các segments gần nhau (cách nhau < 2s)
        merged_segments = []
        merged_segments = [meaningful_segments[0]]
        for seg in meaningful_segments[1:]:
            last_merged = merged_segments[-1]
            gap = seg['start'] - last_merged['end']
            if gap < 2.0:  # Merge nếu cách nhau < 2s
                merged_segments[-1]['end'] = seg['end']
            else:
                merged_segments.append(seg)
        
        logger.info(f"\n📊 THÔNG TIN VOCAL:")
        logger.info(f"   📊 Tổng số đoạn ban đầu: {len(voice_segments)}")
        logger.info(f"   📊 Số đoạn có ý nghĩa (≥1.5s): {len(meaningful_segments)}")
        logger.info(f"   📊 Số đoạn sau merge: {len(merged_segments)}")
        
        if merged_segments:
            # Tìm đoạn vocal dài nhất hoặc đoạn đầu tiên dài hơn 5s
            best_segment = None
            for seg in merged_segments:
                duration = seg['end'] - seg['start']
                if duration >= 5.0:  # Ưu tiên đoạn ≥ 5s
                    best_segment = seg
                    break
            
            # Nếu không có đoạn ≥ 5s, chọn đoạn dài nhất
            if not best_segment:
                best_segment = max(merged_segments, key=lambda x: x['end'] - x['start'])
            
            start_t = best_segment['start']
            end_t = best_segment['end']
            segment_duration = end_t - start_t
            
            logger.info(f"   ✅ Chọn đoạn vocal tốt nhất:")
            logger.info(f"      ⏰ VOCAL XUẤT HIỆN: {start_t:.2f}s ({int(start_t//60):02d}:{int(start_t%60):02d})")
            logger.info(f"      ⏰ VOCAL KẾT THÚC: {end_t:.2f}s ({int(end_t//60):02d}:{int(end_t%60):02d})")
            logger.info(f"      📏 Thời lượng đoạn: {segment_duration:.2f}s")
        else:
            # Fallback: dùng đoạn đầu tiên từ meaningful_segments
            first_seg = meaningful_segments[0]
            start_t = first_seg['start']
            end_t = first_seg['end']
            segment_duration = end_t - start_t
            logger.warning(f"⚠️ Sử dụng đoạn đầu tiên: {start_t:.2f}s - {end_t:.2f}s")
        
        # Load audio để lấy tổng độ dài
        audio, sr = librosa.load(karaoke_file, sr=None, mono=True)
        total_duration = len(audio) / sr
        
        # KIỂM TRA: Nếu vocal xuất hiện ở dưới 20 giây cuối → trả về 0 điểm
        # Ngoại lệ: Nếu file ngắn (≤ 30s), cho phép vocal xuất hiện ở bất kỳ đâu
        seconds_from_end = total_duration - start_t
        min_seconds_from_end = 20.0  # Phải xuất hiện trước 20 giây cuối (cho file dài)
        
        # Xử lý đặc biệt cho file ngắn (≤ 30s)
        is_short_file = total_duration <= 30.0
        
        logger.info(f"   🔍 Kiểm tra file: total_duration={total_duration:.2f}s, is_short_file={is_short_file}")
        
        if is_short_file:
            logger.info(f"   📏 File ngắn ({total_duration:.2f}s ≤ 30s), cho phép vocal xuất hiện ở bất kỳ vị trí nào")
            # Với file ngắn, không kiểm tra vị trí, chỉ cắt từ vocal đến cuối file
            end_t = total_duration
            final_duration = end_t - start_t
            logger.info(f"   📏 Cắt từ vocal bắt đầu ({start_t:.2f}s) đến cuối file ({total_duration:.2f}s)")
            logger.info(f"   📏 Độ dài đoạn cắt: {final_duration:.2f}s")
        else:
            # File dài: kiểm tra vị trí vocal
            if seconds_from_end < min_seconds_from_end:
                logger.error(f"❌ Giọng hát xuất hiện quá muộn trong file!")
                logger.error(f"   Vocal bắt đầu: {start_t:.2f}s ({int(start_t//60):02d}:{int(start_t%60):02d})")
                logger.error(f"   Độ dài file: {total_duration:.2f}s ({int(total_duration//60):02d}:{int(total_duration%60):02d})")
                logger.error(f"   Còn {seconds_from_end:.2f}s đến cuối file (yêu cầu: ≥{min_seconds_from_end:.0f}s)")
                logger.error("   File gốc có thể không phù hợp để phân tích (vocal xuất hiện quá muộn)")
                logger.error("="*70 + "\n")
                elapsed = time.time() - start_time
                end_datetime = datetime.now()
                logger.info("\n" + "="*70)
                logger.info("⏹️ KẾT THÚC PHÂN TÍCH (LỖI)")
                logger.info("="*70)
                logger.info(f"   ⏰ Thời gian kết thúc: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"   ⏱️ Thời gian phân tích: {format_duration(elapsed)} ({elapsed:.2f}s)")
                logger.info("="*70 + "\n")
                return {
                    "success": False,
                    "error": "Giọng hát xuất hiện quá muộn trong file",
                    "score": 0.0,
                    "key_compare": {"match": False, "similarity": 0.0, "score": 0.0},
                    "message": f"Giọng hát xuất hiện ở {seconds_from_end:.1f}s cuối cùng (yêu cầu: ≥{min_seconds_from_end:.0f}s). File không phù hợp để phân tích.",
                    "duration_seconds": elapsed,
                    "start_time": start_datetime.isoformat(),
                    "end_time": end_datetime.isoformat()
                }
            
            logger.info(f"   ✅ Vocal xuất hiện ở vị trí hợp lệ: còn {seconds_from_end:.2f}s đến cuối file (≥{min_seconds_from_end:.0f}s)")
            
            # Cắt từ khi giọng hát xuất hiện đến cuối file hoặc 30 giây (lấy phần ngắn hơn)
            max_duration = 30.0  # Tối đa 30 giây
            
            # Tính thời điểm kết thúc: min(bắt đầu + 30s, cuối file)
            # Không phụ thuộc vào thời điểm kết thúc vocal
            max_end_time = start_t + max_duration
            end_t = min(max_end_time, total_duration)
            
            final_duration = end_t - start_t
            
            logger.info(f"   📏 Cắt từ vocal bắt đầu ({start_t:.2f}s) đến:")
            logger.info(f"      - Bắt đầu + 30s: {max_end_time:.2f}s")
            logger.info(f"      - Cuối file: {total_duration:.2f}s")
            logger.info(f"      → Chọn: {end_t:.2f}s (độ dài: {final_duration:.2f}s)")
        
        logger.info("="*70 + "\n")
        
        logger.info(f"\n✂️ BƯỚC 2: CẮT FILE KARAOKE THEO VỊ TRÍ VOCAL")
        logger.info(f"   Từ: {start_t:.2f}s ({int(start_t//60):02d}:{int(start_t%60):02d})")
        logger.info(f"   Đến: {end_t:.2f}s ({int(end_t//60):02d}:{int(end_t%60):02d})")
        logger.info(f"   Thời lượng: {final_duration:.2f}s (yêu cầu: 30s)")
        
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        end_sample = min(end_sample, len(audio))  # Đảm bảo không vượt quá
        slice_audio = audio[start_sample:end_sample]
        
        logger.info("="*70 + "\n")
        
        # Lưu file đã cắt
        actual_duration = len(slice_audio) / sr
        sliced_path = os.path.join(output_dir, f"{base_stem}_slice_{int(start_t)}s_{int(start_t + actual_duration)}s.wav")
        sf.write(sliced_path, slice_audio, sr)
        
        logger.info(f"✅ Đã cắt audio: {actual_duration:.2f}s từ {start_t:.1f}s")
        
    except Exception as e:
        elapsed = time.time() - start_time
        end_datetime = datetime.now()
        logger.error(f"❌ Lỗi phát hiện vocal hoặc cắt audio: {e}")
        import traceback
        traceback.print_exc()
        logger.info("\n" + "="*70)
        logger.info("⏹️ KẾT THÚC PHÂN TÍCH (LỖI)")
        logger.info("="*70)
        logger.info(f"   ⏰ Thời gian kết thúc: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   ⏱️ Thời gian phân tích: {format_duration(elapsed)} ({elapsed:.2f}s)")
        logger.info("="*70 + "\n")
        return {
            "success": False,
            "error": f"Lỗi cắt audio: {e}",
            "duration_seconds": elapsed,
            "start_time": start_datetime.isoformat(),
            "end_time": end_datetime.isoformat()
        }

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
            elapsed = time.time() - start_time
            end_datetime = datetime.now()
            logger.info("\n" + "="*70)
            logger.info("⏹️ KẾT THÚC PHÂN TÍCH (LỖI)")
            logger.info("="*70)
            logger.info(f"   ⏰ Thời gian kết thúc: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   ⏱️ Thời gian phân tích: {format_duration(elapsed)} ({elapsed:.2f}s)")
            logger.info("="*70 + "\n")
            return {
                "success": False,
                "error": "Tách giọng thất bại",
                "score": 0.0,
                "duration_seconds": elapsed,
                "start_time": start_datetime.isoformat(),
                "end_time": end_datetime.isoformat()
            }
        
        # VALIDATION: Kiểm tra lại file vocal đã tách có thực sự có vocal không
        logger.info("\n" + "="*70)
        logger.info("🔍 BƯỚC 2.5: KIỂM TRA LẠI VOCAL SAU KHI TÁCH")
        logger.info("="*70)
        
        vocals_checker = VocalsPresenceChecker()
        vocals_after_sep_result = vocals_checker.check_vocals_presence(vocals_export)
        
        # Kiểm tra cả confidence và có segments hay không
        has_vocals_after_sep = vocals_after_sep_result.get('has_vocals', False)
        confidence_after_sep = vocals_after_sep_result.get('confidence', 0.0)
        voice_segments_after_sep = vocals_after_sep_result.get('voice_segments', [])
        
        # Lọc các segments có ý nghĩa (≥1.5s) và tính tổng thời lượng vocal
        meaningful_segments_after_sep = [seg for seg in voice_segments_after_sep if (seg['end'] - seg['start']) >= 1.5]
        total_vocal_duration = sum([seg['end'] - seg['start'] for seg in meaningful_segments_after_sep])
        
        logger.info(f"   Confidence sau khi tách: {confidence_after_sep:.2%}")
        logger.info(f"   Có vocal: {has_vocals_after_sep}")
        logger.info(f"   Tổng số segments: {len(voice_segments_after_sep)}")
        logger.info(f"   Số segments có ý nghĩa (≥1.5s): {len(meaningful_segments_after_sep)}")
        logger.info(f"   Tổng thời lượng vocal: {total_vocal_duration:.2f}s")
        
        # Điều kiện cân bằng: phải có vocal và một trong các điều kiện sau:
        # 1. Confidence ≥ 70% → OK (vocal bị tách không tốt nhưng vẫn có) HOẶC
        # 2. Confidence ≥ 50% + có ít nhất 1 segment + tổng thời lượng ≥ 2s HOẶC  
        # 3. Tổng thời lượng ≥ 6s (có nhiều vocal mặc dù segments ngắn)
        # Nếu KHÔNG thỏa bất kỳ điều kiện nào → coi như không có vocal (file chỉ có beat)
        
        min_confidence_very_high = 0.7  # Confidence rất cao (≥70%) = chắc chắn có vocal (có thể tách không tốt)
        min_confidence_high = 0.5  # Confidence cao (≥50%) cần kiểm tra thêm
        min_meaningful_segments = 1  # Ít nhất 1 segment có ý nghĩa
        min_total_duration_short = 2.0  # Tổng thời lượng tối thiểu: 2s (giảm từ 3s)
        min_total_duration_long = 6.0  # Nếu có nhiều vocal tổng cộng ≥6s thì OK (giảm từ 8s)
        
        # Kiểm tra điều kiện 1: Confidence rất cao (≥70%) → chắc chắn có vocal
        condition1 = confidence_after_sep >= min_confidence_very_high
        
        # Kiểm tra điều kiện 2: Confidence cao (≥50%) + có segment có ý nghĩa + tổng duration ≥ 2s
        condition2 = (confidence_after_sep >= min_confidence_high and 
                     len(meaningful_segments_after_sep) >= min_meaningful_segments and
                     total_vocal_duration >= min_total_duration_short)
        
        # Kiểm tra điều kiện 3: Tổng duration dài (≥6s) - có thể là vocal liên tục
        condition3 = total_vocal_duration >= min_total_duration_long
        
        # Nếu không thỏa BẤT KỲ điều kiện nào → không có vocal
        if (not has_vocals_after_sep or 
            (not condition1 and not condition2 and not condition3)):
            logger.error("❌ File vocal đã tách KHÔNG CHỨA GIỌNG HÁT!")
            logger.error(f"   Confidence: {confidence_after_sep:.2%} (điều kiện 1: ≥{min_confidence_very_high:.0%})")
            logger.error(f"   Segments có ý nghĩa: {len(meaningful_segments_after_sep)} (điều kiện 2: ≥{min_meaningful_segments})")
            logger.error(f"   Tổng thời lượng vocal: {total_vocal_duration:.2f}s (điều kiện 2: ≥{min_total_duration_short:.0f}s hoặc điều kiện 3: ≥{min_total_duration_long:.0f}s)")
            logger.error(f"   Điều kiện 1 (confidence ≥{min_confidence_very_high:.0%}): {condition1}")
            logger.error(f"   Điều kiện 2 (confidence ≥{min_confidence_high:.0%} + segments + duration ≥{min_total_duration_short:.0f}s): {condition2}")
            logger.error(f"   Điều kiện 3 (duration ≥{min_total_duration_long:.0f}s): {condition3}")
            logger.error("   File gốc có thể chỉ có beat/instrumental, không có giọng hát thực sự")
            logger.error("="*70 + "\n")
            
            # Kiểm tra thêm RMS energy - nếu quá thấp thì chắc chắn không có vocal
            vocals_check = check_audio_content(vocals_export)
            if not vocals_check.get('has_audio') or vocals_check.get('mean_rms', 0) < 1e-5:
                logger.error("❌ File vocal đã tách gần như im lặng (RMS quá thấp)")
            
            elapsed = time.time() - start_time
            end_datetime = datetime.now()
            logger.info("\n" + "="*70)
            logger.info("⏹️ KẾT THÚC PHÂN TÍCH (LỖI)")
            logger.info("="*70)
            logger.info(f"   ⏰ Thời gian kết thúc: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"   ⏱️ Thời gian phân tích: {format_duration(elapsed)} ({elapsed:.2f}s)")
            logger.info("="*70 + "\n")
            return {
                "success": False,
                "error": "File vocal đã tách không chứa giọng hát",
                "score": 0.0,
                "key_compare": {"match": False, "similarity": 0.0, "score": 0.0},
                "message": f"File gốc chỉ có beat/instrumental, không có giọng hát. Sau khi tách: confidence={confidence_after_sep:.1%}, segments={len(meaningful_segments_after_sep)}, duration={total_vocal_duration:.1f}s - không đạt bất kỳ điều kiện nào.",
                "vocals_check_after_separation": {
                    "has_vocals": has_vocals_after_sep,
                    "confidence": confidence_after_sep,
                    "segments_count": len(voice_segments_after_sep),
                    "meaningful_segments_count": len(meaningful_segments_after_sep),
                    "total_duration": total_vocal_duration,
                    "condition1_met": condition1,
                    "condition2_met": condition2,
                    "condition3_met": condition3,
                    "duration_seconds": elapsed,
                    "start_time": start_datetime.isoformat(),
                    "end_time": end_datetime.isoformat()
                }
            }
        
        logger.info("✅ File vocal đã tách chứa giọng hát (validation passed)")
        logger.info("="*70 + "\n")
        
        # Detect vocals key sau khi separation hoàn thành
        vocals_key = detect_vocals_key(vocals_export)
    
    logger.info("🎉 Hoàn thành tất cả key detection!")
    if not (vocals_key and 'key' in vocals_key and beat_key and 'key' in beat_key):
        elapsed = time.time() - start_time
        end_datetime = datetime.now()
        logger.info("\n" + "="*70)
        logger.info("⏹️ KẾT THÚC PHÂN TÍCH (LỖI)")
        logger.info("="*70)
        logger.info(f"   ⏰ Thời gian kết thúc: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   ⏱️ Thời gian phân tích: {format_duration(elapsed)} ({elapsed:.2f}s)")
        logger.info("="*70 + "\n")
        return {
            "success": False,
            "error": "Phát hiện key thất bại",
            "duration_seconds": elapsed,
            "start_time": start_datetime.isoformat(),
            "end_time": end_datetime.isoformat()
        }

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

    # Tính thời gian phân tích
    elapsed = time.time() - start_time
    end_datetime = datetime.now()
    
    logger.info("\n" + "="*70)
    logger.info("✅ HOÀN THÀNH PHÂN TÍCH")
    logger.info("="*70)
    logger.info(f"   ⏰ Thời gian bắt đầu: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   ⏰ Thời gian kết thúc: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   ⏱️ Tổng thời gian phân tích: {format_duration(elapsed)} ({elapsed:.2f}s)")
    logger.info("="*70 + "\n")

    return {
        "success": True,
        "inputs": {"karaoke_file": karaoke_file, "beat_file": beat_file},
        "sliced_karaoke": sliced_path,
        "vocals_src": vocals_path,
        "vocals_export": vocals_export,
        "vocals_key": vocals_key,
        "beat_key": beat_key,
        "key_compare": {"match": match, "similarity": similarity, "score": score},
        "duration_seconds": elapsed,
        "start_time": start_datetime.isoformat(),
        "end_time": end_datetime.isoformat()
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


