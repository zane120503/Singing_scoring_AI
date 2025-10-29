#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Audio Processor - Workflow tối ưu hóa để giảm thời gian xử lý
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ai.voice_activity_detector import VoiceActivityDetector
from ai.advanced_voice_detector import AdvancedVoiceDetector
from ai.smart_voice_detector import SmartVoiceDetector
from ai.improved_smart_voice_detector import ImprovedSmartVoiceDetector
from ai.final_voice_detector import FinalVoiceDetector
from ai.correct_voice_detector import CorrectVoiceDetector
from ai.advanced_audio_processor import AdvancedAudioProcessor
from ai.advanced_key_detector import AdvancedKeyDetector
from core.scoring_system import KaraokeScoringSystem

logger = logging.getLogger(__name__)

class OptimizedAudioProcessor:
    """Optimized Audio Processor với workflow tối ưu hóa"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        
        # Khởi tạo các components
        self.vad = VoiceActivityDetector(sr)
        self.advanced_vad = AdvancedVoiceDetector(sr)  # Advanced VAD
        self.smart_vad = SmartVoiceDetector(sr)  # Smart VAD
        self.improved_smart_vad = ImprovedSmartVoiceDetector(sr)  # Improved Smart VAD
        self.final_vad = FinalVoiceDetector(sr)  # Final VAD
        self.correct_vad = CorrectVoiceDetector(sr)  # Correct VAD
        self.audio_processor = AdvancedAudioProcessor(fast_mode=False)
        self.key_detector = AdvancedKeyDetector()
        self.scoring_system = KaraokeScoringSystem()
        
        # Cấu hình
        self.min_voice_duration = 1.0  # Tối thiểu 1 giây có giọng
        
        logger.info("✅ Optimized Audio Processor initialized")
    
    def process_karaoke_optimized(self, 
                                karaoke_file: str, 
                                beat_file: str,
                                output_dir: str = None) -> Dict:
        """
        Xử lý karaoke với workflow tối ưu hóa
        
        Args:
            karaoke_file: Đường dẫn file karaoke
            beat_file: Đường dẫn file beat nhạc
            output_dir: Thư mục output (tùy chọn)
            
        Returns:
            Dict: Kết quả xử lý hoàn chỉnh
        """
        try:
            logger.info("🎤 Bắt đầu xử lý karaoke với workflow tối ưu hóa...")
            
            # Tạo output directory nếu chưa có (ưu tiên clean_song_output)
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Audio_separator_ui', 'clean_song_output')
            os.makedirs(output_dir, exist_ok=True)
            
            # Bước 1: Voice Activity Detection với VocalsPresenceChecker
            logger.info("\n" + "="*70)
            logger.info("🔍 BƯỚC 1: PHÁT HIỆN VOCAL TRONG FILE KARAOKE")
            logger.info("="*70)
            
            # Sử dụng VocalsPresenceChecker để phát hiện vocal và log chi tiết
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
                from vocals_presence_checker import VocalsPresenceChecker
                vocals_checker = VocalsPresenceChecker()
                vocals_result = vocals_checker.check_vocals_presence(karaoke_file)
                
                if not vocals_result.get('has_vocals'):
                    logger.error("❌ Không phát hiện được giọng hát trong file karaoke")
                    return {
                        "success": False,
                        "error": "Không phát hiện được giọng hát trong file karaoke",
                        "step": "voice_detection"
                    }
                
                # Lấy voice segments từ kết quả
                voice_segments = vocals_result.get('voice_segments', [])
                
                if not voice_segments:
                    logger.warning("⚠️ Phát hiện vocal nhưng không có voice segments, thử fallback VAD...")
                    voice_segments = self.correct_vad.detect_voice_activity(karaoke_file)
                
                # Log thời gian vocal
                if voice_segments:
                    first_seg = voice_segments[0]
                    last_seg = voice_segments[-1]
                    logger.info(f"\n📊 THÔNG TIN VOCAL:")
                    logger.info(f"   ⏰ VOCAL XUẤT HIỆN: {first_seg['start']:.2f}s ({int(first_seg['start']//60):02d}:{int(first_seg['start']%60):02d})")
                    logger.info(f"   ⏰ VOCAL KẾT THÚC: {last_seg['end']:.2f}s ({int(last_seg['end']//60):02d}:{int(last_seg['end']%60):02d})")
                    logger.info(f"   📏 Tổng thời gian vocal: {last_seg['end'] - first_seg['start']:.2f}s")
                    logger.info(f"   📊 Số đoạn vocal: {len(voice_segments)}")
                    logger.info("="*70 + "\n")
                
                if not voice_segments:
                    return {
                        "success": False,
                        "error": "Không phát hiện được giọng hát trong file karaoke",
                        "step": "voice_detection"
                    }
                
            except Exception as e:
                logger.warning(f"⚠️ VocalsPresenceChecker failed: {e}, using fallback VAD...")
                voice_segments = self.correct_vad.detect_voice_activity(karaoke_file)
                
                if not voice_segments:
                    return {
                        "success": False,
                        "error": "Không phát hiện được giọng hát trong file karaoke",
                        "step": "voice_detection"
                    }
                
                # Log với fallback
                first_voice = voice_segments[0] if voice_segments else None
                if first_voice:
                    logger.info(f"🎯 Tìm thấy đoạn voice (fallback): {first_voice['start']:.2f}s - {first_voice['end']:.2f}s")
            
            # Tìm đoạn voice đầu tiên phù hợp
            first_voice = self._find_optimal_voice_segment(voice_segments)
            if not first_voice:
                return {
                    "success": False,
                    "error": "Không tìm thấy đoạn voice phù hợp",
                    "step": "voice_selection"
                }

            # Bước 2: Cắt file từ lúc vocal bắt đầu đến khi vocal kết thúc
            logger.info("\n" + "="*70)
            logger.info("✂️ BƯỚC 2: CẮT FILE KARAOKE THEO VỊ TRÍ VOCAL")
            logger.info("="*70)
            
            import librosa, soundfile as sf
            from smart_slicing_strategy import SmartSlicingStrategy
            
            base_stem = os.path.splitext(os.path.basename(karaoke_file))[0]
            
            # Lấy thời gian vocal đầu tiên và cuối cùng
            first_seg = voice_segments[0]
            last_seg = voice_segments[-1]
            
            try:
                # Cắt từ lúc vocal xuất hiện đến khi vocal kết thúc
                slice_start = first_seg['start']
                slice_end = last_seg['end']
                
                logger.info(f"📌 Cắt file karaoke:")
                logger.info(f"   Từ: {slice_start:.2f}s ({int(slice_start//60):02d}:{int(slice_start%60):02d})")
                logger.info(f"   Đến: {slice_end:.2f}s ({int(slice_end//60):02d}:{int(slice_end%60):02d})")
                logger.info(f"   Thời lượng: {slice_end - slice_start:.2f}s")
                
                # Load và cắt audio
                audio, sr = librosa.load(karaoke_file, sr=None, mono=True)
                start_sample = int(slice_start * sr)
                end_sample = int(slice_end * sr)
                end_sample = min(end_sample, len(audio))  # Đảm bảo không vượt quá
                
                if start_sample >= len(audio):
                    return {
                        "success": False,
                        "error": f"Karaoke ngắn hơn {slice_start:.2f}s",
                        "step": "audio_slicing"
                    }
                
                slice_audio = audio[start_sample:end_sample]
                sliced_path = os.path.join(output_dir, f"{base_stem}_slice_{int(slice_start)}s_{int(slice_end)}s.wav")
                sf.write(sliced_path, slice_audio, sr)
                
                logger.info(f"✅ Đã cắt file thành công!")
                logger.info(f"   📁 File output: {sliced_path}")
                logger.info("="*70 + "\n")
                
            except Exception as e:
                logger.error(f"❌ Lỗi cắt file: {e}")
                # Fallback: thử smart slicing
                try:
                    logger.warning("⚠️ Thử smart slicing strategy...")
                    slicer = SmartSlicingStrategy()
                    slice_result = slicer.slice_with_voice_guidance(karaoke_file, voice_segments, 
                                                                     os.path.join(output_dir, f"{base_stem}_smart_slice.wav"))
                    sliced_path = slice_result['output_path']
                    slice_start = slice_result['start']
                    slice_end = slice_result['end']
                    logger.info(f"✅ Smart slice created: {slice_start:.2f}s - {slice_end:.2f}s")
                except Exception as e2:
                    logger.error(f"❌ Smart slicing cũng thất bại: {e2}")
                    return {
                        "success": False,
                        "error": f"Failed to slice audio: {str(e)}",
                        "step": "audio_slicing"
                    }

            # Bước 3: Cắt beat (cùng khoảng với karaoke) để đảm bảo key chính xác
            logger.info(f"✂️ Bước 3: Cắt beat cùng khoảng ({slice_start:.1f}s–{slice_end:.1f}s)...")
            beat_audio, beat_sr = librosa.load(beat_file, sr=None, mono=True)
            beat_start_t = slice_start  # Cùng thời điểm với karaoke slice
            beat_end_t = slice_end       # Cùng thời điểm với karaoke slice
            beat_start_sample = int(beat_start_t * beat_sr)
            beat_end_sample = int(beat_end_t * beat_sr)
            if beat_start_sample >= len(beat_audio):
                return {
                    "success": False,
                    "error": "Beat ngắn hơn 15s",
                    "step": "beat_slicing"
                }
            beat_slice = beat_audio[beat_start_sample:min(beat_end_sample, len(beat_audio))]
            beat_sliced_path = os.path.join(output_dir, f"{base_stem}_beat_slice_{int(beat_start_t)}s_{int(beat_end_t)}s.wav")
            sf.write(beat_sliced_path, beat_slice, beat_sr)

            # Bước 4: AI Audio Separator - Tách giọng từ file đã cắt 30s
            logger.info("🎤 Bước 4: Tách giọng hát từ đoạn 30s đã cắt...")
            vocals_file = self.audio_processor.separate_vocals(sliced_path)
            
            if not vocals_file or not os.path.exists(vocals_file):
                return {
                    "success": False,
                    "error": "Lỗi tách giọng hát",
                    "step": "vocal_separation"
                }
            
            # Copy/export vocals 30s về output_dir với tên dễ nhận biết
            vocals_ext = os.path.splitext(vocals_file)[1]
            vocals_export = os.path.join(output_dir, f"{base_stem}_slice_vocals{vocals_ext}")
            try:
                import shutil
                if vocals_file != vocals_export:
                    shutil.copy2(vocals_file, vocals_export)
            except Exception:
                vocals_export = vocals_file

            logger.info(f"✅ Đã tách giọng hát (20s): {vocals_export}")
            
            # Bước 3: Key Detection - Detect key từ file beat gốc (không cắt)
            logger.info("🎹 Bước 3: Phát hiện phím âm nhạc...")
            
            # Key detection cho vocals (file 20s đã tách)
            vocals_key = self.key_detector.detect_key(vocals_export, "vocals")
            
            # Thử nhiều phương pháp detect key cho beat (file gốc)
            beat_key = None
            beat_methods = ['beat', 'instrumental', 'vocals']  # Thử các audio_type khác nhau
            
            for method in beat_methods:
                try:
                    temp_beat_key = self.key_detector.detect_key(beat_file, method)
                    if temp_beat_key and 'key' in temp_beat_key:
                        beat_key = temp_beat_key
                        logger.info(f"✅ Beat key detected với method '{method}': {beat_key['key']}")
                        break
                except Exception as e:
                    logger.warning(f"Method '{method}' failed: {e}")
                    continue
            
            logger.info(f"🎵 Beat key: {beat_key['key']} {beat_key['scale']} (confidence: {beat_key['confidence']:.3f})")
            logger.info(f"🎤 Vocals key: {vocals_key['key']} {vocals_key['scale']} (confidence: {vocals_key['confidence']:.3f})")
            
            # Bước 4: Key Comparison - So sánh key
            logger.info("🔍 Bước 4: So sánh phím âm nhạc...")
            key_comparison = self.key_detector.compare_keys(beat_key, vocals_key)
            
            logger.info(f"📊 Key similarity score: {key_comparison['score']}/100")
            
            # Bước 5: Scoring - Tính điểm
            logger.info("📊 Bước 5: Tính điểm tổng thể...")
            scoring_result = self.scoring_system.calculate_overall_score(
                karaoke_file, beat_file, vocals_export
            )
            
            logger.info(f"🏆 Overall score: {scoring_result['overall_score']}/100")
            
            # Tạo kết quả hoàn chỉnh
            result = {
                "success": True,
                "input_files": {
                    "karaoke_file": karaoke_file,
                    "beat_file": beat_file
                },
                "processed_files": {
                    "karaoke_file": karaoke_file,
                    "sliced_karaoke": sliced_path,
                    "vocals_file": vocals_export
                },
                "voice_detection": {
                    "voice_segments": voice_segments,
                    "selected_voice": first_voice,
                    "slice_start_time": start_t
                },
                "key_detection": {
                    "beat_key": beat_key,
                    "vocals_key": vocals_key,
                    "key_comparison": key_comparison
                },
                "scoring": scoring_result,
                "processing_time": {
                    "voice_detection_time": "N/A",  # Có thể thêm timing
                    "vocal_separation_time": "N/A",
                    "key_detection_time": "N/A",
                    "total_time": "N/A"
                }
            }
            
            logger.info("🎉 Hoàn thành xử lý karaoke với workflow tối ưu hóa!")
            return result
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong quá trình xử lý: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "step": "unknown"
            }
    
    def _find_optimal_voice_segment(self, voice_segments: list) -> Optional[Dict]:
        """Tìm đoạn voice tối ưu để cắt"""
        try:
            for segment in voice_segments:
                duration = segment["end"] - segment["start"]
                
                # Kiểm tra thời lượng tối thiểu
                if duration >= self.min_voice_duration:
                    return {
                        "start": segment["start"],
                        "end": segment["end"],
                        "confidence": segment["confidence"]
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Lỗi tìm optimal voice segment: {e}")
            return None
    
    # Cắt audio không còn được sử dụng trong workflow full audio
    
    def get_processing_summary(self, result: Dict) -> str:
        """Tạo tóm tắt kết quả xử lý"""
        if not result["success"]:
            return f"❌ Xử lý thất bại: {result['error']} (Bước: {result['step']})"
        
        summary = f"""
🎉 XỬ LÝ THÀNH CÔNG!

📁 Files đã xử lý:
   • Karaoke slice: {os.path.basename(result['processed_files']['sliced_karaoke'])}
   • Vocals file: {os.path.basename(result['processed_files']['vocals_file'])}

🎤 Voice Detection:
   • Tìm thấy {len(result['voice_detection']['voice_segments'])} đoạn voice
   • Chọn đoạn: {result['voice_detection']['selected_voice']['start']:.2f}s - {result['voice_detection']['selected_voice']['end']:.2f}s
   • Cắt từ: {result['voice_detection']['slice_start_time']:.2f}s (20s)

🎹 Key Detection:
   • Beat key: {result['key_detection']['beat_key']['key']} {result['key_detection']['beat_key']['scale']} (confidence: {result['key_detection']['beat_key']['confidence']:.3f})
   • Vocals key: {result['key_detection']['vocals_key']['key']} {result['key_detection']['vocals_key']['scale']} (confidence: {result['key_detection']['vocals_key']['confidence']:.3f})
   • Key similarity: {result['key_detection']['key_comparison']['score']}/100

📊 Scoring:
   • Overall score: {result['scoring']['overall_score']}/100
   • Key accuracy: {result['scoring']['detailed_scores']['key_accuracy']:.1f}/100
   • Pitch accuracy: {result['scoring']['detailed_scores']['pitch_accuracy']:.1f}/100
   • Rhythm accuracy: {result['scoring']['detailed_scores']['rhythm_accuracy']:.1f}/100
        """
        
        return summary.strip()
    
    def batch_process_optimized(self, 
                               file_pairs: list, 
                               output_dir: str = None) -> list:
        """
        Xử lý batch nhiều cặp file với workflow tối ưu hóa
        
        Args:
            file_pairs: List of tuples [(karaoke_file, beat_file), ...]
            output_dir: Thư mục output
            
        Returns:
            list: Danh sách kết quả xử lý
        """
        results = []
        
        logger.info(f"🎤 Bắt đầu batch processing {len(file_pairs)} file pairs...")
        
        for i, (karaoke_file, beat_file) in enumerate(file_pairs, 1):
            logger.info(f"📁 Processing pair {i}/{len(file_pairs)}: {Path(karaoke_file).name}")
            
            result = self.process_karaoke_optimized(karaoke_file, beat_file, output_dir)
            results.append(result)
            
            if result["success"]:
                logger.info(f"✅ Pair {i} processed successfully")
            else:
                logger.warning(f"⚠️ Pair {i} failed: {result.get('error', 'Unknown error')}")
        
        # Thống kê
        successful = sum(1 for r in results if r["success"])
        logger.info(f"📊 Batch processing completed: {successful}/{len(file_pairs)} pairs successful")
        
        return results
