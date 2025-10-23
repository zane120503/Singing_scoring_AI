#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Audio Processor - Kết hợp VAD và Audio Slicer để tối ưu hóa tách giọng
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ai.voice_activity_detector import VoiceActivityDetector
from ai.audio_slicer import AudioSlicer

logger = logging.getLogger(__name__)

class SmartAudioProcessor:
    """Smart Audio Processor để tối ưu hóa quá trình tách giọng"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.vad = VoiceActivityDetector(sr)
        self.slicer = AudioSlicer(sr)
        
        # Cấu hình mặc định
        self.default_slice_duration = 20.0  # 20 giây
        self.min_voice_duration = 1.0       # Tối thiểu 1 giây có giọng
        self.voice_buffer = 2.0             # Buffer 2 giây trước khi bắt đầu giọng
    
    def process_karaoke_file(self, 
                           input_path: str, 
                           output_dir: str,
                           slice_duration: float = None) -> Dict:
        """
        Xử lý file karaoke: phát hiện giọng hát và cắt đoạn tối ưu
        
        Args:
            input_path: Đường dẫn file karaoke
            output_dir: Thư mục output
            slice_duration: Thời lượng cắt (mặc định 20s)
            
        Returns:
            Dict: Thông tin kết quả xử lý
        """
        try:
            logger.info(f"🎤 Xử lý file karaoke: {input_path}")
            
            # Sử dụng slice_duration mặc định nếu không được cung cấp
            if slice_duration is None:
                slice_duration = self.default_slice_duration
            
            # Bước 1: Phát hiện voice activity
            logger.info("🔍 Bước 1: Phát hiện voice activity...")
            voice_segments = self.vad.detect_voice_activity(input_path, method="combined")
            
            if not voice_segments:
                logger.warning("⚠️ Không phát hiện được giọng hát trong file")
                return {
                    "success": False,
                    "error": "No voice detected",
                    "voice_segments": [],
                    "output_files": []
                }
            
            # Bước 2: Tìm đoạn voice đầu tiên phù hợp
            logger.info("🎯 Bước 2: Tìm đoạn voice đầu tiên...")
            first_voice = self._find_optimal_voice_segment(voice_segments)
            
            if not first_voice:
                logger.warning("⚠️ Không tìm thấy đoạn voice phù hợp")
                return {
                    "success": False,
                    "error": "No suitable voice segment found",
                    "voice_segments": voice_segments,
                    "output_files": []
                }
            
            # Bước 3: Cắt audio từ đoạn voice
            logger.info("✂️ Bước 3: Cắt audio từ đoạn voice...")
            output_file = self._create_voice_sample(input_path, output_dir, first_voice, slice_duration)
            
            if not output_file:
                logger.error("❌ Lỗi tạo voice sample")
                return {
                    "success": False,
                    "error": "Failed to create voice sample",
                    "voice_segments": voice_segments,
                    "selected_voice": first_voice,
                    "output_files": []
                }
            
            # Bước 4: Preview và kiểm tra
            logger.info("🔍 Bước 4: Kiểm tra kết quả...")
            preview_success = self.slicer.preview_voice_segment(output_file, 0, min(5.0, slice_duration))
            
            result = {
                "success": True,
                "input_file": input_path,
                "output_file": output_file,
                "voice_segments": voice_segments,
                "selected_voice": first_voice,
                "slice_duration": slice_duration,
                "preview_success": preview_success,
                "output_files": [output_file]
            }
            
            logger.info(f"✅ Hoàn thành xử lý file karaoke!")
            logger.info(f"   Input: {input_path}")
            logger.info(f"   Output: {output_file}")
            logger.info(f"   Voice start: {first_voice['start']:.2f}s")
            logger.info(f"   Slice duration: {slice_duration}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Lỗi xử lý file karaoke: {e}")
            return {
                "success": False,
                "error": str(e),
                "voice_segments": [],
                "output_files": []
            }
    
    def _find_optimal_voice_segment(self, voice_segments: list) -> Optional[Dict]:
        """Tìm đoạn voice tối ưu để cắt"""
        try:
            for segment in voice_segments:
                duration = segment["end"] - segment["start"]
                
                # Kiểm tra thời lượng tối thiểu
                if duration >= self.min_voice_duration:
                    # Thêm buffer trước khi bắt đầu giọng
                    adjusted_start = max(0, segment["start"] - self.voice_buffer)
                    
                    return {
                        "start": adjusted_start,
                        "end": segment["end"],
                        "confidence": segment["confidence"],
                        "original_start": segment["start"]
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Lỗi tìm optimal voice segment: {e}")
            return None
    
    def _create_voice_sample(self, 
                           input_path: str, 
                           output_dir: str, 
                           voice_segment: Dict, 
                           slice_duration: float) -> Optional[str]:
        """Tạo voice sample từ đoạn voice được chọn"""
        try:
            # Tạo tên file output
            input_filename = Path(input_path).stem
            voice_start = voice_segment["start"]
            output_filename = f"{input_filename}_voice_{voice_start:.1f}s_{slice_duration}s.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Cắt audio
            success = self.slicer.slice_audio(
                input_path, 
                output_path, 
                voice_start, 
                slice_duration
            )
            
            if success:
                return output_path
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Lỗi tạo voice sample: {e}")
            return None
    
    def batch_process_karaoke_files(self, 
                                  input_files: list, 
                                  output_dir: str,
                                  slice_duration: float = None) -> list:
        """
        Xử lý batch nhiều file karaoke
        
        Args:
            input_files: Danh sách file input
            output_dir: Thư mục output
            slice_duration: Thời lượng cắt
            
        Returns:
            list: Danh sách kết quả xử lý
        """
        results = []
        
        logger.info(f"🎤 Bắt đầu batch processing {len(input_files)} files...")
        
        for i, input_file in enumerate(input_files, 1):
            logger.info(f"📁 Processing file {i}/{len(input_files)}: {Path(input_file).name}")
            
            result = self.process_karaoke_file(input_file, output_dir, slice_duration)
            results.append(result)
            
            if result["success"]:
                logger.info(f"✅ File {i} processed successfully")
            else:
                logger.warning(f"⚠️ File {i} failed: {result.get('error', 'Unknown error')}")
        
        # Thống kê
        successful = sum(1 for r in results if r["success"])
        logger.info(f"📊 Batch processing completed: {successful}/{len(input_files)} files successful")
        
        return results
    
    def get_voice_analysis(self, input_path: str) -> Dict:
        """
        Phân tích voice activity của file (không cắt)
        
        Args:
            input_path: Đường dẫn file input
            
        Returns:
            Dict: Thông tin phân tích voice
        """
        try:
            logger.info(f"🔍 Phân tích voice activity: {input_path}")
            
            # Lấy thông tin audio
            audio_info = self.slicer.get_audio_info(input_path)
            
            # Phát hiện voice segments
            voice_segments = self.vad.detect_voice_activity(input_path, method="combined")
            
            # Tính toán thống kê
            total_duration = audio_info.get("duration", 0)
            voice_duration = sum(seg["end"] - seg["start"] for seg in voice_segments)
            voice_ratio = voice_duration / total_duration if total_duration > 0 else 0
            
            analysis = {
                "input_file": input_path,
                "audio_info": audio_info,
                "voice_segments": voice_segments,
                "voice_duration": voice_duration,
                "total_duration": total_duration,
                "voice_ratio": voice_ratio,
                "voice_count": len(voice_segments)
            }
            
            logger.info(f"📊 Voice analysis completed:")
            logger.info(f"   Total duration: {total_duration:.2f}s")
            logger.info(f"   Voice duration: {voice_duration:.2f}s")
            logger.info(f"   Voice ratio: {voice_ratio:.1%}")
            logger.info(f"   Voice segments: {len(voice_segments)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Lỗi phân tích voice: {e}")
            return {"error": str(e)}
