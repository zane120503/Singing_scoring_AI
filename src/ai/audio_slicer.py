#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Slicer - Cắt file audio theo thời gian
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class AudioSlicer:
    """Audio Slicer để cắt file audio theo thời gian"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def slice_audio(self, 
                   input_path: str, 
                   output_path: str, 
                   start_time: float, 
                   duration: float = 20.0) -> bool:
        """
        Cắt file audio theo thời gian
        
        Args:
            input_path: Đường dẫn file input
            output_path: Đường dẫn file output
            start_time: Thời điểm bắt đầu (giây)
            duration: Thời lượng cắt (giây)
            
        Returns:
            bool: True nếu thành công
        """
        try:
            logger.info(f"✂️ Cắt audio: {start_time:.2f}s - {start_time + duration:.2f}s")
            
            # Load audio
            audio, sr = librosa.load(input_path, sr=self.sr, offset=start_time, duration=duration)
            
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save sliced audio
            sf.write(output_path, audio, sr)
            
            logger.info(f"✅ Đã cắt và lưu audio: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi cắt audio: {e}")
            return False
    
    def slice_from_voice_start(self, 
                              input_path: str, 
                              output_path: str, 
                              voice_start_time: float, 
                              duration: float = 20.0) -> bool:
        """
        Cắt audio từ thời điểm bắt đầu có giọng hát
        
        Args:
            input_path: Đường dẫn file input
            output_path: Đường dẫn file output
            voice_start_time: Thời điểm bắt đầu có giọng hát
            duration: Thời lượng cắt (mặc định 20s)
            
        Returns:
            bool: True nếu thành công
        """
        return self.slice_audio(input_path, output_path, voice_start_time, duration)
    
    def create_voice_sample(self, 
                           input_path: str, 
                           output_dir: str, 
                           voice_start_time: float, 
                           duration: float = 20.0) -> Optional[str]:
        """
        Tạo file sample từ đoạn có giọng hát
        
        Args:
            input_path: Đường dẫn file input
            output_dir: Thư mục output
            voice_start_time: Thời điểm bắt đầu có giọng hát
            duration: Thời lượng cắt
            
        Returns:
            str: Đường dẫn file output hoặc None nếu thất bại
        """
        try:
            # Create output filename
            input_filename = Path(input_path).stem
            output_filename = f"{input_filename}_voice_sample_{voice_start_time:.1f}s_{duration}s.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Slice audio
            if self.slice_audio(input_path, output_path, voice_start_time, duration):
                return output_path
            else:
                return None
                
        except Exception as e:
            logger.error(f"❌ Lỗi tạo voice sample: {e}")
            return None
    
    def get_audio_info(self, audio_path: str) -> dict:
        """
        Lấy thông tin file audio
        
        Args:
            audio_path: Đường dẫn file audio
            
        Returns:
            dict: Thông tin audio
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            duration = len(audio) / sr
            
            return {
                "duration": duration,
                "sample_rate": sr,
                "samples": len(audio),
                "channels": 1 if audio.ndim == 1 else audio.shape[0]
            }
            
        except Exception as e:
            logger.error(f"❌ Lỗi lấy thông tin audio: {e}")
            return {}
    
    def preview_voice_segment(self, audio_path: str, start_time: float, duration: float = 5.0) -> bool:
        """
        Preview đoạn voice (chỉ load và kiểm tra)
        
        Args:
            audio_path: Đường dẫn file audio
            start_time: Thời điểm bắt đầu
            duration: Thời lượng preview
            
        Returns:
            bool: True nếu thành công
        """
        try:
            # Load segment
            audio, sr = librosa.load(audio_path, sr=self.sr, offset=start_time, duration=duration)
            
            logger.info(f"🎵 Preview segment: {start_time:.2f}s - {start_time + duration:.2f}s")
            logger.info(f"   Duration: {duration:.2f}s")
            logger.info(f"   Samples: {len(audio)}")
            logger.info(f"   RMS Energy: {np.sqrt(np.mean(audio**2)):.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi preview segment: {e}")
            return False
