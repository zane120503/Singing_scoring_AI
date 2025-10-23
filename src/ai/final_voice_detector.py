#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Voice Detector - Phát hiện giọng hát cuối cùng với logic đơn giản
"""

import os
import sys
import numpy as np
import librosa
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

logger = logging.getLogger(__name__)

class FinalVoiceDetector:
    """Final Voice Detector - Phát hiện giọng hát cuối cùng với logic đơn giản"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
    
    def detect_voice_activity(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity với logic cuối cùng"""
        try:
            logger.info("🎯 Final Voice Detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Phân tích đơn giản để tìm vị trí giọng hát
            voice_start = self._find_final_voice_start(audio, sr)
            
            if voice_start is None:
                logger.warning("⚠️ Không tìm thấy vị trí giọng hát")
                return []
            
            # Tạo segment từ vị trí tìm được
            segments = [{
                'start': voice_start,
                'end': len(audio) / sr,  # Đến cuối file
                'confidence': 1.0,
                'method': 'final_detection'
            }]
            
            logger.info(f"✅ Final voice detection: {len(segments)} segments")
            logger.info(f"   Voice starts at: {voice_start:.2f}s")
            
            return segments
            
        except Exception as e:
            logger.error(f"❌ Final voice detection failed: {e}")
            return []
    
    def _find_final_voice_start(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Tìm vị trí bắt đầu giọng hát bằng logic cuối cùng"""
        try:
            # Phân tích từng giây
            hop_length = self.hop_length
            
            # Tính RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=hop_length)[0]
            
            # Chuyển frames thành thời gian
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            logger.info(f"🔍 Phân tích {len(rms)} frames để tìm giọng hát...")
            
            # Tính baseline từ 2 giây đầu
            baseline_length = min(2, len(rms))
            baseline_rms = np.mean(rms[:baseline_length])
            
            logger.info(f"📊 Baseline analysis:")
            logger.info(f"   RMS mean: {baseline_rms:.4f}")
            
            # Tìm vị trí giọng hát với logic đơn giản - threshold thấp
            voice_start_candidates = []
            
            # Phân tích 15 giây đầu
            for i in range(min(15, len(rms))):
                start_time = i * hop_length / sr
                segment_rms = rms[i]
                
                # Điều kiện đơn giản: RMS cao hơn baseline + 10%
                threshold = baseline_rms * 1.1  # Chỉ cần cao hơn 10%
                has_voice = segment_rms > threshold
                
                if has_voice:
                    voice_start_candidates.append({
                        'time': start_time,
                        'rms': segment_rms,
                        'confidence': segment_rms / baseline_rms
                    })
                
                # Log chi tiết cho 10 giây đầu
                if start_time < 10:
                    logger.info(f"   {start_time:.1f}s: RMS={segment_rms:.4f}, Threshold={threshold:.4f}, Voice={'YES' if has_voice else 'NO'}")
            
            # Tìm vị trí giọng hát bắt đầu thực sự
            if voice_start_candidates:
                # Lấy vị trí đầu tiên có voice
                first_voice = voice_start_candidates[0]
                
                logger.info(f"\n🎯 Vị trí giọng hát cuối cùng:")
                logger.info(f"   Voice starts at: {first_voice['time']:.2f}s")
                logger.info(f"   RMS: {first_voice['rms']:.4f}")
                logger.info(f"   Confidence: {first_voice['confidence']:.3f}")
                
                return first_voice['time']
            else:
                logger.warning("⚠️ Không tìm thấy vị trí giọng hát!")
                return None
        
        except Exception as e:
            logger.error(f"❌ Lỗi tìm vị trí giọng hát: {e}")
            return None
    
    def get_first_suitable_voice_segment(self, audio_path: str, min_duration: float = 1.0) -> Dict:
        """Returns the first voice segment that meets a minimum duration."""
        segments = self.detect_voice_activity(audio_path)
        for segment in segments:
            if (segment['end'] - segment['start']) >= min_duration:
                logger.info(f"✅ Found first suitable final voice segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
                return segment
        logger.warning("⚠️ No suitable final voice segment found.")
        return None
