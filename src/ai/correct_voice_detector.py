#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct Voice Detector - Phát hiện chính xác vị trí giọng hát thực sự
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

class CorrectVoiceDetector:
    """Correct Voice Detector - Phát hiện chính xác vị trí giọng hát thực sự"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
    
    def detect_voice_activity(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity với độ chính xác cao"""
        try:
            logger.info("🎯 Correct Voice Detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Phân tích để tìm vị trí giọng hát thực sự
            voice_start = self._find_correct_voice_start(audio, sr)
            
            if voice_start is None:
                logger.warning("⚠️ Không tìm thấy vị trí giọng hát")
                return []
            
            # Tạo segment từ vị trí tìm được
            segments = [{
                'start': voice_start,
                'end': len(audio) / sr,  # Đến cuối file
                'confidence': 1.0,
                'method': 'correct_detection'
            }]
            
            logger.info(f"✅ Correct voice detection: {len(segments)} segments")
            logger.info(f"   Voice starts at: {voice_start:.2f}s")
            
            return segments
            
        except Exception as e:
            logger.error(f"❌ Correct voice detection failed: {e}")
            return []
    
    def _find_correct_voice_start(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Tìm vị trí bắt đầu giọng hát thực sự"""
        try:
            # Phân tích từng giây
            hop_length = self.hop_length
            
            # Tính RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=hop_length)[0]
            
            # Chuyển frames thành thời gian
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            logger.info(f"🔍 Phân tích {len(rms)} frames để tìm giọng hát thực sự...")
            
            # Tính baseline từ 5 giây đầu
            baseline_length = min(5, len(rms))
            baseline_rms = np.mean(rms[:baseline_length])
            
            logger.info(f"📊 Baseline RMS: {baseline_rms:.4f}")
            
            # Tìm energy spikes cao hơn 0.08
            energy_spikes = []
            
            for i in range(len(rms)):
                start_time = i * hop_length / sr
                segment_rms = rms[i]
                
                # Tìm các spike cao hơn 0.08
                if segment_rms > 0.08:
                    energy_spikes.append({
                        'time': start_time,
                        'rms': segment_rms
                    })
            
            logger.info(f"📈 Tìm thấy {len(energy_spikes)} energy spikes")
            
            # Tìm spike đầu tiên sau 5 giây (bỏ qua intro)
            for spike in energy_spikes:
                if spike['time'] > 5.0:
                    logger.info(f"\n🎯 Vị trí giọng hát thực sự:")
                    logger.info(f"   Voice starts at: {spike['time']:.2f}s")
                    logger.info(f"   RMS: {spike['rms']:.4f}")
                    return spike['time']
            
            # Fallback: nếu không tìm thấy sau 5s, tìm spike đầu tiên
            if energy_spikes:
                first_spike = energy_spikes[0]
                logger.info(f"\n🎯 Fallback - Vị trí giọng hát đầu tiên:")
                logger.info(f"   Voice starts at: {first_spike['time']:.2f}s")
                logger.info(f"   RMS: {first_spike['rms']:.4f}")
                return first_spike['time']
            
            logger.warning("⚠️ Không tìm thấy energy spikes!")
            return None
        
        except Exception as e:
            logger.error(f"❌ Lỗi tìm vị trí giọng hát: {e}")
            return None
    
    def get_first_suitable_voice_segment(self, audio_path: str, min_duration: float = 1.0) -> Dict:
        """Returns the first voice segment that meets a minimum duration."""
        segments = self.detect_voice_activity(audio_path)
        for segment in segments:
            if (segment['end'] - segment['start']) >= min_duration:
                logger.info(f"✅ Found first suitable correct voice segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
                return segment
        logger.warning("⚠️ No suitable correct voice segment found.")
        return None
