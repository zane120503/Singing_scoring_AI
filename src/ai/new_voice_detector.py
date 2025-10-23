#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New Voice Detector - Phát hiện giọng hát với logic mới
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

class NewVoiceDetector:
    """New Voice Detector - Phát hiện giọng hát với logic mới"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
    
    def detect_voice_activity(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity với logic mới"""
        try:
            logger.info("🎯 New Voice Detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Phân tích để tìm vị trí giọng hát thực sự
            voice_start = self._find_new_voice_start(audio, sr)
            
            if voice_start is None:
                logger.warning("⚠️ Không tìm thấy vị trí giọng hát")
                return []
            
            # Tạo segment từ vị trí tìm được
            segments = [{
                'start': voice_start,
                'end': len(audio) / sr,  # Đến cuối file
                'confidence': 1.0,
                'method': 'new_detection'
            }]
            
            logger.info(f"✅ New voice detection: {len(segments)} segments")
            logger.info(f"   Voice starts at: {voice_start:.2f}s")
            
            return segments
            
        except Exception as e:
            logger.error(f"❌ New voice detection failed: {e}")
            return []
    
    def _find_new_voice_start(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Tìm vị trí bắt đầu giọng hát với logic mới"""
        try:
            # Phân tích từng giây
            hop_length = self.hop_length
            
            # Tính RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=hop_length)[0]
            
            # Tính spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, hop_length=hop_length)[0]
            
            # Chuyển frames thành thời gian
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            logger.info(f"🔍 Phân tích {len(rms)} frames để tìm giọng hát...")
            
            # Tính baseline từ 20 giây đầu (intro)
            baseline_length = min(20, len(rms))
            baseline_rms = np.mean(rms[:baseline_length])
            baseline_centroid = np.mean(spectral_centroids[:baseline_length])
            baseline_rolloff = np.mean(spectral_rolloff[:baseline_length])
            baseline_zcr = np.mean(zcr[:baseline_length])
            
            logger.info(f"📊 Baseline analysis:")
            logger.info(f"   RMS: {baseline_rms:.4f}")
            logger.info(f"   Centroid: {baseline_centroid:.1f} Hz")
            logger.info(f"   Rolloff: {baseline_rolloff:.1f} Hz")
            logger.info(f"   ZCR: {baseline_zcr:.4f}")
            
            # Tìm vị trí giọng hát với logic mới
            voice_start_candidates = []
            
            # Phân tích 120 giây đầu
            for i in range(min(120, len(rms))):
                start_time = i * hop_length / sr
                segment_rms = rms[i]
                segment_centroid = spectral_centroids[i]
                segment_rolloff = spectral_rolloff[i]
                segment_zcr = zcr[i]
                
                # Logic mới: tìm vị trí có energy cao và spectral features khác biệt
                # Điều kiện đơn giản hơn để phát hiện giọng hát
                has_voice = (
                    segment_rms > baseline_rms * 1.2 and  # Energy cao hơn baseline 20%
                    segment_centroid > baseline_centroid * 0.8 and segment_centroid < 4000 and  # Voice frequency range
                    segment_rolloff > baseline_rolloff * 0.8 and  # Có high frequency content
                    segment_zcr > baseline_zcr * 0.7 and segment_zcr < baseline_zcr * 1.8  # ZCR trong khoảng hợp lý
                )
                
                if has_voice:
                    confidence = self._calculate_new_confidence(segment_rms, segment_centroid, baseline_rms, baseline_centroid)
                    voice_start_candidates.append({
                        'time': start_time,
                        'rms': segment_rms,
                        'centroid': segment_centroid,
                        'rolloff': segment_rolloff,
                        'zcr': segment_zcr,
                        'confidence': confidence
                    })
                
                # Log chi tiết cho 30 giây đầu
                if start_time < 30:
                    logger.info(f"   {start_time:.1f}s: RMS={segment_rms:.4f}, Centroid={segment_centroid:.1f}Hz, Rolloff={segment_rolloff:.1f}Hz, ZCR={segment_zcr:.4f}, Voice={'YES' if has_voice else 'NO'}")
            
            # Tìm vị trí giọng hát bắt đầu thực sự
            if voice_start_candidates:
                # Lấy vị trí có confidence cao nhất
                best_candidate = max(voice_start_candidates, key=lambda x: x['confidence'])
                
                logger.info(f"\n🎯 Vị trí giọng hát chính xác:")
                logger.info(f"   Voice starts at: {best_candidate['time']:.2f}s")
                logger.info(f"   Confidence: {best_candidate['confidence']:.3f}")
                logger.info(f"   RMS: {best_candidate['rms']:.4f}")
                logger.info(f"   Centroid: {best_candidate['centroid']:.1f}Hz")
                logger.info(f"   Rolloff: {best_candidate['rolloff']:.1f}Hz")
                logger.info(f"   ZCR: {best_candidate['zcr']:.4f}")
                
                return best_candidate['time']
            else:
                logger.warning("⚠️ Không tìm thấy vị trí giọng hát!")
                return None
        
        except Exception as e:
            logger.error(f"❌ Lỗi tìm vị trí giọng hát: {e}")
            return None
    
    def _calculate_new_confidence(self, rms: float, centroid: float, baseline_rms: float, baseline_centroid: float) -> float:
        """Tính confidence score mới"""
        try:
            # Normalize các features dựa trên baseline
            rms_score = min((rms - baseline_rms) / baseline_rms, 1.0)  # So với baseline
            centroid_score = 1.0 - abs(centroid - baseline_centroid) / baseline_centroid  # So với baseline
            
            # Weighted average
            confidence = (
                max(rms_score, 0) * 0.7 +  # Tăng weight cho RMS
                max(centroid_score, 0) * 0.3
            )
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return 0.5
    
    def get_first_suitable_voice_segment(self, audio_path: str, min_duration: float = 1.0) -> Dict:
        """Returns the first voice segment that meets a minimum duration."""
        segments = self.detect_voice_activity(audio_path)
        for segment in segments:
            if (segment['end'] - segment['start']) >= min_duration:
                logger.info(f"✅ Found first suitable new voice segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
                return segment
        logger.warning("⚠️ No suitable new voice segment found.")
        return None
