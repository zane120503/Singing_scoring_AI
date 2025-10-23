#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precise Voice Detector - Phát hiện chính xác vị trí bắt đầu giọng hát
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

class PreciseVoiceDetector:
    """Precise Voice Detector - Phát hiện chính xác vị trí bắt đầu giọng hát"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
    
    def detect_voice_activity(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity với độ chính xác cao"""
        try:
            logger.info("🎯 Precise Voice Detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Phân tích chi tiết từng giây để tìm vị trí giọng hát
            voice_start = self._find_precise_voice_start(audio, sr)
            
            if voice_start is None:
                logger.warning("⚠️ Không tìm thấy vị trí giọng hát")
                return []
            
            # Tạo segment từ vị trí tìm được
            segments = [{
                'start': voice_start,
                'end': len(audio) / sr,  # Đến cuối file
                'confidence': 1.0,
                'method': 'precise_detection'
            }]
            
            logger.info(f"✅ Precise voice detection: {len(segments)} segments")
            logger.info(f"   Voice starts at: {voice_start:.2f}s")
            
            return segments
            
        except Exception as e:
            logger.error(f"❌ Precise voice detection failed: {e}")
            return []
    
    def _find_precise_voice_start(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Tìm vị trí bắt đầu giọng hát chính xác"""
        try:
            # Phân tích chi tiết từng giây
            hop_length = self.hop_length
            frame_length = self.frame_length
            
            # Tính các features
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Chuyển frames thành thời gian
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
            
            logger.info(f"🔍 Phân tích {len(rms)} frames để tìm giọng hát...")
            
            # Tìm vị trí giọng hát với logic chính xác
            voice_start_candidates = []
            
            # Phân tích 30 giây đầu (đủ để tìm giọng hát)
            for i in range(min(30, len(rms))):
                start_time = i * hop_length / sr
                end_time = min((i + 1) * hop_length / sr, len(audio) / sr)
                
                # Lấy features của frame này
                segment_rms = rms[i]
                segment_centroid = spectral_centroids[i]
                segment_rolloff = spectral_rolloff[i]
                segment_zcr = zcr[i]
                
                # Phân tích harmonic content
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Harmonic-percussive separation
                try:
                    harmonic, percussive = librosa.effects.hpss(segment_audio)
                    harmonic_energy = np.mean(np.abs(harmonic))
                    percussive_energy = np.mean(np.abs(percussive))
                    harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-8)
                except:
                    harmonic_ratio = 0.5
                
                # MFCC để phát hiện voice characteristics
                try:
                    mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfccs[0])
                    mfcc_std = np.std(mfccs[0])
                except:
                    mfcc_mean = 0
                    mfcc_std = 0
                
                # Điều kiện chính xác để xác định có giọng hát
                has_voice = (
                    segment_rms > 0.08 and  # Energy đủ cao
                    segment_centroid > 1000 and segment_centroid < 4000 and  # Voice frequency range
                    segment_rolloff > 1000 and  # Có high frequency content
                    harmonic_ratio > 0.6 and  # Harmonic content cao (voice có harmonic)
                    mfcc_std > 50  # MFCC variation (voice có variation)
                )
                
                if has_voice:
                    voice_start_candidates.append({
                        'time': start_time,
                        'rms': segment_rms,
                        'centroid': segment_centroid,
                        'rolloff': segment_rolloff,
                        'harmonic_ratio': harmonic_ratio,
                        'mfcc_std': mfcc_std,
                        'confidence': self._calculate_voice_confidence(segment_rms, segment_centroid, harmonic_ratio, mfcc_std)
                    })
                
                # Log chi tiết cho 15 giây đầu
                if start_time < 15:
                    logger.info(f"   {start_time:.1f}s: RMS={segment_rms:.4f}, Centroid={segment_centroid:.1f}Hz, Rolloff={segment_rolloff:.1f}Hz, Harmonic={harmonic_ratio:.3f}, MFCC_std={mfcc_std:.1f}, Voice={'YES' if has_voice else 'NO'}")
            
            # Tìm vị trí giọng hát bắt đầu thực sự
            if voice_start_candidates:
                # Lấy vị trí có confidence cao nhất
                best_candidate = max(voice_start_candidates, key=lambda x: x['confidence'])
                
                logger.info(f"\n🎯 Vị trí giọng hát chính xác:")
                logger.info(f"   Voice starts at: {best_candidate['time']:.2f}s")
                logger.info(f"   Confidence: {best_candidate['confidence']:.3f}")
                logger.info(f"   RMS: {best_candidate['rms']:.4f}")
                logger.info(f"   Centroid: {best_candidate['centroid']:.1f}Hz")
                logger.info(f"   Harmonic ratio: {best_candidate['harmonic_ratio']:.3f}")
                
                return best_candidate['time']
            else:
                logger.warning("⚠️ Không tìm thấy vị trí giọng hát!")
                return None
        
        except Exception as e:
            logger.error(f"❌ Lỗi tìm vị trí giọng hát: {e}")
            return None
    
    def _calculate_voice_confidence(self, rms: float, centroid: float, harmonic_ratio: float, mfcc_std: float) -> float:
        """Tính confidence score cho voice detection"""
        try:
            # Normalize các features
            rms_score = min(rms / 0.2, 1.0)  # Normalize RMS
            centroid_score = 1.0 - abs(centroid - 2000) / 2000  # Optimal ở 2000Hz
            harmonic_score = harmonic_ratio  # Đã normalize
            mfcc_score = min(mfcc_std / 100, 1.0)  # Normalize MFCC std
            
            # Weighted average
            confidence = (
                rms_score * 0.3 +
                centroid_score * 0.3 +
                harmonic_score * 0.25 +
                mfcc_score * 0.15
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
                logger.info(f"✅ Found first suitable precise voice segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
                return segment
        logger.warning("⚠️ No suitable precise voice segment found.")
        return None
