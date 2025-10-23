#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accurate Voice Detector - Phát hiện chính xác giọng hát trong file karaoke
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

class AccurateVoiceDetector:
    """Accurate Voice Detector - Phát hiện chính xác giọng hát trong file karaoke"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
    
    def detect_voice_activity(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity với độ chính xác cao"""
        try:
            logger.info("🎯 Accurate Voice Detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Phân tích để tìm vị trí giọng hát thực sự
            voice_start = self._find_accurate_voice_start(audio, sr)
            
            if voice_start is None:
                logger.warning("⚠️ Không tìm thấy vị trí giọng hát")
                return []
            
            # Tạo segment từ vị trí tìm được
            segments = [{
                'start': voice_start,
                'end': len(audio) / sr,  # Đến cuối file
                'confidence': 1.0,
                'method': 'accurate_detection'
            }]
            
            logger.info(f"✅ Accurate voice detection: {len(segments)} segments")
            logger.info(f"   Voice starts at: {voice_start:.2f}s")
            
            return segments
            
        except Exception as e:
            logger.error(f"❌ Accurate voice detection failed: {e}")
            return []
    
    def _find_accurate_voice_start(self, audio: np.ndarray, sr: int) -> Optional[float]:
        """Tìm vị trí bắt đầu giọng hát thực sự"""
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
            
            logger.info(f"🔍 Phân tích {len(rms)} frames để tìm giọng hát thực sự...")
            
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
            
            # Tìm vị trí giọng hát với logic chính xác
            voice_start_candidates = []
            
            # Phân tích 120 giây đầu
            for i in range(min(120, len(rms))):
                start_time = i * hop_length / sr
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
                    segment_rms > baseline_rms * 1.5 and  # Energy cao hơn baseline 50%
                    segment_centroid > baseline_centroid * 0.8 and segment_centroid < 4000 and  # Voice frequency range
                    segment_rolloff > baseline_rolloff * 0.8 and  # Có high frequency content
                    harmonic_ratio > 0.7 and  # Harmonic content cao (voice có harmonic)
                    mfcc_std > 30  # MFCC variation (voice có variation)
                )
                
                if has_voice:
                    voice_start_candidates.append({
                        'time': start_time,
                        'rms': segment_rms,
                        'centroid': segment_centroid,
                        'rolloff': segment_rolloff,
                        'harmonic_ratio': harmonic_ratio,
                        'mfcc_std': mfcc_std,
                        'confidence': self._calculate_voice_confidence(segment_rms, segment_centroid, harmonic_ratio, mfcc_std, baseline_rms, baseline_centroid)
                    })
                
                # Log chi tiết cho 30 giây đầu
                if start_time < 30:
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
    
    def _calculate_voice_confidence(self, rms: float, centroid: float, harmonic_ratio: float, mfcc_std: float, baseline_rms: float, baseline_centroid: float) -> float:
        """Tính confidence score cho voice detection với baseline"""
        try:
            # Normalize các features dựa trên baseline
            rms_score = min((rms - baseline_rms) / baseline_rms, 1.0)  # So với baseline
            centroid_score = 1.0 - abs(centroid - baseline_centroid) / baseline_centroid  # So với baseline
            harmonic_score = harmonic_ratio  # Đã normalize
            mfcc_score = min(mfcc_std / 50, 1.0)  # Normalize MFCC std
            
            # Weighted average
            confidence = (
                max(rms_score, 0) * 0.4 +  # Tăng weight cho RMS
                max(centroid_score, 0) * 0.3 +
                harmonic_score * 0.2 +
                mfcc_score * 0.1
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
                logger.info(f"✅ Found first suitable accurate voice segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
                return segment
        logger.warning("⚠️ No suitable accurate voice segment found.")
        return None
