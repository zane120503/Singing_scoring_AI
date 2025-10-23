#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Voice Detector - phát hiện chính xác vị trí bắt đầu giọng hát
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

class SmartVoiceDetector:
    """Smart Voice Detector - phát hiện chính xác vị trí bắt đầu giọng hát"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
    
    def detect_voice_activity(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity thông minh"""
        try:
            logger.info("🧠 Smart Voice Detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Bước 1: Phân tích baseline (đoạn đầu file)
            baseline_features = self._analyze_baseline(audio, sr)
            
            # Bước 2: Phân tích energy pattern với baseline
            energy_segments = self._detect_energy_with_baseline(audio, sr, baseline_features)
            
            # Bước 3: Phân tích spectral pattern với baseline
            spectral_segments = self._detect_spectral_with_baseline(audio, sr, baseline_features)
            
            # Bước 4: Phân tích harmonic pattern
            harmonic_segments = self._detect_harmonic_pattern(audio, sr)
            
            # Bước 5: Phân tích voice characteristics
            voice_segments = self._detect_voice_characteristics(audio, sr)
            
            # Bước 6: Kết hợp và lọc kết quả thông minh
            combined_segments = self._smart_combine_segments(
                energy_segments, spectral_segments, harmonic_segments, voice_segments, baseline_features
            )
            
            logger.info(f"✅ Smart voice detection: {len(combined_segments)} segments")
            return combined_segments
            
        except Exception as e:
            logger.error(f"❌ Smart voice detection failed: {e}")
            return []
    
    def _analyze_baseline(self, audio: np.ndarray, sr: int) -> Dict:
        """Phân tích baseline của file (đoạn đầu)"""
        try:
            # Phân tích 2 giây đầu
            baseline_length = min(2 * sr, len(audio))
            baseline_audio = audio[:baseline_length]
            
            # Tính toán features của baseline
            baseline_rms = librosa.feature.rms(y=baseline_audio)[0]
            baseline_centroids = librosa.feature.spectral_centroid(y=baseline_audio, sr=sr)[0]
            baseline_rolloff = librosa.feature.spectral_rolloff(y=baseline_audio, sr=sr)[0]
            baseline_zcr = librosa.feature.zero_crossing_rate(baseline_audio)[0]
            
            baseline_features = {
                'rms_mean': np.mean(baseline_rms),
                'rms_std': np.std(baseline_rms),
                'centroid_mean': np.mean(baseline_centroids),
                'centroid_std': np.std(baseline_centroids),
                'rolloff_mean': np.mean(baseline_rolloff),
                'rolloff_std': np.std(baseline_rolloff),
                'zcr_mean': np.mean(baseline_zcr),
                'zcr_std': np.std(baseline_zcr)
            }
            
            logger.info(f"📊 Baseline analysis:")
            logger.info(f"   RMS: {baseline_features['rms_mean']:.4f} ± {baseline_features['rms_std']:.4f}")
            logger.info(f"   Centroid: {baseline_features['centroid_mean']:.1f} ± {baseline_features['centroid_std']:.1f} Hz")
            logger.info(f"   Rolloff: {baseline_features['rolloff_mean']:.1f} ± {baseline_features['rolloff_std']:.1f} Hz")
            logger.info(f"   ZCR: {baseline_features['zcr_mean']:.4f} ± {baseline_features['zcr_std']:.4f}")
            
            return baseline_features
            
        except Exception as e:
            logger.warning(f"Baseline analysis failed: {e}")
            return {}
    
    def _detect_energy_with_baseline(self, audio: np.ndarray, sr: int, baseline_features: Dict) -> List[Dict]:
        """Phát hiện voice dựa trên energy với baseline"""
        try:
            # RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            
            # Threshold dựa trên baseline + margin
            baseline_rms = baseline_features.get('rms_mean', np.mean(rms))
            baseline_std = baseline_features.get('rms_std', np.std(rms))
            
            # Threshold cao hơn baseline đáng kể
            energy_threshold = baseline_rms + 2 * baseline_std
            
            voice_frames = []
            for i, energy in enumerate(rms):
                if energy > energy_threshold:
                    voice_frames.append(i)
            
            return self._frames_to_segments(voice_frames, sr)
            
        except Exception as e:
            logger.warning(f"Energy with baseline detection failed: {e}")
            return []
    
    def _detect_spectral_with_baseline(self, audio: np.ndarray, sr: int, baseline_features: Dict) -> List[Dict]:
        """Phát hiện voice dựa trên spectral với baseline"""
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            
            # Thresholds dựa trên baseline
            baseline_centroid = baseline_features.get('centroid_mean', np.mean(spectral_centroids))
            baseline_rolloff = baseline_features.get('rolloff_mean', np.mean(spectral_rolloff))
            
            # Voice frequency range
            voice_low = 80
            voice_high = 4000
            
            voice_frames = []
            for i in range(len(spectral_centroids)):
                centroid_ok = (voice_low < spectral_centroids[i] < voice_high and 
                             spectral_centroids[i] > baseline_centroid + 200)  # Cao hơn baseline
                rolloff_ok = spectral_rolloff[i] > baseline_rolloff + 500  # Cao hơn baseline
                
                if centroid_ok and rolloff_ok:
                    voice_frames.append(i)
            
            return self._frames_to_segments(voice_frames, sr)
            
        except Exception as e:
            logger.warning(f"Spectral with baseline detection failed: {e}")
            return []
    
    def _detect_harmonic_pattern(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện voice dựa trên harmonic pattern"""
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Phân tích harmonic content
            harmonic_centroids = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
            harmonic_rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr)[0]
            
            # Thresholds cho harmonic content
            harmonic_threshold = np.percentile(harmonic_centroids, 30)
            rolloff_threshold = np.percentile(harmonic_rolloff, 30)
            
            voice_frames = []
            for i in range(len(harmonic_centroids)):
                harmonic_ok = harmonic_centroids[i] > harmonic_threshold
                rolloff_ok = harmonic_rolloff[i] > rolloff_threshold
                
                if harmonic_ok and rolloff_ok:
                    voice_frames.append(i)
            
            return self._frames_to_segments(voice_frames, sr)
            
        except Exception as e:
            logger.warning(f"Harmonic pattern detection failed: {e}")
            return []
    
    def _detect_voice_characteristics(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện voice dựa trên voice characteristics"""
        try:
            # MFCC features (đặc trưng của voice)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            
            # Voice characteristics thresholds
            mfcc_threshold = np.percentile(mfccs[0], 25)  # First MFCC coefficient
            bandwidth_threshold = np.percentile(spectral_bandwidth, 25)
            
            voice_frames = []
            for i in range(len(mfccs[0])):
                mfcc_ok = mfccs[0][i] > mfcc_threshold
                bandwidth_ok = spectral_bandwidth[i] > bandwidth_threshold
                
                if mfcc_ok and bandwidth_ok:
                    voice_frames.append(i)
            
            return self._frames_to_segments(voice_frames, sr)
            
        except Exception as e:
            logger.warning(f"Voice characteristics detection failed: {e}")
            return []
    
    def _smart_combine_segments(self, energy_segments: List[Dict], 
                              spectral_segments: List[Dict], 
                              harmonic_segments: List[Dict], 
                              voice_segments: List[Dict],
                              baseline_features: Dict) -> List[Dict]:
        """Kết hợp segments một cách thông minh"""
        try:
            # Kết hợp tất cả segments
            all_segments = energy_segments + spectral_segments + harmonic_segments + voice_segments
            
            # Merge overlapping segments
            merged_segments = self._merge_overlapping_segments(all_segments)
            
            # Lọc segments thông minh
            filtered_segments = []
            for segment in merged_segments:
                duration = segment['end'] - segment['start']
                
                # Bỏ qua segments quá ngắn
                if duration < 0.5:
                    continue
                
                # Bỏ qua segments ở đầu file quá ngắn (có thể là noise)
                if segment['start'] < 1.0 and duration < 1.0:
                    continue
                
                # Ưu tiên segments có confidence cao
                segment['confidence'] = min(segment.get('confidence', 0.8) + 0.1, 1.0)
                
                filtered_segments.append(segment)
            
            # Sắp xếp theo start time
            filtered_segments.sort(key=lambda x: x['start'])
            
            return filtered_segments
            
        except Exception as e:
            logger.warning(f"Smart combine failed: {e}")
            return []
    
    def _frames_to_segments(self, voice_frames: List[int], sr: int) -> List[Dict]:
        """Chuyển đổi voice frames thành segments"""
        if not voice_frames:
            return []
        
        segments = []
        current_start = voice_frames[0]
        current_end = voice_frames[0]
        
        for i in range(1, len(voice_frames)):
            if voice_frames[i] - voice_frames[i-1] <= 2:  # Liên tục
                current_end = voice_frames[i]
            else:
                # End of current segment
                start_time = current_start * self.hop_length / sr
                end_time = current_end * self.hop_length / sr
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'confidence': 0.95,
                    'method': 'smart_detection'
                })
                current_start = voice_frames[i]
                current_end = voice_frames[i]
        
        # Add last segment
        start_time = current_start * self.hop_length / sr
        end_time = current_end * self.hop_length / sr
        segments.append({
            'start': start_time,
            'end': end_time,
            'confidence': 0.95,
            'method': 'smart_detection'
        })
        
        return segments
    
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge overlapping segments"""
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        merged = [sorted_segments[0]]
        for segment in sorted_segments[1:]:
            last_merged = merged[-1]
            
            # If overlapping, merge
            if segment['start'] <= last_merged['end']:
                last_merged['end'] = max(last_merged['end'], segment['end'])
            else:
                merged.append(segment)
        
        return merged
    
    def find_first_voice_segment(self, audio_path: str, min_duration: float = 1.0) -> Dict:
        """Tìm đoạn voice đầu tiên phù hợp"""
        try:
            segments = self.detect_voice_activity(audio_path)
            
            for segment in segments:
                duration = segment["end"] - segment["start"]
                if duration >= min_duration:
                    logger.info(f"🎯 Found first suitable smart voice segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
                    return segment
            
            logger.warning("⚠️ No suitable smart voice segment found")
            return {"start": 0, "end": 0, "confidence": 0}
            
        except Exception as e:
            logger.error(f"❌ Error finding first voice segment: {e}")
            return {"start": 0, "end": 0, "confidence": 0}
