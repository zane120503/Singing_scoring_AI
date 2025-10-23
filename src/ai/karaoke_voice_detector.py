#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Karaoke Voice Detector - chuyên biệt cho karaoke files
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

class KaraokeVoiceDetector:
    """Voice Detector chuyên biệt cho karaoke files"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
    
    def detect_voice_activity(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity chuyên biệt cho karaoke"""
        try:
            logger.info("🎤 Karaoke Voice Detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Bước 1: Phân tích energy pattern
            energy_segments = self._detect_energy_pattern(audio, sr)
            
            # Bước 2: Phân tích spectral pattern
            spectral_segments = self._detect_spectral_pattern(audio, sr)
            
            # Bước 3: Phân tích harmonic pattern (cho giọng hát)
            harmonic_segments = self._detect_harmonic_pattern(audio, sr)
            
            # Bước 4: Kết hợp và lọc kết quả
            combined_segments = self._combine_and_filter_segments(
                energy_segments, spectral_segments, harmonic_segments
            )
            
            logger.info(f"✅ Karaoke voice detection: {len(combined_segments)} segments")
            return combined_segments
            
        except Exception as e:
            logger.error(f"❌ Karaoke voice detection failed: {e}")
            return []
    
    def _detect_energy_pattern(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện voice dựa trên energy pattern"""
        try:
            # RMS energy với window nhỏ hơn để phát hiện chi tiết
            rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
            
            # Tính toán dynamic threshold
            rms_smooth = librosa.util.normalize(rms)
            energy_threshold = np.percentile(rms_smooth, 20)  # Thấp hơn
            
            # Tìm các đoạn có energy cao liên tục
            voice_frames = []
            for i, energy in enumerate(rms_smooth):
                if energy > energy_threshold:
                    voice_frames.append(i)
            
            return self._frames_to_segments(voice_frames, sr, hop_length=256)
            
        except Exception as e:
            logger.warning(f"Energy pattern detection failed: {e}")
            return []
    
    def _detect_spectral_pattern(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện voice dựa trên spectral pattern"""
        try:
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            
            # Voice frequency range (80-4000 Hz)
            voice_low = 80
            voice_high = 4000
            
            # Adaptive thresholds
            centroid_threshold = np.percentile(spectral_centroids, 15)
            rolloff_threshold = np.percentile(spectral_rolloff, 20)
            bandwidth_threshold = np.percentile(spectral_bandwidth, 25)
            
            voice_frames = []
            for i in range(len(spectral_centroids)):
                centroid_ok = voice_low < spectral_centroids[i] < voice_high and spectral_centroids[i] > centroid_threshold
                rolloff_ok = spectral_rolloff[i] > rolloff_threshold
                bandwidth_ok = spectral_bandwidth[i] > bandwidth_threshold
                
                if sum([centroid_ok, rolloff_ok, bandwidth_ok]) >= 2:
                    voice_frames.append(i)
            
            return self._frames_to_segments(voice_frames, sr)
            
        except Exception as e:
            logger.warning(f"Spectral pattern detection failed: {e}")
            return []
    
    def _detect_harmonic_pattern(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện voice dựa trên harmonic pattern (đặc trưng của giọng hát)"""
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Phân tích harmonic content
            harmonic_centroids = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
            harmonic_rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr)[0]
            
            # Voice có harmonic content cao
            harmonic_threshold = np.percentile(harmonic_centroids, 20)
            rolloff_threshold = np.percentile(harmonic_rolloff, 25)
            
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
    
    def _combine_and_filter_segments(self, energy_segments: List[Dict], 
                                   spectral_segments: List[Dict], 
                                   harmonic_segments: List[Dict]) -> List[Dict]:
        """Kết hợp và lọc các segments"""
        try:
            # Kết hợp tất cả segments
            all_segments = energy_segments + spectral_segments + harmonic_segments
            
            # Merge overlapping segments
            merged_segments = self._merge_overlapping_segments(all_segments)
            
            # Lọc bỏ segments quá ngắn hoặc ở đầu file
            filtered_segments = []
            for segment in merged_segments:
                duration = segment['end'] - segment['start']
                
                # Bỏ qua segments quá ngắn (dưới 0.3s)
                if duration < 0.3:
                    continue
                
                # Bỏ qua segments ở đầu file quá ngắn (có thể là noise)
                if segment['start'] < 0.5 and duration < 0.5:
                    continue
                
                filtered_segments.append(segment)
            
            return filtered_segments
            
        except Exception as e:
            logger.warning(f"Combine and filter failed: {e}")
            return []
    
    def _frames_to_segments(self, voice_frames: List[int], sr: int, hop_length: int = 512) -> List[Dict]:
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
                start_time = current_start * hop_length / sr
                end_time = current_end * hop_length / sr
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'confidence': 0.9,
                    'method': 'karaoke_specialized'
                })
                current_start = voice_frames[i]
                current_end = voice_frames[i]
        
        # Add last segment
        start_time = current_start * hop_length / sr
        end_time = current_end * hop_length / sr
        segments.append({
            'start': start_time,
            'end': end_time,
            'confidence': 0.9,
            'method': 'karaoke_specialized'
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
                    logger.info(f"🎯 Found first suitable karaoke voice segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
                    return segment
            
            logger.warning("⚠️ No suitable karaoke voice segment found")
            return {"start": 0, "end": 0, "confidence": 0}
            
        except Exception as e:
            logger.error(f"❌ Error finding first voice segment: {e}")
            return {"start": 0, "end": 0, "confidence": 0}
