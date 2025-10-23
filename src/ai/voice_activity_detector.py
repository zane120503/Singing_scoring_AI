#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice Activity Detector - Phát hiện giọng hát trong file karaoke
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VoiceActivityDetector:
    """Voice Activity Detector để phát hiện giọng hát trong karaoke"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
        
    def detect_voice_activity(self, audio_path: str, method: str = "spectral") -> List[Dict]:
        """
        Phát hiện voice activity trong file audio
        
        Args:
            audio_path: Đường dẫn file audio
            method: Phương pháp detection ("spectral", "energy", "zero_crossing")
            
        Returns:
            List of voice activity segments: [{"start": float, "end": float, "confidence": float}]
        """
        try:
            logger.info(f"🎤 Phát hiện voice activity trong file: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            logger.info(f"✅ Đã load audio: {len(audio)} samples, {sr} Hz")
            
            if method == "spectral":
                segments = self._detect_spectral_voice(audio, sr)
            elif method == "energy":
                segments = self._detect_energy_voice(audio, sr)
            elif method == "zero_crossing":
                segments = self._detect_zero_crossing_voice(audio, sr)
            else:
                # Combine all methods
                segments = self._detect_combined_voice(audio, sr)
            
            logger.info(f"🎯 Phát hiện {len(segments)} đoạn có giọng hát")
            return segments
            
        except Exception as e:
            logger.error(f"❌ Lỗi phát hiện voice activity: {e}")
            return []
    
    def _detect_spectral_voice(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện voice dựa trên spectral features - cải thiện"""
        try:
            # Extract spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)[0]
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            
            # Voice characteristics - điều chỉnh thresholds
            voice_threshold_centroid = np.percentile(spectral_centroids, 20)  # Thấp hơn để phát hiện voice nhẹ
            voice_threshold_rolloff = np.percentile(spectral_rolloff, 30)    # Thấp hơn
            
            # Voice frequency range (80-4000 Hz) - mở rộng
            voice_low = 80
            voice_high = 4000
            
            # Detect voice frames
            voice_frames = []
            for i in range(len(spectral_centroids)):
                if (spectral_centroids[i] > voice_threshold_centroid and 
                    voice_low < spectral_centroids[i] < voice_high and  # Voice frequency range
                    spectral_rolloff[i] > voice_threshold_rolloff):
                    voice_frames.append(i)
            
            return self._frames_to_segments(voice_frames, sr)
            
        except Exception as e:
            logger.warning(f"Spectral voice detection failed: {e}")
            return []
    
    def _detect_energy_voice(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện voice dựa trên energy - cải thiện v2"""
        try:
            # Calculate RMS energy
            frame_length = self.frame_length
            hop_length = self.hop_length
            
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Adaptive threshold - thấp hơn để phát hiện voice nhẹ
            energy_threshold = np.percentile(rms, 25)  # Thấp hơn nữa
            
            # Detect voice frames với điều kiện nghiêm ngặt hơn
            voice_frames = []
            for i, energy in enumerate(rms):
                if energy > energy_threshold:
                    voice_frames.append(i)
            
            # Lọc bỏ các đoạn quá ngắn ở đầu file (có thể là noise)
            filtered_segments = self._frames_to_segments(voice_frames, sr)
            
            # Lọc bỏ segments ở đầu file quá ngắn (dưới 0.5s)
            filtered_segments = [seg for seg in filtered_segments 
                               if not (seg['start'] < 0.5 and (seg['end'] - seg['start']) < 0.5)]
            
            return filtered_segments
            
        except Exception as e:
            logger.warning(f"Energy voice detection failed: {e}")
            return []
    
    def _detect_zero_crossing_voice(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện voice dựa trên zero crossing rate"""
        try:
            # Calculate zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # Voice has moderate zero crossing rate
            voice_threshold_low = np.percentile(zcr, 20)
            voice_threshold_high = np.percentile(zcr, 80)
            
            # Detect voice frames
            voice_frames = []
            for i, rate in enumerate(zcr):
                if voice_threshold_low < rate < voice_threshold_high:
                    voice_frames.append(i)
            
            return self._frames_to_segments(voice_frames, sr)
            
        except Exception as e:
            logger.warning(f"Zero crossing voice detection failed: {e}")
            return []
    
    def _detect_combined_voice(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Kết hợp nhiều phương pháp để phát hiện voice - cải thiện"""
        try:
            # Get segments from all methods
            spectral_segments = self._detect_spectral_voice(audio, sr)
            energy_segments = self._detect_energy_voice(audio, sr)
            zcr_segments = self._detect_zero_crossing_voice(audio, sr)
            
            # Thêm phương pháp multi-feature detection
            multi_feature_segments = self._detect_multi_feature_voice(audio, sr)
            
            # Combine and merge overlapping segments
            all_segments = spectral_segments + energy_segments + zcr_segments + multi_feature_segments
            merged_segments = self._merge_overlapping_segments(all_segments)
            
            return merged_segments
            
        except Exception as e:
            logger.warning(f"Combined voice detection failed: {e}")
            return []
    
    def _detect_multi_feature_voice(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện voice dựa trên multiple features - phương pháp mới"""
        try:
            # Extract multiple features
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            
            # Adaptive thresholds
            rms_threshold = np.percentile(rms, 25)  # Thấp hơn
            centroid_threshold = np.percentile(spectral_centroids, 15)  # Thấp hơn
            rolloff_threshold = np.percentile(spectral_rolloff, 25)  # Thấp hơn
            zcr_low = np.percentile(zcr, 10)
            zcr_high = np.percentile(zcr, 90)
            
            # Voice frequency range
            voice_low = 80
            voice_high = 4000
            
            # Detect voice frames
            voice_frames = []
            for i in range(len(rms)):
                # Điều kiện phát hiện voice
                rms_ok = rms[i] > rms_threshold
                centroid_ok = voice_low < spectral_centroids[i] < voice_high and spectral_centroids[i] > centroid_threshold
                rolloff_ok = spectral_rolloff[i] > rolloff_threshold
                zcr_ok = zcr_low < zcr[i] < zcr_high
                
                # Cần ít nhất 2/4 điều kiện
                if sum([rms_ok, centroid_ok, rolloff_ok, zcr_ok]) >= 2:
                    voice_frames.append(i)
            
            return self._frames_to_segments(voice_frames, sr)
            
        except Exception as e:
            logger.warning(f"Multi-feature voice detection failed: {e}")
            return []
    
    def _frames_to_segments(self, voice_frames: List[int], sr: int) -> List[Dict]:
        """Chuyển đổi voice frames thành segments"""
        if not voice_frames:
            return []
        
        segments = []
        current_start = voice_frames[0]
        current_end = voice_frames[0]
        
        for i in range(1, len(voice_frames)):
            if voice_frames[i] - voice_frames[i-1] <= 2:  # Consecutive frames
                current_end = voice_frames[i]
            else:
                # End of current segment
                start_time = current_start * self.hop_length / sr
                end_time = (current_end + 1) * self.hop_length / sr
                
                if end_time - start_time >= 0.5:  # Minimum 0.5 seconds
                    segments.append({
                        "start": start_time,
                        "end": end_time,
                        "confidence": 0.8
                    })
                
                # Start new segment
                current_start = voice_frames[i]
                current_end = voice_frames[i]
        
        # Add last segment
        start_time = current_start * self.hop_length / sr
        end_time = (current_end + 1) * self.hop_length / sr
        if end_time - start_time >= 0.5:
            segments.append({
                "start": start_time,
                "end": end_time,
                "confidence": 0.8
            })
        
        return segments
    
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge overlapping voice segments"""
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x["start"])
        
        merged = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            last_merged = merged[-1]
            
            # Check if segments overlap
            if segment["start"] <= last_merged["end"]:
                # Merge segments
                last_merged["end"] = max(last_merged["end"], segment["end"])
                last_merged["confidence"] = max(last_merged["confidence"], segment["confidence"])
            else:
                # Add new segment
                merged.append(segment)
        
        return merged
    
    def find_first_voice_segment(self, audio_path: str, min_duration: float = 1.0) -> Dict:
        """
        Tìm đoạn voice đầu tiên trong file
        
        Args:
            audio_path: Đường dẫn file audio
            min_duration: Thời lượng tối thiểu của đoạn voice
            
        Returns:
            Dict với thông tin đoạn voice đầu tiên
        """
        segments = self.detect_voice_activity(audio_path)
        
        for segment in segments:
            duration = segment["end"] - segment["start"]
            if duration >= min_duration:
                logger.info(f"🎯 Tìm thấy đoạn voice đầu tiên: {segment['start']:.2f}s - {segment['end']:.2f}s")
                return segment
        
        logger.warning("⚠️ Không tìm thấy đoạn voice nào")
        return {"start": 0, "end": 0, "confidence": 0}
