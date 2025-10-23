#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Smart Voice Detector - Phát hiện chính xác vị trí bắt đầu giọng hát
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

class ImprovedSmartVoiceDetector:
    """Improved Smart Voice Detector - Phát hiện chính xác vị trí bắt đầu giọng hát"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
    
    def detect_voice_activity(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity với độ chính xác cao"""
        try:
            logger.info("🧠 Improved Smart Voice Detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Bước 1: Phân tích baseline (đoạn đầu file)
            baseline_features = self._analyze_baseline(audio, sr)
            
            # Bước 2: Phân tích energy pattern với baseline cải thiện
            energy_segments = self._detect_energy_with_improved_baseline(audio, sr, baseline_features)
            
            # Bước 3: Phân tích spectral pattern với baseline cải thiện
            spectral_segments = self._detect_spectral_with_improved_baseline(audio, sr, baseline_features)
            
            # Bước 4: Phân tích harmonic pattern cải thiện
            harmonic_segments = self._detect_harmonic_pattern_improved(audio, sr)
            
            # Bước 5: Phân tích voice characteristics cải thiện
            voice_segments = self._detect_voice_characteristics_improved(audio, sr)
            
            # Bước 6: Kết hợp và lọc kết quả thông minh
            combined_segments = self._smart_combine_segments_improved(
                energy_segments, spectral_segments, harmonic_segments, voice_segments, baseline_features
            )
            
            logger.info(f"✅ Improved Smart voice detection: {len(combined_segments)} segments")
            return combined_segments
            
        except Exception as e:
            logger.error(f"❌ Improved Smart voice detection failed: {e}")
            return []
    
    def _analyze_baseline(self, audio: np.ndarray, sr: int) -> Dict:
        """Phân tích baseline của file (đoạn đầu) - cải thiện"""
        try:
            # Phân tích 3 giây đầu (tăng từ 2s)
            baseline_length = min(3 * sr, len(audio))
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
            
            logger.info(f"📊 Improved Baseline analysis:")
            logger.info(f"   RMS: {baseline_features['rms_mean']:.4f} ± {baseline_features['rms_std']:.4f}")
            logger.info(f"   Centroid: {baseline_features['centroid_mean']:.1f} ± {baseline_features['centroid_std']:.1f} Hz")
            logger.info(f"   Rolloff: {baseline_features['rolloff_mean']:.1f} ± {baseline_features['rolloff_std']:.1f} Hz")
            logger.info(f"   ZCR: {baseline_features['zcr_mean']:.4f} ± {baseline_features['zcr_std']:.4f}")
            
            return baseline_features
            
        except Exception as e:
            logger.warning(f"Baseline analysis failed: {e}")
            return {
                'rms_mean': 0.05, 'rms_std': 0.02,
                'centroid_mean': 1000, 'centroid_std': 200,
                'rolloff_mean': 1200, 'rolloff_std': 300,
                'zcr_mean': 0.08, 'zcr_std': 0.02
            }
    
    def _detect_energy_with_improved_baseline(self, audio: np.ndarray, sr: int, baseline: Dict) -> List[Dict]:
        """Phát hiện energy pattern với baseline cải thiện"""
        try:
            # Tính RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # Adaptive threshold dựa trên baseline - cải thiện
            rms_threshold = baseline['rms_mean'] + baseline['rms_std'] * 0.3  # Giảm threshold để phát hiện voice nhẹ
            
            # Tìm frames có energy cao
            voice_frames = []
            for i, energy in enumerate(rms):
                if energy > rms_threshold:
                    voice_frames.append(i)
            
            # Chuyển frames thành segments
            segments = self._frames_to_segments(voice_frames, sr)
            
            logger.info(f"🔋 Energy detection: {len(segments)} segments, threshold: {rms_threshold:.4f}")
            return segments
            
        except Exception as e:
            logger.warning(f"Energy detection failed: {e}")
            return []
    
    def _detect_spectral_with_improved_baseline(self, audio: np.ndarray, sr: int, baseline: Dict) -> List[Dict]:
        """Phát hiện spectral pattern với baseline cải thiện"""
        try:
            # Tính spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)[0]
            
            # Adaptive thresholds dựa trên baseline - cải thiện
            centroid_threshold = baseline['centroid_mean'] - baseline['centroid_std'] * 0.2  # Giảm threshold
            rolloff_threshold = baseline['rolloff_mean'] - baseline['rolloff_std'] * 0.2  # Giảm threshold
            
            # Tìm frames có spectral features phù hợp với voice
            voice_frames = []
            for i in range(len(spectral_centroids)):
                if (spectral_centroids[i] > centroid_threshold and 
                    spectral_rolloff[i] > rolloff_threshold):
                    voice_frames.append(i)
            
            # Chuyển frames thành segments
            segments = self._frames_to_segments(voice_frames, sr)
            
            logger.info(f"📊 Spectral detection: {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.warning(f"Spectral detection failed: {e}")
            return []
    
    def _detect_harmonic_pattern_improved(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện harmonic pattern cải thiện"""
        try:
            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Tính harmonic energy
            harmonic_rms = librosa.feature.rms(y=harmonic, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # Threshold cho harmonic energy
            harmonic_threshold = np.percentile(harmonic_rms, 25)  # Giảm threshold
            
            # Tìm frames có harmonic energy cao
            voice_frames = []
            for i, energy in enumerate(harmonic_rms):
                if energy > harmonic_threshold:
                    voice_frames.append(i)
            
            # Chuyển frames thành segments
            segments = self._frames_to_segments(voice_frames, sr)
            
            logger.info(f"🎵 Harmonic detection: {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.warning(f"Harmonic detection failed: {e}")
            return []
    
    def _detect_voice_characteristics_improved(self, audio: np.ndarray, sr: int) -> List[Dict]:
        """Phát hiện voice characteristics cải thiện"""
        try:
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, hop_length=self.hop_length)
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=self.hop_length)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # Adaptive thresholds
            mfcc_threshold = np.percentile(mfccs[0], 20)  # Giảm threshold
            contrast_threshold = np.percentile(spectral_contrast[0], 20)  # Giảm threshold
            zcr_threshold = np.percentile(zcr, 20)  # Giảm threshold
            
            # Tìm frames có voice characteristics
            voice_frames = []
            for i in range(len(mfccs[0])):
                if (mfccs[0][i] > mfcc_threshold and 
                    spectral_contrast[0][i] > contrast_threshold and
                    zcr[i] > zcr_threshold):
                    voice_frames.append(i)
            
            # Chuyển frames thành segments
            segments = self._frames_to_segments(voice_frames, sr)
            
            logger.info(f"🎤 Voice characteristics detection: {len(segments)} segments")
            return segments
            
        except Exception as e:
            logger.warning(f"Voice characteristics detection failed: {e}")
            return []
    
    def _smart_combine_segments_improved(self, energy_segments: List[Dict], spectral_segments: List[Dict], 
                                       harmonic_segments: List[Dict], voice_segments: List[Dict], 
                                       baseline: Dict) -> List[Dict]:
        """Kết hợp segments thông minh - cải thiện"""
        try:
            # Kết hợp tất cả segments
            all_segments = energy_segments + spectral_segments + harmonic_segments + voice_segments
            
            # Merge overlapping segments
            merged_segments = self._merge_overlapping_segments(all_segments)
            
            # Lọc segments dựa trên baseline - cải thiện
            filtered_segments = []
            for segment in merged_segments:
                # Kiểm tra duration
                duration = segment['end'] - segment['start']
                if duration < 0.5:  # Bỏ qua segments quá ngắn
                    continue
                
                # Kiểm tra vị trí - bỏ qua segments ở đầu file quá sớm
                if segment['start'] < 1.0:  # Bỏ qua segments ở đầu file
                    continue
                
                # Kiểm tra confidence
                if segment['confidence'] < 0.3:  # Giảm threshold confidence
                    continue
                
                filtered_segments.append(segment)
            
            # Sắp xếp theo thời gian
            filtered_segments.sort(key=lambda x: x['start'])
            
            logger.info(f"🔗 Smart combination: {len(filtered_segments)} segments after filtering")
            return filtered_segments
            
        except Exception as e:
            logger.warning(f"Smart combination failed: {e}")
            return []
    
    def _frames_to_segments(self, voice_frames: List[int], sr: int) -> List[Dict]:
        """Chuyển frames thành segments"""
        if not voice_frames:
            return []
        
        segments = []
        current_start_frame = voice_frames[0]
        current_end_frame = voice_frames[0]
        
        for i in range(1, len(voice_frames)):
            if voice_frames[i] == current_end_frame + 1:
                current_end_frame = voice_frames[i]
            else:
                segments.append({
                    'start': current_start_frame * self.hop_length / sr,
                    'end': (current_end_frame + 1) * self.hop_length / sr,
                    'confidence': 1.0,
                    'method': 'improved_smart_detection'
                })
                current_start_frame = voice_frames[i]
                current_end_frame = voice_frames[i]
        
        segments.append({
            'start': current_start_frame * self.hop_length / sr,
            'end': (current_end_frame + 1) * self.hop_length / sr,
            'confidence': 1.0,
            'method': 'improved_smart_detection'
        })
        
        return segments
    
    def _merge_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Merge overlapping segments"""
        if not segments:
            return []
        
        # Sắp xếp theo start time
        segments.sort(key=lambda x: x['start'])
        
        merged = []
        current_segment = segments[0]
        
        for i in range(1, len(segments)):
            next_segment = segments[i]
            
            # Nếu segments overlap hoặc gần nhau
            if next_segment['start'] - current_segment['end'] <= 0.5:
                current_segment['end'] = max(current_segment['end'], next_segment['end'])
                current_segment['confidence'] = max(current_segment['confidence'], next_segment['confidence'])
            else:
                merged.append(current_segment)
                current_segment = next_segment
        
        merged.append(current_segment)
        return merged
