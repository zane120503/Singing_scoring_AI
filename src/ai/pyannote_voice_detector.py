#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyAnnote Voice Detector - Sử dụng pyannote/voice-activity-detection để phát hiện giọng hát
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

class PyAnnoteVoiceDetector:
    """PyAnnote Voice Detector - Sử dụng pyannote/voice-activity-detection để phát hiện giọng hát"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.frame_length = 2048
        self.hop_length = 512
        self.pipeline = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load PyAnnote voice activity detection pipeline"""
        try:
            logger.info("Loading PyAnnote voice activity detection pipeline...")
            
            # Import pyannote.audio
            from pyannote.audio import Pipeline
            
            # Load voice activity detection pipeline
            # Sử dụng model voice-activity-detection thay vì speaker-diarization
            self.pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
            
            logger.info("✅ PyAnnote voice activity detection pipeline loaded successfully!")
            
        except ImportError as e:
            logger.warning(f"⚠️ PyAnnote.audio not available: {e}")
            self.pipeline = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to load PyAnnote pipeline: {e}")
            self.pipeline = None
    
    def detect_voice_activity(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity sử dụng PyAnnote voice-activity-detection"""
        try:
            logger.info("🎯 PyAnnote Voice Activity Detection...")
            
            if self.pipeline is None:
                logger.warning("⚠️ PyAnnote pipeline not available, using fallback method")
                return self._fallback_voice_detection(audio_path)
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Sử dụng PyAnnote pipeline để phát hiện voice activity
            logger.info("🔍 Running PyAnnote voice activity detection...")
            
            # Tạo temporary audio file cho PyAnnote
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio, sr)
                
                try:
                    # Chạy PyAnnote voice activity detection
                    vad_result = self.pipeline(tmp_file.name)
                    
                    # Chuyển đổi kết quả thành segments
                    segments = []
                    for segment in vad_result.get_timeline():
                        segments.append({
                            'start': segment.start,
                            'end': segment.end,
                            'confidence': 0.9,  # PyAnnote không cung cấp confidence cho VAD
                            'method': 'pyannote_vad'
                        })
                    
                    logger.info(f"✅ PyAnnote detected {len(segments)} voice segments")
                    
                    # Tìm vị trí giọng hát đầu tiên
                    if segments:
                        first_voice_segment = segments[0]
                        logger.info(f"🎯 First voice segment: {first_voice_segment['start']:.2f}s - {first_voice_segment['end']:.2f}s")
                        
                        # Trả về segment đầu tiên
                        return [first_voice_segment]
                    else:
                        logger.warning("⚠️ PyAnnote không phát hiện được voice segments")
                        return self._fallback_voice_detection(audio_path)
                
                finally:
                    # Cleanup temporary file
                    os.unlink(tmp_file.name)
            
        except Exception as e:
            logger.error(f"❌ PyAnnote voice detection failed: {e}")
            return self._fallback_voice_detection(audio_path)
    
    def _fallback_voice_detection(self, audio_path: str) -> List[Dict]:
        """Fallback voice detection method"""
        try:
            logger.info("🔄 Using fallback voice detection method...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Tính RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # Tính spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=self.hop_length)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=self.hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # Chuyển frames thành thời gian
            times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)
            
            logger.info(f"🔍 Fallback analysis: {len(rms)} frames")
            
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
            
            # Tìm vị trí giọng hát với logic đơn giản
            voice_start_candidates = []
            
            # Phân tích 120 giây đầu
            for i in range(min(120, len(rms))):
                start_time = i * self.hop_length / sr
                segment_rms = rms[i]
                segment_centroid = spectral_centroids[i]
                segment_rolloff = spectral_rolloff[i]
                segment_zcr = zcr[i]
                
                # Logic đơn giản: tìm vị trí có energy cao và spectral features khác biệt
                has_voice = (
                    segment_rms > baseline_rms * 1.1 and  # Energy cao hơn baseline 10%
                    segment_centroid > baseline_centroid * 0.7 and segment_centroid < 4000 and  # Voice frequency range
                    segment_rolloff > baseline_rolloff * 0.7 and  # Có high frequency content
                    segment_zcr > baseline_zcr * 0.5 and segment_zcr < baseline_zcr * 2.0  # ZCR trong khoảng hợp lý
                )
                
                if has_voice:
                    confidence = self._calculate_fallback_confidence(segment_rms, segment_centroid, baseline_rms, baseline_centroid)
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
                
                logger.info(f"\n🎯 Fallback voice detection result:")
                logger.info(f"   Voice starts at: {best_candidate['time']:.2f}s")
                logger.info(f"   Confidence: {best_candidate['confidence']:.3f}")
                logger.info(f"   RMS: {best_candidate['rms']:.4f}")
                logger.info(f"   Centroid: {best_candidate['centroid']:.1f}Hz")
                logger.info(f"   Rolloff: {best_candidate['rolloff']:.1f}Hz")
                logger.info(f"   ZCR: {best_candidate['zcr']:.4f}")
                
                # Tạo segment từ vị trí tìm được
                segments = [{
                    'start': best_candidate['time'],
                    'end': len(audio) / sr,  # Đến cuối file
                    'confidence': best_candidate['confidence'],
                    'method': 'fallback_detection'
                }]
                
                return segments
            else:
                logger.warning("⚠️ Fallback method không tìm thấy vị trí giọng hát!")
                return []
        
        except Exception as e:
            logger.error(f"❌ Fallback voice detection failed: {e}")
            return []
    
    def _calculate_fallback_confidence(self, rms: float, centroid: float, baseline_rms: float, baseline_centroid: float) -> float:
        """Tính confidence score cho fallback method"""
        try:
            # Normalize các features dựa trên baseline
            rms_score = min((rms - baseline_rms) / baseline_rms, 1.0)  # So với baseline
            centroid_score = 1.0 - abs(centroid - baseline_centroid) / baseline_centroid  # So với baseline
            
            # Weighted average
            confidence = (
                max(rms_score, 0) * 0.6 +  # Tăng weight cho RMS
                max(centroid_score, 0) * 0.4
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
                logger.info(f"✅ Found first suitable PyAnnote voice segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
                return segment
        logger.warning("⚠️ No suitable PyAnnote voice segment found.")
        return None
