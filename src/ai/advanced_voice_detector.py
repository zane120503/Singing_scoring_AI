#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Voice Detector sử dụng pyannote.audio và các model mạnh
"""

import os
import sys
import numpy as np
import librosa
import torch
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

logger = logging.getLogger(__name__)

class AdvancedVoiceDetector:
    """Advanced Voice Detector sử dụng pyannote.audio và các model mạnh"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.models_loaded = False
        self.vad_model = None
        self.diarization_model = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load các model mạnh cho voice detection"""
        try:
            logger.info("🔄 Loading advanced voice detection models...")
            
            # Try to load pyannote.audio
            try:
                from pyannote.audio import Pipeline
                from pyannote.audio.pipelines import VoiceActivityDetection
                from pyannote.audio.models import SegmentationModel
                
                # Load VAD pipeline
                self.vad_pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
                
                # Load diarization pipeline (for speaker detection)
                self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
                
                logger.info("✅ pyannote.audio models loaded successfully")
                self.models_loaded = True
                
            except ImportError:
                logger.warning("⚠️ pyannote.audio not available, using fallback methods")
                self._load_fallback_models()
            
            # Try to load webrtcvad
            try:
                import webrtcvad
                self.webrtcvad = webrtcvad.Vad(2)  # Aggressiveness level 2
                logger.info("✅ WebRTC VAD loaded")
            except ImportError:
                logger.warning("⚠️ webrtcvad not available")
                self.webrtcvad = None
            
            # Try to load silero-vad
            try:
                import torch
                self.silero_vad_model, self.silero_utils = torch.hub.load(
                    repo_or_dir='silero-models',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                logger.info("✅ Silero VAD loaded")
            except Exception as e:
                logger.warning(f"⚠️ Silero VAD not available: {e}")
                self.silero_vad_model = None
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self._load_fallback_models()
    
    def _load_fallback_models(self):
        """Load fallback models khi pyannote không có sẵn"""
        try:
            logger.info("🔄 Loading fallback voice detection models...")
            
            # Use librosa-based detection as fallback
            self.fallback_detector = True
            logger.info("✅ Fallback models loaded")
            
        except Exception as e:
            logger.error(f"❌ Error loading fallback models: {e}")
    
    def detect_voice_activity_pyannote(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity sử dụng pyannote.audio"""
        try:
            if not self.models_loaded:
                logger.warning("⚠️ pyannote models not loaded, using fallback")
                return self.detect_voice_activity_fallback(audio_path)
            
            logger.info("🎤 Using pyannote.audio for voice detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Apply VAD pipeline
            vad_result = self.vad_pipeline(audio_path)
            
            # Convert to segments
            segments = []
            for segment in vad_result.get_timeline():
                segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'confidence': 0.9,  # pyannote confidence
                    'method': 'pyannote_vad'
                })
            
            logger.info(f"✅ pyannote detected {len(segments)} voice segments")
            return segments
            
        except Exception as e:
            logger.error(f"❌ pyannote voice detection failed: {e}")
            return self.detect_voice_activity_fallback(audio_path)
    
    def detect_voice_activity_silero(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity sử dụng Silero VAD"""
        try:
            if self.silero_vad_model is None:
                logger.warning("⚠️ Silero VAD not available, using fallback")
                return self.detect_voice_activity_fallback(audio_path)
            
            logger.info("🎤 Using Silero VAD for voice detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)  # Silero expects 16kHz
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Get speech timestamps
            speech_timestamps = self.silero_utils[0](audio_tensor, self.silero_vad_model, sampling_rate=16000)
            
            # Convert to segments
            segments = []
            for timestamp in speech_timestamps:
                segments.append({
                    'start': timestamp['start'],
                    'end': timestamp['end'],
                    'confidence': 0.95,  # Silero confidence
                    'method': 'silero_vad'
                })
            
            logger.info(f"✅ Silero detected {len(segments)} voice segments")
            return segments
            
        except Exception as e:
            logger.error(f"❌ Silero voice detection failed: {e}")
            return self.detect_voice_activity_fallback(audio_path)
    
    def detect_voice_activity_webrtc(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity sử dụng WebRTC VAD - cải thiện"""
        try:
            if self.webrtcvad is None:
                logger.warning("⚠️ WebRTC VAD not available, using fallback")
                return self.detect_voice_activity_fallback(audio_path)
            
            logger.info("🎤 Using WebRTC VAD for voice detection...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000)  # WebRTC expects 16kHz
            
            # Convert to 16-bit PCM
            audio_16bit = (audio * 32767).astype(np.int16)
            
            # Frame size for WebRTC VAD (10ms, 20ms, or 30ms)
            frame_size = int(0.02 * sr)  # 20ms
            
            segments = []
            current_segment_start = None
            
            for i in range(0, len(audio_16bit) - frame_size, frame_size):
                frame = audio_16bit[i:i + frame_size]
                
                # Check if frame contains speech
                is_speech = self.webrtcvad.is_speech(frame.tobytes(), sr)
                
                if is_speech and current_segment_start is None:
                    # Start of speech segment
                    current_segment_start = i / sr
                elif not is_speech and current_segment_start is not None:
                    # End of speech segment
                    segments.append({
                        'start': current_segment_start,
                        'end': i / sr,
                        'confidence': 0.85,
                        'method': 'webrtc_vad'
                    })
                    current_segment_start = None
            
            # Handle case where audio ends during speech
            if current_segment_start is not None:
                segments.append({
                    'start': current_segment_start,
                    'end': len(audio_16bit) / sr,
                    'confidence': 0.85,
                    'method': 'webrtc_vad'
                })
            
            # Lọc bỏ các segments quá ngắn ở đầu file (có thể là noise)
            filtered_segments = []
            for segment in segments:
                # Bỏ qua segments ở đầu file quá ngắn (dưới 1.0s) và ở đầu file (dưới 5.0s)
                duration = segment['end'] - segment['start']
                if segment['start'] < 5.0 and duration < 1.0:
                    continue
                filtered_segments.append(segment)
            
            logger.info(f"✅ WebRTC detected {len(filtered_segments)} voice segments (filtered from {len(segments)})")
            return filtered_segments
            
        except Exception as e:
            logger.error(f"❌ WebRTC voice detection failed: {e}")
            return self.detect_voice_activity_fallback(audio_path)
    
    def detect_voice_activity_fallback(self, audio_path: str) -> List[Dict]:
        """Fallback voice detection sử dụng librosa"""
        try:
            logger.info("🎤 Using fallback voice detection...")
            
            # Import fallback detector
            from ai.voice_activity_detector import VoiceActivityDetector
            
            detector = VoiceActivityDetector(self.sr)
            segments = detector.detect_voice_activity(audio_path, method="combined")
            
            # Update method name
            for segment in segments:
                segment['method'] = 'fallback_librosa'
            
            logger.info(f"✅ Fallback detected {len(segments)} voice segments")
            return segments
            
        except Exception as e:
            logger.error(f"❌ Fallback voice detection failed: {e}")
            return []
    
    def detect_voice_activity_advanced(self, audio_path: str, method: str = "auto") -> List[Dict]:
        """Phát hiện voice activity với phương pháp nâng cao"""
        try:
            logger.info(f"🎤 Advanced voice detection with method: {method}")
            
            if method == "auto":
                # Try methods in order of preference - ưu tiên Smart Voice Detector
                methods = ["smart", "pyannote", "silero", "webrtc", "fallback"]
            else:
                methods = [method]
            
            for method_name in methods:
                try:
                    if method_name == "smart":
                        segments = self.detect_voice_activity_smart(audio_path)
                    elif method_name == "pyannote":
                        segments = self.detect_voice_activity_pyannote(audio_path)
                    elif method_name == "silero":
                        segments = self.detect_voice_activity_silero(audio_path)
                    elif method_name == "webrtc":
                        segments = self.detect_voice_activity_webrtc(audio_path)
                    elif method_name == "fallback":
                        segments = self.detect_voice_activity_fallback(audio_path)
                    else:
                        continue
                    
                    if segments:
                        logger.info(f"✅ Successfully detected voice using {method_name}")
                        return segments
                        
                except Exception as e:
                    logger.warning(f"⚠️ Method {method_name} failed: {e}")
                    continue
            
            logger.error("❌ All voice detection methods failed")
            return []
            
        except Exception as e:
            logger.error(f"❌ Advanced voice detection failed: {e}")
            return []
    
    def detect_voice_activity_smart(self, audio_path: str) -> List[Dict]:
        """Phát hiện voice activity sử dụng Smart Voice Detector"""
        try:
            logger.info("🧠 Using Smart Voice Detector...")
            
            # Import Smart Voice Detector
            from ai.smart_voice_detector import SmartVoiceDetector
            
            smart_detector = SmartVoiceDetector(self.sr)
            segments = smart_detector.detect_voice_activity(audio_path)
            
            # Update method name
            for segment in segments:
                segment['method'] = 'smart_vad'
            
            logger.info(f"✅ Smart VAD detected {len(segments)} voice segments")
            return segments
            
        except Exception as e:
            logger.error(f"❌ Smart voice detection failed: {e}")
            return []
    
    def find_first_voice_segment(self, audio_path: str, min_duration: float = 1.0) -> Dict:
        """Tìm đoạn voice đầu tiên phù hợp"""
        try:
            segments = self.detect_voice_activity_advanced(audio_path)
            
            for segment in segments:
                duration = segment["end"] - segment["start"]
                if duration >= min_duration:
                    logger.info(f"🎯 Found first suitable voice segment: {segment['start']:.2f}s - {segment['end']:.2f}s")
                    return segment
            
            logger.warning("⚠️ No suitable voice segment found")
            return {"start": 0, "end": 0, "confidence": 0}
            
        except Exception as e:
            logger.error(f"❌ Error finding first voice segment: {e}")
            return {"start": 0, "end": 0, "confidence": 0}
    
    def get_voice_analysis(self, audio_path: str) -> Dict:
        """Phân tích voice activity chi tiết"""
        try:
            logger.info("🔍 Getting detailed voice analysis...")
            
            # Get voice segments
            segments = self.detect_voice_activity_advanced(audio_path)
            
            # Get audio info
            audio, sr = librosa.load(audio_path, sr=self.sr)
            duration = len(audio) / sr
            
            # Calculate statistics
            voice_duration = sum(seg["end"] - seg["start"] for seg in segments)
            voice_ratio = voice_duration / duration if duration > 0 else 0
            
            analysis = {
                "audio_path": audio_path,
                "total_duration": duration,
                "voice_segments": segments,
                "voice_duration": voice_duration,
                "voice_ratio": voice_ratio,
                "voice_count": len(segments),
                "first_voice": self.find_first_voice_segment(audio_path)
            }
            
            logger.info(f"📊 Voice analysis completed:")
            logger.info(f"   Total duration: {duration:.2f}s")
            logger.info(f"   Voice duration: {voice_duration:.2f}s")
            logger.info(f"   Voice ratio: {voice_ratio:.1%}")
            logger.info(f"   Voice segments: {len(segments)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error getting voice analysis: {e}")
            return {"error": str(e)}
