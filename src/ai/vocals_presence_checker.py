"""
Vocals Presence Checker - Kiểm tra file có giọng hát hay không
"""

import librosa
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class VocalsPresenceChecker:
    """Kiểm tra xem file audio có giọng hát hay không"""
    
    def __init__(self):
        self.sr = 22050
        
    def check_vocals_presence(self, audio_path: str) -> Dict:
        """
        Kiểm tra file có giọng hát hay không
        
        Returns:
            Dict với:
                - has_vocals: bool
                - confidence: float (0-1)
                - method: str
                - details: dict
        """
        try:
            logger.info(f"🔍 Checking vocals presence in: {audio_path}")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            # Method 1: Harmonic-Percussive Separation
            logger.info("📊 Method 1: Analyzing harmonic-percussive separation...")
            method1_result = self._check_with_hpss(audio)
            logger.info(f"   HPSS Result: has_vocals={method1_result['has_vocals']}, confidence={method1_result['confidence']:.3f}")
            
            # Method 2: Spectral centroid analysis
            logger.info("📊 Method 2: Analyzing spectral centroid (frequency analysis)...")
            method2_result = self._check_with_spectral_centroid(audio)
            logger.info(f"   Spectral Result: has_vocals={method2_result['has_vocals']}, confidence={method2_result['confidence']:.3f}")
            
            # Method 3: Zero crossing rate analysis
            logger.info("📊 Method 3: Analyzing zero crossing rate...")
            method3_result = self._check_with_zcr(audio)
            logger.info(f"   ZCR Result: has_vocals={method3_result['has_vocals']}, confidence={method3_result['confidence']:.3f}")
            
            # Combine results
            has_vocals = self._combine_results(method1_result, method2_result, method3_result)
            
            # Calculate overall confidence
            confidence = np.mean([
                method1_result['confidence'],
                method2_result['confidence'],
                method3_result['confidence']
            ])
            
            # Detect voice segments
            voice_segments = []
            if has_vocals:
                voice_segments = self._detect_voice_segments(audio, sr)
            
            # Log final result with emphasis
            if has_vocals:
                print("\n" + "="*60)
                print("🎤 PHÁT HIỆN GIỌNG HÁT!")
                print(f"   Confidence: {confidence:.2%}")
                print(f"   HPSS confidence: {method1_result['confidence']:.2%}")
                print(f"   Spectral confidence: {method2_result['confidence']:.2%}")
                print(f"   ZCR confidence: {method3_result['confidence']:.2%}")
                
                # Log voice segments
                if voice_segments:
                    print(f"\n📊 PHÁT HIỆN {len(voice_segments)} ĐOẠN GIỌNG HÁT:")
                    for i, seg in enumerate(voice_segments, 1):
                        start_min = int(seg['start'] // 60)
                        start_sec = int(seg['start'] % 60)
                        end_min = int(seg['end'] // 60)
                        end_sec = int(seg['end'] % 60)
                        duration = seg['end'] - seg['start']
                        
                        print(f"   Đoạn {i}: {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d} "
                              f"(duration: {duration:.1f}s)")
                    
                    # Log first and last voice
                    first_seg = voice_segments[0]
                    last_seg = voice_segments[-1]
                    print(f"\n⏰ GIỌNG HÁT XUẤT HIỆN: {first_seg['start']:.2f}s "
                          f"({int(first_seg['start']//60):02d}:{int(first_seg['start']%60):02d})")
                    print(f"⏰ GIỌNG HÁT KẾT THÚC: {last_seg['end']:.2f}s "
                          f"({int(last_seg['end']//60):02d}:{int(last_seg['end']%60):02d})")
                print("="*60 + "\n")
            else:
                print("\n" + "="*60)
                print("🔇 KHÔNG PHÁT HIỆN GIỌNG HÁT")
                print(f"   Confidence: {confidence:.2%}")
                print(f"   File này có thể là instrumental/no vocals")
                print("="*60 + "\n")
            
            logger.info(f"✅ Vocals check complete: has_vocals={has_vocals}, confidence={confidence:.3f}")
            
            # Add voice segments to result
            result_dict = {
                'has_vocals': has_vocals,
                'confidence': confidence,
                'method': 'Multi-method analysis',
                'details': {
                    'hpss': method1_result,
                    'spectral_centroid': method2_result,
                    'zcr': method3_result
                }
            }
            
            if voice_segments:
                result_dict['voice_segments'] = voice_segments
                result_dict['first_voice_time'] = voice_segments[0]['start']
                result_dict['last_voice_time'] = voice_segments[-1]['end']
            
            return result_dict
            
        except Exception as e:
            logger.error(f"❌ Error checking vocals presence: {e}")
            return {
                'has_vocals': None,
                'confidence': 0.0,
                'method': 'Error',
                'error': str(e)
            }
    
    def _check_with_hpss(self, audio: np.ndarray) -> Dict:
        """Check using harmonic-percussive separation - improved for vocals + beat"""
        try:
            # Separate harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio, margin=4)
            
            # Calculate energy ratio
            harmonic_energy = np.sum(np.abs(harmonic) ** 2)
            percussive_energy = np.sum(np.abs(percussive) ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            if total_energy == 0:
                return {'has_vocals': False, 'confidence': 0.0}
            
            harmonic_ratio = harmonic_energy / total_energy
            
            # For files with both vocals and beat:
            # - Pure beat/instrumental: harmonic ratio typically 0.2-0.5
            # - Vocals + beat: harmonic ratio typically 0.4-0.8 (higher because vocals add harmonic content)
            # - Pure vocals: harmonic ratio typically 0.6-0.9
            
            # Lower threshold to detect vocals even when beat is present
            has_vocals = harmonic_ratio > 0.35  # Adjusted from 0.3
            confidence = min(harmonic_ratio * 1.8, 1.0)  # Adjusted normalization
            
            return {
                'has_vocals': has_vocals,
                'confidence': confidence,
                'harmonic_ratio': harmonic_ratio,
                'percussive_ratio': percussive_energy / total_energy
            }
        except Exception as e:
            logger.warning(f"HPSS check failed: {e}")
            return {'has_vocals': False, 'confidence': 0.0, 'error': str(e)}
    
    def _check_with_spectral_centroid(self, audio: np.ndarray) -> Dict:
        """Check using spectral centroid with vocal frequency analysis"""
        try:
            # Calculate spectral centroid frame by frame
            stft = librosa.stft(audio)
            spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(stft))
            
            # Mean centroid
            mean_centroid = np.mean(spectral_centroids)
            
            # Voice frequency range is typically 200-4000 Hz
            voice_low_bound = 200
            voice_high_bound = 4000
            
            # Check if centroid is in voice range
            has_vocals = voice_low_bound < mean_centroid < voice_high_bound
            
            # Additional: Analyze frame-by-frame to see vocal characteristic patterns
            # When vocals are present (even with beat), there should be frames with centroid in vocal range
            vocal_frame_count = sum(1 for c in spectral_centroids[0] 
                                    if voice_low_bound < c < voice_high_bound)
            total_frames = len(spectral_centroids[0])
            vocal_frame_ratio = vocal_frame_count / total_frames if total_frames > 0 else 0
            
            # Boost confidence if many frames show vocal characteristics
            if has_vocals and vocal_frame_ratio > 0.3:
                confidence = min(0.8 + vocal_frame_ratio * 0.2, 1.0)
            else:
                # Original confidence calculation
                ideal_centroid = 1000  # Typical voice centroid
                distance = abs(mean_centroid - ideal_centroid)
                max_distance = 3000
                confidence = max(0, 1 - distance / max_distance)
            
            return {
                'has_vocals': has_vocals,
                'confidence': confidence,
                'mean_centroid': mean_centroid,
                'vocal_frame_ratio': vocal_frame_ratio
            }
        except Exception as e:
            logger.warning(f"Spectral centroid check failed: {e}")
            return {'has_vocals': False, 'confidence': 0.0, 'error': str(e)}
    
    def _check_with_zcr(self, audio: np.ndarray) -> Dict:
        """Check using zero crossing rate with temporal pattern analysis"""
        try:
            # Calculate zero crossing rate frame by frame
            zcr = librosa.feature.zero_crossing_rate(audio)
            mean_zcr = np.mean(zcr)
            
            # Voice typically has ZCR in range 0.01-0.1
            # Beat alone typically has more constant, lower or higher ZCR
            # Vocals with beat: mixed pattern with varying ZCR
            
            # Analyze temporal pattern
            # Vocals have more variation in ZCR compared to pure instrumental
            zcr_std = np.std(zcr)
            
            # Voice typically has moderate ZCR with some variation
            has_vocals = 0.01 < mean_zcr < 0.15  # Slightly wider range
            
            # Confidence based on both mean and variation
            ideal_zcr = 0.04  # Typical voice ZCR when mixed with beat
            distance = abs(mean_zcr - ideal_zcr)
            max_distance = 0.15
            base_confidence = max(0, 1 - distance / max_distance)
            
            # Boost confidence if there's variation (indicates vocals + beat mix)
            if 0.005 < zcr_std < 0.05:
                confidence = min(base_confidence + 0.2, 1.0)
            else:
                confidence = base_confidence
            
            return {
                'has_vocals': has_vocals,
                'confidence': confidence,
                'mean_zcr': mean_zcr,
                'zcr_std': zcr_std
            }
        except Exception as e:
            logger.warning(f"ZCR check failed: {e}")
            return {'has_vocals': False, 'confidence': 0.0, 'error': str(e)}
    
    def _detect_voice_segments(self, audio: np.ndarray, sr: int) -> list:
        """Detect time segments where voice appears"""
        try:
            hop_length = 512
            frame_length = 2048
            
            # Extract features
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Improved adaptive thresholds - stricter for files with beat
            # Use higher percentile để tránh false positive từ beat
            rms_threshold = np.percentile(rms, 40)  # Increased from 30
            centroid_low = 200  # Increased from 150 - vocals typically higher
            centroid_high = 4000  # Decreased from 4500 - more focused on vocals
            zcr_low = 0.008  # Increased from 0.005 - stricter lower bound
            zcr_high = 0.12  # Decreased from 0.15 - stricter upper bound
            
            # Additional: Check RMS change (vocals have more variation than steady beat)
            rms_change = np.abs(np.diff(rms))
            rms_change_threshold = np.percentile(rms_change, 50) if len(rms_change) > 0 else 0  # Median change
            
            # Detect voice frames with stricter conditions
            voice_frames = []
            for i in range(len(rms)):
                rms_ok = rms[i] > rms_threshold
                centroid_ok = centroid_low < spectral_centroids[i] < centroid_high
                zcr_ok = zcr_low < zcr[i] < zcr_high
                
                # Check RMS variation (vocals typically have more variation)
                rms_variation_ok = False
                if i > 0 and i-1 < len(rms_change):
                    rms_variation_ok = rms_change[i-1] > rms_change_threshold * 0.5
                
                # Need at least 3/4 conditions (stricter)
                conditions_met = sum([rms_ok, centroid_ok, zcr_ok, rms_variation_ok])
                if conditions_met >= 3:
                    voice_frames.append(i)
            
            if not voice_frames:
                return []
            
            # Convert frames to segments with improved logic
            segments = []
            current_start = voice_frames[0]
            current_end = voice_frames[0]
            
            # Increased gap threshold để merge các đoạn gần nhau
            gap_threshold = 5  # Frames (khoảng 0.1s với hop_length=512)
            
            for i in range(1, len(voice_frames)):
                frame_gap = voice_frames[i] - voice_frames[i-1]
                
                if frame_gap <= gap_threshold:
                    # Continuous or near-continuous: extend current segment
                    current_end = voice_frames[i]
                else:
                    # Gap lớn: end current segment and start new one
                    start_time = current_start * hop_length / sr
                    end_time = (current_end + 1) * hop_length / sr
                    
                    # Increased minimum duration to reduce false positives
                    if end_time - start_time >= 1.0:  # Minimum 1 second
                        segments.append({'start': start_time, 'end': end_time})
                    
                    current_start = voice_frames[i]
                    current_end = voice_frames[i]
            
            # Add last segment
            start_time = current_start * hop_length / sr
            end_time = (current_end + 1) * hop_length / sr
            if end_time - start_time >= 1.0:  # Minimum 1 second
                segments.append({'start': start_time, 'end': end_time})
            
            return segments
            
        except Exception as e:
            logger.warning(f"Voice segments detection failed: {e}")
            return []
    
    def _combine_results(self, result1: Dict, result2: Dict, result3: Dict) -> bool:
        """Combine results from multiple methods"""
        # If any method is very confident, use its result
        # Otherwise use majority vote
        
        if result1['confidence'] > 0.7:
            return result1['has_vocals']
        elif result2['confidence'] > 0.7:
            return result2['has_vocals']
        elif result3['confidence'] > 0.7:
            return result3['has_vocals']
        
        # Majority vote with confidence weighting
        votes = [result1['has_vocals'], result2['has_vocals'], result3['has_vocals']]
        confidences = [result1['confidence'], result2['confidence'], result3['confidence']]
        
        weighted_sum = sum(vote * conf for vote, conf in zip(votes, confidences))
        total_weight = sum(confidences)
        
        if total_weight > 0:
            return weighted_sum / total_weight > 0.5
        else:
            return False
    
    def check_vocals_in_separated_file(self, vocals_path: str, original_path: str) -> Dict:
        """
        Kiểm tra file vocals sau khi tách có thực sự chứa giọng hát không
        So sánh với file gốc để xác định
        """
        try:
            logger.info(f"🔍 Checking separated vocals file: {vocals_path}")
            
            # Check vocals file
            vocals_result = self.check_vocals_presence(vocals_path)
            
            # Check original file
            original_result = self.check_vocals_presence(original_path)
            
            # If original has vocals but separated vocals doesn't, likely error
            if original_result['has_vocals'] and not vocals_result['has_vocals']:
                logger.warning("⚠️ Vocals were detected in original but not in separated file")
                return {
                    'has_vocals': False,
                    'confidence': vocals_result['confidence'],
                    'reason': 'Separation may have failed - no vocals in separated file',
                    'original_has_vocals': True
                }
            
            return vocals_result
            
        except Exception as e:
            logger.error(f"❌ Error checking separated vocals: {e}")
            return {
                'has_vocals': None,
                'confidence': 0.0,
                'error': str(e)
            }

