import librosa
import numpy as np
from typing import Dict, Tuple, List
import warnings
import logging
import subprocess
import os
import tempfile
import shutil

warnings.filterwarnings("ignore")

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedKeyDetector:
    """Advanced Key Detection using Essentia and improved algorithms"""
    
    def __init__(self):
        self.essentia_available = False
        self.docker_available = False
        self._initialize_essentia()
        
        # Define key names
        self.key_names = [
            'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
        ]
        
        # Krumhansl-Schmuckler key profiles
        self.major_profile = np.array([
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
        ])
        
        self.minor_profile = np.array([
            6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
        ])
    
    def _initialize_essentia(self):
        """Initialize Essentia if available"""
        try:
            logger.info("🔄 Đang khởi tạo Essentia AI...")
            import essentia.standard as es
            self.essentia = es
            self.essentia_available = True
            logger.info("✅ Essentia AI initialized successfully!")
            logger.info("🎵 Essentia KeyExtractor sẵn sàng!")
        except ImportError as e:
            logger.warning(f"⚠️ Essentia không khả dụng: {e}")
            logger.info("🔄 Thử Docker Essentia...")
            self._check_docker_essentia()
    
    def _check_docker_essentia(self):
        """Check if Docker Essentia is available"""
        try:
            # Test Docker Essentia
            result = subprocess.run("docker run --rm essentia-karaoke python3 -c \"import essentia.standard as es; print('OK')\"", 
                                  shell=True, capture_output=True, text=True)
            if result.returncode == 0 and "OK" in result.stdout:
                self.docker_available = True
                logger.info("✅ Docker Essentia AI sẵn sàng!")
            else:
                logger.warning("⚠️ Docker Essentia không khả dụng")
        except Exception as e:
            logger.warning(f"⚠️ Docker Essentia check failed: {e}")
    
    def detect_key(self, audio_path: str, audio_type: str = "general") -> Dict:
        """Detect key of audio file with audio type optimization"""
        try:
            logger.info(f"🎹 Bắt đầu phát hiện phím từ file: {audio_path}")
            logger.info(f"📁 Audio type: {audio_type}")
            
            # Load audio
            logger.info("📥 Đang tải file âm thanh...")
            audio, sr = librosa.load(audio_path, sr=22050)
            logger.info(f"✅ Đã tải audio: {len(audio)} samples, {sr} Hz")
            
            # Preprocessing based on audio type
            if audio_type == "beat":
                audio = self._preprocess_beat_audio(audio, sr)
                logger.info("🔧 Applied beat-specific preprocessing")
            elif audio_type == "vocals":
                audio = self._preprocess_vocals_audio(audio, sr)
                logger.info("🔧 Applied vocals-specific preprocessing")
            
            # Use hybrid detector for better accuracy
            logger.info("🔬 Sử dụng Hybrid Key Detector...")
            key_info = self._detect_with_hybrid(audio_path, audio, sr, audio_type)
            logger.info("✅ Hybrid key detection hoàn thành!")
            
            logger.info(f"🎵 Kết quả: {key_info['key']} {key_info['scale']} (confidence: {key_info['confidence']:.3f})")
            return key_info
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi phát hiện phím: {e}")
            return self._get_default_key()
    
    def _preprocess_beat_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess beat audio for better key detection"""
        try:
            # Remove silence at beginning and end
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=15)  # More aggressive trimming
            
            # Normalize audio
            audio_normalized = librosa.util.normalize(audio_trimmed)
            
            # Apply stronger high-pass filter to remove low-frequency noise
            from scipy import signal
            nyquist = sr // 2
            high_pass_freq = 120  # Increased from 80Hz to 120Hz
            b, a = signal.butter(6, high_pass_freq / nyquist, btype='high')  # Increased order
            audio_filtered = signal.filtfilt(b, a, audio_normalized)
            
            # Additional preprocessing: focus on mid-range frequencies
            # Apply band-pass filter to focus on musical frequencies
            low_freq = 200   # Hz
            high_freq = 4000  # Hz
            b2, a2 = signal.butter(4, [low_freq / nyquist, high_freq / nyquist], btype='band')
            audio_bandpass = signal.filtfilt(b2, a2, audio_filtered)
            
            return audio_bandpass
            
        except Exception as e:
            logger.warning(f"Beat preprocessing failed: {e}")
            return audio
    
    def _preprocess_vocals_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess vocals audio for better key detection - lighter processing"""
        try:
            # Remove silence with moderate trimming (less aggressive)
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
            
            # Normalize
            audio_normalized = librosa.util.normalize(audio_trimmed)
            
            # Apply harmonic-percussive separation to isolate harmonic content
            audio_harmonic, audio_percussive = librosa.effects.hpss(audio_normalized, margin=4)
            
            # Use mainly harmonic component but keep some percussive for rhythm
            audio_processed = audio_harmonic + audio_percussive * 0.3
            
            # Apply gentle low-pass filter to focus on musical frequencies
            from scipy import signal
            nyquist = sr // 2
            low_pass_freq = 8000  # Higher frequency to preserve more harmonics
            b, a = signal.butter(4, low_pass_freq / nyquist, btype='low')  # Lower order filter
            audio_filtered = signal.filtfilt(b, a, audio_processed)
            
            # Apply lighter spectral gating to reduce noise without losing key information
            audio_gated = self._apply_light_spectral_gating(audio_filtered, sr)
            
            return audio_gated
            
        except Exception as e:
            logger.warning(f"Vocals preprocessing failed: {e}")
            return audio
    
    def _apply_spectral_gating(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply spectral gating to reduce noise in vocals"""
        try:
            # Compute STFT
            stft = librosa.stft(audio, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply spectral gating threshold
            threshold = np.percentile(magnitude, 85)  # Keep top 15% of energy
            magnitude_gated = np.where(magnitude > threshold, magnitude, magnitude * 0.1)
            
            # Reconstruct audio
            stft_gated = magnitude_gated * np.exp(1j * phase)
            audio_gated = librosa.istft(stft_gated, hop_length=512)
            
            return audio_gated
            
        except Exception as e:
            logger.warning(f"Spectral gating failed: {e}")
            return audio
    
    def _apply_light_spectral_gating(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply lighter spectral gating to reduce noise without losing key information"""
        try:
            # Compute STFT
            stft = librosa.stft(audio, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply lighter spectral gating threshold
            threshold = np.percentile(magnitude, 75)  # Keep top 25% of energy (less aggressive)
            magnitude_gated = np.where(magnitude > threshold, magnitude, magnitude * 0.3)
            
            # Reconstruct audio
            stft_gated = magnitude_gated * np.exp(1j * phase)
            audio_gated = librosa.istft(stft_gated, hop_length=512)
            
            return audio_gated
            
        except Exception as e:
            logger.warning(f"Light spectral gating failed: {e}")
            return audio
    
    def _detect_with_hybrid(self, audio_path: str, audio: np.ndarray, sr: int, audio_type: str = "unknown") -> Dict:
        """Hybrid key detection combining multiple methods"""
        try:
            results = []
            
            # Adjust weights based on audio type
            if audio_type == "vocals":
                # For vocals, prioritize traditional methods that work better
                essentia_weight = 0.1  # Very low weight for AI (it's often wrong for vocals)
                traditional_weight = 0.7  # Highest weight for traditional
                vocals_weight = 0.5  # Medium weight for vocals-specific
                chroma_weight = 0.6  # High weight for chroma
            else:
                # For other types, use balanced weights
                essentia_weight = 0.3
                traditional_weight = 0.3
                vocals_weight = 0.3
                chroma_weight = 0.3
            
            # Method 1: Docker Essentia AI (if available) - Skip for vocals
            if self.docker_available and audio_type != "vocals":
                try:
                    essentia_result = self._detect_with_docker_essentia(audio_path)
                    if essentia_result:
                        results.append({
                            'key': essentia_result['key'],
                            'scale': essentia_result['scale'],
                            'confidence': essentia_result['confidence'],
                            'method': 'Docker Essentia AI',
                            'weight': essentia_weight
                        })
                except Exception as e:
                    logger.warning(f"Docker Essentia failed: {e}")
            
            # Method 2: Traditional librosa + Krumhansl
            try:
                traditional_result = self._detect_with_improved_traditional(audio, sr)
                if traditional_result:
                        results.append({
                            'key': traditional_result['key'],
                            'scale': traditional_result['scale'],
                            'confidence': traditional_result['confidence'],
                            'method': 'Traditional Librosa',
                            'weight': traditional_weight
                        })
            except Exception as e:
                logger.warning(f"Traditional method failed: {e}")
            
            # Method 3: Vocals-specific key detection
            try:
                vocals_result = self._detect_with_vocals_specific(audio, sr)
                if vocals_result:
                    results.append({
                        'key': vocals_result['key'],
                        'scale': vocals_result['scale'],
                        'confidence': vocals_result['confidence'],
                        'method': 'Vocals-Specific Analysis',
                        'weight': vocals_weight
                    })
            except Exception as e:
                logger.warning(f"Vocals-specific method failed: {e}")
            
            # Method 4: Enhanced chroma analysis
            try:
                chroma_result = self._detect_with_enhanced_chroma(audio, sr)
                if chroma_result:
                        results.append({
                            'key': chroma_result['key'],
                            'scale': chroma_result['scale'],
                            'confidence': chroma_result['confidence'],
                            'method': 'Enhanced Chroma',
                            'weight': 0.4  # Increased weight
                        })
            except Exception as e:
                logger.warning(f"Enhanced chroma failed: {e}")
            
            # Method 5: Beat-specific harmonic analysis (if beat type)
            if audio_type == "beat":
                try:
                    beat_result = self._detect_with_beat_harmonic_analysis(audio, sr)
                    if beat_result:
                        results.append({
                            'key': beat_result['key'],
                            'scale': beat_result['scale'],
                            'confidence': beat_result['confidence'],
                            'method': 'Beat Harmonic Analysis',
                            'weight': 0.5  # High weight for beat-specific method
                        })
                except Exception as e:
                    logger.warning(f"Beat harmonic analysis failed: {e}")
            
            # Voting mechanism with weights
            if results:
                logger.info(f"🔬 Hybrid voting: {len(results)} methods")
                for result in results:
                    logger.info(f"   {result['method']}: {result['key']} {result['scale']} (conf: {result['confidence']:.3f})")
                
                # Weighted voting
                best_result = self._weighted_voting(results)
                logger.info(f"🏆 Best result: {best_result['method']} - {best_result['key']} {best_result['scale']}")
                
                return {
                    'key': best_result['key'],
                    'scale': best_result['scale'],
                    'confidence': best_result['confidence'],
                    'method': f"Hybrid ({best_result['method']})"
                }
            else:
                logger.warning("⚠️ All methods failed, using fallback")
                return self._detect_with_improved_traditional(audio, sr)
                
        except Exception as e:
            logger.error(f"❌ Hybrid detection failed: {e}")
            return self._detect_with_improved_traditional(audio, sr)
    
    def _detect_with_enhanced_chroma(self, audio: np.ndarray, sr: int) -> Dict:
        """Enhanced chroma-based key detection"""
        try:
            # Extract chroma with different parameters
            chroma1 = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)
            chroma2 = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=512)
            chroma3 = librosa.feature.chroma_cens(y=audio, sr=sr, hop_length=512)
            
            # Combine chroma features
            chroma_combined = np.mean([chroma1, chroma2, chroma3], axis=0)
            chroma_mean = np.mean(chroma_combined, axis=1)
            
            # Normalize
            chroma_mean = chroma_mean / np.sum(chroma_mean)
            
            # Calculate correlations with key profiles
            major_correlations = []
            minor_correlations = []
            
            for i in range(12):
                major_rotated = np.roll(self.major_profile, i)
                minor_rotated = np.roll(self.minor_profile, i)
                
                major_rotated = major_rotated / np.sum(major_rotated)
                minor_rotated = minor_rotated / np.sum(minor_rotated)
                
                major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
                minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
                
                major_correlations.append(major_corr)
                minor_correlations.append(minor_corr)
            
            # Find best matches
            major_max_idx = np.argmax(major_correlations)
            minor_max_idx = np.argmax(minor_correlations)
            
            major_max_corr = major_correlations[major_max_idx]
            minor_max_corr = minor_correlations[minor_max_idx]
            
            # Choose between major and minor
            if major_max_corr > minor_max_corr:
                key_name = self.key_names[major_max_idx]
                scale = 'major'
                confidence = major_max_corr
            else:
                key_name = self.key_names[minor_max_idx]
                scale = 'minor'
                confidence = minor_max_corr
            
            return {
                'key': key_name,
                'scale': scale,
                'confidence': confidence,
                'method': 'Enhanced Chroma'
            }
            
        except Exception as e:
            logger.error(f"Enhanced chroma detection failed: {e}")
            return None
    
    def _detect_with_beat_harmonic_analysis(self, audio: np.ndarray, sr: int) -> Dict:
        """Beat-specific harmonic analysis using HPSS"""
        try:
            # Separate harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio, margin=8)
            
            # Focus on harmonic component for key detection
            harmonic_audio = harmonic
            
            # Extract chroma features from harmonic component
            chroma_harmonic = librosa.feature.chroma_stft(y=harmonic_audio, sr=sr, hop_length=512)
            chroma_cqt_harmonic = librosa.feature.chroma_cqt(y=harmonic_audio, sr=sr, hop_length=512)
            
            # Combine harmonic chroma features
            chroma_combined = np.mean([chroma_harmonic, chroma_cqt_harmonic], axis=0)
            chroma_mean = np.mean(chroma_combined, axis=1)
            
            # Normalize
            chroma_mean = chroma_mean / np.sum(chroma_mean)
            
            # Calculate correlations with key profiles
            major_correlations = []
            minor_correlations = []
            
            for i in range(12):
                major_rotated = np.roll(self.major_profile, i)
                minor_rotated = np.roll(self.minor_profile, i)
                
                major_rotated = major_rotated / np.sum(major_rotated)
                minor_rotated = minor_rotated / np.sum(minor_rotated)
                
                major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
                minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
                
                major_correlations.append(major_corr)
                minor_correlations.append(minor_corr)
            
            # Find best matches
            major_max_idx = np.argmax(major_correlations)
            minor_max_idx = np.argmax(minor_correlations)
            
            major_max_corr = major_correlations[major_max_idx]
            minor_max_corr = minor_correlations[minor_max_idx]
            
            # Choose between major and minor
            if major_max_corr > minor_max_corr:
                key_name = self.key_names[major_max_idx]
                scale = 'major'
                confidence = major_max_corr
            else:
                key_name = self.key_names[minor_max_idx]
                scale = 'minor'
                confidence = minor_max_corr
            
            return {
                'key': key_name,
                'scale': scale,
                'confidence': confidence,
                'method': 'Beat Harmonic Analysis'
            }
            
        except Exception as e:
            logger.error(f"Beat harmonic analysis failed: {e}")
            return None
    
    def _detect_with_music21(self, audio_path: str) -> Dict:
        """Key detection using Music21"""
        try:
            import music21
            
            # Convert audio to Music21 stream
            stream = music21.converter.parse(audio_path)
            
            # Analyze key
            key = stream.analyze('key')
            
            return {
                'key': str(key.tonic),
                'scale': str(key.mode),
                'confidence': 0.7,  # Music21 doesn't provide confidence
                'method': 'Music21'
            }
            
        except ImportError:
            logger.warning("Music21 not available")
            return None
        except Exception as e:
            logger.warning(f"Music21 detection failed: {e}")
            return None
    
    def _weighted_voting(self, results: List[Dict]) -> Dict:
        """Weighted voting mechanism with consensus priority"""
        try:
            # Group results by key+scale
            key_groups = {}
            for result in results:
                key_scale = f"{result['key']} {result['scale']}"
                if key_scale not in key_groups:
                    key_groups[key_scale] = []
                key_groups[key_scale].append(result)
            
            # Calculate weighted scores with consensus bonus
            weighted_scores = {}
            for key_scale, group in key_groups.items():
                total_weight = 0
                weighted_confidence = 0
                
                for result in group:
                    weight = result.get('weight', 0.1)
                    total_weight += weight
                    weighted_confidence += result['confidence'] * weight
                
                if total_weight > 0:
                    # Add consensus bonus: more methods agreeing = higher score
                    consensus_bonus = len(group) * 0.3  # Increased to 0.3 per agreeing method
                    
                    # Special bonus for vocals-specific methods
                    vocals_bonus = 0
                    for result in group:
                        if 'Vocals-Specific' in result.get('method', '') or 'Docker Essentia' in result.get('method', ''):
                            vocals_bonus += 0.2
                    
                    weighted_scores[key_scale] = (weighted_confidence / total_weight) + consensus_bonus + vocals_bonus
            
            # Find best key
            if weighted_scores:
                best_key_scale = max(weighted_scores, key=weighted_scores.get)
                best_score = weighted_scores[best_key_scale]
                
                # Find the result with highest confidence for this key
                best_group = key_groups[best_key_scale]
                best_result = max(best_group, key=lambda x: x['confidence'])
                
                logger.info(f"🏆 Voting scores: {dict(weighted_scores)}")
                
                return {
                    'key': best_result['key'],
                    'scale': best_result['scale'],
                    'confidence': best_score,
                    'method': best_result['method']
                }
            else:
                # Fallback to highest confidence
                return max(results, key=lambda x: x['confidence'])
                
        except Exception as e:
            logger.error(f"Weighted voting failed: {e}")
            return max(results, key=lambda x: x['confidence'])
    
    def _detect_with_essentia(self, audio: np.ndarray, sr: int) -> Dict:
        """Detect key using Essentia"""
        try:
            logger.info("🔄 Đang sử dụng Essentia KeyExtractor...")
            
            # Create Essentia algorithms
            loader = self.essentia.MonoLoader()
            key_extractor = self.essentia.KeyExtractor()
            
            logger.info("🤖 Đang chạy Essentia AI KeyExtractor...")
            # Process audio
            key, scale, strength = key_extractor(audio)
            logger.info(f"✅ Essentia AI output: {key} {scale} (strength: {strength:.3f})")
            
            return {
                'key': key,
                'scale': scale,
                'confidence': strength,
                'method': 'Essentia AI'
            }
            
        except Exception as e:
            logger.error(f"❌ Essentia detection failed: {e}")
            logger.warning("⚠️ Chuyển sang phương pháp fallback...")
            return self._detect_with_improved_traditional(audio, sr)
    
    def _detect_with_docker_essentia(self, audio_path: str) -> Dict:
        """Detect key using Docker Essentia with improved accuracy"""
        try:
            logger.info("🐳 Đang sử dụng Docker Essentia KeyExtractor với độ chính xác cao...")
            
            # Copy to temp ASCII name to avoid Unicode/special chars issues
            temp_ascii_name = "temp_input.mp3"
            docker_path = f"/app/{temp_ascii_name}"
            copy_cmd = f'docker cp "{audio_path}" essentia-karaoke:{docker_path}'
            subprocess.run(copy_cmd, shell=True, check=True)
            
            # Run multiple key detections with different parameters for voting
            results = []
            
            # Method 1: Standard key detection
            cmd1 = f"docker exec essentia-karaoke python3 -c \"import essentia.standard as es; audio = es.MonoLoader(filename='{docker_path}')(); key, scale, strength = es.KeyExtractor()(audio); print(f'{{key}} {{scale}} {{strength}}')\""
            result1 = subprocess.run(cmd1, shell=True, capture_output=True, text=True)
            
            if result1.returncode == 0:
                parts1 = result1.stdout.strip().split()
                if len(parts1) >= 3:
                    results.append({
                        'key': parts1[0],
                        'scale': parts1[1],
                        'confidence': float(parts1[2]),
                        'method': 'Docker Essentia Standard'
                    })
            
            # Method 2: High resolution key detection
            cmd2 = f"docker exec essentia-karaoke python3 -c \"import essentia.standard as es; audio = es.MonoLoader(filename='{docker_path}', sampleRate=44100)(); key, scale, strength = es.KeyExtractor()(audio); print(f'{{key}} {{scale}} {{strength}}')\""
            result2 = subprocess.run(cmd2, shell=True, capture_output=True, text=True)
            
            if result2.returncode == 0:
                parts2 = result2.stdout.strip().split()
                if len(parts2) >= 3:
                    results.append({
                        'key': parts2[0],
                        'scale': parts2[1],
                        'confidence': float(parts2[2]),
                        'method': 'Docker Essentia High-Res'
                    })
            
            # Method 3: Multiple segments voting
            cmd3 = f"docker exec essentia-karaoke python3 -c \"import essentia.standard as es; import numpy as np; audio = es.MonoLoader(filename='{docker_path}')(); segments = [audio[i:i+len(audio)//3] for i in range(0, len(audio), len(audio)//3)]; keys = []; for seg in segments: key, scale, strength = es.KeyExtractor()(seg); keys.append((key, scale, strength)); from collections import Counter; most_common = Counter(keys).most_common(1)[0][0]; print(f'{{most_common[0]}} {{most_common[1]}} {{most_common[2]}}')\""
            result3 = subprocess.run(cmd3, shell=True, capture_output=True, text=True)
            
            if result3.returncode == 0:
                parts3 = result3.stdout.strip().split()
                if len(parts3) >= 3:
                    results.append({
                        'key': parts3[0],
                        'scale': parts3[1],
                        'confidence': float(parts3[2]),
                        'method': 'Docker Essentia Voting'
                    })
            
            # Voting mechanism: choose result with highest confidence
            if results:
                best_result = max(results, key=lambda x: x['confidence'])
                logger.info(f"✅ Docker Essentia AI output: {best_result['key']} {best_result['scale']} (strength: {best_result['confidence']:.3f})")
                logger.info(f"📊 Voting results: {len(results)} methods, best: {best_result['method']}")
                return {
                    'key': best_result['key'],
                    'scale': best_result['scale'],
                    'confidence': best_result['confidence'],
                    'method': 'Docker Essentia AI (Enhanced)'
                }
            
            logger.error("❌ Docker Essentia detection failed")
            return self._detect_with_improved_traditional(
                librosa.load(audio_path, sr=22050)[0], 22050
            )
            
        except Exception as e:
            logger.error(f"❌ Docker Essentia detection failed: {e}")
            logger.warning("⚠️ Chuyển sang phương pháp fallback...")
            return self._detect_with_improved_traditional(
                librosa.load(audio_path, sr=22050)[0], 22050
            )
    
    def _detect_with_improved_traditional(self, audio: np.ndarray, sr: int) -> Dict:
        """Improved traditional key detection"""
        try:
            # Extract chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # Compute mean chroma
            chroma_mean = np.mean(chroma, axis=1)
            
            # Normalize chroma
            chroma_mean = chroma_mean / np.sum(chroma_mean)
            
            # Calculate correlations with key profiles
            major_correlations = []
            minor_correlations = []
            
            for i in range(12):
                # Rotate profiles
                major_rotated = np.roll(self.major_profile, i)
                minor_rotated = np.roll(self.minor_profile, i)
                
                # Normalize profiles
                major_rotated = major_rotated / np.sum(major_rotated)
                minor_rotated = minor_rotated / np.sum(minor_rotated)
                
                # Calculate correlation
                major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
                minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
                
                major_correlations.append(major_corr)
                minor_correlations.append(minor_corr)
            
            # Find best matches
            major_max_idx = np.argmax(major_correlations)
            minor_max_idx = np.argmax(minor_correlations)
            
            major_max_corr = major_correlations[major_max_idx]
            minor_max_corr = minor_correlations[minor_max_idx]
            
            # Choose the better match
            if major_max_corr > minor_max_corr:
                key_name = self.key_names[major_max_idx]
                mode = 'major'
                confidence = major_max_corr
            else:
                key_name = self.key_names[minor_max_idx]
                mode = 'minor'
                confidence = minor_max_corr
            
            return {
                'key': key_name,
                'scale': mode,
                'confidence': confidence,
                'method': 'Improved Traditional'
            }
            
        except Exception as e:
            print(f"Traditional detection failed: {e}")
            return self._get_default_key()
    
    def _get_default_key(self) -> Dict:
        """Return default key when detection fails"""
        return {
            'key': 'C',
            'scale': 'major',
            'confidence': 0.0,
            'method': 'Default'
        }
    
    def compare_keys(self, key1: Dict, key2: Dict) -> Dict:
        """Compare two keys and return similarity score"""
        # Calculate similarity
        key_match = key1['key'] == key2['key']
        mode_match = key1['scale'] == key2['scale']
        
        # Calculate score based on similarity
        score = 0
        if key_match and mode_match:
            score = 100  # Perfect match
        elif key_match:
            score = 70   # Same key, different mode
        elif self._is_relative_key(key1, key2):
            score = 50   # Relative keys
        elif self._is_parallel_key(key1, key2):
            score = 30   # Parallel keys
        else:
            score = 0    # No similarity
        
        return {
            'score': score,
            'key_match': key_match,
            'mode_match': mode_match,
            'key1': f"{key1['key']} {key1['scale']}",
            'key2': f"{key2['key']} {key2['scale']}",
            'similarity': self._calculate_similarity(key1, key2)
        }
    
    def _is_relative_key(self, key1: Dict, key2: Dict) -> bool:
        """Check if keys are relative"""
        relative_pairs = [
            ('C', 'major', 'A', 'minor'),
            ('G', 'major', 'E', 'minor'),
            ('D', 'major', 'B', 'minor'),
            ('A', 'major', 'F#', 'minor'),
            ('E', 'major', 'C#', 'minor'),
            ('B', 'major', 'G#', 'minor'),
            ('F#', 'major', 'D#', 'minor'),
            ('C#', 'major', 'A#', 'minor'),
            ('F', 'major', 'D', 'minor'),
            ('Bb', 'major', 'G', 'minor'),
            ('Eb', 'major', 'C', 'minor'),
            ('Ab', 'major', 'F', 'minor')
        ]
        
        for k1, m1, k2, m2 in relative_pairs:
            if ((key1['key'] == k1 and key1['scale'] == m1 and 
                 key2['key'] == k2 and key2['scale'] == m2) or
                (key2['key'] == k1 and key2['scale'] == m1 and 
                 key1['key'] == k2 and key1['scale'] == m2)):
                return True
        return False
    
    def _detect_with_vocals_specific(self, audio: np.ndarray, sr: int) -> Dict:
        """Vocals-specific key detection using multiple approaches"""
        try:
            results = []
            
            # Method 1: Fundamental frequency analysis
            f0_result = self._analyze_vocals_fundamental_frequencies(audio, sr)
            if f0_result:
                results.append(f0_result)
            
            # Method 2: Harmonic analysis
            harmonic_result = self._analyze_vocals_harmonics(audio, sr)
            if harmonic_result:
                results.append(harmonic_result)
            
            # Method 3: Chroma analysis with vocals-specific parameters
            chroma_result = self._analyze_vocals_chroma(audio, sr)
            if chroma_result:
                results.append(chroma_result)
            
            # Voting among vocals-specific methods
            if results:
                # Simple voting - most common result wins
                key_votes = {}
                for result in results:
                    key_name = f"{result['key']} {result['scale']}"
                    if key_name not in key_votes:
                        key_votes[key_name] = {'count': 0, 'confidence': 0, 'result': result}
                    key_votes[key_name]['count'] += 1
                    key_votes[key_name]['confidence'] += result['confidence']
                
                # Find most voted key
                best_key = max(key_votes.items(), key=lambda x: x[1]['count'])
                best_result = best_key[1]['result']
                best_result['confidence'] = best_key[1]['confidence'] / best_key[1]['count']
                
                return best_result
            
            return None
            
        except Exception as e:
            logger.warning(f"Vocals-specific detection failed: {e}")
            return None
    
    def _analyze_vocals_fundamental_frequencies(self, audio: np.ndarray, sr: int) -> Dict:
        """Analyze vocals using fundamental frequency analysis"""
        try:
            # Extract fundamental frequencies using YIN algorithm
            f0 = librosa.yin(audio, fmin=80, fmax=1000, sr=sr)
            
            # Remove NaN values and outliers
            f0_clean = f0[~np.isnan(f0)]
            f0_clean = f0_clean[f0_clean > 80]  # Remove very low frequencies
            f0_clean = f0_clean[f0_clean < 1000]  # Remove very high frequencies
            
            if len(f0_clean) == 0:
                return None
            
            # Convert fundamental frequencies to MIDI notes
            midi_notes = librosa.hz_to_midi(f0_clean)
            
            # Round to nearest semitone
            midi_notes_rounded = np.round(midi_notes)
            
            # Count note occurrences
            note_counts = np.bincount(midi_notes_rounded.astype(int))
            
            # Find most common notes
            most_common_notes = np.argsort(note_counts)[-5:][::-1]  # Top 5 notes
            
            # Convert MIDI notes to note names
            note_names = [librosa.midi_to_note(note) for note in most_common_notes]
            
            # Analyze key based on most common notes
            key_result = self._analyze_key_from_notes(note_names)
            
            if key_result:
                return {
                    'key': key_result['key'],
                    'scale': key_result['scale'],
                    'confidence': key_result['confidence'] * 0.8,  # Reduce confidence for F0 method
                    'method': 'Vocals Fundamental Analysis'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Fundamental frequency analysis failed: {e}")
            return None
    
    def _analyze_vocals_harmonics(self, audio: np.ndarray, sr: int) -> Dict:
        """Analyze vocals using harmonic analysis"""
        try:
            # Extract harmonics using STFT
            stft = librosa.stft(audio, hop_length=512)
            magnitude = np.abs(stft)
            
            # Focus on lower frequencies where vocals are strongest
            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
            vocal_range = (freq_bins >= 80) & (freq_bins <= 2000)
            vocal_magnitude = magnitude[vocal_range, :]
            
            # Find peaks in frequency spectrum
            from scipy.signal import find_peaks
            mean_spectrum = np.mean(vocal_magnitude, axis=1)
            peaks, _ = find_peaks(mean_spectrum, height=np.max(mean_spectrum) * 0.1)
            
            if len(peaks) == 0:
                return None
            
            # Convert peak frequencies to notes
            peak_freqs = freq_bins[vocal_range][peaks]
            midi_notes = librosa.hz_to_midi(peak_freqs)
            note_names = [librosa.midi_to_note(int(note)) for note in midi_notes]
            
            # Analyze key
            key_result = self._analyze_key_from_notes(note_names)
            
            if key_result:
                return {
                    'key': key_result['key'],
                    'scale': key_result['scale'],
                    'confidence': key_result['confidence'] * 0.7,
                    'method': 'Vocals Harmonic Analysis'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Harmonic analysis failed: {e}")
            return None
    
    def _analyze_vocals_chroma(self, audio: np.ndarray, sr: int) -> Dict:
        """Analyze vocals using chroma with vocals-specific parameters"""
        try:
            # Use smaller hop length for better time resolution
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=256, n_fft=1024)
            
            # Focus on stronger chroma values
            chroma_mean = np.mean(chroma, axis=1)
            chroma_strong = chroma_mean > np.max(chroma_mean) * 0.3  # Only strong chroma values
            
            if not np.any(chroma_strong):
                return None
            
            # Get strong chroma indices
            strong_indices = np.where(chroma_strong)[0]
            chroma_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            strong_notes = [chroma_notes[i] for i in strong_indices]
            
            # Analyze key
            key_result = self._analyze_key_from_notes(strong_notes)
            
            if key_result:
                return {
                    'key': key_result['key'],
                    'scale': key_result['scale'],
                    'confidence': key_result['confidence'] * 0.6,
                    'method': 'Vocals Chroma Analysis'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Vocals chroma analysis failed: {e}")
            return None
    
    def _analyze_key_from_notes(self, note_names: list) -> Dict:
        """Analyze key from list of note names - improved for minor keys"""
        try:
            # Define key signatures for both major and minor
            major_keys = {
                'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
                'G': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
                'D': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
                'A': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
                'E': ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
                'B': ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'],
                'F#': ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'E#'],
                'C#': ['C#', 'D#', 'E#', 'F#', 'G#', 'A#', 'B#'],
                'F': ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
                'Bb': ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'],
                'Eb': ['Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D'],
                'Ab': ['Ab', 'Bb', 'C', 'Db', 'Eb', 'F', 'G'],
                'Db': ['Db', 'Eb', 'F', 'Gb', 'Ab', 'Bb', 'C']
            }
            
            minor_keys = {
                'A': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                'E': ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
                'B': ['B', 'C#', 'D', 'E', 'F#', 'G', 'A'],
                'F#': ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E'],
                'C#': ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B'],
                'G#': ['G#', 'A#', 'B', 'C#', 'D#', 'E', 'F#'],
                'D#': ['D#', 'E#', 'F#', 'G#', 'A#', 'B', 'C#'],
                'A#': ['A#', 'B#', 'C#', 'D#', 'E#', 'F#', 'G#'],
                'D': ['D', 'E', 'F', 'G', 'A', 'Bb', 'C'],
                'G': ['G', 'A', 'Bb', 'C', 'D', 'Eb', 'F'],
                'C': ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'Bb'],
                'F': ['F', 'G', 'Ab', 'Bb', 'C', 'Db', 'Eb'],
                'Bb': ['Bb', 'C', 'Db', 'Eb', 'F', 'Gb', 'Ab']
            }
            
            # Count matches for major keys
            major_scores = {}
            for key, notes in major_keys.items():
                score = 0
                for note in note_names:
                    if note in notes:
                        score += 1
                major_scores[key] = score
            
            # Count matches for minor keys
            minor_scores = {}
            for key, notes in minor_keys.items():
                score = 0
                for note in note_names:
                    if note in notes:
                        score += 1
                minor_scores[key] = score
            
            # Find best matching keys
            best_major_key = max(major_scores, key=major_scores.get)
            best_minor_key = max(minor_scores, key=minor_scores.get)
            best_major_score = major_scores[best_major_key]
            best_minor_score = minor_scores[best_minor_key]
            
            # Choose between major and minor
            if best_minor_score > best_major_score:
                best_key = best_minor_key
                best_scale = 'minor'
                best_score = best_minor_score
            else:
                best_key = best_major_key
                best_scale = 'major'
                best_score = best_major_score
            
            # Calculate confidence
            confidence = min(best_score / len(note_names), 1.0)
            
            if confidence > 0.2:  # Lower threshold to catch minor keys
                return {
                    'key': best_key,
                    'scale': best_scale,
                    'confidence': confidence
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Key analysis from notes failed: {e}")
            return None
    
    def _is_parallel_key(self, key1: Dict, key2: Dict) -> bool:
        """Check if keys are parallel"""
        return (key1['key'] == key2['key'] and key1['scale'] != key2['scale'])
    
    def _calculate_similarity(self, key1: Dict, key2: Dict) -> float:
        """Calculate similarity between keys using circle of fifths"""
        circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
        try:
            idx1 = circle_of_fifths.index(key1['key'])
            idx2 = circle_of_fifths.index(key2['key'])
            
            distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
            similarity = 1 - (distance / 6)
            
            if key1['scale'] != key2['scale']:
                similarity *= 0.7
            
            return similarity
        except ValueError:
            return 0.0

