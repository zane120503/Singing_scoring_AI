"""
Smart Slicing Strategy - Cắt file audio thông minh dựa trên vị trí giọng hát
"""

import librosa
import soundfile as sf
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class SmartSlicingStrategy:
    """Chiến lược cắt file audio thông minh"""
    
    def __init__(self):
        self.min_slice_duration = 20.0  # Tối thiểu 20 giây
        self.max_slice_duration = 60.0  # Tối đa 60 giây
        self.preferred_slice_duration = 30.0  # Ưu tiên 30 giây
    
    def slice_with_voice_guidance(self, audio_path: str, voice_segments: List[Dict], 
                                   output_path: Optional[str] = None) -> Dict:
        """
        Cắt file audio dựa trên vị trí giọng hát
        
        Args:
            audio_path: Đường dẫn file audio
            voice_segments: Danh sách các đoạn có giọng hát
            output_path: Đường dẫn output (optional)
            
        Returns:
            Dict với thông tin đoạn đã cắt
        """
        try:
            logger.info("🎯 Smart slicing with voice guidance...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            audio_duration = len(audio) / sr
            
            logger.info(f"📊 Audio duration: {audio_duration:.2f}s")
            logger.info(f"🎤 Found {len(voice_segments)} voice segments")
            
            # Strategy 1: Tìm đoạn tốt nhất dựa trên voice segments
            best_segment = self._find_best_slice_position(audio_duration, voice_segments)
            
            if not best_segment:
                logger.warning("⚠️ No suitable voice segment found, using fallback")
                best_segment = self._get_fallback_slice(audio_duration)
            
            # Extract slice
            start_sample = int(best_segment['start'] * sr)
            end_sample = int(best_segment['end'] * sr)
            
            # Ensure we don't exceed audio length
            end_sample = min(end_sample, len(audio))
            
            slice_audio = audio[start_sample:end_sample]
            
            # Save sliced audio
            if output_path is None:
                import os
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = f"{base_name}_smart_slice_{int(best_segment['start'])}s.wav"
            
            sf.write(output_path, slice_audio, sr)
            
            logger.info(f"✅ Smart slicing complete: {best_segment['start']:.2f}s - {best_segment['end']:.2f}s")
            
            return {
                'output_path': output_path,
                'start': best_segment['start'],
                'end': best_segment['end'],
                'duration': best_segment['end'] - best_segment['start'],
                'strategy': best_segment['strategy'],
                'voice_segment_used': best_segment.get('voice_segment')
            }
            
        except Exception as e:
            logger.error(f"❌ Smart slicing failed: {e}")
            raise
    
    def _find_best_slice_position(self, audio_duration: float, voice_segments: List[Dict]) -> Optional[Dict]:
        """Tìm vị trí cắt tốt nhất"""
        try:
            if not voice_segments:
                return None
            
            candidates = []
            
            # Strategy 1: Tìm đoạn voice trong 60 giây đầu
            for seg in voice_segments:
                if seg['start'] < 60:  # Trong 60s đầu
                    # Tạo slice 30s bắt đầu từ voice
                    slice_start = max(0, seg['start'] - 5)  # Trước 5s để lấy intro
                    slice_end = min(audio_duration, slice_start + self.preferred_slice_duration)
                    
                    if slice_end - slice_start >= self.min_slice_duration:
                        candidates.append({
                            'start': slice_start,
                            'end': slice_end,
                            'duration': slice_end - slice_start,
                            'strategy': 'voice_in_first_60s',
                            'voice_segment': seg,
                            'priority': 1  # Highest priority
                        })
            
            # Strategy 2: Tìm đoạn có voice dài nhất trong đầu file
            if not candidates:
                for seg in voice_segments:
                    if seg['start'] < 120:  # Trong 2 phút đầu
                        slice_start = seg['start'] - 10  # Trước 10s
                        slice_end = min(audio_duration, seg['end'] + 20)  # Sau voice 20s
                        
                        # Ensure minimum duration
                        if slice_end - slice_start < self.min_slice_duration:
                            slice_end = slice_start + self.preferred_slice_duration
                        
                        candidates.append({
                            'start': slice_start,
                            'end': slice_end,
                            'duration': slice_end - slice_start,
                            'strategy': 'voice_based',
                            'voice_segment': seg,
                            'priority': 2
                        })
            
            # Strategy 3: Slice fix ở giữa nếu voice ở cuối file
            if not candidates:
                for seg in voice_segments:
                    if seg['start'] > audio_duration / 2:  # Voice ở nửa sau
                        # Cắt từ 15-45s (chiến lược mặc định)
                        slice_start = 15.0
                        slice_end = min(audio_duration, 45.0)
                        
                        candidates.append({
                            'start': slice_start,
                            'end': slice_end,
                            'duration': slice_end - slice_start,
                            'strategy': 'default_fallback',
                            'voice_segment': seg,
                            'priority': 3,
                            'warning': 'Voice detected late in file, using default slice'
                        })
            
            # Sort by priority and select best
            if candidates:
                # Sort by priority, then by duration
                candidates.sort(key=lambda x: (x['priority'], -x['duration']))
                best = candidates[0]
                
                logger.info(f"📌 Selected slice: {best['start']:.2f}s - {best['end']:.2f}s ({best['strategy']})")
                return best
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding best slice: {e}")
            return None
    
    def _get_fallback_slice(self, audio_duration: float) -> Dict:
        """Fallback slicing strategy"""
        logger.info("🔄 Using fallback slicing strategy")
        
        # Slice from 15-45s if possible
        start = 15.0
        end = min(audio_duration, 45.0)
        
        # Ensure minimum duration
        if end - start < self.min_slice_duration:
            end = min(audio_duration, start + self.min_slice_duration)
        
        return {
            'start': start,
            'end': end,
            'duration': end - start,
            'strategy': 'simple_fallback',
            'warning': 'No voice detected, using simple slice'
        }
    
    def handle_no_vocals(self, audio_path: str) -> Dict:
        """Xử lý trường hợp không có giọng hát"""
        try:
            logger.warning("⚠️ No vocals detected in file")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            audio_duration = len(audio) / sr
            
            # Use first 30 seconds
            slice_start = 0
            slice_end = min(30.0, audio_duration)
            
            slice_audio = audio[:int(slice_end * sr)]
            
            # Save
            import os
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = f"{base_name}_no_vocals_{int(slice_start)}s.wav"
            
            sf.write(output_path, slice_audio, sr)
            
            logger.info(f"✅ Created slice for no-vocals file: {slice_start:.2f}s - {slice_end:.2f}s")
            
            return {
                'output_path': output_path,
                'start': slice_start,
                'end': slice_end,
                'duration': slice_end - slice_start,
                'strategy': 'no_vocals_fallback',
                'has_vocals': False
            }
            
        except Exception as e:
            logger.error(f"❌ Error handling no-vocals file: {e}")
            raise

