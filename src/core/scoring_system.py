import librosa
import numpy as np
from typing import Dict, List, Tuple
import math

class KaraokeScoringSystem:
    """Hệ thống chấm điểm karaoke với nhiều tiêu chí"""
    
    def __init__(self):
        self.weights = {
            'key_accuracy': 0.30,      # Độ chính xác về phím
            'pitch_accuracy': 0.25,   # Độ chính xác về cao độ
            'rhythm_accuracy': 0.20,  # Độ chính xác về nhịp điệu
            'timing_accuracy': 0.15,  # Độ chính xác về thời gian
            'vocal_quality': 0.10     # Chất lượng giọng hát
        }
    
    def calculate_overall_score(self, karaoke_path: str, beat_path: str, vocals_path: str) -> Dict[str, any]:
        """Tính điểm tổng thể cho bài hát karaoke"""
        try:
            # Tải các file âm thanh
            karaoke_audio, karaoke_sr = librosa.load(karaoke_path, sr=22050)
            beat_audio, beat_sr = librosa.load(beat_path, sr=22050)
            vocals_audio, vocals_sr = librosa.load(vocals_path, sr=22050)
            
            # Tính các điểm số thành phần
            scores = {}
            
            # 1. Độ chính xác về phím (đã được tính từ KeyDetector)
            scores['key_accuracy'] = self._calculate_key_accuracy(vocals_audio, vocals_sr, beat_audio, beat_sr)
            
            # 2. Độ chính xác về cao độ
            scores['pitch_accuracy'] = self._calculate_pitch_accuracy(vocals_audio, vocals_sr, beat_audio, beat_sr)
            
            # 3. Độ chính xác về nhịp điệu
            scores['rhythm_accuracy'] = self._calculate_rhythm_accuracy(vocals_audio, vocals_sr, beat_audio, beat_sr)
            
            # 4. Độ chính xác về thời gian
            scores['timing_accuracy'] = self._calculate_timing_accuracy(vocals_audio, vocals_sr, beat_audio, beat_sr)
            
            # 5. Chất lượng giọng hát
            scores['vocal_quality'] = self._calculate_vocal_quality(vocals_audio, vocals_sr)
            
            # Tính điểm tổng thể
            overall_score = sum(scores[key] * self.weights[key] for key in scores.keys())
            
            return {
                'overall_score': round(overall_score, 2),
                'detailed_scores': scores,
                'weights': self.weights,
                'grade': self._get_grade(overall_score),
                'feedback': self._generate_feedback(scores)
            }
            
        except Exception as e:
            raise Exception(f"Lỗi khi tính điểm: {e}")
    
    def _calculate_key_accuracy(self, vocals: np.ndarray, vocals_sr: int, beat: np.ndarray, beat_sr: int) -> float:
        """Tính độ chính xác về phím âm nhạc"""
        try:
            # Trích xuất chroma features
            vocals_chroma = librosa.feature.chroma_stft(y=vocals, sr=vocals_sr)
            beat_chroma = librosa.feature.chroma_stft(y=beat, sr=beat_sr)
            
            # Tính correlation giữa chroma của giọng hát và beat
            vocals_mean = np.mean(vocals_chroma, axis=1)
            beat_mean = np.mean(beat_chroma, axis=1)
            
            correlation = np.corrcoef(vocals_mean, beat_mean)[0, 1]
            
            # Chuyển đổi correlation thành điểm (0-100)
            score = max(0, min(100, (correlation + 1) * 50))
            
            return round(score, 2)
        except:
            return 50.0  # Điểm mặc định
    
    def _calculate_pitch_accuracy(self, vocals: np.ndarray, vocals_sr: int, beat: np.ndarray, beat_sr: int) -> float:
        """Tính độ chính xác về cao độ"""
        try:
            # Trích xuất pitch
            vocals_pitch = librosa.yin(vocals, fmin=50, fmax=4000)
            beat_pitch = librosa.yin(beat, fmin=50, fmax=4000)
            
            # Loại bỏ các giá trị NaN
            vocals_pitch = vocals_pitch[~np.isnan(vocals_pitch)]
            beat_pitch = beat_pitch[~np.isnan(beat_pitch)]
            
            if len(vocals_pitch) == 0 or len(beat_pitch) == 0:
                return 50.0
            
            # Tính độ lệch pitch
            vocals_mean = np.mean(vocals_pitch)
            beat_mean = np.mean(beat_pitch)
            
            # Tính phần trăm lệch
            pitch_deviation = abs(vocals_mean - beat_mean) / beat_mean
            
            # Chuyển đổi thành điểm (0-100)
            score = max(0, min(100, 100 - (pitch_deviation * 200)))
            
            return round(score, 2)
        except:
            return 50.0
    
    def _calculate_rhythm_accuracy(self, vocals: np.ndarray, vocals_sr: int, beat: np.ndarray, beat_sr: int) -> float:
        """Tính độ chính xác về nhịp điệu"""
        try:
            # Trích xuất tempo và beat tracking
            vocals_tempo, vocals_beats = librosa.beat.beat_track(y=vocals, sr=vocals_sr)
            beat_tempo, beat_beats = librosa.beat.beat_track(y=beat, sr=beat_sr)
            
            # Tính độ lệch tempo
            tempo_deviation = abs(vocals_tempo - beat_tempo) / beat_tempo
            
            # Tính độ chính xác beat alignment
            if len(vocals_beats) > 0 and len(beat_beats) > 0:
                # Chuyển đổi beat frames thành thời gian
                vocals_beat_times = librosa.frames_to_time(vocals_beats, sr=vocals_sr)
                beat_beat_times = librosa.frames_to_time(beat_beats, sr=beat_sr)
                
                # Tính độ lệch trung bình
                min_length = min(len(vocals_beat_times), len(beat_beat_times))
                if min_length > 0:
                    beat_alignment_error = np.mean(np.abs(vocals_beat_times[:min_length] - beat_beat_times[:min_length]))
                    beat_score = max(0, 100 - (beat_alignment_error * 10))
                else:
                    beat_score = 50
            else:
                beat_score = 50
            
            # Tính điểm tổng hợp
            tempo_score = max(0, 100 - (tempo_deviation * 100))
            overall_score = (tempo_score + beat_score) / 2
            
            return round(overall_score, 2)
        except:
            return 50.0
    
    def _calculate_timing_accuracy(self, vocals: np.ndarray, vocals_sr: int, beat: np.ndarray, beat_sr: int) -> float:
        """Tính độ chính xác về thời gian"""
        try:
            # Tính độ dài của các file
            vocals_duration = len(vocals) / vocals_sr
            beat_duration = len(beat) / beat_sr
            
            # Tính độ lệch thời gian
            duration_deviation = abs(vocals_duration - beat_duration) / beat_duration
            
            # Tính điểm dựa trên độ lệch
            score = max(0, min(100, 100 - (duration_deviation * 50)))
            
            return round(score, 2)
        except:
            return 50.0
    
    def _calculate_vocal_quality(self, vocals: np.ndarray, vocals_sr: int) -> float:
        """Tính chất lượng giọng hát"""
        try:
            # Tính các đặc trưng âm thanh
            rms_energy = np.sqrt(np.mean(vocals**2))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(vocals)[0])
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=vocals, sr=vocals_sr)[0])
            
            # Tính điểm dựa trên các đặc trưng
            energy_score = min(100, rms_energy * 1000)  # Normalize energy
            clarity_score = min(100, (1 - zero_crossing_rate) * 100)  # Lower ZCR = clearer
            brightness_score = min(100, spectral_centroid / 50)  # Normalize spectral centroid
            
            # Tính điểm trung bình
            overall_score = (energy_score + clarity_score + brightness_score) / 3
            
            return round(overall_score, 2)
        except:
            return 50.0
    
    def _get_grade(self, score: float) -> str:
        """Chuyển đổi điểm số thành thang điểm"""
        if score >= 90:
            return "A+ (Xuất sắc)"
        elif score >= 80:
            return "A (Giỏi)"
        elif score >= 70:
            return "B+ (Khá)"
        elif score >= 60:
            return "B (Trung bình khá)"
        elif score >= 50:
            return "C (Trung bình)"
        elif score >= 40:
            return "D (Yếu)"
        else:
            return "F (Kém)"
    
    def _generate_feedback(self, scores: Dict[str, float]) -> List[str]:
        """Tạo phản hồi dựa trên điểm số"""
        feedback = []
        
        for criterion, score in scores.items():
            if score >= 80:
                feedback.append(f"✅ {self._get_criterion_name(criterion)}: Tuyệt vời!")
            elif score >= 60:
                feedback.append(f"👍 {self._get_criterion_name(criterion)}: Khá tốt")
            elif score >= 40:
                feedback.append(f"⚠️ {self._get_criterion_name(criterion)}: Cần cải thiện")
            else:
                feedback.append(f"❌ {self._get_criterion_name(criterion)}: Cần luyện tập nhiều hơn")
        
        return feedback
    
    def _get_criterion_name(self, criterion: str) -> str:
        """Chuyển đổi tên tiêu chí sang tiếng Việt"""
        names = {
            'key_accuracy': 'Độ chính xác phím',
            'pitch_accuracy': 'Độ chính xác cao độ',
            'rhythm_accuracy': 'Độ chính xác nhịp điệu',
            'timing_accuracy': 'Độ chính xác thời gian',
            'vocal_quality': 'Chất lượng giọng hát'
        }
        return names.get(criterion, criterion)

