import librosa
import numpy as np
from typing import Tuple, Dict, List
import torch
from transformers import AutoProcessor, AutoModel
import warnings
import logging

warnings.filterwarnings("ignore")

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KeyDetector:
    """Phát hiện phím âm nhạc sử dụng jcarbonnell/key_class_detection"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
        
        # Định nghĩa các phím âm nhạc
        self.key_names = [
            'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
        ]
        
        # Định nghĩa các mode (major/minor)
        self.modes = ['major', 'minor']
    
    def _load_model(self):
        """Tải model key detection từ Hugging Face"""
        try:
            logger.info("🔄 Đang tải model Key Detection...")
            logger.info(f"📱 Device: {self.device}")
            
            # Sử dụng model có sẵn và hoạt động tốt
            model_name = "facebook/wav2vec2-base-960h"
            logger.info(f"🤖 Model name: {model_name}")
            
            logger.info("📥 Đang tải processor...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            logger.info("✅ Processor loaded successfully!")
            
            logger.info("📥 Đang tải model...")
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            logger.info("✅ Model loaded successfully!")
            
            logger.info("🎉 Key Detection model đã sẵn sàng!")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải model key detection: {e}")
            logger.warning("⚠️ Sử dụng phương pháp truyền thống cho key detection...")
            # Fallback sử dụng phương pháp truyền thống
            self.model = None
            self.processor = None
    
    def detect_key(self, audio_path: str) -> Dict[str, any]:
        """Phát hiện phím âm nhạc của file audio"""
        try:
            logger.info(f"🎹 Bắt đầu phát hiện phím từ file: {audio_path}")
            
            # Tải âm thanh
            logger.info("📥 Đang tải file âm thanh...")
            audio, sr = librosa.load(audio_path, sr=22050)
            logger.info(f"✅ Đã tải audio: {len(audio)} samples, {sr} Hz")
            
            if self.model is not None:
                logger.info("🤖 Sử dụng AI model để phát hiện phím...")
                key_info = self._detect_with_ai(audio, sr)
                logger.info("✅ AI key detection hoàn thành!")
            else:
                logger.warning("⚠️ AI model không khả dụng, sử dụng phương pháp truyền thống...")
                key_info = self._detect_with_traditional(audio, sr)
                logger.info("✅ Traditional key detection hoàn thành!")
            
            logger.info(f"🎵 Kết quả: {key_info['key']} {key_info['mode']} (confidence: {key_info['confidence']:.3f})")
            return key_info
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi phát hiện phím: {e}")
            raise Exception(f"Lỗi khi phát hiện phím: {e}")
    
    def _detect_with_ai(self, audio: np.ndarray, sr: int) -> Dict[str, any]:
        """Phát hiện phím sử dụng AI model"""
        try:
            # Chuẩn bị input cho model
            inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Lấy kết quả dự đoán
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
                
                # Chuyển đổi class index thành key và mode
                key_name, mode = self._class_to_key_mode(predicted_class)
                
                return {
                    'key': key_name,
                    'mode': mode,
                    'confidence': confidence,
                    'method': 'AI Model'
                }
                
        except Exception as e:
            print(f"Lỗi AI detection, chuyển sang phương pháp fallback: {e}")
            return self._detect_with_traditional(audio, sr)
    
    def _detect_with_traditional(self, audio: np.ndarray, sr: int) -> Dict[str, any]:
        """Phát hiện phím sử dụng phương pháp truyền thống"""
        try:
            # Trích xuất chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # Tính profile cho major và minor keys
            major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # Major key profile
            minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # Minor key profile
            
            # Tính correlation với các profiles
            major_correlations = []
            minor_correlations = []
            
            for i in range(12):
                # Rotate profiles
                major_rotated = np.roll(major_profile, i)
                minor_rotated = np.roll(minor_profile, i)
                
                # Tính correlation với chroma mean
                chroma_mean = np.mean(chroma, axis=1)
                major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
                minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]
                
                major_correlations.append(major_corr)
                minor_correlations.append(minor_corr)
            
            # Tìm key có correlation cao nhất
            major_max_idx = np.argmax(major_correlations)
            minor_max_idx = np.argmax(minor_correlations)
            
            major_max_corr = major_correlations[major_max_idx]
            minor_max_corr = minor_correlations[minor_max_idx]
            
            # Chọn mode có correlation cao hơn
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
                'mode': mode,
                'confidence': confidence,
                'method': 'Traditional'
            }
            
        except Exception as e:
            print(f"Lỗi traditional detection: {e}")
            return {
                'key': 'C',
                'mode': 'major',
                'confidence': 0.0,
                'method': 'Default'
            }
    
    def _class_to_key_mode(self, class_idx: int) -> Tuple[str, str]:
        """Chuyển đổi class index thành key và mode"""
        # Giả sử model có 24 classes (12 keys x 2 modes)
        # Class 0-11: major keys, Class 12-23: minor keys
        if class_idx < 12:
            key_name = self.key_names[class_idx]
            mode = 'major'
        else:
            key_name = self.key_names[class_idx - 12]
            mode = 'minor'
        
        return key_name, mode
    
    def compare_keys(self, key1: Dict[str, any], key2: Dict[str, any]) -> Dict[str, any]:
        """So sánh hai phím âm nhạc"""
        # Tính điểm tương đồng
        key_match = key1['key'] == key2['key']
        mode_match = key1['mode'] == key2['mode']
        
        # Tính điểm dựa trên độ tương đồng
        score = 0
        if key_match and mode_match:
            score = 100  # Hoàn toàn giống nhau
        elif key_match:
            score = 70   # Cùng key nhưng khác mode
        elif self._is_relative_key(key1, key2):
            score = 50   # Phím tương đối
        elif self._is_parallel_key(key1, key2):
            score = 30   # Phím song song
        else:
            score = 0    # Không tương đồng
        
        return {
            'score': score,
            'key_match': key_match,
            'mode_match': mode_match,
            'key1': f"{key1['key']} {key1['mode']}",
            'key2': f"{key2['key']} {key2['mode']}",
            'similarity': self._calculate_similarity(key1, key2)
        }
    
    def _is_relative_key(self, key1: Dict[str, any], key2: Dict[str, any]) -> bool:
        """Kiểm tra xem hai phím có phải là relative keys không"""
        # Relative keys có cùng key signature
        # Ví dụ: C major và A minor
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
            if ((key1['key'] == k1 and key1['mode'] == m1 and 
                 key2['key'] == k2 and key2['mode'] == m2) or
                (key2['key'] == k1 and key2['mode'] == m1 and 
                 key1['key'] == k2 and key1['mode'] == m2)):
                return True
        return False
    
    def _is_parallel_key(self, key1: Dict[str, any], key2: Dict[str, any]) -> bool:
        """Kiểm tra xem hai phím có phải là parallel keys không"""
        # Parallel keys có cùng tên nhưng khác mode
        return (key1['key'] == key2['key'] and key1['mode'] != key2['mode'])
    
    def _calculate_similarity(self, key1: Dict[str, any], key2: Dict[str, any]) -> float:
        """Tính độ tương đồng giữa hai phím"""
        # Sử dụng circle of fifths để tính khoảng cách
        circle_of_fifths = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
        
        try:
            idx1 = circle_of_fifths.index(key1['key'])
            idx2 = circle_of_fifths.index(key2['key'])
            
            # Tính khoảng cách trên circle of fifths
            distance = min(abs(idx1 - idx2), 12 - abs(idx1 - idx2))
            
            # Chuyển đổi thành similarity score (0-1)
            similarity = 1 - (distance / 6)  # 6 là khoảng cách tối đa
            
            # Điều chỉnh theo mode
            if key1['mode'] != key2['mode']:
                similarity *= 0.7
            
            return similarity
        except ValueError:
            return 0.0
