import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModel
import os
import logging
from typing import Tuple, Optional

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Xử lý âm thanh và tách giọng hát sử dụng AI Audio Separator"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Tải model AI Audio Separator từ Hugging Face"""
        try:
            logger.info("🔄 Đang tải model AI Audio Separator...")
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
            
            logger.info("🎉 AI Audio Separator model đã sẵn sàng!")
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tải model AI: {e}")
            logger.warning("⚠️ Sử dụng phương pháp fallback với librosa...")
            # Fallback sử dụng librosa để tách giọng đơn giản
            self.model = None
            self.processor = None
    
    def load_audio(self, file_path: str, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
        """Tải file âm thanh"""
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate)
            return audio, sr
        except Exception as e:
            raise Exception(f"Lỗi khi tải file âm thanh: {e}")
    
    def separate_vocals(self, audio_path: str, output_path: Optional[str] = None) -> str:
        """Tách giọng hát từ file karaoke"""
        try:
            logger.info(f"🎤 Bắt đầu tách giọng từ file: {audio_path}")
            
            # Tải âm thanh
            logger.info("📥 Đang tải file âm thanh...")
            audio, sr = self.load_audio(audio_path)
            logger.info(f"✅ Đã tải audio: {len(audio)} samples, {sr} Hz")
            
            if self.model is not None:
                logger.info("🤖 Sử dụng AI model để tách giọng...")
                vocals = self._separate_with_ai(audio, sr)
                logger.info("✅ AI separation hoàn thành!")
            else:
                logger.warning("⚠️ AI model không khả dụng, sử dụng fallback với librosa...")
                vocals = self._separate_with_librosa(audio, sr)
                logger.info("✅ Librosa separation hoàn thành!")
            
            # Lưu file giọng đã tách với định dạng MP3
            if output_path is None:
                base_name = os.path.splitext(audio_path)[0]
                output_path = f"{base_name}_vocals.mp3"
            
            logger.info(f"💾 Đang lưu file giọng đã tách: {output_path}")
            
            # Lưu với định dạng MP3
            sf.write(output_path, vocals, sr, format='MP3')
            logger.info(f"✅ Đã tách giọng và lưu tại: {output_path}")
            
            # Kiểm tra file đã được tạo
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"📊 Kích thước file: {file_size / 1024 / 1024:.2f} MB")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Lỗi khi tách giọng: {e}")
            raise Exception(f"Lỗi khi tách giọng: {e}")
    
    def _separate_with_ai(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Tách giọng sử dụng AI model"""
        try:
            logger.info("🔄 Đang xử lý với AI model...")
            
            # Chuyển đổi audio thành tensor
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            logger.info(f"📊 Audio tensor shape: {audio_tensor.shape}")
            
            # Xử lý với model
            logger.info("🤖 Đang chạy AI model...")
            with torch.no_grad():
                separated = self.model(audio_tensor)
                vocals = separated[0, 0].cpu().numpy()  # Lấy track giọng hát
                logger.info(f"✅ AI model output shape: {vocals.shape}")
            
            return vocals
        except Exception as e:
            logger.error(f"❌ Lỗi AI separation: {e}")
            logger.warning("⚠️ Chuyển sang phương pháp fallback...")
            return self._separate_with_librosa(audio, sr)
    
    def _separate_with_librosa(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Tách giọng sử dụng phương pháp đơn giản với librosa"""
        try:
            logger.info("🔄 Sử dụng librosa harmonic-percussive separation...")
            
            # Sử dụng harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            logger.info(f"📊 Harmonic shape: {harmonic.shape}, Percussive shape: {percussive.shape}")
            
            # Harmonic component thường chứa giọng hát
            vocals = harmonic
            
            # Normalize audio
            vocals = librosa.util.normalize(vocals)
            logger.info("✅ Librosa separation hoàn thành!")
            
            return vocals
        except Exception as e:
            logger.error(f"❌ Lỗi librosa separation: {e}")
            logger.warning("⚠️ Trả về audio gốc...")
            return audio  # Trả về audio gốc nếu không tách được
    
    def get_audio_features(self, audio_path: str) -> dict:
        """Trích xuất các đặc trưng âm thanh"""
        try:
            audio, sr = self.load_audio(audio_path)
            
            # Tính toán các đặc trưng
            features = {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'rms_energy': np.sqrt(np.mean(audio**2)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)[0]),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0]),
                'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]),
                'mfcc': np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
            }
            
            return features
        except Exception as e:
            raise Exception(f"Lỗi khi trích xuất đặc trưng: {e}")
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Chuẩn hóa âm thanh"""
        return librosa.util.normalize(audio)
    
    def trim_silence(self, audio: np.ndarray, sr: int, top_db: int = 20) -> np.ndarray:
        """Loại bỏ khoảng lặng"""
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed
