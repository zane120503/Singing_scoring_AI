import os
import sys
import json
import librosa
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
from typing import Tuple, Optional, Union
import warnings
import logging
warnings.filterwarnings("ignore")

# Import Audio Separator Integration
from src.ai.audio_separator_integration import AudioSeparatorIntegration

logger = logging.getLogger(__name__)

class AdvancedAudioProcessor:
    """Xử lý âm thanh nâng cao sử dụng Audio Separator UI"""
    
    def __init__(self, fast_mode=False):
        # Force GPU usage
        from src.core.gpu_config import get_device, force_cuda
        force_cuda()
        self.device = get_device()
        self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'temp_output')
        self.fast_mode = fast_mode
        
        # Tạo thư mục output nếu chưa có
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Khởi tạo Audio Separator Integration
        if not fast_mode:
            self.audio_separator = AudioSeparatorIntegration(fast_mode=fast_mode)
            self.model = "Audio_Separator_AI" if self.audio_separator.available else None
        else:
            self.audio_separator = None
            self.model = "Fast_Mode"
        
        logger.info(f"✅ AdvancedAudioProcessor initialized {'with Fast Mode' if fast_mode else 'with Audio Separator Integration'}")
    
    def _initialize_models(self):
        """Khởi tạo các models từ Audio Separator"""
        try:
            if os.path.exists(os.path.join(self.mdxnet_models_dir, "data.json")):
                with open(os.path.join(self.mdxnet_models_dir, "data.json")) as infile:
                    self.mdx_model_params = json.load(infile)
                print("Loaded model configuration from Audio Separator")
                self.model = "Audio_Separator_UI"  # Set model để GUI nhận diện
            else:
                print("Model configuration not found, using fallback")
                self.mdx_model_params = None
                self.model = None
        except Exception as e:
            print(f"Error initializing models: {e}")
            self.mdx_model_params = None
            self.model = None
    
    def load_audio(self, file_path: str, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
        """Tải file âm thanh"""
        try:
            audio, sr = librosa.load(file_path, sr=sample_rate)
            return audio, sr
        except Exception as e:
            raise Exception(f"Lỗi khi tải file âm thanh: {e}")
    
    def convert_to_stereo_and_wav(self, input_path: str) -> str:
        """Chuyển đổi file âm thanh thành stereo WAV"""
        try:
            # Tải âm thanh
            audio, sr = librosa.load(input_path, sr=44100, mono=False)
            
            # Đảm bảo là stereo
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio])
            elif audio.shape[0] == 1:
                audio = np.repeat(audio, 2, axis=0)
            
            # Tạo file output
            output_path = input_path.replace('.mp3', '_converted.wav').replace('.m4a', '_converted.wav')
            if not output_path.endswith('.wav'):
                output_path += '.wav'
            
            # Lưu file
            sf.write(output_path, audio.T, sr)
            return output_path
            
        except Exception as e:
            print(f"Lỗi khi chuyển đổi file: {e}")
            return input_path
    
    def separate_vocals_advanced(self, audio_path: str, output_path: Union[str, None] = None) -> str:
        """Tách giọng hát sử dụng Audio Separator UI"""
        try:
            print("Separating vocals with Audio Separator...")
            
            # Chuyển đổi file thành định dạng phù hợp
            converted_path = self.convert_to_stereo_and_wav(audio_path)
            
            # Tạo hash cho file
            import hashlib
            with open(converted_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            # Tạo thư mục output
            song_output_dir = os.path.join(self.output_dir, f"{file_hash}_mdx")
            os.makedirs(song_output_dir, exist_ok=True)
            
            # Sử dụng phương pháp đơn giản hóa từ Audio Separator
            vocals_path = self._extract_vocals_simple(converted_path, song_output_dir)
            
            # Nếu có output_path được chỉ định, copy file
            if output_path and vocals_path:
                import shutil
                shutil.copy2(vocals_path, output_path)
                return output_path
            
            return vocals_path
            
        except Exception as e:
            print(f"❌ Lỗi khi tách giọng với Audio Separator: {e}")
            # Fallback về phương pháp đơn giản
            return self._separate_vocals_fallback(audio_path, output_path)
    
    def _extract_vocals_simple(self, audio_path: str, output_dir: str) -> str:
        """Sử dụng AI model thực sự để tách giọng"""
        try:
            # Import AI Audio Separator wrapper
            from ai_audio_separator import AIAudioSeparator
            
            print("Using AI Audio Separator model...")
            
            # Sử dụng AI model để tách giọng
            vocals_path = os.path.join(output_dir, "vocals.wav")
            
            # Tạo separator instance
            separator = AIAudioSeparator()
            
            # Gọi AI model
            result_path = separator.separate_vocals(audio_path, vocals_path)
            
            if result_path and os.path.exists(result_path):
                print(f"AI Vocals separated and saved at: {result_path}")
                return result_path
            else:
                print("AI model failed, using fallback...")
                return self._extract_vocals_fallback(audio_path, output_dir)
                
        except ImportError as e:
            print(f"AI model not available: {e}")
            return self._extract_vocals_fallback(audio_path, output_dir)
        except Exception as e:
            print(f"Error in AI vocal extraction: {e}")
            return self._extract_vocals_fallback(audio_path, output_dir)
    
    def _extract_vocals_fallback(self, audio_path: str, output_dir: str) -> str:
        """Phương pháp tách giọng fallback"""
        try:
            # Tải âm thanh
            audio, sr = librosa.load(audio_path, sr=44100)
            
            # Sử dụng harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Harmonic component thường chứa giọng hát
            vocals = harmonic
            
            # Normalize
            vocals = librosa.util.normalize(vocals)
            
            # Lưu file vocals
            vocals_path = os.path.join(output_dir, "vocals.wav")
            sf.write(vocals_path, vocals, sr)
            
            print(f"Fallback vocals separated and saved at: {vocals_path}")
            return vocals_path
            
        except Exception as e:
            print(f"Error in fallback vocal extraction: {e}")
            return None
    
    def _separate_vocals_fallback(self, audio_path: str, output_path: Union[str, None] = None) -> str:
        """Phương pháp fallback để tách giọng"""
        try:
            # Tải âm thanh
            audio, sr = self.load_audio(audio_path)
            
            # Sử dụng harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Harmonic component thường chứa giọng hát
            vocals = harmonic
            
            # Normalize audio
            vocals = librosa.util.normalize(vocals)
            
            # Lưu file giọng đã tách
            if output_path is None:
                output_path = audio_path.replace('.wav', '_vocals.wav').replace('.mp3', '_vocals.wav')
            
            sf.write(output_path, vocals, sr)
            print(f"Vocals separated (fallback) and saved at: {output_path}")
            return output_path
            
        except Exception as e:
            raise Exception(f"Error in fallback vocal separation: {e}")
    
    def separate_vocals(self, audio_path: str, output_path: Union[str, None] = None) -> str:
        """Tách giọng hát - Fast Mode hoặc AI Mode"""
        try:
            if self.fast_mode:
                logger.info("🚀 Sử dụng Fast Mode để tách giọng hát...")
                return self._separate_vocals_fast(audio_path, output_path)
            else:
                logger.info("🎤 Bắt đầu tách giọng hát bằng AI Audio Separator...")
                
                # Kiểm tra Audio Separator có khả dụng không
                if not self.audio_separator.available:
                    logger.warning("⚠️ Audio Separator không khả dụng, sử dụng fallback method")
                    return self._separate_vocals_fallback(audio_path, output_path)
                
                # Sử dụng AI Audio Separator
                logger.info("✅ Sử dụng AI Audio Separator model...")
                vocals_path = self.audio_separator.separate_vocals_ai(audio_path, "mp3")
                
                # Nếu có output_path được chỉ định, copy file
                if output_path and vocals_path != output_path:
                    import shutil
                    shutil.copy2(vocals_path, output_path)
                    vocals_path = output_path
                
                logger.info(f"✅ AI Vocals separated and saved at: {vocals_path}")
                return vocals_path
                
        except Exception as e:
            logger.error(f"❌ Error in AI vocal separation: {e}")
            logger.info("🔄 Chuyển sang fallback method...")
            return self._separate_vocals_fallback(audio_path, output_path)
    
    def _separate_vocals_fast(self, audio_path: str, output_path: Union[str, None] = None) -> str:
        """Fast Mode - Tách giọng nhanh với chất lượng chấp nhận được"""
        try:
            logger.info("⚡ Fast Mode: Tách giọng với tốc độ cao...")
            
            # Load audio với sample rate thấp hơn để tăng tốc
            audio, sr = librosa.load(audio_path, sr=16000, mono=False)  # Giảm từ 22050 xuống 16000
            
            # Chuyển đổi sang mono nếu cần
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Sử dụng HPSS với tham số tối ưu cho tốc độ
            harmonic, percussive = librosa.effects.hpss(audio, margin=(1, 1))  # Giảm margin để tăng tốc
            
            # Kết hợp harmonic và một phần percussive để giữ vocals
            vocals_audio = harmonic + 0.3 * percussive  # Thêm một chút percussive để giữ vocals
            
            # Tạo output path
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(self.output_dir, f"{base_name}_vocals_fast.wav")
            
            # Save vocals với sample rate thấp hơn
            sf.write(output_path, vocals_audio, sr)
            
            # Đảm bảo đường dẫn là absolute path
            output_path = os.path.abspath(output_path)
            
            logger.info(f"⚡ Fast vocals saved at: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error in fast vocal separation: {e}")
            return self._separate_vocals_fallback(audio_path, output_path)
    
    def _separate_vocals_fallback(self, audio_path: str, output_path: Union[str, None] = None) -> str:
        """Fallback method sử dụng librosa"""
        try:
            logger.info("🔄 Sử dụng fallback method (librosa)...")
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050, mono=False)
            
            # Chuyển đổi sang mono nếu cần
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Harmonic-percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Sử dụng harmonic component làm vocals
            vocals_audio = harmonic
            
            # Tạo output path
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                output_path = os.path.join(self.output_dir, f"{base_name}_vocals_fallback.wav")
            
            # Save vocals
            sf.write(output_path, vocals_audio, sr)
            
            # Đảm bảo đường dẫn là absolute path
            output_path = os.path.abspath(output_path)
            
            logger.info(f"✅ Fallback vocals saved at: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Error in fallback vocal separation: {e}")
            raise
    
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
    
    def cleanup_temp_files(self):
        """Dọn dẹp các file tạm"""
        try:
            if os.path.exists(self.output_dir):
                import shutil
                shutil.rmtree(self.output_dir)
                os.makedirs(self.output_dir, exist_ok=True)
                print("Temporary files cleaned up")
        except Exception as e:
            print(f"Error during cleanup: {e}")
