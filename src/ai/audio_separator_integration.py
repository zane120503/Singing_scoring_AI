"""
Audio Separator Integration Module
Tích hợp Audio Separator UI vào hệ thống karaoke scoring
"""

import os
import sys
import logging
import json
import hashlib
import shutil
import librosa
import soundfile as sf
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path

# Thêm Audio_separator_ui vào Python path
audio_separator_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Audio_separator_ui')
sys.path.insert(0, audio_separator_path)

try:
    from app import process_uvr_task, MDX, MDXModel, convert_to_stereo_and_wav
    from utils import create_directories, remove_directory_contents
    AUDIO_SEPARATOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Audio Separator UI khong kha dung: {e}")
    AUDIO_SEPARATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

class AudioSeparatorIntegration:
    """Tích hợp Audio Separator UI vào hệ thống"""
    
    def __init__(self, fast_mode=False):
        self.available = AUDIO_SEPARATOR_AVAILABLE
        self.model_params = None
        self.models_dir = None
        self.output_dir = None
        self.fast_mode = fast_mode  # Tùy chọn chế độ nhanh
        
        if self.available:
            self._initialize_audio_separator()
    
    def _initialize_audio_separator(self):
        """Khởi tạo Audio Separator"""
        try:
            # Đường dẫn đến Audio_separator_ui
            audio_separator_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Audio_separator_ui')
            self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets', 'models', 'mdx_models')
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output', 'clean_song_output')
            
            # Tạo thư mục output nếu chưa có
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Load model parameters
            data_json_path = os.path.join(self.models_dir, 'data.json')
            if os.path.exists(data_json_path):
                with open(data_json_path, 'r') as f:
                    self.model_params = json.load(f)
                logger.info("Audio Separator model parameters loaded successfully")
            else:
                logger.error("Khong tim thay data.json trong mdx_models")
                self.available = False
                
        except Exception as e:
            logger.error(f"Loi khoi tao Audio Separator: {e}")
            self.available = False
    
    def separate_vocals_ai(self, input_file, output_format="wav"):
        """
        Tách giọng hát sử dụng AI Audio Separator
        
        Args:
            input_file (str): Đường dẫn file âm thanh đầu vào
            output_format (str): Định dạng file đầu ra (wav, mp3)
            
        Returns:
            str: Đường dẫn file vocals đã tách
        """
        if not self.available:
            raise Exception("Audio Separator không khả dụng")
        
        try:
            logger.info("Bat dau tach giong hat bang AI Audio Separator...")
            
            # Tạo unique song ID
            song_id = self._get_file_hash(input_file)
            
            # Chuyển đổi sang stereo WAV nếu cần
            stereo_file = convert_to_stereo_and_wav(input_file)
            
            # Thay đổi working directory để process_uvr_task hoạt động đúng
            original_cwd = os.getcwd()
            audio_separator_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Audio_separator_ui')
            os.chdir(audio_separator_dir)
            
            try:
                # Sử dụng wrapper tùy chỉnh chỉ tách vocals
                final_vocals_path = self._separate_vocals_only(stereo_file, song_id, audio_separator_dir)
            finally:
                # Khôi phục working directory
                os.chdir(original_cwd)
            
            # Chuyển đổi đường dẫn thành absolute path
            if not os.path.isabs(final_vocals_path):
                final_vocals_path = os.path.abspath(final_vocals_path)
            
            # Đảm bảo đường dẫn đúng
            if not os.path.exists(final_vocals_path):
                # Thử đường dẫn trong Audio_separator_ui
                alt_path = os.path.join(audio_separator_dir, final_vocals_path)
                if os.path.exists(alt_path):
                    final_vocals_path = alt_path
                else:
                    # Thử đường dẫn tương đối từ clean_song_output
                    rel_path = os.path.join("clean_song_output", final_vocals_path)
                    alt_path = os.path.join(audio_separator_dir, rel_path)
                    if os.path.exists(alt_path):
                        final_vocals_path = alt_path
                    else:
                        # Thử đường dẫn với clean_song_output prefix
                        clean_song_path = os.path.join(audio_separator_dir, "clean_song_output")
                        alt_path = os.path.join(clean_song_path, os.path.basename(final_vocals_path))
                        if os.path.exists(alt_path):
                            final_vocals_path = alt_path
                        else:
                            # Tìm file vocals trong clean_song_output
                            clean_song_path = os.path.join(audio_separator_dir, "clean_song_output")
                            if os.path.exists(clean_song_path):
                                for root, dirs, files in os.walk(clean_song_path):
                                    for file in files:
                                        if "Vocals_DeReverb" in file and file.endswith('.wav'):
                                            full_path = os.path.join(root, file)
                                            final_vocals_path = full_path
                                            logger.info(f"Found vocals file: {final_vocals_path}")
                                            break
                                    if final_vocals_path and os.path.exists(final_vocals_path):
                                        break
            
            # Chuyển đổi sang MP3 và xóa file WAV gốc
            if output_format.lower() != "wav":
                mp3_path = self._convert_format(final_vocals_path, output_format)
                # Xóa file WAV gốc sau khi convert thành công
                if mp3_path != final_vocals_path and os.path.exists(mp3_path):
                    try:
                        os.remove(final_vocals_path)
                        logger.info(f"Removed original WAV file: {final_vocals_path}")
                        final_vocals_path = mp3_path
                    except Exception as e:
                        logger.warning(f"Could not remove WAV file: {e}")
                        final_vocals_path = mp3_path
            
            logger.info(f"AI Vocal separation hoan thanh: {final_vocals_path}")
            return final_vocals_path
            
        except Exception as e:
            logger.error(f"Loi trong AI vocal separation: {e}")
            raise
    
    def _separate_vocals_only(self, stereo_file, song_id, audio_separator_dir):
        """
        Tách vocals chỉ tạo ra file vocals cần thiết, không tạo các file thừa
        """
        try:
            # Load model parameters
            with open(os.path.join(self.models_dir, "data.json")) as infile:
                mdx_model_params = json.load(infile)
            
            # Tạo thư mục output
            song_output_dir = os.path.join(audio_separator_dir, "clean_song_output", f"{song_id}_mdx")
            os.makedirs(song_output_dir, exist_ok=True)
            
            # Device
            # Force GPU usage
            from src.core.gpu_config import force_cuda, CUDA_AVAILABLE
            force_cuda()
            device_base = "cuda" if CUDA_AVAILABLE else "cpu"
            
            # Chuẩn hóa tên input về ASCII để tránh lỗi tên file Unicode
            ascii_input_path = os.path.join(song_output_dir, "input.wav")
            try:
                if os.path.exists(ascii_input_path):
                    os.remove(ascii_input_path)
                shutil.copyfile(stereo_file, ascii_input_path)
                source_path_for_mdx = ascii_input_path
            except Exception:
                # Nếu copy thất bại, fallback dùng đường dẫn gốc
                source_path_for_mdx = stereo_file

            # Step 1: Tách vocals từ file gốc
            logger.info("Vocal Track Isolation...")
            vocals_path, instrumentals_path = self._run_mdx_vocals_only(
                mdx_model_params,
                song_output_dir,
                os.path.join(self.models_dir, "UVR-MDX-NET-Voc_FT.onnx"),
                source_path_for_mdx,
                device_base
            )
            
            # Đợi một chút để đảm bảo file được tạo hoàn chỉnh (giảm delay để tăng tốc)
            import time
            time.sleep(0.1 if self.fast_mode else 0.5)
            
            # Xóa file instrumentals ngay
            if os.path.exists(instrumentals_path):
                os.remove(instrumentals_path)
                logger.info(f"Removed instrumentals file: {os.path.basename(instrumentals_path)}")
            
            # Step 2: Áp dụng dereverb
            logger.info("Vocal Clarity Enhancement through De-Reverberation...")
            _, vocals_dereverb_path = self._run_mdx_dereverb_only(
                mdx_model_params,
                song_output_dir,
                os.path.join(self.models_dir, "Reverb_HQ_By_FoxJoy.onnx"),
                vocals_path,
                device_base
            )
            
            # Đợi một chút để đảm bảo file được tạo hoàn chỉnh (giảm delay để tăng tốc)
            import time
            time.sleep(0.1 if self.fast_mode else 0.5)
            
            # Xóa file vocals thô ngay
            if os.path.exists(vocals_path):
                os.remove(vocals_path)
                logger.info(f"Removed raw vocals file: {os.path.basename(vocals_path)}")
            
            # Đảm bảo file vocals_dereverb_path tồn tại và có thể đọc được
            if not os.path.exists(vocals_dereverb_path):
                raise Exception(f"Vocals dereverb file not found: {vocals_dereverb_path}")
            
            # Kiểm tra file có thể đọc được không
            try:
                with open(vocals_dereverb_path, 'rb') as f:
                    f.read(1024)  # Đọc 1KB để kiểm tra
            except Exception as e:
                raise Exception(f"Cannot read vocals dereverb file: {e}")
            
            return vocals_dereverb_path
            
        except Exception as e:
            logger.error(f"Error in _separate_vocals_only: {e}")
            raise
    
    def _run_mdx_vocals_only(self, mdx_model_params, song_output_dir, model_path, orig_song_path, device_base):
        """Chạy MDX chỉ để tách vocals"""
        try:
            # Import các hàm cần thiết
            from app import run_mdx
            
            # Chạy MDX để tách vocals
            vocals_path, instrumentals_path = run_mdx(
                mdx_model_params,
                song_output_dir,
                model_path,
                orig_song_path,
                denoise=True,
                keep_orig=True,
                device_base=device_base,
            )
            
            return vocals_path, instrumentals_path
            
        except Exception as e:
            logger.error(f"Error in _run_mdx_vocals_only: {e}")
            raise
    
    def _run_mdx_dereverb_only(self, mdx_model_params, song_output_dir, model_path, vocals_path, device_base):
        """Chạy MDX chỉ để áp dụng dereverb"""
        try:
            # Import các hàm cần thiết
            from app import run_mdx
            
            # Chạy MDX để áp dụng dereverb
            _, vocals_dereverb_path = run_mdx(
                mdx_model_params,
                song_output_dir,
                model_path,
                vocals_path,
                invert_suffix="DeReverb",
                exclude_main=True,
                denoise=True,
                device_base=device_base,
            )
            
            return _, vocals_dereverb_path
            
        except Exception as e:
            logger.error(f"Error in _run_mdx_dereverb_only: {e}")
            raise
    
    def _get_file_hash(self, file_path):
        """Tạo hash cho file"""
        with open(file_path, 'rb') as f:
            file_hash = hashlib.blake2b()
            while chunk := f.read(8192):
                file_hash.update(chunk)
        return file_hash.hexdigest()[:18]
    
    def _convert_format(self, input_path, target_format):
        """Chuyển đổi format file"""
        try:
            # Load audio
            audio, sr = sf.read(input_path)
            
            # Tạo output path
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_converted.{target_format.lower()}"
            
            # Save với format mới
            sf.write(output_path, audio, sr, format=target_format.upper())
            
            logger.info(f"Converted to {target_format}: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Loi convert format: {e}")
            return input_path
    
    def get_status(self):
        """Trả về trạng thái của Audio Separator"""
        if not self.available:
            return "Audio Separator: Khong kha dung"
        
        # Kiểm tra models
        model_files = [
            "UVR-MDX-NET-Voc_FT.onnx",
            "UVR_MDXNET_KARA_2.onnx", 
            "Reverb_HQ_By_FoxJoy.onnx",
            "UVR-MDX-NET-Inst_HQ_4.onnx"
        ]
        
        missing_models = []
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            if not os.path.exists(model_path):
                missing_models.append(model_file)
        
        if missing_models:
            return f"Audio Separator: Thieu models ({len(missing_models)} files)"
        else:
            return "Audio Separator: San sang"
    
    def cleanup_temp_files(self, song_id):
        """Dọn dẹp file tạm"""
        try:
            song_output_dir = os.path.join(self.output_dir, f"{song_id}_mdx")
            if os.path.exists(song_output_dir):
                remove_directory_contents(song_output_dir)
                logger.info(f"Cleaned up temp files: {song_output_dir}")
        except Exception as e:
            logger.warning(f"Khong the cleanup temp files: {e}")

# Test function
def test_audio_separator_integration():
    """Test tích hợp Audio Separator"""
    print("Testing Audio Separator Integration...")
    
    separator = AudioSeparatorIntegration()
    print(f"Status: {separator.get_status()}")
    
    if separator.available:
        print("Audio Separator Integration san sang!")
        
        # Test với file mẫu nếu có
        test_file = "test.mp3"
        if os.path.exists(test_file):
            try:
                vocals_path = separator.separate_vocals_ai(test_file, "mp3")
                print(f"Test thanh cong: {vocals_path}")
            except Exception as e:
                print(f"Test that bai: {e}")
        else:
            print("Khong co file test.mp3 de test")
    else:
        print("Audio Separator Integration khong kha dung")

if __name__ == "__main__":
    test_audio_separator_integration()
