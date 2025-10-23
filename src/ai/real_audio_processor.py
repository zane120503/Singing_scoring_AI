#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import librosa
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
from typing import Tuple, Union
import warnings
warnings.filterwarnings("ignore")

# Thêm đường dẫn đến Audio_separator_ui
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Audio_separator_ui'))

class RealAudioProcessor:
    """Real Audio Processor using actual Audio_separator_ui AI models"""
    
    def __init__(self, audio_separator_path='Audio_separator_ui'):
        self.audio_separator_path = audio_separator_path
        self.mdxnet_models_dir = os.path.join(audio_separator_path, 'mdx_models')
        self.output_dir = os.path.join(os.getcwd(), 'temp_output')
        os.makedirs(self.output_dir, exist_ok=True)
        self.mdx_model_params = None
        # Force GPU usage
        from src.core.gpu_config import force_cuda, CUDA_AVAILABLE
        force_cuda()
        self.device_base = "cuda" if CUDA_AVAILABLE else "cpu"
        
        # Import Audio Separator classes
        self._import_audio_separator_classes()
        self._initialize_models()
        
        print(f"RealAudioProcessor initialized with device: {self.device_base}")
    
    def _import_audio_separator_classes(self):
        """Import necessary classes from Audio Separator"""
        try:
            # Import MDXModel and MDX classes
            from Audio_separator_ui.app import MDXModel, MDX, convert_to_stereo_and_wav
            self.MDXModel = MDXModel
            self.MDX = MDX
            self.convert_to_stereo_and_wav = convert_to_stereo_and_wav
            print("✅ Audio Separator classes imported successfully!")
        except ImportError as e:
            print(f"❌ Failed to import Audio Separator classes: {e}")
            raise
    
    def _initialize_models(self):
        """Initialize model configuration"""
        try:
            if os.path.exists(os.path.join(self.mdxnet_models_dir, "data.json")):
                with open(os.path.join(self.mdxnet_models_dir, "data.json")) as infile:
                    self.mdx_model_params = json.load(infile)
                print("✅ Model configuration loaded successfully!")
            else:
                print("❌ Model configuration not found!")
                self.mdx_model_params = None
        except Exception as e:
            print(f"❌ Error loading model configuration: {e}")
            self.mdx_model_params = None
    
    def load_audio(self, audio_path: str, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=False)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def _get_hash(self, file_path: str) -> str:
        """Get hash of file (simplified version)"""
        try:
            return self.MDX.get_hash(file_path)
        except:
            # Fallback hash calculation
            import hashlib
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
    
    def separate_vocals_real(self, audio_path: str, output_path: Union[str, None] = None) -> str:
        """Real vocal separation using Audio Separator AI models"""
        try:
            print("🎵 Starting real AI vocal separation...")
            
            # Convert to stereo WAV format
            converted_path = self.convert_to_stereo_and_wav(audio_path)
            print(f"✅ Audio converted: {converted_path}")
            
            # Generate unique song ID
            hash_audio = str(self._get_hash(converted_path))
            song_id = hash_audio + "mdx"
            song_output_dir = os.path.join(self.output_dir, song_id)
            os.makedirs(song_output_dir, exist_ok=True)
            
            # Load model parameters
            if not self.mdx_model_params:
                raise Exception("Model parameters not loaded")
            
            print("🎤 Processing vocal separation with AI models...")
            
            # Use the main vocal separation model
            vocals_path, instrumentals_path = self._run_mdx_separation(
                converted_path, 
                song_output_dir,
                "UVR-MDX-NET-Voc_FT.onnx"  # Main vocal model
            )
            
            if output_path is None:
                output_path = os.path.join(self.output_dir, f"{os.path.basename(audio_path).split('.')[0]}_vocals_real.wav")
            
            # Copy vocals to desired output path
            if vocals_path and os.path.exists(vocals_path):
                import shutil
                shutil.copy2(vocals_path, output_path)
                print(f"✅ Real AI vocal separation completed: {output_path}")
                return output_path
            else:
                raise Exception("Vocal separation failed - no output generated")
                
        except Exception as e:
            print(f"❌ Error in real vocal separation: {e}")
            print("🔄 Falling back to traditional method...")
            return self._separate_vocals_fallback(audio_path, output_path)
    
    def _run_mdx_separation(self, audio_path: str, output_dir: str, model_name: str) -> Tuple[str, str]:
        """Run MDX separation using real AI models"""
        try:
            # Model path
            model_path = os.path.join(self.mdxnet_models_dir, model_name)
            
            if not os.path.exists(model_path):
                # Try alternative model names
                alternative_models = [
                    "UVR-MDX-NET-Voc_FT.onnx",
                    "UVR-MDX-NET-Inst_HQ_4.onnx", 
                    "UVR_MDXNET_KARA_2.onnx"
                ]
                for alt_model in alternative_models:
                    alt_path = os.path.join(self.mdxnet_models_dir, alt_model)
                    if os.path.exists(alt_path):
                        model_path = alt_path
                        model_name = alt_model
                        break
                else:
                    raise Exception(f"Model {model_name} not found")
            
            print(f"🎯 Using model: {model_name}")
            
            # Get model parameters
            model_hash = self._get_hash(model_path)
            mp = self.mdx_model_params.get(model_hash)
            
            if not mp:
                raise Exception(f"Model parameters not found for hash: {model_hash}")
            
            # Setup device
            if self.device_base == "cuda":
                device = torch.device("cuda:0")
                processor_num = 0
                provider = ["CUDAExecutionProvider"]
            else:
                device = torch.device("cpu")
                processor_num = -1
                provider = ["CPUExecutionProvider"]
            
            # Create model
            model = self.MDXModel(
                device,
                dim_f=mp["mdx_dim_f_set"],
                dim_t=2 ** mp["mdx_dim_t_set"],
                n_fft=mp["mdx_n_fft_scale_set"],
                stem_name=mp["primary_stem"],
                compensation=mp["compensate"],
            )
            
            # Create MDX instance
            mdx_sess = self.MDX(model_path, model, processor=processor_num)
            
            # Load and process audio
            wave, sr = librosa.load(audio_path, mono=False, sr=44100)
            
            # Normalize input
            peak = max(np.max(wave), abs(np.min(wave)))
            if peak > 0:
                wave /= peak
            
            # Process with denoising
            print("🔄 Processing audio with AI model...")
            wave_processed = -(mdx_sess.process_wave(-wave, 1)) + (mdx_sess.process_wave(wave, 1))
            
            # Generate output paths
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            vocals_path = os.path.join(output_dir, f"{base_name}_Vocals.wav")
            instrumentals_path = os.path.join(output_dir, f"{base_name}_Instrumental.wav")
            
            # Save vocals (inverted)
            vocals = -wave_processed
            sf.write(vocals_path, vocals.T, sr)
            
            # Save instrumentals
            instrumentals = wave_processed
            sf.write(instrumentals_path, instrumentals.T, sr)
            
            print(f"✅ AI separation completed:")
            print(f"   Vocals: {vocals_path}")
            print(f"   Instrumentals: {instrumentals_path}")
            
            return vocals_path, instrumentals_path
            
        except Exception as e:
            print(f"❌ MDX separation error: {e}")
            raise
    
    def _separate_vocals_fallback(self, audio_path: str, output_path: Union[str, None] = None) -> str:
        """Fallback vocal separation using traditional methods"""
        try:
            print("🔄 Using fallback vocal separation...")
            
            # Load audio
            audio, sr = self.load_audio(audio_path)
            if audio is None:
                raise Exception(" Speech separation: Failed to load audio")
            
            # Simple harmonic-percussive separation
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Harmonic-percussive source separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Use harmonic component as vocals (simplified)
            vocals = y_harmonic
            
            if output_path is None:
                output_path = os.path.join(self.output_dir, f"{os.path.basename(audio_path).split('.')[0]}_vocals_fallback.wav")
            
            # Save vocals
            sf.write(output_path, vocals, sr)
            print(f"✅ Fallback vocal separation completed: {output_path}")
            
            return output_path
            
        except Exception as e:
            raise Exception(f"Error in fallback vocal separation: {e}")
    
    def separate_vocals(self, audio_path: str, output_path: Union[str, None] = None) -> str:
        """Main vocal separation method - tries real AI first, then fallback"""
        return self.separate_vocals_real(audio_path, output_path)
    
    def get_audio_features(self, audio_path: str) -> dict:
        """Extract audio features"""
        try:
            audio, sr = self.load_audio(audio_path)
            if audio is None:
                return {}
            
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Extract features
            duration = len(audio) / sr
            rms_energy = np.mean(librosa.feature.rms(y=audio))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'rms_energy': float(rms_energy),
                'zero_crossing_rate': float(zero_crossing_rate),
                'spectral_centroid': float(spectral_centroid),
                'spectral_rolloff': float(spectral_rolloff),
                'mfcc': mfcc.tolist()
            }
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return {}
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import shutil
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
                os.makedirs(self.output_dir, exist_ok=True)
                print("✅ Temporary files cleaned up")
        except Exception as e:
            print(f"Error cleaning up temp files: {e}")
