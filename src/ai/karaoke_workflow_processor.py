#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Karaoke Workflow Processor - Xử lý workflow đầy đủ cho hệ thống karaoke
Workflow:
1. Input 2 file: karaoke (beat+vocal), beat
2. Xử lý song song:
   - Phát hiện key của beat file
   - Phát hiện vocal trong karaoke file
   - Cắt karaoke từ lúc vocal bắt đầu đến khi kết thúc
   - Tách vocal từ file đã cắt bằng AI
   - Phát hiện key của vocal đã tách
   - So sánh 2 key
"""

import os
import sys
import librosa
import soundfile as sf
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# Thêm path để import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.ai.advanced_key_detector import AdvancedKeyDetector
from src.ai.real_audio_processor import RealAudioProcessor
from src.ai.smart_slicing_strategy import SmartSlicingStrategy
from vocals_presence_checker import VocalsPresenceChecker

logger = logging.getLogger(__name__)

class KaraokeWorkflowProcessor:
    """Xử lý workflow đầy đủ cho hệ thống karaoke"""
    
    def __init__(self):
        """Khởi tạo processor"""
        logger.info("🔄 Khởi tạo Karaoke Workflow Processor...")
        
        # Khởi tạo các components
        self.key_detector = AdvancedKeyDetector()
        self.vocals_checker = VocalsPresenceChecker()
        self.audio_processor = RealAudioProcessor()
        self.slicing_strategy = SmartSlicingStrategy()
        
        # Thư mục tạm
        self.temp_dir = os.path.join(os.getcwd(), 'temp_workflow')
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info("✅ Karaoke Workflow Processor đã sẵn sàng!")
    
    def process(self, karaoke_file: str, beat_file: str) -> Dict:
        """
        Xử lý workflow đầy đủ
        
        Args:
            karaoke_file: Đường dẫn file karaoke (beat + vocal)
            beat_file: Đường dẫn file beat nhạc
            
        Returns:
            Dict chứa kết quả phân tích và so sánh
        """
        try:
            print("\n" + "="*70)
            print("🎵 BẮT ĐẦU WORKFLOW XỬ LÝ KARAOKE")
            print("="*70)
            
            # Validate files
            if not os.path.exists(karaoke_file):
                raise FileNotFoundError(f"File karaoke không tồn tại: {karaoke_file}")
            if not os.path.exists(beat_file):
                raise FileNotFoundError(f"File beat không tồn tại: {beat_file}")
            
            result = {
                'karaoke_file': karaoke_file,
                'beat_file': beat_file,
                'steps': {}
            }
            
            # Bước 1: Xử lý song song - Phát hiện key beat và vocal trong karaoke
            print("\n📊 BƯỚC 1: Xử lý song song - Phát hiện key beat và vocal...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Task 1: Phát hiện key của beat file
                future_beat_key = executor.submit(self._detect_beat_key, beat_file)
                
                # Task 2: Phát hiện vocal trong karaoke file
                future_vocals = executor.submit(self._detect_vocals_in_karaoke, karaoke_file)
                
                # Đợi cả 2 task hoàn thành
                beat_key_result = future_beat_key.result()
                vocals_result = future_vocals.result()
            
            result['steps']['beat_key_detection'] = beat_key_result
            result['steps']['vocals_detection'] = vocals_result
            
            if not vocals_result['has_vocals']:
                print("⚠️ Không phát hiện giọng hát trong file karaoke!")
                result['error'] = 'No vocals detected in karaoke file'
                return result
            
            # Bước 2: Cắt file karaoke từ lúc vocal bắt đầu đến khi kết thúc
            print("\n✂️ BƯỚC 2: Cắt file karaoke theo vị trí vocal...")
            sliced_file = self._slice_karaoke_file(karaoke_file, vocals_result['voice_segments'])
            result['steps']['slicing'] = sliced_file
            
            if not sliced_file['success']:
                result['error'] = 'Failed to slice karaoke file'
                return result
            
            # Bước 3: Tách vocal từ file đã cắt bằng AI
            print("\n🎤 BƯỚC 3: Tách vocal bằng AI Audio Separator...")
            separated_vocals = self._separate_vocals(sliced_file['output_path'])
            result['steps']['vocal_separation'] = separated_vocals
            
            if not separated_vocals['success']:
                result['error'] = 'Failed to separate vocals'
                return result
            
            # Bước 4: Phát hiện key của vocal đã tách
            print("\n🎹 BƯỚC 4: Phát hiện key của vocal đã tách...")
            vocal_key_result = self._detect_vocal_key(separated_vocals['vocals_path'])
            result['steps']['vocal_key_detection'] = vocal_key_result
            
            # Bước 5: So sánh 2 key
            print("\n🔍 BƯỚC 5: So sánh key giữa beat và vocal...")
            key_comparison = self._compare_keys(beat_key_result, vocal_key_result)
            result['steps']['key_comparison'] = key_comparison
            
            # Tổng hợp kết quả
            result['summary'] = {
                'beat_key': f"{beat_key_result['key']} {beat_key_result['scale']}",
                'vocal_key': f"{vocal_key_result['key']} {vocal_key_result['scale']}",
                'key_match': key_comparison['key_match'],
                'mode_match': key_comparison['mode_match'],
                'similarity_score': key_comparison['score'],
                'first_voice_time': vocals_result.get('first_voice_time', 0),
                'last_voice_time': vocals_result.get('last_voice_time', 0),
                'vocal_duration': vocals_result.get('last_voice_time', 0) - vocals_result.get('first_voice_time', 0)
            }
            
            print("\n" + "="*70)
            print("✅ HOÀN THÀNH WORKFLOW XỬ LÝ KARAOKE")
            print("="*70)
            print(f"\n📊 KẾT QUẢ:")
            print(f"   Beat Key: {result['summary']['beat_key']}")
            print(f"   Vocal Key: {result['summary']['vocal_key']}")
            print(f"   Key Match: {'✅ CÓ' if key_comparison['key_match'] else '❌ KHÔNG'}")
            print(f"   Mode Match: {'✅ CÓ' if key_comparison['mode_match'] else '❌ KHÔNG'}")
            print(f"   Similarity Score: {key_comparison['score']}/100")
            print(f"   Vocal Duration: {result['summary']['vocal_duration']:.2f}s")
            print("="*70 + "\n")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Lỗi trong workflow: {e}")
            return {
                'error': str(e),
                'karaoke_file': karaoke_file,
                'beat_file': beat_file
            }
    
    def _detect_beat_key(self, beat_file: str) -> Dict:
        """Phát hiện key của beat file"""
        try:
            print(f"   🎹 Đang phát hiện key của beat file: {os.path.basename(beat_file)}")
            key_result = self.key_detector.detect_key(beat_file, audio_type="beat")
            print(f"   ✅ Beat key: {key_result['key']} {key_result['scale']} (confidence: {key_result['confidence']:.3f})")
            return key_result
        except Exception as e:
            logger.error(f"Lỗi phát hiện beat key: {e}")
            return {'key': 'C', 'scale': 'major', 'confidence': 0.0, 'error': str(e)}
    
    def _detect_vocals_in_karaoke(self, karaoke_file: str) -> Dict:
        """Phát hiện vocal trong file karaoke"""
        try:
            print(f"   🎤 Đang phát hiện vocal trong karaoke file: {os.path.basename(karaoke_file)}")
            vocals_result = self.vocals_checker.check_vocals_presence(karaoke_file)
            return vocals_result
        except Exception as e:
            logger.error(f"Lỗi phát hiện vocal: {e}")
            return {'has_vocals': False, 'error': str(e)}
    
    def _slice_karaoke_file(self, karaoke_file: str, voice_segments: list) -> Dict:
        """Cắt file karaoke từ lúc vocal bắt đầu đến khi kết thúc"""
        try:
            if not voice_segments:
                return {'success': False, 'error': 'No voice segments found'}
            
            # Tìm đoạn vocal đầu tiên và cuối cùng
            first_seg = voice_segments[0]
            last_seg = voice_segments[-1]
            
            # Tạo output path
            base_name = os.path.splitext(os.path.basename(karaoke_file))[0]
            output_path = os.path.join(self.temp_dir, f"{base_name}_sliced.wav")
            
            # Load audio
            audio, sr = librosa.load(karaoke_file, sr=None)
            
            # Tính sample indices
            start_sample = int(first_seg['start'] * sr)
            end_sample = int(last_seg['end'] * sr)
            
            # Đảm bảo không vượt quá độ dài audio
            end_sample = min(end_sample, len(audio))
            
            # Cắt audio
            sliced_audio = audio[start_sample:end_sample]
            
            # Lưu file đã cắt
            sf.write(output_path, sliced_audio, sr)
            
            print(f"   ✅ Đã cắt file: {first_seg['start']:.2f}s - {last_seg['end']:.2f}s")
            print(f"   📁 File output: {output_path}")
            
            return {
                'success': True,
                'output_path': output_path,
                'start_time': first_seg['start'],
                'end_time': last_seg['end'],
                'duration': last_seg['end'] - first_seg['start']
            }
            
        except Exception as e:
            logger.error(f"Lỗi cắt file: {e}")
            return {'success': False, 'error': str(e)}
    
    def _separate_vocals(self, audio_path: str) -> Dict:
        """Tách vocal từ file đã cắt bằng AI"""
        try:
            # Tạo output path
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            vocals_path = os.path.join(self.temp_dir, f"{base_name}_vocals.wav")
            
            # Tách vocal
            result_path = self.audio_processor.separate_vocals_real(
                audio_path, 
                output_path=vocals_path,
                check_vocals=False  # Đã check rồi
            )
            
            if os.path.exists(result_path):
                print(f"   ✅ Đã tách vocal thành công")
                print(f"   📁 File vocals: {result_path}")
                return {
                    'success': True,
                    'vocals_path': result_path
                }
            else:
                return {'success': False, 'error': 'Vocal separation failed - no output file'}
                
        except Exception as e:
            logger.error(f"Lỗi tách vocal: {e}")
            return {'success': False, 'error': str(e)}
    
    def _detect_vocal_key(self, vocals_path: str) -> Dict:
        """Phát hiện key của vocal đã tách"""
        try:
            print(f"   🎹 Đang phát hiện key của vocal đã tách...")
            key_result = self.key_detector.detect_key(vocals_path, audio_type="vocals")
            print(f"   ✅ Vocal key: {key_result['key']} {key_result['scale']} (confidence: {key_result['confidence']:.3f})")
            return key_result
        except Exception as e:
            logger.error(f"Lỗi phát hiện vocal key: {e}")
            return {'key': 'C', 'scale': 'major', 'confidence': 0.0, 'error': str(e)}
    
    def _compare_keys(self, beat_key: Dict, vocal_key: Dict) -> Dict:
        """So sánh 2 key"""
        try:
            comparison = self.key_detector.compare_keys(beat_key, vocal_key)
            print(f"   📊 So sánh kết quả:")
            print(f"      Beat: {comparison['key1']}")
            print(f"      Vocal: {comparison['key2']}")
            print(f"      Key Match: {'✅' if comparison['key_match'] else '❌'}")
            print(f"      Mode Match: {'✅' if comparison['mode_match'] else '❌'}")
            print(f"      Similarity Score: {comparison['score']}/100")
            return comparison
        except Exception as e:
            logger.error(f"Lỗi so sánh key: {e}")
            return {
                'score': 0,
                'key_match': False,
                'mode_match': False,
                'error': str(e)
            }
    
    def cleanup_temp_files(self):
        """Dọn dẹp các file tạm"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                os.makedirs(self.temp_dir, exist_ok=True)
                logger.info("✅ Đã dọn dẹp file tạm")
        except Exception as e:
            logger.warning(f"Lỗi dọn dẹp file tạm: {e}")

