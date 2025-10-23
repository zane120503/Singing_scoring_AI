#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Essentia Docker wrapper
"""

import subprocess
import os
import tempfile
import numpy as np
import librosa
import soundfile as sf

class EssentiaDockerWrapper:
    """Wrapper to use Essentia via Docker"""
    
    def __init__(self):
        self.container_name = "essentia-karaoke"
        self.setup_docker()
    
    def setup_docker(self):
        """Setup Docker container"""
        try:
            # Check if container exists
            result = subprocess.run(f"docker ps -a --filter name={self.container_name}", 
                                  shell=True, capture_output=True, text=True)
            if self.container_name not in result.stdout:
                print("Creating Docker container...")
                # Create container
                subprocess.run(f"docker run -d --name {self.container_name} essentia-karaoke", 
                             shell=True)
            else:
                print("Docker container already exists")
        except:
            pass
    
    def detect_key(self, audio_path):
        """Detect key using Essentia in Docker"""
        try:
            # Convert Windows path to Docker path
            docker_path = f"/app/{os.path.basename(audio_path)}"
            
            # Copy file to container
            subprocess.run(f"docker cp {audio_path} {self.container_name}:{docker_path}", 
                         shell=True, check=True)
            
            # Run key detection
            cmd = f"docker exec {self.container_name} python3 -c \"import essentia.standard as es; audio = es.MonoLoader(filename='{docker_path}')(); key, scale, strength = es.KeyExtractor()(audio); print(f'{{key}} {{scale}} {{strength}}')\""
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                if len(parts) >= 3:
                    return {
                        'key': parts[0],
                        'scale': parts[1],
                        'confidence': float(parts[2]),
                        'method': 'Essentia Docker'
                    }
            
            return None
            
        except Exception as e:
            print(f"Error: {e}")
            return None

def create_test_audio():
    """Create test audio file"""
    print("Creating test audio...")
    
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Beat nhạc (C major chord)
    beat = (0.3 * np.sin(2 * np.pi * 261.63 * t) +  # C4
            0.2 * np.sin(2 * np.pi * 329.63 * t) +  # E4
            0.2 * np.sin(2 * np.pi * 392.00 * t))    # G4
    
    # Giọng hát (C major scale)
    vocals = np.zeros_like(t)
    scale_notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]  # C-D-E-F-G-A-B
    for i, freq in enumerate(scale_notes):
        start_time = i * duration / len(scale_notes)
        end_time = (i + 1) * duration / len(scale_notes)
        mask = (t >= start_time) & (t < end_time)
        vocals[mask] = 0.3 * np.sin(2 * np.pi * freq * t[mask])
    
    # Karaoke (beat + vocals)
    karaoke = beat + vocals
    karaoke = karaoke / np.max(np.abs(karaoke))
    
    # Lưu file
    sf.write('test_beat.wav', beat, sr)
    sf.write('test_karaoke.wav', karaoke, sr)
    print("Test audio files created!")

def test_essentia_docker():
    """Test Essentia Docker wrapper"""
    print("TESTING ESSENTIA DOCKER WRAPPER")
    print("=" * 40)
    
    # Create test audio
    create_test_audio()
    
    # Test wrapper
    wrapper = EssentiaDockerWrapper()
    
    # Test beat detection
    print("\nTesting beat key detection...")
    beat_result = wrapper.detect_key('test_beat.wav')
    if beat_result:
        print(f"Beat key: {beat_result['key']} {beat_result['scale']} (confidence: {beat_result['confidence']:.3f})")
    else:
        print("Failed to detect beat key")
    
    # Test karaoke detection
    print("\nTesting karaoke key detection...")
    karaoke_result = wrapper.detect_key('test_karaoke.wav')
    if karaoke_result:
        print(f"Karaoke key: {karaoke_result['key']} {karaoke_result['scale']} (confidence: {karaoke_result['confidence']:.3f})")
    else:
        print("Failed to detect karaoke key")
    
    # Cleanup
    for file in ['test_beat.wav', 'test_karaoke.wav']:
        if os.path.exists(file):
            os.remove(file)
    
    print("\nEssentia Docker test completed!")

if __name__ == "__main__":
    test_essentia_docker()
