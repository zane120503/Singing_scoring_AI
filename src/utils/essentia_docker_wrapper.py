
import subprocess
import os
import tempfile

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
                # Create container
                subprocess.run(f"docker run -d --name {self.container_name} mtgupf/essentia:latest", 
                             shell=True)
        except:
            pass
    
    def detect_key(self, audio_path):
        """Detect key using Essentia in Docker"""
        try:
            # Copy file to container
            subprocess.run(f"docker cp {audio_path} {self.container_name}:/tmp/audio.wav", 
                         shell=True)
            
            # Run key detection
            cmd = f"docker exec {self.container_name} python -c "import essentia.standard as es; audio = es.MonoLoader(filename='/tmp/audio.wav')(); key, scale, strength = es.KeyExtractor()(audio); print(f'{{key}} {{scale}} {{strength}}')""
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                parts = result.stdout.strip().split()
                return {
                    'key': parts[0],
                    'scale': parts[1],
                    'confidence': float(parts[2]),
                    'method': 'Essentia Docker'
                }
            else:
                return None
        except:
            return None

# Usage example
if __name__ == "__main__":
    wrapper = EssentiaDockerWrapper()
    result = wrapper.detect_key("test_audio.wav")
    print(result)
