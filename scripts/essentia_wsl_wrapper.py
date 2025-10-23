import subprocess
import os
import tempfile

class EssentiaWSLWrapper:
    """Wrapper to use Essentia via WSL2"""
    
    def __init__(self):
        self.wsl_python = "wsl python3"
    
    def detect_key(self, audio_path):
        """Detect key using Essentia in WSL2"""
        try:
            # Convert Windows path to WSL path
            wsl_path = audio_path.replace("C:", "/mnt/c").replace("\", "/")
            
            # Create Python script for WSL
            script = f"""
import essentia.standard as es
import sys

try:
    audio = es.MonoLoader(filename='{wsl_path}')()
    key, scale, strength = es.KeyExtractor()(audio)
    print(f"{{key}} {{scale}} {{strength}}")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
            
            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script)
                temp_script = f.name
            
            # Convert temp file path to WSL
            wsl_script = temp_script.replace("C:", "/mnt/c").replace("\", "/")
            
            # Run script in WSL
            cmd = f"wsl python3 {wsl_script}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            # Clean up
            os.unlink(temp_script)
            
            if result.returncode == 0 and "ERROR" not in result.stdout:
                parts = result.stdout.strip().split()
                if len(parts) >= 3:
                    return {
                        'key': parts[0],
                        'scale': parts[1],
                        'confidence': float(parts[2]),
                        'method': 'Essentia WSL2'
                    }
            
            return None
            
        except Exception as e:
            print(f"Error: {e}")
            return None

# Usage example
if __name__ == "__main__":
    wrapper = EssentiaWSLWrapper()
    result = wrapper.detect_key("test_audio.wav")
    print(result)
