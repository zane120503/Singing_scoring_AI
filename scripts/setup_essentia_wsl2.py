#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WSL2 Essentia Installation Guide
"""

def print_wsl2_guide():
    """Print WSL2 installation guide"""
    print("ESSENTIA INSTALLATION WITH WSL2")
    print("=" * 40)
    
    print("\nSTEP 1: Install WSL2")
    print("1. Open PowerShell as Administrator")
    print("2. Run: wsl --install")
    print("3. Restart your computer")
    print("4. Set up Ubuntu when prompted")
    
    print("\nSTEP 2: Install Essentia in WSL2")
    print("1. Open Ubuntu terminal")
    print("2. Update system:")
    print("   sudo apt update")
    print("   sudo apt upgrade -y")
    
    print("\n3. Install Python and pip:")
    print("   sudo apt install python3 python3-pip -y")
    
    print("\n4. Install Essentia:")
    print("   pip install essentia")
    
    print("\nSTEP 3: Use Essentia from Windows")
    print("1. Access WSL2 files from Windows:")
    print("   \\\\wsl$\\Ubuntu\\home\\username")
    
    print("\n2. Run Python scripts in WSL2:")
    print("   wsl python your_script.py")
    
    print("\nSTEP 4: Alternative - Use Windows Terminal")
    print("1. Install Windows Terminal from Microsoft Store")
    print("2. Open Ubuntu tab")
    print("3. Run your Python scripts there")

def create_wsl2_script():
    """Create script to run in WSL2"""
    script_content = '''#!/bin/bash
# Essentia installation script for WSL2 Ubuntu

echo "Installing Essentia in WSL2 Ubuntu..."

# Update system
sudo apt update
sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-dev -y

# Install build tools
sudo apt install build-essential cmake git -y

# Install Essentia
pip install essentia

# Test installation
python3 -c "import essentia.standard as es; print('Essentia installed successfully!')"

echo "Installation complete!"
'''
    
    with open("install_essentia_wsl2.sh", "w") as f:
        f.write(script_content)
    
    print("WSL2 installation script created: install_essentia_wsl2.sh")

def create_windows_wrapper():
    """Create Windows wrapper to use WSL2 Essentia"""
    wrapper_content = '''import subprocess
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
            wsl_path = audio_path.replace("C:", "/mnt/c").replace("\\", "/")
            
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
            wsl_script = temp_script.replace("C:", "/mnt/c").replace("\\", "/")
            
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
'''
    
    with open("essentia_wsl_wrapper.py", "w") as f:
        f.write(wrapper_content)
    
    print("WSL2 wrapper created: essentia_wsl_wrapper.py")

def main():
    """Main function"""
    print_wsl2_guide()
    create_wsl2_script()
    create_windows_wrapper()
    
    print("\n" + "="*50)
    print("QUICK START:")
    print("1. Install WSL2: wsl --install")
    print("2. Run in Ubuntu: bash install_essentia_wsl2.sh")
    print("3. Use wrapper: python essentia_wsl_wrapper.py")

if __name__ == "__main__":
    main()
