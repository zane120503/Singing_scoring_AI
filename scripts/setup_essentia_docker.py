#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docker-based Essentia solution
"""

import subprocess
import os
import json

def check_docker():
    """Check if Docker is installed"""
    try:
        result = subprocess.run("docker --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Docker found: {result.stdout.strip()}")
            return True
        else:
            print("Docker not found")
            return False
    except:
        print("Docker not found")
        return False

def create_docker_setup():
    """Create Docker setup for Essentia"""
    print("CREATING DOCKER SETUP FOR ESSENTIA")
    print("=" * 40)
    
    # Create Dockerfile
    dockerfile_content = """
FROM mtgupf/essentia:latest

# Install additional Python packages
RUN pip install librosa soundfile numpy scipy matplotlib

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Expose port for web interface
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Create docker-compose.yml
    compose_content = """
version: '3.8'

services:
  essentia-app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./temp_output:/app/temp_output
      - ./input_files:/app/input_files
    environment:
      - PYTHONUNBUFFERED=1
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose_content)
    
    print("Docker files created!")
    return True

def run_essentia_docker():
    """Run Essentia in Docker"""
    print("RUNNING ESSENTIA IN DOCKER")
    print("=" * 30)
    
    if not check_docker():
        print("Please install Docker Desktop first:")
        print("https://www.docker.com/products/docker-desktop/")
        return False
    
    # Create directories
    os.makedirs("temp_output", exist_ok=True)
    os.makedirs("input_files", exist_ok=True)
    
    # Build and run Docker container
    commands = [
        "docker build -t essentia-karaoke .",
        "docker run -it -p 8000:8000 -v ${PWD}/temp_output:/app/temp_output -v ${PWD}/input_files:/app/input_files essentia-karaoke"
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"Command failed: {cmd}")
                return False
        except Exception as e:
            print(f"Exception: {e}")
            return False
    
    return True

def create_essentia_wrapper():
    """Create a wrapper to use Essentia via Docker"""
    print("CREATING ESSENTIA WRAPPER")
    print("=" * 30)
    
    wrapper_content = '''
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
            cmd = f"docker exec {self.container_name} python -c \"import essentia.standard as es; audio = es.MonoLoader(filename='/tmp/audio.wav')(); key, scale, strength = es.KeyExtractor()(audio); print(f'{{key}} {{scale}} {{strength}}')\""
            
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
'''
    
    with open("essentia_docker_wrapper.py", "w") as f:
        f.write(wrapper_content)
    
    print("Essentia Docker wrapper created!")
    return True

def main():
    """Main function"""
    print("ESSENTIA DOCKER SOLUTION")
    print("=" * 30)
    
    print("This will create a Docker-based solution for Essentia.")
    print("Requirements:")
    print("- Docker Desktop installed")
    print("- Internet connection")
    
    if not check_docker():
        print("\nPlease install Docker Desktop first:")
        print("https://www.docker.com/products/docker-desktop/")
        return
    
    print("\nCreating Docker setup...")
    create_docker_setup()
    create_essentia_wrapper()
    
    print("\nDocker setup complete!")
    print("To use Essentia:")
    print("1. docker-compose up")
    print("2. Or use essentia_docker_wrapper.py")

if __name__ == "__main__":
    main()
