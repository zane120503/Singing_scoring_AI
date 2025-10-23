#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Essentia Installation Script for Windows
"""

import subprocess
import sys
import os

def run_command(command):
    """Run command and show output"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("SUCCESS!")
            if result.stdout:
                print(f"Output: {result.stdout}")
        else:
            print(f"ERROR: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Exception: {e}")
        return False

def install_essentia_from_source():
    """Install Essentia from source"""
    print("ESSENTIA INSTALLATION FROM SOURCE")
    print("=" * 40)
    
    # Check if we have required tools
    print("\n1. Checking required tools...")
    
    # Check Git
    if not run_command("git --version"):
        print("ERROR: Git not found. Please install Git first.")
        return False
    
    # Check CMake
    if not run_command("cmake --version"):
        print("ERROR: CMake not found. Please install CMake first.")
        return False
    
    # Check Visual Studio Build Tools
    if not run_command("where cl"):
        print("ERROR: Visual Studio Build Tools not found in PATH.")
        print("Please run Visual Studio Developer Command Prompt.")
        return False
    
    print("\n2. Cloning Essentia repository...")
    if not run_command("git clone https://github.com/MTG/essentia.git"):
        print("ERROR: Failed to clone Essentia repository.")
        return False
    
    print("\n3. Building Essentia...")
    os.chdir("essentia")
    
    # Configure
    if not run_command("python waf configure --build-static --with-example=streaming_extractor_music"):
        print("ERROR: Failed to configure Essentia.")
        return False
    
    # Build
    if not run_command("python waf"):
        print("ERROR: Failed to build Essentia.")
        return False
    
    # Install
    if not run_command("python waf install"):
        print("ERROR: Failed to install Essentia.")
        return False
    
    print("\n4. Testing installation...")
    if run_command("python -c \"import essentia.standard as es; print('Essentia installed successfully!')\""):
        print("SUCCESS: Essentia installed and working!")
        return True
    else:
        print("ERROR: Essentia installation test failed.")
        return False

def install_essentia_conda():
    """Try installing Essentia via conda"""
    print("TRYING CONDA INSTALLATION")
    print("=" * 30)
    
    # Check if conda is available
    if not run_command("conda --version"):
        print("Conda not found. Installing Miniconda...")
        
        # Download and install Miniconda
        print("Please download and install Miniconda from:")
        print("https://docs.conda.io/en/latest/miniconda.html")
        return False
    
    # Try installing Essentia via conda
    if run_command("conda install -c conda-forge essentia -y"):
        print("SUCCESS: Essentia installed via conda!")
        return True
    else:
        print("ERROR: Failed to install Essentia via conda.")
        return False

def install_essentia_wheel():
    """Try installing Essentia from wheel files"""
    print("TRYING WHEEL INSTALLATION")
    print("=" * 30)
    
    # Get Python version
    python_version = f"{sys.version_info.major}{sys.version_info.minor}"
    print(f"Python version: {python_version}")
    
    # Try different wheel URLs
    wheel_urls = [
        f"https://github.com/MTG/essentia/releases/download/v2.1b6.dev1030/essentia-2.1b6.dev1030-cp{python_version}-cp{python_version}-win_amd64.whl",
        "https://github.com/MTG/essentia/releases/download/v2.1b6.dev1030/essentia-2.1b6.dev1030-cp39-cp39-win_amd64.whl",
        "https://github.com/MTG/essentia/releases/download/v2.1b6.dev1030/essentia-2.1b6.dev1030-cp38-cp38-win_amd64.whl",
    ]
    
    for url in wheel_urls:
        print(f"Trying: {url}")
        if run_command(f"pip install {url}"):
            print("SUCCESS: Essentia installed from wheel!")
            return True
    
    print("ERROR: All wheel installations failed.")
    return False

def main():
    """Main function"""
    print("ESSENTIA INSTALLATION FOR WINDOWS")
    print("=" * 40)
    
    # Method 1: Try wheel installation first
    print("\nMETHOD 1: Wheel Installation")
    if install_essentia_wheel():
        return
    
    # Method 2: Try conda installation
    print("\nMETHOD 2: Conda Installation")
    if install_essentia_conda():
        return
    
    # Method 3: Build from source
    print("\nMETHOD 3: Build from Source")
    print("WARNING: This will take a long time and requires:")
    print("- Git")
    print("- CMake")
    print("- Visual Studio Build Tools")
    print("- Python development headers")
    
    choice = input("\nDo you want to try building from source? (y/n): ").lower()
    if choice == 'y':
        if install_essentia_from_source():
            return
    
    print("\nAll installation methods failed.")
    print("You can still use the system with fallback methods.")

if __name__ == "__main__":
    main()
