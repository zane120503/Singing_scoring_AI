#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install Miniconda and Essentia
"""

import subprocess
import sys
import os
import urllib.request
import tempfile

def download_file(url, filename):
    """Download file from URL"""
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return False

def install_miniconda():
    """Install Miniconda"""
    print("INSTALLING MINICONDA")
    print("=" * 20)
    
    # Miniconda download URL for Windows
    miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    installer_name = "Miniconda3-latest-Windows-x86_64.exe"
    
    # Download Miniconda installer
    if not download_file(miniconda_url, installer_name):
        return False
    
    # Install Miniconda silently
    print("Installing Miniconda...")
    install_command = f"{installer_name} /InstallationType=JustMe /RegisterPython=0 /S /D=C:\\Miniconda3"
    
    try:
        result = subprocess.run(install_command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Miniconda installed successfully!")
            return True
        else:
            print(f"Error installing Miniconda: {result.stderr}")
            return False
    except Exception as e:
        print(f"Exception installing Miniconda: {e}")
        return False

def install_essentia_with_conda():
    """Install Essentia using conda"""
    print("INSTALLING ESSENTIA WITH CONDA")
    print("=" * 30)
    
    # Add conda to PATH
    conda_path = "C:\\Miniconda3\\Scripts\\conda.exe"
    
    if not os.path.exists(conda_path):
        print("Conda not found. Please install Miniconda first.")
        return False
    
    # Create new environment with Essentia
    commands = [
        f'"{conda_path}" create -n essentia python=3.9 -y',
        f'"{conda_path}" activate essentia',
        f'"{conda_path}" install -c conda-forge essentia -y'
    ]
    
    for cmd in commands:
        print(f"Running: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("SUCCESS!")
            else:
                print(f"ERROR: {result.stderr}")
                return False
        except Exception as e:
            print(f"Exception: {e}")
            return False
    
    print("Essentia installed successfully with conda!")
    return True

def create_essentia_environment():
    """Create a conda environment with Essentia"""
    print("CREATING ESSENTIA ENVIRONMENT")
    print("=" * 30)
    
    # Check if conda is available
    try:
        result = subprocess.run("conda --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("Conda is available!")
        else:
            print("Conda not found. Installing Miniconda...")
            if not install_miniconda():
                return False
    except:
        print("Conda not found. Installing Miniconda...")
        if not install_miniconda():
            return False
    
    # Install Essentia
    return install_essentia_with_conda()

def main():
    """Main function"""
    print("ESSENTIA INSTALLATION WITH CONDA")
    print("=" * 40)
    
    print("This will:")
    print("1. Install Miniconda (if not available)")
    print("2. Create a conda environment with Python 3.9")
    print("3. Install Essentia in that environment")
    
    choice = input("\nProceed? (y/n): ").lower()
    if choice != 'y':
        print("Installation cancelled.")
        return
    
    if create_essentia_environment():
        print("\nSUCCESS!")
        print("To use Essentia:")
        print("1. conda activate essentia")
        print("2. python your_script.py")
    else:
        print("\nFAILED!")
        print("You can still use the system with fallback methods.")

if __name__ == "__main__":
    main()
