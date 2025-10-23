#!/bin/bash
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
