#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

print("🎤 Testing Docker Environment 🎵")
print("=" * 50)

# Test imports
try:
    import essentia
    print("✅ Essentia imported successfully!")
except ImportError as e:
    print(f"❌ Essentia import failed: {e}")

try:
    import librosa
    print("✅ Librosa imported successfully!")
except ImportError as e:
    print(f"❌ Librosa import failed: {e}")

try:
    import tkinter
    print("✅ Tkinter imported successfully!")
except ImportError as e:
    print(f"❌ Tkinter import failed: {e}")

try:
    from advanced_audio_processor import AdvancedAudioProcessor
    print("✅ AdvancedAudioProcessor imported successfully!")
except ImportError as e:
    print(f"❌ AdvancedAudioProcessor import failed: {e}")

try:
    from advanced_key_detector import AdvancedKeyDetector
    print("✅ AdvancedKeyDetector imported successfully!")
except ImportError as e:
    print(f"❌ AdvancedKeyDetector import failed: {e}")

print("\n🎉 Docker environment test completed!")
