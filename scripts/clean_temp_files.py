#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de sua loi file lock
"""

import os
import shutil
from pathlib import Path

def clean_temp_files():
    """Clean temp files and directories"""
    base_path = Path(__file__).parent.parent
    
    # Clean directories
    temp_dirs = [
        base_path / "Audio_separator_ui" / "clean_song_output",
        base_path / "output" / "clean_song_output",
        base_path / "data" / "temp_output"
    ]
    
    print("Cleaning temp files...")
    
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print(f"Removed: {temp_dir}")
            except Exception as e:
                print(f"Error removing {temp_dir}: {e}")
    
    # Create fresh directories
    output_dirs = [
        base_path / "output" / "clean_song_output",
        base_path / "data" / "temp_output"
    ]
    
    for output_dir in output_dirs:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created: {output_dir}")
        except Exception as e:
            print(f"Error creating {output_dir}: {e}")

if __name__ == "__main__":
    clean_temp_files()
    print("Done!")
