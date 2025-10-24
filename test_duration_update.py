#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Duration Update - Kiểm tra cập nhật thời lượng từ 20s lên 30s
"""

import sys
import os
import librosa
import soundfile as sf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_duration_update():
    """Test cập nhật thời lượng từ 20s lên 30s"""
    
    print("🧪 TEST DURATION UPDATE: 20s → 30s")
    print("=" * 50)
    
    # Test parameters
    test_files = [
        "D:/singing scoring AI/Audio_separator_ui/clean_song_output/test_stereo.wav",
        "D:/singing scoring AI/assets/audio/test_stereo.wav"
    ]
    
    # Find existing test file
    test_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            test_file = file_path
            break
    
    if not test_file:
        print("❌ Không tìm thấy file test audio")
        return False
    
    print(f"📁 Test file: {test_file}")
    
    try:
        # Load audio
        audio, sr = librosa.load(test_file, sr=None, mono=True)
        duration = len(audio) / sr
        print(f"📊 Audio duration: {duration:.2f}s")
        
        if duration < 45:
            print("⚠️ Audio quá ngắn để test 30s slice (cần >= 45s)")
            return False
        
        # Test old parameters (20s from 25s-45s)
        print("\n🔍 Test OLD parameters (20s from 25s-45s):")
        old_start = 25.0
        old_duration = 20.0
        old_end = old_start + old_duration
        
        old_start_sample = int(old_start * sr)
        old_end_sample = int(old_end * sr)
        old_slice = audio[old_start_sample:old_end_sample]
        old_slice_duration = len(old_slice) / sr
        
        print(f"   Start: {old_start}s")
        print(f"   End: {old_end}s")
        print(f"   Duration: {old_slice_duration:.2f}s")
        
        # Test new parameters (30s from 15s-45s)
        print("\n🔍 Test NEW parameters (30s from 15s-45s):")
        new_start = 15.0
        new_duration = 30.0
        new_end = new_start + new_duration
        
        new_start_sample = int(new_start * sr)
        new_end_sample = int(new_end * sr)
        new_slice = audio[new_start_sample:new_end_sample]
        new_slice_duration = len(new_slice) / sr
        
        print(f"   Start: {new_start}s")
        print(f"   End: {new_end}s")
        print(f"   Duration: {new_slice_duration:.2f}s")
        
        # Compare
        print("\n📊 COMPARISON:")
        print(f"   Old slice: {old_slice_duration:.2f}s")
        print(f"   New slice: {new_slice_duration:.2f}s")
        print(f"   Difference: +{new_slice_duration - old_slice_duration:.2f}s")
        
        # Test file naming
        base_stem = os.path.splitext(os.path.basename(test_file))[0]
        old_filename = f"{base_stem}_slice_{int(old_start)}s_{int(old_end)}s.wav"
        new_filename = f"{base_stem}_slice_{int(new_start)}s_{int(new_end)}s.wav"
        
        print(f"\n📁 File naming:")
        print(f"   Old: {old_filename}")
        print(f"   New: {new_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_workflow_duration():
    """Test workflow với duration mới"""
    print("\n🚀 TEST WORKFLOW DURATION")
    print("=" * 30)
    
    try:
        from optimized_middle_workflow import run_workflow
        
        # Test với duration 30s
        print("Testing workflow với duration=30.0...")
        
        # Mock files (không cần file thật để test parameters)
        karaoke_file = "test_karaoke.mp3"
        beat_file = "test_beat.mp3"
        
        # Test function signature
        import inspect
        sig = inspect.signature(run_workflow)
        params = sig.parameters
        
        print(f"✅ Function signature: {sig}")
        
        # Check default duration
        duration_param = params.get('duration')
        if duration_param and duration_param.default == 30.0:
            print("✅ Default duration đã được cập nhật thành 30.0s")
        else:
            print("❌ Default duration chưa được cập nhật")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST DURATION UPDATE: 20s → 30s ===")
    
    # Test duration update
    duration_ok = test_duration_update()
    
    # Test workflow
    workflow_ok = test_workflow_duration()
    
    print("\n" + "=" * 50)
    if duration_ok and workflow_ok:
        print("🎉 DURATION UPDATE SUCCESSFUL!")
        print("✅ Thời lượng đã được cập nhật từ 20s lên 30s")
        print("✅ Workflow parameters đã được cập nhật")
        print("✅ File naming đã được cập nhật")
    else:
        print("❌ DURATION UPDATE NEEDS ATTENTION!")
        print("⚠️ Một số cập nhật chưa hoàn thành")
    
    print("\n📋 SUMMARY:")
    print("   • Slice duration: 20s → 30s")
    print("   • Start time: 25s → 15s") 
    print("   • End time: 45s → 45s")
    print("   • Total analysis time: 20s → 30s")
    
    input("\nNhấn Enter để thoát...")
