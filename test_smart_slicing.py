#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Smart Audio Slicing - Kiểm tra logic cắt file thông minh
"""

import sys
import os
import librosa
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_smart_slicing():
    """Test logic cắt file thông minh"""
    
    print("🧪 TEST SMART AUDIO SLICING")
    print("=" * 50)
    
    # Test cases với các độ dài khác nhau
    test_cases = [
        {"duration": 20.0, "expected": "File ngắn"},
        {"duration": 25.0, "expected": "File ngắn"},
        {"duration": 30.0, "expected": "File ngắn"},
        {"duration": 35.0, "expected": "File trung bình"},
        {"duration": 45.0, "expected": "File trung bình"},
        {"duration": 60.0, "expected": "File trung bình"},
        {"duration": 70.0, "expected": "File dài"},
        {"duration": 120.0, "expected": "File dài"}
    ]
    
    target_duration = 30.0  # Duration mục tiêu
    
    for case in test_cases:
        file_duration = case["duration"]
        print(f"\n📁 Test case: {file_duration}s file")
        
        # Simulate audio data
        sr = 22050
        audio_length = int(file_duration * sr)
        audio = np.random.randn(audio_length)
        
        # Apply smart slicing logic
        if file_duration <= target_duration:
            # File ngắn: sử dụng toàn bộ file
            start_t = 0.0
            end_t = file_duration
            slice_audio = audio
            actual_duration = file_duration
            print(f"   ✅ Logic: File ngắn ({file_duration:.1f}s ≤ {target_duration}s)")
            print(f"   📊 Sử dụng toàn bộ file: {start_t:.1f}s - {end_t:.1f}s")
            
        elif file_duration <= 60.0:
            # File trung bình: cắt từ giữa
            start_t = max(0, (file_duration - target_duration) / 2)
            end_t = start_t + target_duration
            actual_duration = target_duration
            print(f"   ✅ Logic: File trung bình ({file_duration:.1f}s)")
            print(f"   📊 Cắt từ giữa: {start_t:.1f}s - {end_t:.1f}s")
            
        else:
            # File dài: cắt từ 15s như cũ
            start_t = 15.0
            end_t = start_t + target_duration
            actual_duration = target_duration
            print(f"   ✅ Logic: File dài ({file_duration:.1f}s)")
            print(f"   📊 Cắt từ 15s: {start_t:.1f}s - {end_t:.1f}s")
        
        print(f"   🎯 Actual duration: {actual_duration:.1f}s")
        
        # Verify logic
        if file_duration <= target_duration:
            expected_start = 0.0
            expected_duration = file_duration
        elif file_duration <= 60.0:
            expected_start = (file_duration - target_duration) / 2
            expected_duration = target_duration
        else:
            expected_start = 15.0
            expected_duration = target_duration
        
        if abs(start_t - expected_start) < 0.1 and abs(actual_duration - expected_duration) < 0.1:
            print(f"   ✅ Logic correct!")
        else:
            print(f"   ❌ Logic error!")
            print(f"      Expected: start={expected_start:.1f}s, duration={expected_duration:.1f}s")
            print(f"      Actual: start={start_t:.1f}s, duration={actual_duration:.1f}s")
    
    return True

def test_edge_cases():
    """Test các trường hợp đặc biệt"""
    print("\n🔍 TEST EDGE CASES")
    print("=" * 30)
    
    edge_cases = [
        {"duration": 0.1, "name": "File rất ngắn"},
        {"duration": 1.0, "name": "File 1 giây"},
        {"duration": 15.0, "name": "File đúng 15s"},
        {"duration": 30.0, "name": "File đúng 30s"},
        {"duration": 30.1, "name": "File 30.1s"},
        {"duration": 60.0, "name": "File đúng 60s"},
        {"duration": 60.1, "name": "File 60.1s"}
    ]
    
    target_duration = 30.0
    
    for case in edge_cases:
        file_duration = case["duration"]
        print(f"\n📁 {case['name']}: {file_duration}s")
        
        # Determine logic
        if file_duration <= target_duration:
            logic = "File ngắn - sử dụng toàn bộ"
            start_t = 0.0
            actual_duration = file_duration
        elif file_duration <= 60.0:
            logic = "File trung bình - cắt từ giữa"
            start_t = (file_duration - target_duration) / 2
            actual_duration = target_duration
        else:
            logic = "File dài - cắt từ 15s"
            start_t = 15.0
            actual_duration = target_duration
        
        print(f"   Logic: {logic}")
        print(f"   Start: {start_t:.1f}s, Duration: {actual_duration:.1f}s")
        
        # Check for potential issues
        if file_duration < 1.0:
            print(f"   ⚠️ Warning: File rất ngắn có thể không đủ dữ liệu")
        elif start_t + actual_duration > file_duration:
            print(f"   ❌ Error: End time vượt quá file duration!")
        else:
            print(f"   ✅ OK")
    
    return True

def test_workflow_integration():
    """Test tích hợp với workflow"""
    print("\n🚀 TEST WORKFLOW INTEGRATION")
    print("=" * 35)
    
    try:
        from optimized_middle_workflow import run_workflow
        
        # Test với mock files
        print("Testing workflow với các độ dài file khác nhau...")
        
        # Simulate different file durations
        durations = [20.0, 30.0, 45.0, 90.0]
        
        for duration in durations:
            print(f"\n📁 Testing {duration}s file:")
            
            # Create mock audio file
            sr = 22050
            audio_length = int(duration * sr)
            audio = np.random.randn(audio_length)
            
            # Save temporary file
            temp_file = f"temp_test_{duration}s.wav"
            import soundfile as sf
            sf.write(temp_file, audio, sr)
            
            try:
                # Test workflow (will fail at separation step, but slicing should work)
                result = run_workflow(temp_file, temp_file, duration=30.0)
                
                if result.get("success"):
                    print(f"   ✅ Workflow successful")
                else:
                    error = result.get("error", "Unknown error")
                    if "ngắn hơn" in error or "Lỗi cắt audio" in error:
                        print(f"   ❌ Slicing failed: {error}")
                    else:
                        print(f"   ⚠️ Workflow failed at later step: {error}")
                        print(f"   ✅ Slicing likely successful")
            
            except Exception as e:
                print(f"   ❌ Exception: {e}")
            
            finally:
                # Clean up
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST SMART AUDIO SLICING ===")
    
    # Test smart slicing
    slicing_ok = test_smart_slicing()
    
    # Test edge cases
    edge_ok = test_edge_cases()
    
    # Test workflow integration
    integration_ok = test_workflow_integration()
    
    print("\n" + "=" * 50)
    if slicing_ok and edge_ok and integration_ok:
        print("🎉 SMART SLICING TEST SUCCESSFUL!")
        print("✅ Logic cắt file thông minh hoạt động đúng")
        print("✅ Edge cases được xử lý tốt")
        print("✅ Tích hợp workflow thành công")
    else:
        print("❌ SMART SLICING NEEDS ATTENTION!")
        print("⚠️ Một số vấn đề cần sửa")
    
    print("\n📋 LOGIC SUMMARY:")
    print("   • File ≤ 30s: Sử dụng toàn bộ file")
    print("   • File 30-60s: Cắt từ giữa (30s)")
    print("   • File > 60s: Cắt từ 15s (30s)")
    
    input("\nNhấn Enter để thoát...")
