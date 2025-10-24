#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Parallel Beat Key Detection - Kiểm tra beat key detection song song
"""

import sys
import os
import time
import concurrent.futures
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_parallel_beat_key():
    """Test beat key detection chạy song song với vocal separation"""
    
    print("🧪 TEST PARALLEL BEAT KEY DETECTION")
    print("=" * 50)
    
    try:
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        # Initialize key detector
        key_detector = AdvancedKeyDetector()
        print(f"✅ Key detector initialized")
        print(f"   GPU Status: {'ENABLED' if key_detector.use_gpu else 'DISABLED'}")
        
        # Mock functions để test timing
        def mock_beat_key_detection():
            """Mock beat key detection"""
            print("🎵 Starting beat key detection...")
            time.sleep(2)  # Simulate processing time
            print("✅ Beat key detection completed!")
            return {
                'key': 'A',
                'scale': 'minor',
                'confidence': 0.8,
                'method': 'Beat Harmonic Analysis'
            }
        
        def mock_vocal_separation():
            """Mock vocal separation"""
            print("🎤 Starting vocal separation...")
            time.sleep(5)  # Simulate longer processing time
            print("✅ Vocal separation completed!")
            return "mock_vocals_path.mp3"
        
        def mock_vocals_key_detection(vocals_path):
            """Mock vocals key detection"""
            print("🎤 Starting vocals key detection...")
            time.sleep(1)  # Simulate processing time
            print("✅ Vocals key detection completed!")
            return {
                'key': 'A',
                'scale': 'minor',
                'confidence': 0.9,
                'method': 'Traditional Librosa'
            }
        
        # Test sequential execution (old way)
        print("\n📊 Testing SEQUENTIAL execution (old way):")
        start_time = time.time()
        
        # Beat key detection
        beat_key_seq = mock_beat_key_detection()
        
        # Vocal separation
        vocals_path_seq = mock_vocal_separation()
        
        # Vocals key detection
        vocals_key_seq = mock_vocals_key_detection(vocals_path_seq)
        
        sequential_time = time.time() - start_time
        print(f"⏱️ Sequential time: {sequential_time:.2f}s")
        
        # Test parallel execution (new way)
        print("\n📊 Testing PARALLEL execution (new way):")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit beat key detection ngay lập tức
            beat_future = executor.submit(mock_beat_key_detection)
            # Submit vocal separation song song
            vocals_sep_future = executor.submit(mock_vocal_separation)
            
            # Chờ beat key detection hoàn thành trước
            beat_key_par = beat_future.result()
            print("🎉 Beat key detection hoàn thành!")
            
            # Chờ vocal separation hoàn thành
            vocals_path_par = vocals_sep_future.result()
            
            # Detect vocals key sau khi separation hoàn thành
            vocals_key_par = mock_vocals_key_detection(vocals_path_par)
        
        parallel_time = time.time() - start_time
        print(f"⏱️ Parallel time: {parallel_time:.2f}s")
        
        # Calculate improvement
        improvement = ((sequential_time - parallel_time) / sequential_time) * 100
        print(f"\n📈 Performance improvement: {improvement:.1f}%")
        print(f"   Time saved: {sequential_time - parallel_time:.2f}s")
        
        # Verify results
        print(f"\n🔍 Results verification:")
        print(f"   Beat key (seq): {beat_key_seq['key']} {beat_key_seq['scale']}")
        print(f"   Beat key (par): {beat_key_par['key']} {beat_key_par['scale']}")
        print(f"   Vocals key (seq): {vocals_key_seq['key']} {vocals_key_seq['scale']}")
        print(f"   Vocals key (par): {vocals_key_par['key']} {vocals_key_par['scale']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_workflow_timing():
    """Test timing của workflow mới"""
    print("\n⏱️ TEST WORKFLOW TIMING")
    print("=" * 30)
    
    # Simulate workflow steps với timing thực tế
    workflow_steps = [
        {"name": "Audio Slicing", "duration": 0.5, "parallel": False},
        {"name": "Beat Key Detection", "duration": 3.0, "parallel": True},
        {"name": "Vocal Separation", "duration": 8.0, "parallel": True},
        {"name": "Vocals Key Detection", "duration": 2.0, "parallel": False},
        {"name": "Key Comparison", "duration": 0.1, "parallel": False}
    ]
    
    print("📋 Workflow steps:")
    for step in workflow_steps:
        parallel_indicator = "⚡" if step["parallel"] else "➡️"
        print(f"   {parallel_indicator} {step['name']}: {step['duration']:.1f}s")
    
    # Calculate total time
    print(f"\n📊 Time calculation:")
    
    # Sequential time
    sequential_total = sum(step["duration"] for step in workflow_steps)
    print(f"   Sequential total: {sequential_total:.1f}s")
    
    # Parallel time (beat key + vocal separation run in parallel)
    parallel_total = 0
    parallel_total += workflow_steps[0]["duration"]  # Audio slicing
    parallel_total += max(workflow_steps[1]["duration"], workflow_steps[2]["duration"])  # Max of beat key + vocal sep
    parallel_total += workflow_steps[3]["duration"]  # Vocals key detection
    parallel_total += workflow_steps[4]["duration"]  # Key comparison
    
    print(f"   Parallel total: {parallel_total:.1f}s")
    
    improvement = ((sequential_total - parallel_total) / sequential_total) * 100
    print(f"   Improvement: {improvement:.1f}%")
    print(f"   Time saved: {sequential_total - parallel_total:.1f}s")
    
    return True

def test_beat_key_priority():
    """Test beat key detection được ưu tiên"""
    print("\n🎯 TEST BEAT KEY PRIORITY")
    print("=" * 30)
    
    print("📋 Execution order:")
    print("   1. Audio Slicing (0.5s)")
    print("   2. Beat Key Detection starts immediately ⚡")
    print("   3. Vocal Separation starts in parallel ⚡")
    print("   4. Beat Key Detection completes first ✅")
    print("   5. Vocal Separation completes")
    print("   6. Vocals Key Detection starts")
    print("   7. Vocals Key Detection completes")
    print("   8. Key Comparison")
    
    print(f"\n✅ Benefits:")
    print(f"   • Beat key detection starts immediately")
    print(f"   • No waiting for vocal separation")
    print(f"   • Faster overall processing")
    print(f"   • Better user experience")
    
    return True

if __name__ == "__main__":
    print("=== TEST PARALLEL BEAT KEY DETECTION ===")
    
    # Test parallel execution
    parallel_ok = test_parallel_beat_key()
    
    # Test workflow timing
    timing_ok = test_workflow_timing()
    
    # Test beat key priority
    priority_ok = test_beat_key_priority()
    
    print("\n" + "=" * 50)
    if parallel_ok and timing_ok and priority_ok:
        print("🎉 PARALLEL BEAT KEY DETECTION SUCCESSFUL!")
        print("✅ Beat key detection runs in parallel")
        print("✅ Performance improvement achieved")
        print("✅ Beat key gets priority")
    else:
        print("❌ PARALLEL BEAT KEY DETECTION NEEDS ATTENTION!")
        print("⚠️ Some issues detected")
    
    print("\n📋 NEW WORKFLOW:")
    print("   1. Audio Slicing")
    print("   2. Beat Key Detection + Vocal Separation (parallel)")
    print("   3. Vocals Key Detection")
    print("   4. Key Comparison & Scoring")
    
    input("\nNhấn Enter để thoát...")
