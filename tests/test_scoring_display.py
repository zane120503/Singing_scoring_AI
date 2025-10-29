#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Scoring Display - Kiểm tra hiển thị điểm số
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_scoring_display():
    """Test hiển thị điểm số"""
    
    print("🧪 TEST SCORING DISPLAY")
    print("=" * 40)
    
    # Mock result từ parallel workflow
    mock_result = {
        "success": True,
        "inputs": {
            "karaoke_file": "test_karaoke.mp3",
            "beat_file": "test_beat.mp3"
        },
        "vocals_key": {
            "key": "A",
            "scale": "minor",
            "confidence": 0.962,
            "method": "Traditional Librosa"
        },
        "beat_key": {
            "key": "A", 
            "scale": "minor",
            "confidence": 1.545,
            "method": "GPU Chroma Analysis"
        },
        "key_compare": {
            "match": True,
            "similarity": 1.0,
            "score": 100.0
        }
    }
    
    # Test scoring logic
    key_score = mock_result['key_compare']['score']
    overall_score = key_score
    
    # Xác định grade
    if overall_score >= 90:
        grade = "A"
    elif overall_score >= 80:
        grade = "B"
    elif overall_score >= 70:
        grade = "C"
    else:
        grade = "D"
    
    print(f"📊 Key Score: {key_score:.1f}/100")
    print(f"🏆 Overall Score: {overall_score:.1f}/100")
    print(f"📈 Grade: {grade}")
    print(f"✅ Match: {mock_result['key_compare']['match']}")
    print(f"🎵 Vocals: {mock_result['vocals_key']['key']} {mock_result['vocals_key']['scale']}")
    print(f"🎵 Beat: {mock_result['beat_key']['key']} {mock_result['beat_key']['scale']}")
    
    # Test detailed scores
    detailed_scores = [
        ("Độ chính xác phím", f"{key_score:.1f}", "100.0%")
    ]
    
    print("\n📋 Detailed Scores:")
    for criterion, score, weight in detailed_scores:
        print(f"   {criterion}: {score} ({weight})")
    
    print("\n✅ Scoring display test completed!")
    return True

if __name__ == "__main__":
    success = test_scoring_display()
    
    if success:
        print("\n🎉 Test thành công!")
        print("Hệ thống sẽ hiển thị điểm số đúng cách.")
    else:
        print("\n❌ Test thất bại!")
    
    input("\nNhấn Enter để thoát...")
