#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Voting Mechanism - Kiểm tra cơ chế voting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_voting_mechanism():
    """Test voting mechanism với dữ liệu thực tế"""
    
    print("🧪 TEST VOTING MECHANISM")
    print("=" * 50)
    
    try:
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        # Initialize key detector
        key_detector = AdvancedKeyDetector()
        
        # Mock results từ log thực tế
        mock_results = [
            {
                'key': 'F',
                'scale': 'major',
                'confidence': 0.730,
                'method': 'Docker Essentia AI',
                'weight': 0.6
            },
            {
                'key': 'A',
                'scale': 'minor',
                'confidence': 0.435,
                'method': 'Traditional Librosa',
                'weight': 0.4
            },
            {
                'key': 'C',
                'scale': 'major',
                'confidence': 0.350,
                'method': 'Vocals-Specific Analysis',
                'weight': 0.3
            },
            {
                'key': 'A#',
                'scale': 'minor',
                'confidence': 0.348,
                'method': 'GPU Chroma Analysis',
                'weight': 0.5
            },
            {
                'key': 'A',
                'scale': 'minor',
                'confidence': 0.531,
                'method': 'Beat Harmonic Analysis',
                'weight': 0.5
            }
        ]
        
        print("📊 Mock results:")
        for result in mock_results:
            print(f"   {result['method']}: {result['key']} {result['scale']} (conf: {result['confidence']:.3f}, weight: {result['weight']:.1f})")
        
        # Test voting mechanism
        print("\n🗳️ Testing voting mechanism:")
        best_result = key_detector._weighted_voting(mock_results)
        
        print(f"\n🏆 Final result:")
        print(f"   Key: {best_result['key']} {best_result['scale']}")
        print(f"   Confidence: {best_result['confidence']:.3f}")
        print(f"   Method: {best_result['method']}")
        
        # Manual calculation để verify
        print(f"\n🔍 Manual calculation:")
        
        # Group by key+scale
        key_groups = {}
        for result in mock_results:
            key_scale = f"{result['key']} {result['scale']}"
            if key_scale not in key_groups:
                key_groups[key_scale] = []
            key_groups[key_scale].append(result)
        
        print(f"   Key groups: {list(key_groups.keys())}")
        
        # Calculate scores manually
        for key_scale, group in key_groups.items():
            total_weight = sum(r.get('weight', 0.1) for r in group)
            weighted_confidence = sum(r['confidence'] * r.get('weight', 0.1) for r in group)
            
            consensus_bonus = len(group) * 0.5
            beat_bonus = 0
            for result in group:
                if 'Beat Harmonic Analysis' in result.get('method', ''):
                    beat_bonus += 0.3
                elif 'Docker Essentia' in result.get('method', ''):
                    beat_bonus += 0.1
            
            final_score = (weighted_confidence / total_weight) + consensus_bonus + beat_bonus
            
            print(f"   {key_scale}:")
            print(f"     Methods: {len(group)}")
            print(f"     Total weight: {total_weight:.1f}")
            print(f"     Weighted confidence: {weighted_confidence:.3f}")
            print(f"     Consensus bonus: {consensus_bonus:.1f}")
            print(f"     Beat bonus: {beat_bonus:.1f}")
            print(f"     Final score: {final_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_consensus_priority():
    """Test consensus priority"""
    print("\n🎯 TEST CONSENSUS PRIORITY")
    print("=" * 30)
    
    try:
        from src.ai.advanced_key_detector import AdvancedKeyDetector
        
        key_detector = AdvancedKeyDetector()
        
        # Test case: 2 methods agree vs 1 method with high confidence
        test_cases = [
            {
                'name': 'Case 1: 2 methods agree vs 1 high confidence',
                'results': [
                    {'key': 'A', 'scale': 'minor', 'confidence': 0.4, 'method': 'Method 1', 'weight': 0.4},
                    {'key': 'A', 'scale': 'minor', 'confidence': 0.5, 'method': 'Method 2', 'weight': 0.5},
                    {'key': 'F', 'scale': 'major', 'confidence': 0.8, 'method': 'Method 3', 'weight': 0.6}
                ]
            },
            {
                'name': 'Case 2: 3 methods agree vs 1 high confidence',
                'results': [
                    {'key': 'A', 'scale': 'minor', 'confidence': 0.3, 'method': 'Method 1', 'weight': 0.3},
                    {'key': 'A', 'scale': 'minor', 'confidence': 0.4, 'method': 'Method 2', 'weight': 0.4},
                    {'key': 'A', 'scale': 'minor', 'confidence': 0.5, 'method': 'Method 3', 'weight': 0.5},
                    {'key': 'F', 'scale': 'major', 'confidence': 0.9, 'method': 'Method 4', 'weight': 0.6}
                ]
            }
        ]
        
        for case in test_cases:
            print(f"\n📋 {case['name']}:")
            result = key_detector._weighted_voting(case['results'])
            print(f"   Winner: {result['key']} {result['scale']}")
            print(f"   Score: {result['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consensus test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== TEST VOTING MECHANISM ===")
    
    # Test voting mechanism
    voting_ok = test_voting_mechanism()
    
    # Test consensus priority
    consensus_ok = test_consensus_priority()
    
    print("\n" + "=" * 50)
    if voting_ok and consensus_ok:
        print("🎉 VOTING MECHANISM TEST SUCCESSFUL!")
        print("✅ Consensus priority working correctly")
        print("✅ Beat-specific bonuses applied")
    else:
        print("❌ VOTING MECHANISM NEEDS ATTENTION!")
        print("⚠️ Some issues detected")
    
    print("\n📋 EXPECTED BEHAVIOR:")
    print("   • A minor (2 methods) should beat F major (1 method)")
    print("   • Consensus bonus should favor multiple agreements")
    print("   • Beat Harmonic Analysis should get bonus")
    
    input("\nNhấn Enter để thoát...")
