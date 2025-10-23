#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script để kiểm tra Essentia hoặc Madmom
"""

def test_essentia():
    """Test Essentia installation"""
    try:
        import essentia.standard as es
        print("✅ Essentia installed successfully!")
        
        # Test key detection
        loader = es.MonoLoader()
        key_extractor = es.KeyExtractor()
        
        print("✅ Essentia key detection ready!")
        return True
        
    except ImportError as e:
        print(f"❌ Essentia not installed: {e}")
        return False

def test_madmom():
    """Test Madmom installation"""
    try:
        import madmom
        print("✅ Madmom installed successfully!")
        
        # Test key detection
        from madmom.features.key import KeyExtractor
        key_extractor = KeyExtractor()
        
        print("✅ Madmom key detection ready!")
        return True
        
    except ImportError as e:
        print(f"❌ Madmom not installed: {e}")
        return False

def main():
    """Main test function"""
    print("Testing Music Analysis Libraries")
    print("=" * 40)
    
    essentia_ok = test_essentia()
    madmom_ok = test_madmom()
    
    if essentia_ok or madmom_ok:
        print("\n🎉 At least one library is ready!")
        if essentia_ok:
            print("📊 You can use Essentia for key detection")
        if madmom_ok:
            print("📊 You can use Madmom for key detection")
    else:
        print("\n❌ Neither library is installed")
        print("Please follow the installation instructions")

if __name__ == "__main__":
    main()

