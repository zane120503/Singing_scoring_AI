"""
Test Runner for Singing Scoring AI System
Chạy các test cases và tạo report
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List
import json

# Setup paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_results.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TestRunner:
    """Test Runner cho hệ thống"""
    
    def __init__(self):
        self.test_results = []
        self.stats = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
    
    def run_test(self, test_name: str, test_func, *args, **kwargs):
        """Run a single test"""
        self.stats['total'] += 1
        logger.info(f"\n{'='*70}")
        logger.info(f"TEST: {test_name}")
        logger.info(f"{'='*70}")
        
        try:
            start_time = datetime.now()
            result = test_func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result.get('status') == 'passed':
                self.stats['passed'] += 1
                logger.info(f"✅ PASSED in {duration:.2f}s")
            elif result.get('status') == 'skipped':
                self.stats['skipped'] += 1
                logger.info(f"⏭️  SKIPPED: {result.get('reason', 'No reason')}")
            else:
                self.stats['failed'] += 1
                logger.error(f"❌ FAILED: {result.get('error', 'Unknown error')}")
            
            self.test_results.append({
                'name': test_name,
                'status': result.get('status', 'failed'),
                'duration': duration,
                'message': result.get('message', ''),
                'error': result.get('error', '')
            })
            
            return result
            
        except Exception as e:
            self.stats['failed'] += 1
            logger.error(f"❌ TEST CRASHED: {e}")
            self.test_results.append({
                'name': test_name,
                'status': 'crashed',
                'duration': 0,
                'error': str(e)
            })
            return {'status': 'failed', 'error': str(e)}
    
    def run_gpu_tests(self):
        """Run GPU-related tests"""
        logger.info("\n" + "="*70)
        logger.info("GPU USAGE TESTS")
        logger.info("="*70)
        
        def test_gpu_detection():
            """TC-GPU-001: Detect GPU Availability"""
            import torch
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                return {
                    'status': 'passed',
                    'message': f'GPU available: {torch.cuda.get_device_name(0)}'
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'CUDA not available'
                }
        
        self.run_test("GPU Detection", test_gpu_detection)
        
        def test_onnx_providers():
            """TC-GPU-002: ONNX CUDA Provider"""
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            if 'CUDAExecutionProvider' in providers:
                return {
                    'status': 'passed',
                    'message': f'CUDA provider available in: {providers}'
                }
            else:
                return {
                    'status': 'failed',
                    'error': f'CUDA provider not available in: {providers}'
                }
        
        self.run_test("ONNX CUDA Provider", test_onnx_providers)
    
    def run_key_detection_tests(self):
        """Run key detection tests"""
        logger.info("\n" + "="*70)
        logger.info("KEY DETECTION TESTS")
        logger.info("="*70)
        
        def test_key_detector_init():
            """Test key detector initialization"""
            from src.ai.advanced_key_detector import AdvancedKeyDetector
            
            try:
                detector = AdvancedKeyDetector()
                return {
                    'status': 'passed',
                    'message': 'Key detector initialized successfully'
                }
            except Exception as e:
                return {
                    'status': 'failed',
                    'error': str(e)
                }
        
        self.run_test("Key Detector Init", test_key_detector_init)
    
    def run_vocals_detection_tests(self):
        """Run vocals detection tests"""
        logger.info("\n" + "="*70)
        logger.info("VOCALS DETECTION TESTS")
        logger.info("="*70)
        
        def test_vocals_checker_init():
            """Test vocals presence checker"""
            try:
                from src.ai.vocals_presence_checker import VocalsPresenceChecker
                checker = VocalsPresenceChecker()
                return {
                    'status': 'passed',
                    'message': 'Vocals checker initialized'
                }
            except Exception as e:
                return {
                    'status': 'failed',
                    'error': str(e)
                }
        
        self.run_test("Vocals Checker Init", test_vocals_checker_init)
    
    def run_slicing_tests(self):
        """Run slicing tests"""
        logger.info("\n" + "="*70)
        logger.info("FILE SLICING TESTS")
        logger.info("="*70)
        
        def test_smart_slicing_init():
            """Test smart slicing strategy"""
            try:
                from src.ai.smart_slicing_strategy import SmartSlicingStrategy
                slicer = SmartSlicingStrategy()
                return {
                    'status': 'passed',
                    'message': 'Smart slicer initialized'
                }
            except Exception as e:
                return {
                    'status': 'failed',
                    'error': str(e)
                }
        
        self.run_test("Smart Slicing Init", test_smart_slicing_init)
    
    def run_audio_processor_tests(self):
        """Run audio processor tests"""
        logger.info("\n" + "="*70)
        logger.info("AUDIO PROCESSOR TESTS")
        logger.info("="*70)
        
        def test_audio_processor_init():
            """Test audio processor initialization"""
            try:
                from src.ai.real_audio_processor import RealAudioProcessor
                processor = RealAudioProcessor()
                return {
                    'status': 'passed',
                    'message': 'Audio processor initialized'
                }
            except Exception as e:
                return {
                    'status': 'failed',
                    'error': str(e)
                }
        
        self.run_test("Audio Processor Init", test_audio_processor_init)
    
    def run_all_tests(self):
        """Run all test suites"""
        logger.info("\n" + "="*70)
        logger.info("STARTING TEST SUITE")
        logger.info("="*70)
        
        self.stats['start_time'] = datetime.now()
        
        # Run test suites
        self.run_gpu_tests()
        self.run_key_detection_tests()
        self.run_vocals_detection_tests()
        self.run_slicing_tests()
        self.run_audio_processor_tests()
        
        self.stats['end_time'] = datetime.now()
        self.generate_report()
    
    def generate_report(self):
        """Generate test report"""
        logger.info("\n" + "="*70)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*70)
        
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        logger.info(f"\nTotal Tests: {self.stats['total']}")
        logger.info(f"✅ Passed: {self.stats['passed']}")
        logger.info(f"❌ Failed: {self.stats['failed']}")
        logger.info(f"⏭️  Skipped: {self.stats['skipped']}")
        logger.info(f"⏱️  Duration: {duration:.2f}s")
        logger.info(f"📊 Pass Rate: {(self.stats['passed']/self.stats['total']*100) if self.stats['total'] > 0 else 0:.1f}%")
        
        # Save results to JSON
        report = {
            'summary': self.stats,
            'tests': self.test_results,
            'duration_seconds': duration
        }
        
        with open('test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("\n📄 Detailed report saved to: test_report.json")
        logger.info("📄 Log file: test_results.log")

if __name__ == "__main__":
    runner = TestRunner()
    runner.run_all_tests()

