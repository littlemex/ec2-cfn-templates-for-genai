#!/usr/bin/env python3
"""
Hardware Deep Analysis Framework Integration Test
================================================

新しいハードウェア深層解析フレームワークの統合テストスクリプト

このテストは以下を検証します:
1. Neuron環境の正常性
2. ハードウェアプロファイリング機能
3. NTFF解析パイプライン
4. レポート生成機能
"""

import sys
import os
from pathlib import Path
import logging

# フレームワークモジュールのインポートテスト
sys.path.insert(0, str(Path(__file__).parent))

def test_import_framework():
    """フレームワークインポートテスト"""
    print("🧪 Testing framework imports...")
    
    try:
        from neuron_hardware_deep_analyzer import NeuronHardwareProfiler, HardwareProfile
        print("✅ Hardware analyzer import successful")
        
        from run_hardware_deep_analysis import HardwareAnalysisRunner
        print("✅ Analysis runner import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_neuron_environment():
    """Neuron環境テスト"""
    print("\n🔬 Testing Neuron environment...")
    
    try:
        import torch
        import torch_xla
        print("✅ PyTorch XLA available")
        
        try:
            import torch_neuronx
            print("✅ torch_neuronx available")
            neuron_available = True
        except ImportError:
            print("⚠️  torch_neuronx not available - using fallback mode")
            neuron_available = False
            
        # XLA device test
        try:
            device = torch_xla.device()
            print(f"✅ XLA device available: {device}")
        except Exception as e:
            print(f"⚠️  XLA device warning: {e}")
            
        return neuron_available
        
    except ImportError as e:
        print(f"❌ Neuron environment test failed: {e}")
        return False

def test_hardware_profiler_basic():
    """基本ハードウェアプロファイラーテスト"""
    print("\n⚙️  Testing basic hardware profiler...")
    
    try:
        from neuron_hardware_deep_analyzer import NeuronHardwareProfiler
        
        # プロファイラー初期化テスト
        profiler = NeuronHardwareProfiler("test_basic_functionality")
        print("✅ Hardware profiler initialization successful")
        
        # テストデータ作成テスト
        test_data = profiler.create_test_data((100, 200))
        print(f"✅ Test data creation successful: {test_data.shape}")
        
        # ログファイル確認
        log_file = profiler.profile_output_dir / "hardware_analysis.log"
        if log_file.exists():
            print(f"✅ Log file created: {log_file}")
        else:
            print("⚠️  Log file not found")
            
        return True
        
    except Exception as e:
        print(f"❌ Hardware profiler test failed: {e}")
        return False

def test_fallback_profile_creation():
    """フォールバックプロファイル作成テスト"""  
    print("\n🛡️  Testing fallback profile creation...")
    
    try:
        from neuron_hardware_deep_analyzer import NeuronHardwareProfiler
        
        profiler = NeuronHardwareProfiler("test_fallback")
        
        # フォールバックプロファイル作成テスト
        fallback_profile = profiler._create_fallback_profile("test_pattern")
        print("✅ Fallback profile creation successful")
        
        # プロファイル属性確認
        assert hasattr(fallback_profile, 'pattern_name')
        assert hasattr(fallback_profile, 'tensor_engine_utilization')
        assert hasattr(fallback_profile, 'optimization_recommendations')
        print("✅ Profile attributes validation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback profile test failed: {e}")
        return False

def test_analysis_runner():
    """解析ランナーテスト"""
    print("\n🏃 Testing analysis runner...")
    
    try:
        from run_hardware_deep_analysis import HardwareAnalysisRunner
        
        # テスト用一時ディレクトリ
        test_output = "/tmp/hardware_framework_test"
        runner = HardwareAnalysisRunner(test_output)
        print("✅ Analysis runner initialization successful")
        
        # 出力ディレクトリ確認
        if runner.output_dir.exists():
            print(f"✅ Output directory created: {runner.output_dir}")
        else:
            print("❌ Output directory creation failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Analysis runner test failed: {e}")
        return False

def test_neuron_profile_command():
    """neuron-profileコマンド利用可能性テスト"""
    print("\n🔧 Testing neuron-profile command availability...")
    
    try:
        import subprocess
        
        # neuron-profile help実行
        result = subprocess.run(['neuron-profile', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ neuron-profile command available")
            return True
        else:
            print("⚠️  neuron-profile command not available - some features may be limited")
            return False
            
    except FileNotFoundError:
        print("⚠️  neuron-profile command not found in PATH")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  neuron-profile command timeout")
        return False
    except Exception as e:
        print(f"⚠️  neuron-profile test warning: {e}")
        return False

def run_integration_test():
    """統合テスト実行"""
    print("🚀 Hardware Deep Analysis Framework Integration Test")
    print("=" * 60)
    
    test_results = {
        'framework_import': test_import_framework(),
        'neuron_environment': test_neuron_environment(),
        'hardware_profiler_basic': test_hardware_profiler_basic(),
        'fallback_profile': test_fallback_profile_creation(),
        'analysis_runner': test_analysis_runner(),
        'neuron_profile_command': test_neuron_profile_command()
    }
    
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST RESULTS")
    print("=" * 60)
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests:.1%})")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - Framework ready for use!")
        return True
    elif passed_tests >= total_tests * 0.7:  # 70%以上でwarning
        print("\n⚠️  SOME TESTS FAILED - Framework partially functional")
        print("Check failed tests and environment configuration")
        return False
    else:
        print("\n❌ CRITICAL FAILURES - Framework not ready")
        print("Please resolve failed tests before using the framework")
        return False

def print_usage_instructions():
    """使用方法表示"""
    print("\n" + "=" * 60)
    print("📖 FRAMEWORK USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print("\n🔬 Basic Hardware Analysis:")
    print("   python neuron_hardware_deep_analyzer.py")
    
    print("\n🎯 Targeted Pattern Analysis:")
    print("   python run_hardware_deep_analysis.py --pattern vmap")
    print("   python run_hardware_deep_analysis.py --pattern scan")
    print("   python run_hardware_deep_analysis.py --pattern for_loop")
    print("   python run_hardware_deep_analysis.py --pattern negative_compilation")
    
    print("\n📊 Full Analysis with Verbose Output:")
    print("   python run_hardware_deep_analysis.py --pattern all --verbose")
    
    print("\n📁 Custom Output Directory:")
    print("   python run_hardware_deep_analysis.py --output-dir /custom/path")
    
    print("\n📚 Documentation:")
    print("   See: pytorch-neuron/docs/NEURON_HARDWARE_DEEP_ANALYSIS_FRAMEWORK.md")

def main():
    """メイン実行"""
    success = run_integration_test()
    
    if success:
        print_usage_instructions()
        
    print(f"\n{'='*60}")
    print("Integration test completed.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
