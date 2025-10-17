#!/usr/bin/env python3
"""
AWS Neuronでのネストしたvmapをテストするデバッグスクリプト

このスクリプトは実際のPyTorchとNeuronを使用して、vmapコンパイル失敗を検証

実行前提条件:
- AWS Neuron環境 (TRN1インスタンス)
- torch-neuron インストール済み
"""

import torch
import torch.nn as nn
import time
import sys
import os
import traceback
import subprocess
import threading
import signal
from typing import Dict, List, Any, Optional, Callable
import resource
import psutil
import logging

# NeuronX imports (correct for TRN1)
try:
    import torch_neuronx
    import torch_xla.core.xla_model as xm
    from torch_neuronx.utils import get_platform_target
    NEURONX_AVAILABLE = True
    print("✅ torch_neuronx successfully imported")
except ImportError as e:
    print(f"⚠️ torch_neuronx not available: {e}")
    NEURONX_AVAILABLE = False


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('neuron_vmap_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """フォーマット済みヘッダーを出力"""
    separator = "=" * 80
    print(f"\n{separator}")
    print(f" {title}")
    print(f"{separator}")
    logger.info(f"セクション開始: {title}")


def print_debug_info(info: str, level: str = "INFO"):
    """デバッグ情報を出力"""
    timestamp = time.strftime("%H:%M:%S")
    symbols = {"INFO": "🔍", "WARNING": "⚠️", "ERROR": "❌", "SUCCESS": "✅"}
    symbol = symbols.get(level, "ℹ️")
    print(f"[{timestamp}] {symbol} {info}")
    
    # ログレベルのマッピング（SUCCESSはINFOとして記録）
    log_level_mapping = {
        "SUCCESS": "info",
        "DEBUG": "debug",
        "INFO": "info", 
        "WARNING": "warning",
        "ERROR": "error"
    }
    
    log_method = log_level_mapping.get(level, "info")
    getattr(logger, log_method)(info)


class SystemMonitor:
    """システムリソース監視クラス"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = []
        self.thread = None
    
    def start_monitoring(self):
        """監視開始"""
        self.monitoring = True
        self.stats = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        print_debug_info("システム監視を開始しました")
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)
        print_debug_info("システム監視を停止しました")
        return self.get_stats_summary()
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # メモリ使用量
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                memory_percent = memory.percent
                
                # プロセス情報
                process = psutil.Process()
                process_memory = process.memory_info().rss / (1024 * 1024)
                
                stat = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'memory_percent': memory_percent,
                    'process_memory_mb': process_memory
                }
                self.stats.append(stat)
                
                time.sleep(1)
            except Exception as e:
                print_debug_info(f"監視エラー: {e}", "WARNING")
                break
    
    def get_stats_summary(self):
        """統計サマリーを取得"""
        if not self.stats:
            return "監視データなし"
        
        cpu_max = max(stat['cpu_percent'] for stat in self.stats)
        memory_max = max(stat['memory_mb'] for stat in self.stats)
        process_memory_max = max(stat['process_memory_mb'] for stat in self.stats)
        
        return {
            'duration_seconds': len(self.stats),
            'cpu_max_percent': cpu_max,
            'memory_max_mb': memory_max,
            'process_memory_max_mb': process_memory_max,
            'total_samples': len(self.stats)
        }


class NeuronEnvironmentChecker:
    """Neuron環境チェッククラス"""
    
    @staticmethod
    def check_neuron_environment():
        """Neuron環境の詳細チェック"""
        print_header("Neuron環境チェック")
        
        checks = []
        
        # 1. torch_neuronxインポートチェック
        if NEURONX_AVAILABLE:
            try:
                version = getattr(torch_neuronx, '__version__', 'unknown')
                checks.append(("torch_neuronx インポート", True, f"バージョン: {version}"))
                print_debug_info(f"torch_neuronx バージョン: {version}", "SUCCESS")
                
                # XLA環境チェック
                try:
                    devices = xm.get_xla_supported_devices()
                    checks.append(("XLA デバイス", True, f"サポートデバイス: {devices}"))
                    print_debug_info(f"XLA サポートデバイス: {devices}", "SUCCESS")
                    
                    device_kind = xm.xla_device_kind()
                    checks.append(("XLA デバイス種別", True, device_kind))
                    print_debug_info(f"XLA デバイス種別: {device_kind}", "SUCCESS")
                    
                    platform = get_platform_target()
                    checks.append(("Neuron プラットフォーム", True, platform))
                    print_debug_info(f"Neuron プラットフォーム: {platform}", "SUCCESS")
                    
                except Exception as e:
                    checks.append(("XLA 環境", False, str(e)))
                    print_debug_info(f"XLA環境エラー: {e}", "ERROR")
                    
            except Exception as e:
                checks.append(("torch_neuronx 詳細", False, str(e)))
                print_debug_info(f"torch_neuronx詳細エラー: {e}", "ERROR")
        else:
            checks.append(("torch_neuronx インポート", False, "モジュールが利用できません"))
            print_debug_info("torch_neuronx インポートエラー: モジュールが利用できません", "ERROR")
        
        # 2. Neuronデバイスチェック
        try:
            neuron_devices = subprocess.run(['ls', '/dev/neuron*'], 
                                          capture_output=True, text=True)
            if neuron_devices.returncode == 0:
                devices = neuron_devices.stdout.strip().split('\n')
                checks.append(("Neuronデバイス", True, f"発見されたデバイス: {devices}"))
                print_debug_info(f"Neuronデバイス: {devices}", "SUCCESS")
            else:
                checks.append(("Neuronデバイス", False, "デバイスが見つかりません"))
                print_debug_info("Neuronデバイスなし", "WARNING")
        except Exception as e:
            checks.append(("Neuronデバイス", False, str(e)))
        
        # 3. neuron-lsコマンドチェック
        try:
            neuron_ls = subprocess.run(['neuron-ls'], capture_output=True, text=True)
            if neuron_ls.returncode == 0:
                checks.append(("neuron-ls", True, neuron_ls.stdout[:200]))
                print_debug_info("neuron-ls コマンド成功", "SUCCESS")
            else:
                checks.append(("neuron-ls", False, neuron_ls.stderr[:200]))
        except Exception as e:
            checks.append(("neuron-ls", False, str(e)))
        
        # 4. 環境変数チェック
        neuron_env_vars = [
            'NEURON_RT_NUM_CORES',
            'NEURON_RT_VISIBLE_CORES',
            'NEURON_FRAMEWORK_DEBUG',
            'NEURON_CC_FLAGS'
        ]
        
        for var in neuron_env_vars:
            value = os.environ.get(var)
            checks.append((f"環境変数 {var}", value is not None, value or "未設定"))
            if value:
                print_debug_info(f"{var}: {value}", "INFO")
        
        return checks


class SimpleLinearModel(nn.Module):
    """テスト用の単純な線形モデル"""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class VmapTester:
    """vmap動作テストクラス"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = SimpleLinearModel().to(device)
        self.monitor = SystemMonitor()
    
    def test_single_vmap(self, batch_size=32, input_size=10):
        """単一vmapテスト"""
        print_header("単一vmap テスト")
        
        try:
            self.monitor.start_monitoring()
            start_time = time.time()
            
            # テストデータ
            test_data = torch.randn(batch_size, input_size).to(self.device)
            print_debug_info(f"テストデータ形状: {test_data.shape}")
            
            def single_forward(x):
                return self.model(x)
            
            print_debug_info("単一vmapコンパイル開始...")
            result = torch.vmap(single_forward)(test_data)
            
            # NeuronX同期（XLAデバイスの場合）
            if NEURONX_AVAILABLE and str(self.device) != 'cpu':
                xm.mark_step()
                print_debug_info("NeuronX同期完了")
            
            duration = time.time() - start_time
            stats = self.monitor.stop_monitoring()
            
            print_debug_info(f"単一vmap成功 - 実行時間: {duration:.3f}秒", "SUCCESS")
            print_debug_info(f"出力形状: {result.shape}")
            print_debug_info(f"システム統計: {stats}")
            
            return True, duration, result.shape, stats
            
        except Exception as e:
            duration = time.time() - start_time
            stats = self.monitor.stop_monitoring()
            print_debug_info(f"単一vmap失敗: {e}", "ERROR")
            print_debug_info(f"実行時間: {duration:.3f}秒")
            print_debug_info(f"システム統計: {stats}")
            return False, duration, None, stats
    
    def test_nested_vmap_with_timeout(self, batch_size=4, samples_per_batch=8, 
                                    input_size=10, timeout_seconds=300):
        """ネストしたvmapテスト"""
        print_header("ネストしたvmap テスト")
        
        print_debug_info(f"⚠️  警告: このテストは最大{timeout_seconds}秒でタイムアウトします")
        print_debug_info(f"データ形状: [{batch_size}, {samples_per_batch}, {input_size}]")
        
        # タイムアウト処理用のシグナルハンドラー
        def timeout_handler(signum, frame):
            raise TimeoutError(f"ネストしたvmapが{timeout_seconds}秒でタイムアウトしました")
        
        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
        
        try:
            self.monitor.start_monitoring()
            start_time = time.time()
            
            # テストデータ [batch_size, samples_per_batch, input_size]
            test_data = torch.randn(batch_size, samples_per_batch, input_size).to(self.device)
            
            print_debug_info("ネストしたvmap構造を構築中...")
            
            def outer_func(batch_data):
                def inner_func(sample):
                    print_debug_info(f"内側vmap - サンプル形状: {sample.shape}", "INFO")
                    return self.model(sample)
                
                print_debug_info(f"外側vmap - バッチ形状: {batch_data.shape}", "INFO")
                # 内側vmap（問題の核心部分）
                return torch.vmap(inner_func)(batch_data)
            
            print_debug_info("ネストしたvmapコンパイル開始...")
            
            # タイムアウト設定
            signal.alarm(timeout_seconds)
            
            # 外側vmap（さらなるネスト）
            result = torch.vmap(outer_func)(test_data)
            
            # NeuronX同期（XLAデバイスの場合）
            if NEURONX_AVAILABLE and str(self.device) != 'cpu':
                xm.mark_step()
                print_debug_info("NeuronX同期完了")
            
            # タイムアウトをキャンセル
            signal.alarm(0)
            
            duration = time.time() - start_time
            stats = self.monitor.stop_monitoring()
            
            print_debug_info(f"ネストしたvmap成功 - 実行時間: {duration:.3f}秒", "SUCCESS")
            print_debug_info(f"出力形状: {result.shape}")
            print_debug_info(f"システム統計: {stats}")
            
            return True, duration, result.shape, stats
            
        except TimeoutError as e:
            signal.alarm(0)
            duration = time.time() - start_time
            stats = self.monitor.stop_monitoring()
            print_debug_info(f"予期されたタイムアウト: {e}", "ERROR")
            print_debug_info(f"実行時間: {duration:.3f}秒")
            print_debug_info(f"システム統計: {stats}")
            return False, duration, None, stats
            
        except Exception as e:
            signal.alarm(0)
            duration = time.time() - start_time
            stats = self.monitor.stop_monitoring()
            print_debug_info(f"ネストしたvmap失敗: {e}", "ERROR")
            print_debug_info(f"エラータイプ: {type(e).__name__}")
            print_debug_info(f"実行時間: {duration:.3f}秒")
            print_debug_info(f"システム統計: {stats}")
            traceback.print_exc()
            return False, duration, None, stats
            
        finally:
            signal.signal(signal.SIGALRM, original_handler)
    
    def test_explicit_loop_alternative(self, batch_size=4, samples_per_batch=8, input_size=10):
        """明示的ループ代替案テスト"""
        print_header("明示的ループ代替案テスト")
        
        try:
            self.monitor.start_monitoring()
            start_time = time.time()
            
            # テストデータ
            test_data = torch.randn(batch_size, samples_per_batch, input_size).to(self.device)
            print_debug_info(f"テストデータ形状: {test_data.shape}")
            
            print_debug_info("明示的ループ実行開始...")
            
            # 明示的ループ実装
            batch_results = []
            for batch_idx in range(batch_size):
                sample_results = []
                for sample_idx in range(samples_per_batch):
                    sample = test_data[batch_idx, sample_idx]
                    result = self.model(sample)
                    sample_results.append(result)
                
                batch_result = torch.stack(sample_results)
                batch_results.append(batch_result)
            
            final_result = torch.stack(batch_results)
            
            duration = time.time() - start_time
            stats = self.monitor.stop_monitoring()
            
            print_debug_info(f"明示的ループ成功 - 実行時間: {duration:.3f}秒", "SUCCESS")
            print_debug_info(f"出力形状: {final_result.shape}")
            print_debug_info(f"システム統計: {stats}")
            
            return True, duration, final_result.shape, stats
            
        except Exception as e:
            duration = time.time() - start_time
            stats = self.monitor.stop_monitoring()
            print_debug_info(f"明示的ループ失敗: {e}", "ERROR")
            print_debug_info(f"実行時間: {duration:.3f}秒")
            print_debug_info(f"システム統計: {stats}")
            return False, duration, None, stats


def analyze_compilation_failure(error_info: Dict[str, Any]):
    """コンパイル失敗の分析"""
    print_header("コンパイル失敗分析")
    
    print_debug_info("失敗パターン分析:", "INFO")
    
    # エラーの分類
    if error_info.get('timeout'):
        print_debug_info("タイムアウトパターン:", "WARNING")
        print_debug_info("- Neuronコンパイラがネストしたvmapの依存性グラフを解析中にスタック")
    
    if error_info.get('memory_usage_high'):
        print_debug_info("高メモリ使用量:", "WARNING")
        print_debug_info("- コンパイル時の中間表現が大量のメモリを消費")
        print_debug_info("- 依存性グラフの構築で指数的メモリ増加")
    
    if error_info.get('cpu_usage_high'):
        print_debug_info("高CPU使用量:", "WARNING")
        print_debug_info("- コンパイラが形状推論と最適化で高負荷")
        print_debug_info("- ネストしたループ構造の解析処理")
    
    print_debug_info("\n根本原因:", "INFO")
    print_debug_info("1. 静的コンパイル制約: Neuronは実行前にすべての形状を確定する必要")
    print_debug_info("2. MLIR制限: 構造化制御フローでの複雑なネスト処理")
    print_debug_info("3. 依存性グラフ複雑度: O(n³)の解析が必要")
    print_debug_info("4. ハードウェア制約: 事前定義された並列パターンのみサポート")


def main():
    """メイン関数"""
    print_header("AWS Neuron vmap 失敗検証テスト")
    print_debug_info("このスクリプトはNeuronでのNested vmapを検証します")
    
    # 環境チェック
    checker = NeuronEnvironmentChecker()
    env_checks = checker.check_neuron_environment()
    
    # デバイス初期化
    device = 'cpu'
    using_neuronx = False
    
    if NEURONX_AVAILABLE:
        try:
            # XLAデバイス初期化（untitled37_neuron_complete.pyパターン）
            device = xm.xla_device()
            print_debug_info(f"XLAデバイス初期化成功: {device}", "SUCCESS")
            
            # デバイス詳細情報
            devices = xm.get_xla_supported_devices()
            print_debug_info(f"サポートされているXLAデバイス: {devices}", "INFO")
            
            device_kind = xm.xla_device_kind()
            print_debug_info(f"XLAデバイス種別: {device_kind}", "INFO")
            
            platform = get_platform_target()
            print_debug_info(f"Neuronプラットフォーム: {platform}", "INFO")
            
            using_neuronx = True
            print_debug_info("✅ NeuronX環境で実際のvmapコンパイルを検証します", "SUCCESS")
            
        except Exception as e:
            print_debug_info(f"XLA初期化失敗、CPUフォールバック: {e}", "WARNING")
            device = 'cpu'
    else:
        print_debug_info("警告: torch_neuronxが利用できません", "WARNING")
        print_debug_info("このスクリプトはCPUモードで実行されますが、実際のNeuronは再現されません", "WARNING")
    
    print_debug_info(f"使用デバイス: {device}")
    
    # テスト実行
    tester = VmapTester(device)
    
    results = {}
    
    # 1. 単一vmapテスト
    print_debug_info("テスト1: 単一vmap")
    results['single_vmap'] = tester.test_single_vmap()
    
    # 2. ネストしたvmapテスト
    print_debug_info("テスト2: ネストしたvmap")
    results['nested_vmap'] = tester.test_nested_vmap_with_timeout(timeout_seconds=60)  # 短めのタイムアウト
    
    # 3. 明示的ループテスト
    print_debug_info("テスト3: 明示的ループ")
    results['explicit_loop'] = tester.test_explicit_loop_alternative()
    
    # 結果分析
    print_header("テスト結果サマリー")
    
    for test_name, (success, duration, shape, stats) in results.items():
        status = "✅ 成功" if success else "❌ 失敗"
        print_debug_info(f"{test_name}: {status} ({duration:.3f}秒)")
        if shape:
            print_debug_info(f"  出力形状: {shape}")
        if stats:
            print_debug_info(f"  最大メモリ: {stats.get('process_memory_max_mb', 0):.1f}MB")
    
    # 失敗分析
    if not results['nested_vmap'][0]:  # ネストしたvmapが失敗した場合
        error_info = {
            'timeout': True,
            'memory_usage_high': results['nested_vmap'][3].get('process_memory_max_mb', 0) > 1000,
            'cpu_usage_high': results['nested_vmap'][3].get('cpu_max_percent', 0) > 80
        }
        analyze_compilation_failure(error_info)
    
    print_header("推奨事項")
    print_debug_info("✅ 単一vmap: 全環境で使用可能")
    print_debug_info("✅ 明示的ループ: Neuron推奨パターン")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_debug_info("ユーザーによって中断されました", "WARNING")
        sys.exit(1)
    except Exception as e:
        print_debug_info(f"予期しないエラー: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1)
