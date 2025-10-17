#!/usr/bin/env python3
"""
TRN1 Neuron PyTorch 環境チェックスクリプト
AWS Trainium (TRN1) インスタンスでNeuron PyTorchが正しく動作するかを確認します
"""

import sys
import os
import subprocess
import time

def print_section(title):
    """セクションタイトルを表示"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_check(item, status, details=""):
    """チェック結果を表示"""
    status_symbol = "✅" if status else "❌"
    print(f"{status_symbol} {item}")
    if details:
        print(f"   {details}")

def check_python_environment():
    """Pythonバージョンと実行環境を確認"""
    print_section("1. Python実行環境の確認")
    
    print(f"Pythonバージョン: {sys.version}")
    print(f"Python実行パス: {sys.executable}")
    print(f"プラットフォーム: {sys.platform}")
    
    # Python 3.8以上が推奨
    version_ok = sys.version_info >= (3, 8)
    print_check(
        "Pythonバージョン",
        version_ok,
        "Python 3.8以上が推奨されています" if version_ok else "Python 3.8以上にアップグレードしてください"
    )
    
    return version_ok

def check_required_packages():
    """必須パッケージのインストール状況を確認"""
    print_section("2. 必須パッケージの確認")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torch_neuronx', 'PyTorch NeuronX'),
        ('torch_xla', 'PyTorch XLA'),
        ('torch_xla.core.xla_model', 'XLA Model'),
        ('torchvision', 'TorchVision'),
        ('numpy', 'NumPy'),
    ]
    
    all_ok = True
    for module_name, display_name in required_packages:
        try:
            __import__(module_name)
            print_check(display_name, True, f"モジュール '{module_name}' が利用可能")
        except ImportError as e:
            print_check(display_name, False, f"モジュール '{module_name}' がインストールされていません")
            all_ok = False
    
    return all_ok

def check_neuron_devices():
    """物理的なNeuronデバイスの存在を確認"""
    print_section("3. Neuronデバイスの検出")
    
    # /dev/neuron* デバイスの確認
    try:
        result = subprocess.run(['ls', '/dev/neuron*'], 
                              capture_output=True, 
                              text=True, 
                              shell=True)
        if result.returncode == 0 and result.stdout.strip():
            devices = result.stdout.strip().split('\n')
            print_check(
                "Neuronデバイスファイル",
                True,
                f"{len(devices)}個のデバイスが検出されました"
            )
            for dev in devices:
                print(f"   - {dev}")
            device_ok = True
        else:
            print_check(
                "Neuronデバイスファイル",
                False,
                "/dev/neuron* が見つかりません。TRN1インスタンスで実行していますか？"
            )
            device_ok = False
    except Exception as e:
        print_check("Neuronデバイスファイル", False, f"エラー: {e}")
        device_ok = False
    
    # neuron-ls コマンドの確認
    try:
        result = subprocess.run(['neuron-ls'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            print_check("neuron-ls コマンド", True, "Neuronツールが正しくインストールされています")
            print("\nNeuronデバイス情報:")
            print(result.stdout)
            neuron_ls_ok = True
        else:
            print_check("neuron-ls コマンド", False, "neuron-ls コマンドが失敗しました")
            neuron_ls_ok = False
    except FileNotFoundError:
        print_check("neuron-ls コマンド", False, "neuron-ls コマンドが見つかりません")
        neuron_ls_ok = False
    except Exception as e:
        print_check("neuron-ls コマンド", False, f"エラー: {e}")
        neuron_ls_ok = False
    
    return device_ok and neuron_ls_ok

def check_xla_device():
    """XLAデバイスの初期化と設定を確認"""
    print_section("4. XLAデバイスの初期化")
    
    try:
        import torch_xla.core.xla_model as xm
        
        # XLAデバイスの取得
        device = xm.xla_device()
        print_check("XLAデバイスの取得", True, f"デバイス: {device}")
        
        # 利用可能なデバイスの確認
        devices = xm.get_xla_supported_devices()
        print(f"\n利用可能なXLAデバイス: {devices}")
        
        # デバイスの種類を確認
        device_kind = xm.xla_device_kind()
        # TRN1ではNC_v2 (NeuronCore v2)と表示される
        is_neuron = device_kind in ['NEURON', 'NC_v2']
        print_check(
            "デバイス種別",
            is_neuron,
            f"デバイス種別: {device_kind}" + (" (Neuronデバイスです)" if is_neuron else " (Neuronデバイスではありません)")
        )
        
        # プラットフォームの確認
        try:
            from torch_neuronx.utils import get_platform_target
            platform = get_platform_target()
            print(f"プラットフォーム: {platform}")
        except:
            print("プラットフォーム情報を取得できませんでした")
        
        return is_neuron
        
    except Exception as e:
        print_check("XLAデバイスの初期化", False, f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_tensor_operations():
    """基本的なテンソル操作を確認"""
    print_section("5. 基本的なテンソル操作")
    
    try:
        import torch
        import torch_xla.core.xla_model as xm
        
        device = xm.xla_device()
        
        # CPU上でテンソルを作成
        print("テンソルを作成中...")
        x_cpu = torch.randn(2, 3, 4)
        print_check("CPU上でのテンソル作成", True, f"形状: {x_cpu.shape}")
        
        # Neuronデバイスに転送
        print("Neuronデバイスにテンソルを転送中...")
        x_neuron = x_cpu.to(device)
        print_check("Neuronデバイスへの転送", True, f"デバイス: {x_neuron.device}")
        
        # 簡単な計算
        print("テンソル演算を実行中...")
        y_neuron = x_neuron * 2 + 1
        xm.mark_step()  # 同期
        print_check("テンソル演算", True, f"結果形状: {y_neuron.shape}")
        
        # CPUに戻す
        y_cpu = y_neuron.cpu()
        print_check("CPUへの転送", True, "正常に完了")
        
        return True
        
    except Exception as e:
        print_check("テンソル操作", False, f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_simple_model():
    """シンプルなニューラルネットワークの動作を確認"""
    print_section("6. ニューラルネットワークの動作確認")
    
    try:
        import torch
        import torch.nn as nn
        import torch_xla.core.xla_model as xm
        
        device = xm.xla_device()
        
        # シンプルなモデルを定義
        print("シンプルなモデルを作成中...")
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(20, 5)
            
            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x
        
        model = SimpleModel().to(device)
        print_check("モデルの作成", True, "3層のニューラルネットワーク")
        
        # テストデータを作成
        print("テストデータで推論を実行中...")
        x = torch.randn(4, 10).to(device)
        
        # フォワードパス
        start_time = time.time()
        output = model(x)
        xm.mark_step()  # 同期
        end_time = time.time()
        
        print_check(
            "モデル推論",
            True,
            f"入力形状: {x.shape}, 出力形状: {output.shape}, 実行時間: {end_time - start_time:.3f}秒"
        )
        
        # 出力の妥当性確認
        output_cpu = output.cpu()
        if output_cpu.shape == (4, 5) and not torch.isnan(output_cpu).any():
            print_check("出力の妥当性", True, "正常な出力が得られました")
            return True
        else:
            print_check("出力の妥当性", False, "出力に異常があります")
            return False
        
    except Exception as e:
        print_check("モデル推論", False, f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    print("\n" + "="*70)
    print("  TRN1 Neuron PyTorch 環境チェック")
    print("  AWS Trainium (TRN1) 向け PyTorch Neuron 環境の診断")
    print("="*70)
    
    results = {}
    
    # 各チェックを実行
    results['python'] = check_python_environment()
    results['packages'] = check_required_packages()
    results['devices'] = check_neuron_devices()
    results['xla'] = check_xla_device()
    results['tensor'] = check_tensor_operations()
    results['model'] = check_simple_model()
    
    # 最終結果のサマリー
    print_section("チェック結果サマリー")
    
    all_passed = all(results.values())
    
    print("\n各項目の結果:")
    check_items = [
        ('python', 'Python実行環境'),
        ('packages', '必須パッケージ'),
        ('devices', 'Neuronデバイス'),
        ('xla', 'XLAデバイス初期化'),
        ('tensor', '基本テンソル操作'),
        ('model', 'モデル推論'),
    ]
    
    for key, name in check_items:
        status = "✅ 合格" if results[key] else "❌ 不合格"
        print(f"  {name}: {status}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ すべてのチェックに合格しました！")
        print("TRN1 Neuron PyTorch環境は正常に動作しています。")
        print("neuron_test.py等のプログラムを実行する準備が整っています。")
    else:
        print("❌ 一部のチェックに失敗しました。")
        print("上記のエラーメッセージを確認し、環境を修正してください。")
        print("\n推奨される対処方法:")
        if not results['packages']:
            print("  - 必須パッケージをインストール: pip install torch-neuronx torch-xla")
        if not results['devices']:
            print("  - TRN1インスタンス上で実行していることを確認")
            print("  - Neuronドライバーが正しくインストールされているか確認")
        if not results['xla']:
            print("  - PyTorch XLAの設定を確認")
            print("  - 環境変数の設定を確認")
    print("="*70 + "\n")
    
    # 終了コード
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
