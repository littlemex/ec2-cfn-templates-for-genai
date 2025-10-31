#!/usr/bin/env python3
"""
TRN1 Neuron PyTorch 環境チェックスクリプト
AWS Trainium (TRN1) インスタンスでNeuron PyTorchが正しく動作するかを確認します
バージョン情報と利用可能な機能の詳細チェックを含む
"""

import sys
import os
import subprocess
import time
import json
import importlib

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

def safe_import_and_get_version(module_name, version_attr='__version__'):
    """安全にモジュールをインポートしてバージョンを取得"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, version_attr, 'unknown')
        return True, str(version), module
    except ImportError as e:
        return False, f"Import Error: {e}", None
    except Exception as e:
        return False, f"Error: {e}", None

def check_command_version(command):
    """コマンドのバージョンを確認"""
    version_flags = ['--version', '-V', '-v', 'version']
    
    for flag in version_flags:
        try:
            result = subprocess.run([command, flag], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            if result.returncode == 0:
                return True, result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        except Exception as e:
            continue
    
    # バージョンフラグが効かない場合はhelpを試す
    try:
        result = subprocess.run([command, '--help'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            return True, "Available (help output successful)"
    except:
        pass
    
    return False, "Command not found or no version info"

def check_detailed_versions():
    """詳細なバージョン情報をチェック"""
    print_section("7. 詳細バージョン情報")
    
    version_info = {}
    
    # PyTorch関連パッケージ
    print("🔥 PyTorch Related Versions:")
    pytorch_packages = [
        ('torch', '__version__'),
        ('torch_neuronx', '__version__'),
        ('torch_xla', '__version__'),
        ('torchvision', '__version__'),
    ]
    
    version_info['pytorch'] = {}
    for package, attr in pytorch_packages:
        success, version, module = safe_import_and_get_version(package, attr)
        version_info['pytorch'][package] = {
            'available': success,
            'version': version if success else None,
            'error': version if not success else None
        }
        status = "✅" if success else "❌"
        print(f"  {status} {package}: {version}")
    
    return version_info

def check_neuron_tools():
    """Neuronツールのバージョン確認"""
    print_section("8. Neuronツール確認")
    
    tools_info = {}
    
    # 一般的なNeuronツール
    neuron_commands = [
        'neuron-profile',
        'neuronx-cc',
        'neuron-monitor'
    ]
    
    print("🛠️ Neuron Command Line Tools:")
    for cmd in neuron_commands:
        success, info = check_command_version(cmd)
        tools_info[cmd] = {
            'available': success,
            'info': info
        }
        status = "✅" if success else "❌"
        print(f"  {status} {cmd}: {info}")
    
    return tools_info

def check_scan_layers_support():
    """scan_layers機能の対応状況を確認"""
    print_section("9. scan_layers機能確認")
    
    scan_support = {}
    
    # torch_xla.experimental.scan_layers の確認
    print("🔍 PyTorch/XLA scan_layers Support:")
    
    try:
        from torch_xla.experimental import scan_layers
        print("  ✅ torch_xla.experimental.scan_layers: Available")
        scan_support['scan_layers_available'] = True
        
        # scan_layers関数の存在確認
        if hasattr(scan_layers, 'scan_layers'):
            print("  ✅ scan_layers function: Available")
            scan_support['scan_layers_function'] = True
        else:
            print("  ❌ scan_layers function: Not found")
            scan_support['scan_layers_function'] = False
            
    except ImportError as e:
        print(f"  ❌ torch_xla.experimental.scan_layers: Not available ({e})")
        scan_support['scan_layers_available'] = False
        scan_support['import_error'] = str(e)
    
    # torch_xla.experimental.scan の確認
    try:
        from torch_xla.experimental import scan
        print("  ✅ torch_xla.experimental.scan: Available")
        scan_support['scan_available'] = True
    except ImportError as e:
        print(f"  ❌ torch_xla.experimental.scan: Not available ({e})")
        scan_support['scan_available'] = False
    
    return scan_support

def check_vmap_support():
    """vmap機能の対応状況を確認"""
    print_section("10. vmap機能確認")
    
    vmap_support = {}
    
    print("🗺️ vmap Support:")
    
    try:
        import torch
        
        # torch.vmap の確認
        if hasattr(torch, 'vmap'):
            print("  ✅ torch.vmap: Available")
            vmap_support['torch_vmap'] = True
        else:
            print("  ❌ torch.vmap: Not available")
            vmap_support['torch_vmap'] = False
        
        # torch.func.vmap の確認 (newer versions)
        try:
            from torch.func import vmap
            print("  ✅ torch.func.vmap: Available")
            vmap_support['torch_func_vmap'] = True
        except ImportError:
            print("  ❌ torch.func.vmap: Not available")
            vmap_support['torch_func_vmap'] = False
        
    except Exception as e:
        print(f"  ❌ Error checking vmap: {e}")
        vmap_support['error'] = str(e)
    
    return vmap_support

def save_environment_info(basic_results, version_info, tools_info, scan_support, vmap_support):
    """環境情報をJSONファイルに保存"""
    print_section("11. 環境情報の保存")
    
    full_info = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'basic_checks': basic_results,
        'versions': version_info,
        'tools': tools_info,
        'scan_support': scan_support,
        'vmap_support': vmap_support
    }
    
    output_file = '/tmp/neuron_environment_info.json'
    
    try:
        with open(output_file, 'w') as f:
            json.dump(full_info, f, indent=2)
        print(f"✅ Environment info saved to: {output_file}")
        
        # 重要な情報のサマリーも表示
        print("\n📋 Key Information Summary:")
        if version_info.get('pytorch', {}).get('torch_neuronx', {}).get('available'):
            torch_neuronx_ver = version_info['pytorch']['torch_neuronx']['version']
            print(f"  • torch_neuronx: {torch_neuronx_ver}")
        
        if version_info.get('pytorch', {}).get('torch_xla', {}).get('available'):
            torch_xla_ver = version_info['pytorch']['torch_xla']['version']
            print(f"  • torch_xla: {torch_xla_ver}")
        
        if tools_info.get('neuron-profile', {}).get('available'):
            print(f"  • neuron-profile: Available")
        
        if scan_support.get('scan_layers_available'):
            print(f"  • scan_layers: Available")
        else:
            print(f"  • scan_layers: Not Available")
        
        return output_file, full_info
        
    except Exception as e:
        print(f"❌ Failed to save environment info: {e}")
        return None, full_info

def main():
    """メイン実行関数"""
    import argparse
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='TRN1 Neuron PyTorch 環境チェック')
    parser.add_argument('--detailed', '-d', action='store_true', 
                       help='詳細なバージョン情報と機能チェックを実行')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  TRN1 Neuron PyTorch 環境チェック")
    print("  AWS Trainium (TRN1) 向け PyTorch Neuron 環境の診断")
    if args.detailed:
        print("  詳細モード: バージョン情報と機能チェックを含む")
    print("="*70)
    
    results = {}
    
    # 基本チェックを実行
    results['python'] = check_python_environment()
    results['packages'] = check_required_packages()
    results['devices'] = check_neuron_devices()
    results['xla'] = check_xla_device()
    results['tensor'] = check_tensor_operations()
    results['model'] = check_simple_model()
    
    # 詳細モードの場合は追加チェックを実行
    if args.detailed:
        version_info = check_detailed_versions()
        tools_info = check_neuron_tools()
        scan_support = check_scan_layers_support()
        vmap_support = check_vmap_support()
        
        # 環境情報を保存
        output_file, full_info = save_environment_info(
            results, version_info, tools_info, scan_support, vmap_support
        )
    
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
        if args.detailed:
            print("詳細な環境情報が /tmp/neuron_environment_info.json に保存されました。")
            print("この情報を使用して性能解析ツールの実装を決定できます。")
        else:
            print("neuron_test.py等のプログラムを実行する準備が整っています。")
            print("詳細な環境情報を確認するには --detailed オプションを使用してください。")
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
