#!/usr/bin/env python3
"""
Equinox MNIST CNN サンプルコード検証スクリプト
JAX + Equinox + Optax + jaxtyping を使用したMNIST分類の動作確認
"""

import sys
import json
import traceback
from typing import Dict, Any
import warnings

def make_json_serializable(obj):
    """オブジェクトをJSONシリアライズ可能に変換"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # シリアライズできないオブジェクトは文字列表現に変換
        return f"<{type(obj).__name__} object>"

def check_imports():
    """オブジェクトをJSONシリアライズ可能に変換"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        # シリアライズできないオブジェクトは文字列表現に変換
        return f"<{type(obj).__name__} object>"

# 警告を抑制（NCCL/OFI関連）
warnings.filterwarnings("ignore", category=UserWarning)

def check_imports() -> Dict[str, Any]:
    """必要なライブラリのインポート確認"""
    results = {}
    
    # 基本ライブラリ
    libraries = {
        'jax': 'jax',
        'jax.numpy': 'jax.numpy as jnp',
        'equinox': 'equinox as eqx',
        'optax': 'optax',
        'jaxtyping': 'jaxtyping',
        'torch': 'torch',
        'torchvision': 'torchvision'
    }
    
    for name, import_str in libraries.items():
        try:
            exec(f"import {import_str}")
            results[name] = {"available": True, "error": None}
            print(f"✓ {name} インポート成功")
        except ImportError as e:
            results[name] = {"available": False, "error": str(e)}
            print(f"✗ {name} インポート失敗: {e}")
        except Exception as e:
            results[name] = {"available": False, "error": str(e)}
            print(f"✗ {name} 予期しないエラー: {e}")
    
    return results

def verify_jax_setup() -> Dict[str, Any]:
    """JAX環境の確認"""
    results = {}
    
    try:
        import jax
        import jax.numpy as jnp
        
        # デバイス確認
        devices = jax.devices()
        results["devices"] = {
            "count": len(devices),
            "types": [str(d) for d in devices],
            "default_backend": jax.default_backend()
        }
        
        # 基本動作テスト
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.sum(x)
        results["basic_test"] = {"success": True, "result": float(y)}
        
        print(f"✓ JAX デバイス: {len(devices)}個")
        print(f"✓ JAX バックエンド: {jax.default_backend()}")
        
    except Exception as e:
        results["error"] = str(e)
        print(f"✗ JAX セットアップエラー: {e}")
    
    return results

def create_mnist_dataset():
    """MNIST データセットの作成テスト"""
    try:
        import torch
        import torchvision
        
        print("MNIST データセット作成中...")
        
        # データ正規化
        normalise_data = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ])
        
        # データセット作成（小さなサンプルサイズで高速化）
        train_dataset = torchvision.datasets.MNIST(
            "MNIST_test", train=True, download=True, transform=normalise_data
        )
        test_dataset = torchvision.datasets.MNIST(
            "MNIST_test", train=False, download=True, transform=normalise_data
        )
        
        # データローダー作成
        BATCH_SIZE = 32  # 小さなバッチサイズでテスト
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        
        # サンプルデータの確認
        sample_x, sample_y = next(iter(trainloader))
        
        print(f"✓ 訓練データセット: {len(train_dataset)} サンプル")
        print(f"✓ テストデータセット: {len(test_dataset)} サンプル")
        print(f"✓ バッチ形状: {sample_x.shape}, ラベル形状: {sample_y.shape}")
        
        return {
            "success": True,
            "train_size": len(train_dataset),
            "test_size": len(test_dataset),
            "batch_shape": list(sample_x.shape),
            "label_shape": list(sample_y.shape)
            # DataLoaderはJSONシリアライズできないため除外
        }
        
    except Exception as e:
        print(f"✗ MNIST データセット作成エラー: {e}")
        return {"success": False, "error": str(e)}

def create_cnn_model():
    """CNN モデルの作成テスト"""
    try:
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        from jaxtyping import Float, Int, Array
        
        print("CNN モデル作成中...")
        
        class CNN(eqx.Module):
            layers: list

            def __init__(self, key):
                key1, key2, key3, key4 = jax.random.split(key, 4)
                self.layers = [
                    eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
                    eqx.nn.MaxPool2d(kernel_size=2),
                    jax.nn.relu,
                    jnp.ravel,
                    eqx.nn.Linear(1728, 512, key=key2),
                    jax.nn.sigmoid,
                    eqx.nn.Linear(512, 64, key=key3),
                    jax.nn.relu,
                    eqx.nn.Linear(64, 10, key=key4),
                    jax.nn.log_softmax,
                ]

            def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
                for layer in self.layers:
                    x = layer(x)
                return x
        
        # モデル作成
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key, 2)
        model = CNN(subkey)
        
        # ダミーデータでテスト
        dummy_x = jax.random.normal(subkey, (1, 28, 28))
        output = model(dummy_x)
        
        print(f"✓ CNN モデル作成成功")
        print(f"✓ 出力形状: {output.shape}")
        print(f"✓ モデル構造確認完了")
        
        return {
            "success": True,
            "output_shape": list(output.shape)
            # modelとkeyオブジェクトはJSONシリアライズできないため除外
        }
        
    except Exception as e:
        print(f"✗ CNN モデル作成エラー: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def test_loss_and_gradients():
    """損失関数と勾配計算のテスト"""
    try:
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        from jaxtyping import Float, Int, Array
        
        print("損失関数と勾配計算テスト中...")
        
        # モデルを再作成
        class CNN(eqx.Module):
            layers: list

            def __init__(self, key):
                key1, key2, key3, key4 = jax.random.split(key, 4)
                self.layers = [
                    eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
                    eqx.nn.MaxPool2d(kernel_size=2),
                    jax.nn.relu,
                    jnp.ravel,
                    eqx.nn.Linear(1728, 512, key=key2),
                    jax.nn.sigmoid,
                    eqx.nn.Linear(512, 64, key=key3),
                    jax.nn.relu,
                    eqx.nn.Linear(64, 10, key=key4),
                    jax.nn.log_softmax,
                ]

            def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
                for layer in self.layers:
                    x = layer(x)
                return x
        
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key, 2)
        model = CNN(subkey)
        
        def cross_entropy(y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]) -> Float[Array, ""]:
            pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
            return -jnp.mean(pred_y)

        def loss(model, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]) -> Float[Array, ""]:
            pred_y = jax.vmap(model)(x)
            return cross_entropy(y, pred_y)
        
        # ダミーデータ作成
        batch_size = 8
        dummy_x = jax.random.normal(key, (batch_size, 1, 28, 28))
        dummy_y = jax.random.randint(key, (batch_size,), 0, 10)
        
        # 損失計算
        loss_value = loss(model, dummy_x, dummy_y)
        
        # 勾配計算（Equinoxのfilter機能使用）
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, dummy_x, dummy_y)
        
        print(f"✓ 損失値: {float(loss_value):.4f}")
        print(f"✓ 勾配計算成功")
        
        return {
            "success": True,
            "loss_value": float(loss_value),
            "gradients_computed": True
        }
        
    except Exception as e:
        print(f"✗ 損失・勾配計算エラー: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def test_training_step():
    """訓練ステップのテスト"""
    try:
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        import optax
        from jaxtyping import Float, Int, Array
        
        print("訓練ステップテスト中...")
        
        # モデルを再作成
        class CNN(eqx.Module):
            layers: list

            def __init__(self, key):
                key1, key2, key3, key4 = jax.random.split(key, 4)
                self.layers = [
                    eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
                    eqx.nn.MaxPool2d(kernel_size=2),
                    jax.nn.relu,
                    jnp.ravel,
                    eqx.nn.Linear(1728, 512, key=key2),
                    jax.nn.sigmoid,
                    eqx.nn.Linear(512, 64, key=key3),
                    jax.nn.relu,
                    eqx.nn.Linear(64, 10, key=key4),
                    jax.nn.log_softmax,
                ]

            def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
                for layer in self.layers:
                    x = layer(x)
                return x
        
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key, 2)
        model = CNN(subkey)
        
        def cross_entropy(y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]) -> Float[Array, ""]:
            pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
            return -jnp.mean(pred_y)

        def loss(model, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]) -> Float[Array, ""]:
            pred_y = jax.vmap(model)(x)
            return cross_entropy(y, pred_y)
        
        # オプティマイザー設定
        optim = optax.adamw(learning_rate=1e-3)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        
        # 訓練ステップ関数
        @eqx.filter_jit
        def make_step(model, opt_state, x, y):
            loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
            updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value
        
        # ダミーデータで訓練ステップ実行
        batch_size = 8
        dummy_x = jax.random.normal(key, (batch_size, 1, 28, 28))
        dummy_y = jax.random.randint(key, (batch_size,), 0, 10)
        
        # 初期損失
        initial_loss = loss(model, dummy_x, dummy_y)
        
        # 訓練ステップ実行
        model, opt_state, train_loss = make_step(model, opt_state, dummy_x, dummy_y)
        
        print(f"✓ 初期損失: {float(initial_loss):.4f}")
        print(f"✓ 訓練後損失: {float(train_loss):.4f}")
        print(f"✓ 訓練ステップ成功")
        
        return {
            "success": True,
            "initial_loss": float(initial_loss),
            "train_loss": float(train_loss),
            "model_updated": True
        }
        
    except Exception as e:
        print(f"✗ 訓練ステップエラー: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def main():
    """メイン検証関数"""
    print("=== Equinox MNIST CNN サンプルコード検証開始 ===\n")
    
    results = {
        "timestamp": str(jax.numpy.datetime64('now') if 'jax' in sys.modules else "unknown"),
        "verification_results": {}
    }
    
    # 1. インポート確認
    print("1. ライブラリインポート確認")
    import_results = check_imports()
    results["verification_results"]["imports"] = import_results
    
    # 必要なライブラリが不足している場合は終了
    required_libs = ['jax', 'equinox', 'optax', 'torch', 'torchvision']
    missing_libs = [lib for lib in required_libs if not import_results.get(lib, {}).get("available", False)]
    
    if missing_libs:
        print(f"\n✗ 必要なライブラリが不足しています: {missing_libs}")
        print("pip install jax equinox optax torch torchvision jaxtyping でインストールしてください")
        results["verification_results"]["status"] = "failed_missing_libraries"
        return results
    
    print("\n2. JAX環境確認")
    jax_results = verify_jax_setup()
    results["verification_results"]["jax_setup"] = jax_results
    
    print("\n3. MNIST データセット作成テスト")
    dataset_results = create_mnist_dataset()
    results["verification_results"]["dataset"] = dataset_results
    
    if not dataset_results["success"]:
        print("データセット作成に失敗したため、以降のテストをスキップします")
        results["verification_results"]["status"] = "failed_dataset"
        return results
    
    print("\n4. CNN モデル作成テスト")
    model_results = create_cnn_model()
    results["verification_results"]["model"] = model_results
    
    if not model_results["success"]:
        print("モデル作成に失敗したため、以降のテストをスキップします")
        results["verification_results"]["status"] = "failed_model"
        return results
    
    print("\n5. 損失関数・勾配計算テスト")
    loss_results = test_loss_and_gradients()
    results["verification_results"]["loss_gradients"] = loss_results
    
    print("\n6. 訓練ステップテスト")
    training_results = test_training_step()
    results["verification_results"]["training"] = training_results
    
    # 全体結果の判定
    all_tests_passed = all([
        import_results.get(lib, {}).get("available", False) for lib in required_libs
    ]) and all([
        dataset_results["success"],
        model_results["success"],
        loss_results["success"],
        training_results["success"]
    ])
    
    results["verification_results"]["status"] = "passed" if all_tests_passed else "partial_failure"
    
    print(f"\n=== 検証完了 ===")
    print(f"総合結果: {'✓ 全テスト成功' if all_tests_passed else '⚠ 一部テスト失敗'}")
    
    # 結果をJSONシリアライズ可能に変換
    serializable_results = make_json_serializable(results)
    
    # 結果をJSONファイルに保存
    with open("equinox_mnist_verification_results.json", "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"詳細結果: equinox_mnist_verification_results.json に保存しました")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0 if results["verification_results"]["status"] == "passed" else 1)
    except Exception as e:
        print(f"検証スクリプト実行エラー: {e}")
        traceback.print_exc()
        sys.exit(1)