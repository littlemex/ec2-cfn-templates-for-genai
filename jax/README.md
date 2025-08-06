# JAX 検証 in AWS Neuron

JAX 検証のために単一の Amazon EC2 Trn1/Inf2 インスタンスを起動する Cloudformation テンプレートと実行のためのスクリプトを提供します。
デフォルトで vscode を入れているため localhost:18080 で vscode server にアクセスできます。

## 事前準備

- [ローカル PC に Session Manager プラグインを入れてください](https://docs.aws.amazon.com/ja_jp/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)
- AWS CLI v2 が利用できる状態を期待します
- 適切な AWS の実行権限が付与されていること

```bash
bash stack_manager.sh --help
AWS Neuron DLAMI Stack Manager

使用方法:
  stack_manager.sh <command> [options]

コマンド:
  create      - スタックを作成
  status      - スタック状態を確認
  monitor     - スタック作成/削除の進捗を監視
  connect     - インスタンスにSSH接続
  jupyter     - Jupyterポートフォワーディング
  vscode      - VS Codeポートフォワーディング
  ports       - 両方のポートフォワーディング
  delete      - スタックを削除
  list        - 全スタック一覧
  validate    - テンプレート検証

オプション:
  -n, --name NAME         スタック名 (デフォルト: neuron-dev-USERNAME)
  -r, --region REGION     AWSリージョン (デフォルト: us-east-1)
  -t, --type TYPE         インスタンスタイプ (デフォルト: trn1.2xlarge)
  -d, --dlami TYPE        DLAMIタイプ (デフォルト: jax-0.6)
  -u, --user USER         ユーザー名 (デフォルト: 現在のユーザー)
  -p, --port PORT         ローカルポート番号 (vscodeコマンド用, デフォルト: 18080)
  -h, --help              このヘルプを表示

インスタンスタイプ:
  Trn1: trn1.2xlarge, trn1.32xlarge
  Inf2: inf2.xlarge, inf2.8xlarge, inf2.24xlarge, inf2.48xlarge

DLAMIタイプ:
  multi-framework, jax-0.6, pytorch-2.7, tensorflow-2.10

例:
  stack_manager.sh create -n my-jax-dev -t trn1.2xlarge -d jax-0.6
  stack_manager.sh status -n my-jax-dev
  stack_manager.sh vscode -n my-jax-dev                    # デフォルトポート 18080
  stack_manager.sh vscode -n my-jax-dev -p 28080           # カスタムポート 28080
  stack_manager.sh ports -n my-jax-dev -p 28080            # 複数ポートでVS Codeは28080
  stack_manager.sh delete -n my-jax-dev
```

## 基本的な使い方

ローカル PC や AWS CloudShell から適切な権限を付与した上で以下のコマンドを実行します。

```bash
export AWS_PROFILE=default
# Trn1 インスタンス作成
bash stack_manager.sh create -n jax
# 作成のモニタリング
bash stack_manager.sh monitor -n jax -r us-east-1
# ポートフォワード(AWS CLI 用の Session Manager プラグインのインストールが必要です)、28080 で接続する場合 -p を指定します。
bash stack_manager.sh vscode -p 28080 -n jax
```

ブラウザから localhost:28080 を開くと vscode が起動します。必要に応じて jupyter 等をインストールしてください。

## Amazon EC2 にログインしてからの検証作業

scripts/ ディレクトリに検証のためのインフラストラクチャの構成情報等を取得するスクリプトが含まれているため実行してください。

```bash
cd scripts
# 仮想環境の切り替えを含みます。
source setup.sh
# JAX の動作確認を実施します。refs: https://docs.kidger.site/equinox/examples/mnist/
python equinox_mnist_verification.py
```

以下は setup.sh の実行で生成されるインフラ構成状況の結果ファイルの一例です。

```bash
cat neuron_verification_results.json 
{
  "system_info": {
    "hostname": "ip-10-0-1-188",
    "kernel": "6.8.0-1031-aws",
    "os_release": "Ubuntu 22.04.5 LTS",
    "architecture": "x86_64",
    "cpu_info": "Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz",
    "memory_total": "30Gi",
    "disk_usage": "22G/497G (5% used)"
  },
  "neuron_driver_status": "loaded",
  "neuron_driver_info": "neuron                450560  6",
  "neuron_ls_available": "true",
  "neuron_ls_output": "instance-type: trn1.2xlarge\ninstance-id: i-09e7a3b60c6a6eac8\n+--------+--------+----------+--------+--------------+-------+----------+------+--------------------------------------+---------+\n| NEURON | NEURON |  NEURON  | NEURON |     PCI      |  PID  |   CPU    | NUMA |               COMMAND                | RUNTIME |\n| DEVICE | CORES  | CORE IDS | MEMORY |     BDF      |       | AFFINITY | NODE |                                      | VERSION |\n+--------+--------+----------+--------+--------------+-------+----------+------+--------------------------------------+---------+\n| 0      | 2      | 0-1      | 32 GB  | 0000:00:1e.0 | 16237 | 0-7      | -1   | python equinox_mnist_verification.py | 2.27.23 |\n+--------+--------+----------+--------+--------------+-------+----------+------+--------------------------------------+---------+",
  "virtual_environments": {
    "aws_neuronx_venv_jax_0_6": {
      "path": "/opt/aws_neuronx_venv_jax_0_6",
      "python_version": "Python 3.10.12",
      "status": "available"
    }
  },
  "jax_environment": {
    "environment_path": "/opt/aws_neuronx_venv_jax_0_6",
    "python_version": "Python 3.10.12",
    "python_path": "/opt/aws_neuronx_venv_jax_0_6/bin/python",
    "jax": {
      "available": true,
      "version": "0.6.1",
      "jaxlib_version": "0.6.1",
      "devices": "Device query failed",
      "device_count": 0,
      "neuron_device_count": 0,
      "default_backend": "unknown"
    },
    "jax_neuronx": {
      "available": true,
      "version": "unknown"
    },
    "libneuronxla": {
      "available": true,
      "version": "2.2.8201.0+f46ac1ef",
      "supported_clients": "['jaxlib 0.4.31 (PJRT C-API 0.54)', 'jaxlib 0.4.33 (PJRT C-API 0.54)', 'jaxlib 0.4.34 (PJRT C-API 0.54)', 'jaxlib 0.4.35 (PJRT C-API 0.55)', 'jaxlib 0.4.36 (PJRT C-API 0.57)', 'jaxlib 0.4.37 (PJRT C-API 0.57)', 'jaxlib 0.4.38 (PJRT C-API 0.58)', 'jaxlib 0.5.0 (PJRT C-API 0.64)', 'jaxlib 0.5.1 (PJRT C-API 0.67)', 'jaxlib 0.5.2 (PJRT C-API 0.67)', 'jaxlib 0.5.3 (PJRT C-API 0.67)', 'jaxlib 0.6.0 (PJRT C-API 0.68)', 'jaxlib 0.6.1 (PJRT C-API 0.68)', 'torch_xla 2.6.0 (PJRT C-API 0.55)', 'torch_xla 2.6.1 (PJRT C-API 0.55)', 'torch_xla 2.7.0 (PJRT C-API 0.61)', 'torch_xla 2.7.1 (PJRT C-API 0.61)']"
    },
    "neuronx_cc": {
      "available": true,
      "version": "2.20.9961.0+0acef03a"
    },
    "neuron_packages": "jax-neuronx               0.6.1.1.0.3499+2edccbed\nlibneuronxla              2.2.8201.0+f46ac1ef\nneuronx-cc                2.20.9961.0+0acef03a"
  },
  "environment_variables": {
    "warning_suppression_enabled": "true",
    "NEURON_RT_LOG_LEVEL": "ERROR",
    "NEURON_CC_FLAGS": "--model-type=transformer",
    "JAX_PLATFORMS": "neuron",
    "JAX_NEURON_EXPERIMENTAL_PYTHON_CACHE": "1",
    "NCCL_DEBUG": "ERROR",
    "OFI_NCCL_DISABLE_WARN": "1",
    "PATH_contains_neuron": "true",
    "VIRTUAL_ENV": "not_set"
  },
  "active_environment": "jax-0.6",
  "active_python_path": "/opt/aws_neuronx_venv_jax_0_6/bin/python",
  "active_python_version": "Python 3.10.12",
  "permission_check": {
    "pip_permissions": "ok"
  },
  "test_results": {
    "jax_basic_test": "failed",
    "jax_devices_test": "failed",
    "jax_devices_output": "No JAX devices found",
    "device_count": 0,
    "neuron_device_count": 0
  }
}
```

以下は、Equinox(JAX) の動作検証結果ファイルの例です。

```bash
{
  "timestamp": "unknown",
  "verification_results": {
    "imports": {
      "jax": {
        "available": true,
        "error": null
      },
      "jax.numpy": {
        "available": true,
        "error": null
      },
      "equinox": {
        "available": true,
        "error": null
      },
      "optax": {
        "available": true,
        "error": null
      },
      "jaxtyping": {
        "available": true,
        "error": null
      },
      "torch": {
        "available": true,
        "error": null
      },
      "torchvision": {
        "available": true,
        "error": null
      }
    },
    "jax_setup": {
      "devices": {
        "count": 2,
        "types": [
          "NeuronCore(id=0, process_index=0, local_id=0)",
          "NeuronCore(id=1, process_index=0, local_id=1)"
        ],
        "default_backend": "neuron"
      },
      "basic_test": {
        "success": true,
        "result": 6.0
      }
    },
    "dataset": {
      "success": true,
      "train_size": 60000,
      "test_size": 10000,
      "batch_shape": [
        32,
        1,
        28,
        28
      ],
      "label_shape": [
        32
      ]
    },
    "model": {
      "success": true,
      "output_shape": [
        10
      ]
    },
    "loss_gradients": {
      "success": true,
      "loss_value": 2.5412025451660156,
      "gradients_computed": true
    },
    "training": {
      "success": true,
      "initial_loss": 2.5412025451660156,
      "train_loss": 2.5412025451660156,
      "model_updated": true
    },
    "status": "passed"
  }
}
```