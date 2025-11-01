# PyTorch Neuron on TRN1 - CloudFormation Template

AWS Trainium (TRN1) インスタンス上でPyTorch Neuronを使用する環境をCloudFormationで構築します。

## 🚀 概要

このプロジェクトは、AWS TRN1インスタンス上にcode-server環境を構築し、PyTorch Neuronを使用した機械学習ワークロードを実行するためのインフラストラクチャを提供します。

### 主な特徴

- **🧠 AWS Trainium対応**: TRN1インスタンスでNeuronアクセラレーターを活用
- **🔬 PyTorch Neuron**: Neuron SDK対応のPyTorchライブラリ
- **☁️ CloudFormation**: インフラストラクチャをコードで管理
- **💻 Code-server**: ブラウザベースのVS Code IDE環境
- **📦 ネストスタック**: 管理しやすいモジュラー構成
- **🛠️ 検証ツール**: 技術的エラーパターンの検証とトラブルシューティング支援

## 🏗️ アーキテクチャ

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CloudFront    │────│   Code-server    │────│ TRN1 Instance   │
│   Distribution  │    │   (Port 80)      │    │ + Neuron DLAMI  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                       │
         └────────────────────────┼───────────────────────┘
                                  │
                         ┌──────────────────┐
                         │ SSM Session Mgr  │
                         │ (secure access)  │
                         └──────────────────┘
```

### 構成要素

- **TRN1インスタンス**: Neuron Deep Learning AMI搭載
- **Code-server**: ポート80でWebIDEを提供
- **CloudFront**: グローバル配信とセキュリティ
- **Lambda関数**: 自動化とヘルスチェック
- **SSM**: セキュアなインスタンス管理

## ⚡ クイックスタート

### 前提条件

- AWS CLI設定済み
- 適切なIAM権限
- Session Manager Plugin (オプション)

### デプロイと環境チェック

[DEPLOYMENT.md](./docs/DEPLOYMENT.md) をご確認ください。

### VMAP 検証

[VMAP.md](./docs/VMAP.md) をご確認ください。

## VMAP vs Scan vs For 性能比較

[PERFORMANCE_PATTERN_ANALYZER.md](./docs/PERFORMANCE_PATTERN_ANALYZER.md) をご確認ください。

## VMAP vs Scan vs For Neuron Profiler 分析

[docs/NEURON_HARDWARE_DEEP_ANALYSIS.md](./docs/docs/NEURON_HARDWARE_DEEP_ANALYSIS.md) をご確認ください。

### 警告メッセージの理解

TRN1での実行時に表示される警告メッセージ（正常動作）

```bash
# PJRT API警告（無害）
W neuron/pjrt-api/neuronpjrt.cc:1972] Use PJRT C-API 0.73 as client did not specify a PJRT C-API version

# 分散処理警告（単一ノードでは無害）
NET/OFI Failed to initialize sendrecv protocol

# メモリリーク警告（正常）
nrtucode: internal error: XX object(s) leaked, improper teardown
```

これらの警告は**計算結果に影響しません**。
