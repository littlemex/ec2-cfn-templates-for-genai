# AWS Neuron ナレッジ抽出レポート

## 概要

AWS Neuron に関する実装上のナレッジを集約することを目的としています。

## 1. 環境セットアップ・設定知識

### 1.1 ハードウェア・プラットフォーム
- **TRN1インスタンス**: Intel x86_64アーキテクチャ（ARMではない）
- **プラットフォーム確認**: `get_platform_target()` → "trn1"
- **デバイス初期化**: `xm.xla_device()` → "xla:0" (Neuronデバイス)

### 1.2 重要な警告メッセージとその意味
```
DeprecationWarning: Use torch_xla.device instead
```
- **意味**: `xm.xla_device()`の非推奨警告
- **対処**: 動作には影響なし、将来的に`torch_xla.device()`に移行

```
W neuron/pjrt-api/neuronpjrt.cc:1972] Use PJRT C-API 0.73 as client did not specify a PJRT C-API version
```
- **意味**: PJRT C-APIバージョン未指定
- **影響**: 動作に影響なし、APIバージョンが自動選択される

```
NET/OFI Failed to initialize sendrecv protocol
```
- **意味**: 分散処理用ネットワーク初期化失敗
- **影響**: 単一ノードテストには影響なし、分散処理時のみ関連

### 1.3 環境変数・設定
- Deep Learning AMI with Neuron使用推奨
- SSMパラメータによる最新AMI自動取得
- Neuronデバイス検出: `/dev/neuron*`

## 2. コンパイラー動作・最適化知識

### 2.1 NeuronXコンパイラ動作
```
INFO ||NEURON_CC_WRAPPER||: Call compiler with cmd: neuronx-cc compile --framework=XLA
```
- **プロセス**: XLA HLO → Neuron実行形式(.neff)変換
- **パラメータ**: `--target=trn1`, `--verbose=35`
- **成功指標**: "Compiler status PASS"

### 2.2 コンパイルキャッシュ機能
```
INFO ||NEURON_CC_WRAPPER||: Using a cached neff at /var/tmp/neuron-compile-cache/
```
- **効果**: 2回目以降の実行で大幅高速化
- **location**: `/var/tmp/neuron-compile-cache/neuronxcc-2.21.18209.0+043b1bf7/`
- **命名**: `MODULE_{hash}+{version}.neff`

### 2.3 コンパイル時間特性
- **初回**: 各モジュールで2-3秒のコンパイル時間
- **キャッシュ後**: ほぼ瞬時（0.001秒レベル）
- **複雑度依存**: モデル複雑度に比例してコンパイル時間増加

## 3. XLA統合・制限事項知識

### 3.1 View Operator制限（重要）
```
RuntimeError: The operator aten::unfold appears to be a view operator, but it has no implementation for the backend "xla:0"
```
- **問題**: `tensor.unfold()`がXLAバックエンドで未実装
- **解決策**: `nn.Unfold()`モジュール使用に変更
- **修正例**: 
  ```python
  # 問題のあるコード
  patches = img.unfold(-2, patch_size, patch_size)
  
  # 解決済みコード  
  self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
  patches = self.unfold(img_flat)
  ```

### 3.2 API互換性問題
```
AttributeError: module 'torch_xla.core.xla_model' has no attribute 'sync'
```
- **原因**: `xm.sync()`のAPI変更
- **解決策**: `xm.wait_device_ops()`に変更またはスキップ

## 4. vmap機能・パフォーマンス知識

### 4.1 Randomness制限（重要発見）
```
RuntimeError: vmap: called random operation while in randomness error mode
```
- **問題**: vmap内でDropout等のランダム操作が制限
- **解決策**: Dropoutを削除またはvmap外で実行
- **推奨**: LayerNormで代替

### 4.2 複雑ネストvmap性能特性
- **小規模テスト**: 0.08秒（超高速）
- **中規模テスト**: 2.43秒（高速）
- **大規模テスト**: 10.64秒（実用的）
- **元notebook級**: 11.65秒（実用レベル）

### 4.3 明示的ループ vs vmap比較
- **明示的ループ**: 1.45秒（キャッシュ効果）
- **複雑vmap**: 11.65秒（初回コンパイル込み）
- **結論**: どちらも実用的、キャッシュ後は明示的ループが高速

## 5. エラーパターン・トラブルシューティング

### 5.1 メモリリーク警告
```
nrtucode: internal error: 54 object(s) leaked, improper teardown
```
- **種類**: NeuronRuntime内部リークエラー
- **影響**: 計算結果には影響なし
- **対策**: プロセス終了時の正常クリーンアップで回避可能

### 5.2 段階的問題解決プロセス
1. **result-001**: aten::unfold問題発見
2. **result-002**: Dropoutランダム操作制限発見  
3. **result-003**: XLA対応修正による改善
4. **result-004**: 完全解決と性能確認

### 5.3 デバッグ手法
- **段階的複雑度テスト**: 小→中→大→元notebook級
- **タイムアウト設定**: 各段階で適切な時間制限
- **システム監視**: CPU、メモリ使用量の追跡

## 6. 性能最適化・ベストプラクティス

### 6.1 メモリ使用量パターン
- **小規模**: 1058.3MB
- **中規模**: 1078.6MB  
- **大規模**: 1087.1MB
- **元notebook級**: 1111.2MB
- **特徴**: 複雑度にほぼ比例、メモリ効率良好

### 6.2 実装推奨事項
1. **XLA対応**: View operatorの代替実装必須
2. **Randomness制限**: vmap内でのランダム操作回避
3. **API互換性**: 非推奨API使用時の適切な代替手段
4. **段階的テスト**: 複雑度を徐々に上げる検証手法

### 6.3 性能最適化戦略
- **コンパイルキャッシュ活用**: 2回目以降の高速化
- **バッチサイズ調整**: メモリ制約に合わせた最適化
- **明示的ループ選択**: シンプルで確実な実装

## 7. 技術進化の洞察

### 7.1 問題解決の進展
- **初期**: 基本的なXLA互換性問題
- **中期**: ランダム操作制限の理解
- **後期**: 適切な修正による完全解決
- **最終**: 実用レベルの性能達成

### 7.2 Neuron固有の制約理解
- **View Operator**: 従来PyTorchとの互換性制限
- **Randomness**: 並列化との整合性制約  
- **API Evolution**: 急速な開発による互換性変化

### 7.3 実用性評価
- **結論**: 適切な修正により実用的性能を達成
- **推奨**: 明示的ループとネストvmapの選択的使用
- **将来性**:継続的なAPI改善により更なる向上期待

## 8. 具体的コード修正パターン

### 8.1 XLA対応修正
```python
# 修正前
patches = img.unfold(-2, self.patch_size, self.patch_size)

# 修正後  
self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
patches_flat = self.unfold(img_flat)
```

### 8.2 Dropout削除
```python
# 修正前
nn.Dropout(0.1)

# 修正後
# Dropout削除、LayerNormで代替
nn.LayerNorm(hidden_size)
```

### 8.3 同期API修正
```python
# 修正前
xm.sync()

# 修正後
try:
    xm.wait_device_ops()
except AttributeError:
    pass  # API変更対応
```

## 9. TEN404コンパイラエラー知識

### 9.1 TEN404エラーの分類

**TEN404エラー**は、AWS Neuronコンパイラ（neuronx-cc）の内部処理限界を示す重要なエラーパターンです。

#### 主要エラータイプ:
- **SundaSizeTiling: tuple index out of range**
  - テンソル最適化処理での配列範囲外アクセス
  - 複雑なテンソル形状変換時に発生

- **InferIntrinsicOnCC: Internal tensorizer error**
  - Collective Communication操作の内部処理失敗
  - 分散処理最適化の限界

- **NeuronValueNumbering: insertAtEnd() incompatible function arguments**
  - 命令最適化での競合状態
  - IR（Intermediate Representation）変換エラー

- **TensorInitialization: Expect NeuronReduceMacro**
  - テンソル初期化の制約違反
  - Reduce操作の期待値不整合

### 9.2 発生パターンと根本原因

#### 典型的な発生ケース:
```python
# 複雑なfor-loop構造（TEN404発生しやすい）
for i in range(large_iterations):
    intermediate = torch.matmul(data[i].unsqueeze(0), data[i].unsqueeze(1))
    processed = torch.sum(intermediate) * complex_scalar_ops
    result = result + processed  # 複雑な累積処理
```

#### 根本原因:
1. **XLA graph fragmentation**: 複雑なループによるグラフ分割
2. **Tensorizer optimization limits**: 多層最適化の処理限界
3. **Memory layout conflicts**: メモリレイアウト最適化の競合
4. **Instruction dependency chains**: 長い依存関係チェーン

### 9.3 実際のエラーメッセージ例

```bash
2025-10-31T07:17:12Z [TEN404] Internal tensorizer error: InferIntrinsicOnCC
- Please open a support ticket at https://github.com/aws-neuron/aws-neuron-sdk/issues/new

ERROR ||NEURON_CC_WRAPPER||: Compilation failed for /tmp/.../model.hlo_module.pb after 0 retries

RuntimeError: RunNeuronCCImpl: error condition error != 0
Command died with <Signals.SIGHUP: 1>
returned non-zero exit status 70
```

### 9.4 ワークアラウンド戦略

#### 即座の対策:
1. **ループ構造の単純化**:
```python
# 修正前（TEN404発生）
for i in range(50):
    complex_tensor_ops(data[i])

# 修正後（成功率向上）  
for i in range(5):  # iteration数削減
    simple_tensor_ops(data[i])
```

2. **テンソル形状の最適化**:
```python
# 修正前
intermediate = torch.matmul(data[i].unsqueeze(0), data[i].unsqueeze(1))

# 修正後
intermediate = torch.sum(data[i] * data[i])  # 単純化
```

3. **Graceful Degradation実装**:
```python
try:
    hardware_result = analyze_complex_pattern(data)
except CompilationError:
    return create_simplified_fallback(pattern_name)
```

### 9.5 GitHub Issues参照リンク

**TEN404エラー事例の詳細情報**:

- **SundaSizeTiling問題**: [Issue #1101](https://github.com/aws-neuron/aws-neuron-sdk/issues/1101)
  - PyTorchモデルのtrace失敗事例
  - tuple index out of range詳細

- **NeuronValueNumbering問題**: [Issue #947](https://github.com/aws-neuron/aws-neuron-sdk/issues/947)  
  - insertAtEnd()互換性問題
  - 大規模モデルでの発生パターン

- **一般的なTensorizer問題**: [Issue #881](https://github.com/aws-neuron/aws-neuron-sdk/issues/881)
  - CNN実装での発生事例
  - シンプルなモデルでの問題

- **OptimizeNKIKernels問題**: [Issue #1114](https://github.com/aws-neuron/aws-neuron-sdk/issues/1114)
  - max()空シーケンス問題
  - NKI kernel最適化限界

- **TensorInitialization問題**: [Issue #1058](https://github.com/aws-neuron/aws-neuron-sdk/issues/1058)
  - NeuronReduceMacro期待値エラー
  - Stanford CS149課題での発生

### 9.6 実用的な回避指針

#### 開発時の推奨事項:
1. **段階的複雑度テスト**: 小→中→大規模での検証
2. **ループ数制限**: iterations < 10での初期テスト
3. **エラーハンドリング**: try-except による graceful fallback
4. **代替実装準備**: 複雑パターンの単純化版準備

#### 本番デプロイ時:
1. **フォールバック機構**: TEN404時の自動代替処理
2. **監視とアラート**: コンパイル失敗の自動検知
3. **パフォーマンス分析**: 成功パターンとの比較分析

## まとめ

このナレッジ抽出により、AWS Neuron環境での深層学習モデル実装における重要な制約、最適化手法、トラブルシューティング方法が体系化されました。特にXLA互換性、vmap制限、コンパイルキャッシュ、そして**TEN404エラー対策**の理解は実用的な開発において重要な知見となります。

**TEN404エラーは正常な動作範囲内の限界**であり、適切な回避策により実用的なシステム構築が可能です。
