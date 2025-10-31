# PyTorch/XLA + AWS Neuron 性能パターン解析フレームワーク

## 概要

本文書は、PyTorch/XLA + AWS Neuron環境における性能パターン解析フレームワークの包括的な技術文書です。このフレームワークは、測定ツール（performance_pattern_analyzer.py）から構成されています。システムは、自然なforループ、依存性のあるforループ、vmap操作、scan操作といった異なる実行パターンを計測することを目的としています。

性能解析フレームワークは、理論的な仮定よりも実証的なデータ収集を重視した測定中心のアプローチを採用しています。

### 主要コンポーネント

#### CompilationTimeAnalyzerクラス

CompilationTimeAnalyzerは、中核的な測定エンジンとして機能し、異なるパフォーマンス特性を系統的に分離する4段階の解析パイプラインを実装しています。

## 測定手法

### 同期ベースライン確立

測定フレームワークは、XLA同期操作の分離測定を通じて、ベースライン同期コストの確立から開始されます。このベースライン測定は、総実行時間をコンパイル時間、実際の実行時間、同期(sync(), 古いメソッドはxm.mark_step())オーバーヘッドを正確に分解するために重要です。
同期測定手法は、新しいtorch_xla.sync() APIと従来のxm.mark_step()を考慮し、利用可能なAPIを自動検出して測定技術を適応させます。

### コンパイル時間分離技術

初回実行（コンパイルオーバーヘッドを含む）とキャッシュ実行（コンパイル既完了）間の差分解析を採用して、純粋なコンパイル時間を分離します。
各測定パターンは、導出されたコンパイル時間値に対する統計的信頼性を確立するために複数の試行を実行します。システムは、確立されたベースライン測定に基づく同期オーバーヘッドを考慮して、初回実行時間とキャッシュ実行時間の差としてコンパイル時間を計算します。

### グラフ複雑性スケーリング解析

フレームワークは、コンパイル複雑性と実行特性の異なる側面を探索するために8つの異なる計算グラフパターンを実装しています。これらのパターンは、最小限の単一層ネットワークから複雑なマルチヘッドアーキテクチャまで範囲し、本番環境で遭遇する一般的なニューラルネットワーク計算パターンのいくつかをカバーします。

#### グラフアーキテクチャ仕様

TinyGraphパターンは、最小コンパイルオーバーヘッドを理解するためのベースラインとして機能する、最小限の64次元単一層線形変換を実装します。SmallGraphパターンは単一層アーキテクチャを維持しながら128次元に拡張し、コンパイル時間に対するパラメータスケーリング効果の洞察を提供します。

MediumGraphは256次元層でマルチ層アーキテクチャを導入し、コンパイル複雑性における深さ対幅のトレードオフの分析を可能にします。LargeGraphは128次元層を維持しながら3層の深度に拡張し、パラメータ数スケーリングから計算深度の効果を分離します。

WideGraphは、SiLU活性化関数を組み込んだボトルネックアーキテクチャ（128→512→128）を実装し、一般的なトランスフォーマー式計算パターンを表現します。DeepGraphは一貫した次元数の5層に拡張し、大幅な深度増加でのコンパイルスケーリングを探索します。

MLPGraphは、トランスフォーマーフィードフォワードネットワークで典型的な拡張比率を持つ一般的な多層パーセプトロンアーキテクチャを複製します。MultiHeadGraphは、アテンションメカニズムの計算パターンを表現する連結付き並列計算パスを実装します。

### 実行パターン測定

#### 自然なForループパターン

自然なforループ測定パターンは、複数の操作が完了時の同期のみで順次実行される最も一般的なコーディングアプローチを実装します。このパターンは同期オーバーヘッドを最小化しながら、より大きな複合計算グラフを作成します。測定戦略は、直截的なニューラルネットワーク実装で典型的な融合操作シーケンスのコンパイル特性を捕捉します。

**実装例**:
```python
# 自然なforループパターン
x = input_tensor.clone()
for i in range(3):
    x = model(x)  # 3回の連続操作
xla_sync()  # 最後に1回だけ同期
```

#### 依存性のあるForループパターン

依存性のあるforループパターンは中間同期点を導入し、後続の処理決定のために中間結果を実体化する必要があるシナリオをシミュレートします。このパターンは、自然なforループアプローチと比較して、より小さく、より頻繁なグラフコンパイルに関連するコンパイルオーバーヘッドを探索します。

**実装例**:
```python
# 依存性のあるforループパターン
x = input_tensor.clone()
for i in range(3):
    x = model(x)
    xla_sync()  # 毎回同期（中間結果が必要）
```

#### ベクトル化マップ（vmap）パターン

vmap測定パターンは、PyTorchの関数変換システムを活用してバッチデータ全体で同一操作を適用します。このアプローチは通常、逐次実行パターンでは利用できないベクトル化機会を活用できる最適化されたコンパイルパスを生み出します。測定では、ベクトル化実装のコンパイル時間特性と実行効率の両方を捕捉します。

**実装例**:
```python
# vmapパターン（ベクトル化）
def single_layer_func(x):
    return model(x)

batched_func = vmap(single_layer_func, in_dims=0)
batch_input = input_tensor.unsqueeze(0).repeat(3, 1, 1)
result = batched_func(batch_input)
xla_sync()
```

#### 逐次スキャンパターン

スキャンパターン測定は、リカレントニューラルネットワーク実装で典型的な状態伝播を伴う逐次計算に対処します。フレームワークは、利用可能な場合はネイティブtorch.func.scan実装をサポートし、異なるPyTorchインストール間での測定一貫性を保証するためのフォールバック手動実装を提供します。

**実装例**:
```python
# scanパターン（状態付き逐次処理）
def scan_func(carry, x):
    return model(carry), carry

init_carry = input_tensor
scan_inputs = input_tensor.unsqueeze(0).repeat(3, 1, 1)
final_carry, all_outputs = scan(scan_func, init_carry, scan_inputs)
xla_sync()
```

## Performance Pattern Analyzerスクリプト詳細

### 全体処理フロー

performance_pattern_analyzer.pyスクリプトは、段階的な測定アプローチを採用して、PyTorch/XLA + AWS Neuron環境での実行パターンを包括的に分析します。スクリプトの実行フローは以下の4つの主要段階で構成されています。

#### Phase 1: 同期時間ベースライン測定
XLA同期操作の純粋なコストを分離測定し、後続の解析で使用する基準値を確立します。この段階では何も実際の計算を行わず、同期操作のみを反復実行して統計的に信頼できるベースライン値を算出します。

#### Phase 2: コンパイルパターン測定
8種類のグラフアーキテクチャと4種類の実行パターンを組み合わせて、コンパイル時間と実行時間を分離測定します。各パターンで初回実行（コンパイル込み）とキャッシュ実行（コンパイル済み）を比較し、差分からコンパイル時間を抽出します。

#### Phase 3: コンパイル支配性解析
測定されたデータを統計的に処理し、各実行パターンでのコンパイル時間支配性を定量化します。同期オーバーヘッドを除去した実行時間、コンパイル時間比率、キャッシュ効果などを算出します。

#### Phase 4: 理論モデル構築
実測データに基づいて理論的パフォーマンスモデルを構築し、実測値との整合性を検証します。この段階では、異なる実行アプローチの理論的優位性を数学的に証明します。

### グローバル関数詳細

#### xla_sync()関数
```python
def xla_sync():
    """XLA同期処理（新旧API対応）"""
```
**目的**: PyTorch/XLAバージョンに依存しない統一同期インターフェースの提供

**処理内容**:
- USE_NEW_SYNC_APIフラグに基づいてAPI選択
- 新API: torch_xla.sync()を使用
- 旧API: xm.mark_step()にフォールバック
- バージョン互換性を保証して一貫した同期動作を実現

#### ensure_device()関数
```python
def ensure_device():
    """XLAデバイスが利用可能か確認"""
```
**目的**: 実行環境でのXLAデバイス可用性検証と取得

**処理内容**:
- PyTorch/XLAライブラリの利用可能性確認
- 新旧APIでのデバイス取得試行
- デバイス取得成功時はデバイスオブジェクト返却
- 失敗時はNoneを返却してエラーハンドリング支援

### CompilationTimeAnalyzerクラス関数詳細

#### __init__(device)メソッド
**目的**: アナライザーインスタンスの初期化と状態管理構造の構築

**処理内容**:
- XLAデバイス参照の保存
- グラフ情報表示フラグの初期化
- 結果格納辞書の構造化初期化
- タイムスタンプ記録による測定セッション識別

#### measure_pure_sync_time(iterations=10)メソッド
**目的**: XLA同期操作の純粋なコストをベースライン測定として確立

**処理内容**:
- 指定回数の同期操作時間測定ループ
- 各試行でtime.perf_counter()による高精度時間測定
- 統計値算出（平均、中央値、最小値、最大値、標準偏差）
- 測定結果の内部結果辞書への格納
- 後続解析で使用する同期オーバーヘッド基準値の提供

#### create_computation_graphs()メソッド
**目的**: 複数の複雑度レベルの計算グラフを系統的に生成

**処理内容**:
- 8種類のニューラルネットワークアーキテクチャクラス定義
  - TinyGraph: 64次元単一層（最小ベースライン）
  - SmallGraph: 128次元単一層（パラメータ効果測定）
  - MediumGraph: 256次元2層（幅と深さの効果）
  - LargeGraph: 128次元3層（深度効果分離）
  - WideGraph: 128→512→128ボトルネック（幅優先）
  - DeepGraph: 128次元5層（深度優先）
  - MLPGraph: トランスフォーマー式MLP
  - MultiHeadGraph: 並列処理パターン
- 各グラフのパラメータ数とレイヤー数算出
- 初回実行時のみ詳細情報表示（重複防止制御）
- 全グラフインスタンスの辞書形式返却

#### measure_compilation_patterns()メソッド
**目的**: 4種類の実行パターンでコンパイル時間と実行時間を分離測定

**処理内容**:

**自然なforループパターン測定**:
- 各グラフタイプで3回の操作を連続実行
- 最後に単一同期実行（典型的なコーディングパターン）
- 初回実行とキャッシュ実行の時間差からコンパイル時間抽出
- 5つのグラフタイプ（tiny, small, medium, large, wide）で測定

**依存性forループパターン測定**:
- 各操作後に中間同期実行（依存性シミュレーション）
- smallグラフのみで効率的測定
- 3回の操作で3回の同期によるオーバーヘッド測定

**vmapパターン測定**:
- torch.func.vmapによるバッチ並列処理
- 単一の大きなグラフへの最適化効果測定
- ベクトル化による実行効率向上の定量化

**scanパターン測定**:
- torch.func.scanまたは手動実装による状態付き逐次処理
- RNN式の計算パターンシミュレーション
- 利用可能API自動検出による互換性保証

各パターンで統計的信頼性のため3回試行実行し、平均値と個別測定値を記録

#### analyze_compilation_dominance()メソッド
**目的**: 測定データの統計処理によるパフォーマンス特性の定量化

**処理内容**:
- 各実行パターンの測定データ分解
- 同期オーバーヘッドの分離計算（ベースライン × 同期回数）
- 実測実行時間の算出（キャッシュ時間 - 同期オーバーヘッド）
- コンパイル時間比率の計算（コンパイル時間 / 総実行時間）
- キャッシュ高速化比率の算出（初回時間 / キャッシュ時間）
- グラフ数対同期回数比率の効率性指標算出
- 全パターンの比較可能形式での結果構造化

#### analyze_theoretical_model()メソッド
**目的**: 実測データに基づく理論的パフォーマンスモデルの構築と検証

**処理内容**:
- 理論的仮定値の設定（小グラフ vs 大グラフコンパイル時間）
- 明示的ループとvmapの理論的総時間計算
- 理論的高速化倍率の導出
- 実測データとの比較による理論精度評価

#### generate_measurement_summary()メソッド
**目的**: 実測データの人間可読形式サマリー生成

**処理内容**:
- 同期ベースライン統計の整理表示
- グラフサイズ別パフォーマンス特性の階層表示
- 実行パターン別比較結果の構造化出力
- 統計値の適切な精度での数値フォーマット
- 階層的マークダウン形式による視認性向上

#### save_results(filename)メソッド
**目的**: 全測定結果と解析結果の永続化

**処理内容**:
- 測定サマリーの結果辞書への追加
- JSON形式での構造化データシリアライゼーション
- UTF-8エンコーディングによるマルチバイト文字対応
- ファイル出力エラーハンドリング
- 出力ファイルパスの返却

### main()関数処理フロー

**目的**: スクリプト全体の実行制御とエラーハンドリング

**処理内容**:
1. XLAデバイス可用性確認とエラー終了制御
2. CompilationTimeAnalyzerインスタンス生成
3. 4段階の解析フェーズ順次実行
4. 各フェーズでの例外捕捉と診断情報出力
5. 結果ファイル保存と完了メッセージ表示
6. 包括的例外ハンドリングによる障害診断支援

## 結果の解釈

### 1. 全実行パターンでのデバイス初期化現象

**観測された共通パターン**:
全ての実行パターンで以下の現象が観測されました。おそらく初回実行時にデバイス初期化とメモリ転送処理などの時間がかかっている可能性があり、以前のTag: v0.0.1-pytorch-neuron計測ではこれを正しく考慮していませんでした。

**natural_for_loop_small**:
- 初回実行: [13.030ms, 0.346ms, 0.305ms]
- キャッシュ実行: [0.300ms, 0.219ms, 0.251ms]

**dependent_for_loop_small**:
- 初回実行: [22.418ms, 0.608ms, 0.614ms]  
- キャッシュ実行: [0.590ms, 0.523ms, 0.550ms]

**vmap_small**:
- 初回実行: [10.944ms, 0.409ms, 0.354ms]
- キャッシュ実行: [11.317ms, 0.431ms, 0.386ms]

**scan_small**:
- 初回実行: [12.165ms, 0.397ms, 0.359ms]
- キャッシュ実行: [0.473ms, 0.391ms, 0.397ms]

### 2. 前回測定手法の欠陥

前回測定ではコンパイル時間の導出に問題がありました。上述したデバイス初期化と思われる時間を考慮していなかったため、今回は Worm Up と複数回計測を導入しました。
ただしこれも今後より詳細な Profiling が必要です。

**従来の誤った仮定**:
```
コンパイル時間 = 初回実行時間 - キャッシュ実行時間
```

**実際の構成要素**:
```
初回第1試行 = コンパイル + 実行 + デバイス初期化・データ転送・ハードウェアキャッシング?
初回第2試行以降 = 実行のみ（デバイスレベル既キャッシュ）
キャッシュ第1試行 = 実行のみ（グラフレベル既キャッシュ）
```

### 3. vmapパターンの挙動

- vmapキャッシュ実行平均(4.045ms) > 初回実行平均(3.903ms)
- 他パターンではキャッシュ実行 < 初回実行の関係が成立
- vmapのみこの関係が逆転している

vmapの関数変換により生成されるHLOグラフは、forループより実行時間が長く、ベクトル化の理論的利益が実際には実現されていない可能性があります。
これは前回計測から引き続き観測されており原因解明に至っていません。

### 4. scanパターンの実行特性

**scanの挙動**:
- 初回実行: [12.165ms, 0.397ms, 0.359ms]
- キャッシュ実行: [0.473ms, 0.391ms, 0.397ms]

scanパターンは従来のforループと類似した実行特性を示し、デバイス初期化後は安定した実行時間を維持しています。manual_simulation実装でも期待通りの挙動を示しています。
scanは安定的に利用できる可能性が高いです。

### 5. 真のパフォーマンス特性

**実測に基づく実行時間比較**（デバイス初期化除去後）
- natural_for_loop: ~0.26ms
- dependent_for_loop: ~0.55ms（3回同期のオーバーヘッド）
- vmap: ~4.04ms（最も重い実行コスト）
- scan: ~0.42ms

1. **自然なforループが最も効率的**
2. **vmapは理論的期待に反して最も重い**
3. **scanは中間的なコスト**
4. **依存性ありループは同期オーバーヘッドが支配的**

### 6. AWS Neuron + XLA環境での制約

**ハードウェア/デバイスレベル制約（と思われる）**:
1. **初回デバイス初期化**: 10-20ms程度の固定コスト
2. **データ転送・メモリ配置**: SRAM/VRAMへの初期配置
3. **ハードウェアキャッシング**: デバイスレベルでの自動最適化

**ソフトウェアレベル制約**:
1. **グラフコンパイル**: 実際の影響は小さい
2. **実行パターン差**: vmapの予想外のオーバーヘッド
3. **同期頻度**: 依存性パターンでの累積コスト

### 7. 仮説

**デバイス初期化仮説**:
各測定セッションの最初の計算でのみ、Neuronデバイスの初期化・データ転送・ハードウェアキャッシュ構築が発生し、以降の計算では既初期化状態のデバイスを使用すると想定されます。

**vmap実行コスト仮説**:
functorch.vmapによる関数変換は、期待されるベクトル化効果ではなく、実際には追加的な実行オーバーヘッドを生成している可能性があります。

### 結論

端的にいうと今回の計測ではパフォーマンスのより低レイヤの状況を解析するには至らず、発生している事象をより詳細化できた、という結果となりました。

より詳細に計測をする中で、1/ NEFF のキャッシュ有無に関わらず、初回処理実行時とそれ以降で必ず実行時間に差が出ました。そしてこれはおそらくハードウェアレベルでのキャッシュ、VRAM へのデータ転送処理、などが影響しているはずですが、これを Perf/Profiler 等で確認するに至れておらず、前回 Tag で計測した結果についてはこの点を考慮していない単発の実行であったためデータ信用性が低く、vmap の方が性能が劣る、という点自体をまずは疑った上でより詳細な計測をしなければなりません。そして、実利用状況によって性能は変動すると思われ、確定的な結論を出すのは困難です。

### 計測スクリプトの主要改良点

#### 1. ウォームアップ機能
```python
# デバイス初期化と思われる処理時間を除去するウォームアップ
def perform_warmup_execution(self, pattern_name, model, input_tensor, execution_func):
    for i in range(self.warmup_iterations):
        execution_func(model, input_tensor)
        xla_sync()
```

- **目的**: デバイス初期化時間（10-20ms）の除去
- **効果**: 真の実行パフォーマンスの分離測定
- **設定**: デフォルト5回、`--warmup`オプションで調整可能

#### 2. 統計的信頼性向上
```python
# 試行回数増加
measurement_trials: int = 10
exclude_first_trial: bool = True
```

- **試行回数**: 3回から10回に増加（`--trials`オプション）
- **初回異常値除外**: 第1試行の自動除外機能（`--no-exclude-first`で無効化）
- **統計精度**: 標準偏差、信頼区間の改良

#### 3. デバイス再初期化機能
```python
def device_reinitialize():
    torch_xla.sync()
    gc.collect()
    print("🔄 Device reinitialization attempted")
```

- **制御実験**: デバイス状態リセット機能
- **用途**: デバイス初期化効果の検証
- **有効化**: `--enable-reinit`オプション

#### 4. 詳細ログ機能
```python
detailed_timings.append({
    'trial': trial + 1,
    'first_run': first_time,
    'cached_run': cached_time,
    'difference': first_time - cached_time
})
```

- **全試行記録**: 各測定の詳細データ保存
- **ウォームアップ履歴**: 安定化過程の記録
- **統計分析**: 除外前後の比較データ

### 使用方法

#### 基本実行
```bash
# デフォルト設定
python performance_pattern_analyzer.py

# 設定カスタマイズ
python performance_pattern_analyzer.py --warmup 10 --trials 15

# 制御実験モード
python performance_pattern_analyzer.py --enable-reinit --no-exclude-first
```

#### コマンドライン オプション

| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--warmup` | ウォームアップ試行回数 | 5 |
| `--trials` | 測定試行回数 | 10 |
| `--no-exclude-first` | 初回試行除外を無効化 | False |
| `--enable-reinit` | デバイス再初期化を有効化 | False |
| `--output` | 出力ファイル名 | improved_compilation_dominance_analysis.json |

### 改良版測定結果の導出方法

**デバイス初期化時間**
```python
device_init_time = first_trials[0] - statistics.mean(first_trials[1:])
```
- 初回第1試行の異常値から算出
- 通常10-20ms程度の固定コスト

**真の実行時間**
```python
true_execution_time = warmup_data.get('warmup_mean', stats['cached_run_mean'])
```
- ウォームアップ後の安定した実行時間
- デバイス初期化を除去した純粋な計算時間

**ウォームアップ安定化**
```python
warmup_stabilization = warmup_times[-1] - warmup_times[0]
```
- ウォームアップ過程での時間変化

#### 結果ファイル構造

```json
{
  "configuration": {
    "warmup_iterations": 5,
    "measurement_trials": 10,
    "exclude_first_trial": true,
    "enable_device_reinit": false
  },
  "warmup_data": {
    "pattern_name": {
      "warmup_times": [...],
      "warmup_mean": 0.001234,
      "warmup_trend": -0.000056
    }
  },
  "theoretical_analysis": {
    "pattern_name": {
      "device_initialization_time": 0.012345,
      "true_execution_time": 0.000678,
      "warmup_stabilization": -0.000056
    }
  }
}
```

### 参考文献

1. [AWS Neuron Compiler Documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/index.html)
2. [PyTorch/XLA Overview](https://docs.pytorch.org/xla/master/learn/xla-overview.html)
3. [XLA Recompilation Sources](https://docs.pytorch.org/xla/release/r2.7/perf/recompilation.html)
4. [functorch.vmap Documentation](https://docs.pytorch.org/functorch/stable/generated/functorch.vmap.html)
5. [XLA:GPU Architecture](https://openxla.org/xla/gpu_architecture)
6. [Operator Fusion in XLA](https://arxiv.org/pdf/2301.13062)
7. [AI Model Optimization on AWS](https://medium.com/data-science/ai-model-optimization-on-aws-inferentia-and-trainium-cfd48e85d5ac)
8. [PyTorch/XLA Performance Debugging](https://cloud.google.com/blog/topics/developers-practitioners/pytorchxla-performance-debugging-cloud-tpu-vm-part-ii)
