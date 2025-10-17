# VMAP 動作確認スクリプトの実行結果


VMAP 動作確認スクリプトは `../scripts/test_neuron_vmap.py` です。

```
✅ torch_neuronx successfully imported

================================================================================
 AWS Neuron vmap 失敗検証テスト
================================================================================
[07:39:24] 🔍 このスクリプトはNeuronでのNested vmapを検証します

================================================================================
 Neuron環境チェック
================================================================================
[07:39:24] ✅ torch_neuronx バージョン: 2.8.0.2.10.13553+1e4dd6ca
2025-10-17 07:39:24.838173: W neuron/pjrt-api/neuronpjrt.cc:1972] Use PJRT C-API 0.73 as client did not specify a PJRT C-API version
2025-Oct-17 07:39:30.0403 478555:478624 [1] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):213 CCOM WARN NET/OFI Failed to initialize sendrecv protocol
2025-Oct-17 07:39:30.0413 478555:478624 [1] int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t**):354 CCOM WARN NET/OFI aws-ofi-nccl initialization failed
2025-Oct-17 07:39:30.0422 478555:478624 [1] ncclResult_t nccl_net_ofi_init_no_atexit_fini_v6(ncclDebugLogger_t):183 CCOM WARN NET/OFI Initializing plugin failed
2025-Oct-17 07:39:30.0432 478555:478624 [1] net_plugin.cc:97 CCOM WARN OFI plugin initNet() failed is EFA enabled?
[07:39:30] ✅ XLA サポートデバイス: ['xla:0', 'xla:1']
[07:39:30] ✅ XLA デバイス種別: NC_v2
[07:39:30] ✅ Neuron プラットフォーム: trn1
[07:39:30] ⚠️ Neuronデバイスなし
__main__: Neuronデバイスなし
[07:39:30] ✅ neuron-ls コマンド成功
/work/test.py:470: DeprecationWarning: Use torch_xla.device instead
  device = xm.xla_device()
[07:39:30] ✅ XLAデバイス初期化成功: xla:0
[07:39:30] 🔍 サポートされているXLAデバイス: ['xla:0', 'xla:1']
[07:39:30] 🔍 XLAデバイス種別: NC_v2
[07:39:30] 🔍 Neuronプラットフォーム: trn1
[07:39:30] ✅ ✅ NeuronX環境で実際のvmapコンパイルを検証します
[07:39:30] 🔍 使用デバイス: xla:0
[07:39:30] 🔍 テスト1: 単一vmap

================================================================================
 単一vmap テスト
================================================================================
[07:39:30] 🔍 システム監視を開始しました
[07:39:30] 🔍 テストデータ形状: torch.Size([32, 10])
[07:39:30] 🔍 単一vmapコンパイル開始...
/work/test.py:280: DeprecationWarning: Use torch_xla.sync instead
  xm.mark_step()
2025-10-17 07:39:30.000474:  478555  INFO ||NEURON_CC_WRAPPER||: Using a cached neff at /var/tmp/neuron-compile-cache/neuronxcc-2.21.18209.0+043b1bf7/MODULE_6431493834373381678+e30acd3a/model.neff
[07:39:30] 🔍 NeuronX同期完了
[07:39:31] 🔍 システム監視を停止しました
[07:39:31] ✅ 単一vmap成功 - 実行時間: 0.018秒
[07:39:31] 🔍 出力形状: torch.Size([32, 5])
[07:39:31] 🔍 システム統計: {'duration_seconds': 1, 'cpu_max_percent': 2.5, 'memory_max_mb': 3196.82421875, 'process_memory_max_mb': 1038.01953125, 'total_samples': 1}
[07:39:31] 🔍 テスト2: ネストしたvmap

================================================================================
 ネストしたvmap テスト
================================================================================
[07:39:31] 🔍 ⚠️  警告: このテストは最大60秒でタイムアウトします
[07:39:31] 🔍 データ形状: [4, 8, 10]
[07:39:31] 🔍 システム監視を開始しました
[07:39:31] 🔍 ネストしたvmap構造を構築中...
[07:39:31] 🔍 ネストしたvmapコンパイル開始...
[07:39:31] 🔍 外側vmap - バッチ形状: torch.Size([8, 10])
[07:39:31] 🔍 内側vmap - サンプル形状: torch.Size([10])
/work/test.py:342: DeprecationWarning: Use torch_xla.sync instead
  xm.mark_step()
2025-10-17 07:39:31.000490:  478555  INFO ||NEURON_CC_WRAPPER||: Using a cached neff at /var/tmp/neuron-compile-cache/neuronxcc-2.21.18209.0+043b1bf7/MODULE_12918015898290422879+e30acd3a/model.neff
[07:39:31] 🔍 NeuronX同期完了
[07:39:32] 🔍 システム監視を停止しました
[07:39:32] ✅ ネストしたvmap成功 - 実行時間: 0.015秒
[07:39:32] 🔍 出力形状: torch.Size([4, 8, 5])
[07:39:32] 🔍 システム統計: {'duration_seconds': 1, 'cpu_max_percent': 1.2, 'memory_max_mb': 3205.23046875, 'process_memory_max_mb': 1046.51953125, 'total_samples': 1}
[07:39:32] 🔍 テスト3: 明示的ループ

================================================================================
 明示的ループ代替案テスト
================================================================================
[07:39:32] 🔍 システム監視を開始しました
[07:39:32] 🔍 テストデータ形状: torch.Size([4, 8, 10])
[07:39:32] 🔍 明示的ループ実行開始...
[07:39:33] 🔍 システム監視を停止しました
[07:39:33] ✅ 明示的ループ成功 - 実行時間: 0.008秒
[07:39:33] 🔍 出力形状: torch.Size([4, 8, 5])
[07:39:33] 🔍 システム統計: {'duration_seconds': 1, 'cpu_max_percent': 1.3, 'memory_max_mb': 3205.6953125, 'process_memory_max_mb': 1048.26953125, 'total_samples': 1}

================================================================================
 テスト結果サマリー
================================================================================
[07:39:33] 🔍 single_vmap: ✅ 成功 (0.018秒)
[07:39:33] 🔍   出力形状: torch.Size([32, 5])
[07:39:33] 🔍   最大メモリ: 1038.0MB
[07:39:33] 🔍 nested_vmap: ✅ 成功 (0.015秒)
[07:39:33] 🔍   出力形状: torch.Size([4, 8, 5])
[07:39:33] 🔍   最大メモリ: 1046.5MB
[07:39:33] 🔍 explicit_loop: ✅ 成功 (0.008秒)
[07:39:33] 🔍   出力形状: torch.Size([4, 8, 5])
[07:39:33] 🔍   最大メモリ: 1048.3MB

================================================================================
 推奨事項
================================================================================
[07:39:33] 🔍 ✅ 単一vmap: 全環境で使用可能
[07:39:33] 🔍 ✅ 明示的ループ: Neuron推奨パターン
nrtucode: internal error: 54 object(s) leaked, improper teardown
```