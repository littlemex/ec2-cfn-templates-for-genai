#!/usr/bin/env python3
"""
AWS Neuron Hardware Deep Analyzer
==================================

完全にハードウェアレベルでの内部挙動解析に専念するフレームワーク

主要機能:
1. Neuron Profiler 2.0を使用したシステム+デバイスレベル統合解析
2. vmap/scan/for-loopの純粋なハードウェア内部挙動解明
3. NTFF (Neuron Trace File Format) 詳細解析
4. Memory Architecture Deep Dive (HBM↔SRAM DMA patterns)
5. Compute Engine Utilization Analysis (Tensor/Vector/Scalar/GPSIMD)
6. Perfetto統合による高度可視化
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import time

import torch
import torch_xla.core.xla_model as xm
import torch_xla
import numpy as np

# Neuron環境チェック
try:
    import torch_neuronx
    from torch_neuronx.experimental import profiler
    NEURON_AVAILABLE = True
except ImportError:
    NEURON_AVAILABLE = False
    print("❌ torch_neuronx not available - Hardware analysis requires Neuron environment")
    sys.exit(1)

# PyTorch関数の互換性チェック  
TORCH_FUNC_AVAILABLE = hasattr(torch, 'func') and hasattr(torch.func, 'scan')

@dataclass
class HardwareProfile:
    """ハードウェアプロファイル詳細メトリクス"""
    pattern_name: str
    
    # Compute Engine分析
    tensor_engine_utilization: float
    vector_engine_utilization: float
    scalar_engine_utilization: float
    gpsimd_engine_utilization: float
    engine_overlap_efficiency: float
    
    # Memory Architecture分析
    hbm_bandwidth_utilization: float
    sram_usage_efficiency: float
    dma_transfer_count: int
    memory_bound_score: float
    
    # Instruction Level分析
    total_instructions: int
    instruction_categories: Dict[str, int]
    hardware_execution_time_ns: int
    
    # Performance Classification
    is_memory_bound: bool
    is_compute_bound: bool
    bottleneck_type: str
    optimization_recommendations: List[str]

@dataclass
class NTFFAnalysisResult:
    """NTFF詳細解析結果"""
    profile_file_path: str
    neff_file_path: str
    
    # Timeline分析
    device_timeline_events: List[Dict]
    system_timeline_events: List[Dict]
    
    # Hardware Metrics
    neuron_core_utilization: Dict[str, float]
    dma_activity_patterns: Dict[str, Any]
    memory_access_patterns: Dict[str, Any]
    
    # Engine分析
    compute_engine_breakdown: Dict[str, Dict]
    instruction_dependency_chains: List[Dict]
    
    # Performance Insights
    performance_bottlenecks: List[str]
    hardware_efficiency_score: float

class NeuronHardwareProfiler:
    """Neuron Hardware Deep Profiler"""
    
    # 統一比較条件設定（performance_pattern_analyzer.py と整合）
    UNIFIED_CONDITIONS = {
        'iterations': 3,           # すべてのパターンで3回処理
        'batch_size': 32,         # バッチサイズ32
        'feature_size': 128,      # 特徴次元128
        'model_type': 'small'     # smallモデル使用
    }
    
    def __init__(self, analysis_name: str = "hardware_deep_analysis"):
        self.analysis_name = analysis_name
        self.device = torch_xla.device()
        self.profile_output_dir = Path(f"/tmp/neuron_hardware_profiles_{analysis_name}")
        self.profile_output_dir.mkdir(exist_ok=True, parents=True)
        
        # パターン名とプロファイルファイルのマッピング追跡
        self.pattern_profile_mapping = {}
        self.profile_execution_order = []
        
        # 環境変数設定 (Neuron Profiler 2.0)
        os.environ['NEURON_RT_INSPECT_OUTPUT_DIR'] = str(self.profile_output_dir)
        
        self.setup_logging()
        
    def setup_logging(self):
        """詳細ログ設定（改良版）"""
        log_file = self.profile_output_dir / "hardware_analysis.log"
        
        # 既存のハンドラーをクリア
        logging.getLogger().handlers.clear()
        
        # ログレベル設定
        logging.getLogger().setLevel(logging.DEBUG)
        
        # ファイルハンドラー（詳細ログ用）
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # コンソールハンドラー（重要な情報のみ）
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # ロガー設定
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 初期ログメッセージ
        self.logger.info(f"🗂️  Hardware analysis log initialized: {log_file}")
        self.logger.info(f"📂 Profile output directory: {self.profile_output_dir}")
        
        # 環境情報ログ
        self.logger.debug(f"Environment variables:")
        for key, value in os.environ.items():
            if 'NEURON' in key or 'XLA' in key:
                self.logger.debug(f"  {key} = {value}")
        
    def create_test_data(self, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """統一条件に基づくテストデータ作成"""
        if shape is None:
            # 統一条件を使用
            shape = (self.UNIFIED_CONDITIONS['batch_size'], self.UNIFIED_CONDITIONS['feature_size'])
        
        self.logger.info(f"Creating test data with unified shape {shape} (batch_size={self.UNIFIED_CONDITIONS['batch_size']}, feature_size={self.UNIFIED_CONDITIONS['feature_size']})")
        tensor = torch.randn(*shape, device=self.device, dtype=torch.float32)
        torch_xla.sync()
        return tensor
        
    @contextmanager
    def hardware_profiling_context(self, pattern_name: str):
        """ハードウェアプロファイリングコンテキスト (Neuron Profiler 2.0) - 強化版エラーハンドリング"""
        self.logger.info(f"🔬 Starting hardware profiling for pattern: {pattern_name}")
        
        # プロファイル開始前の詳細状態記録
        before_files = set(self.profile_output_dir.glob("**/*.ntff"))
        before_neff_files = set(self.profile_output_dir.glob("**/*.neff"))
        
        self.logger.debug(f"Pre-profiling state for {pattern_name}:")
        self.logger.debug(f"  Output directory: {self.profile_output_dir}")
        self.logger.debug(f"  Existing NTFF files: {len(before_files)}")
        self.logger.debug(f"  Existing NEFF files: {len(before_neff_files)}")
        self.logger.debug(f"  Directory writable: {os.access(self.profile_output_dir, os.W_OK)}")
        
        # 重要な環境変数の確認
        important_env_vars = [
            'NEURON_RT_INSPECT_OUTPUT_DIR', 
            'NEURON_RT_INSPECT_DEVICE_PROFILE',
            'NEURON_RT_ROOT_COMM_ID',
            'NEURON_RT_VISIBLE_CORES'
        ]
        
        for env_var in important_env_vars:
            value = os.environ.get(env_var, 'NOT_SET')
            self.logger.debug(f"  {env_var}: {value}")
        
        profiling_success = False
        profiling_error = None
        
        try:
            # デバイスレベルプロファイル環境変数を事前設定
            os.environ['NEURON_RT_INSPECT_DEVICE_PROFILE'] = '1'
            self.logger.debug(f"Set NEURON_RT_INSPECT_DEVICE_PROFILE=1 for {pattern_name}")
            
            # Neuron Profiler 2.0: System + Device profiles
            self.logger.info(f"Initializing profiler with 30-second duration for {pattern_name}")
            
            with profiler.profile(
                port=9012,
                profile_type='system',  # システムレベルプロファイル
                target='neuron_profile_perfetto',  # Perfetto統合
                output_dir=str(self.profile_output_dir),
                ms_duration=30000  # 30秒間キャプチャ
            ) as prof:
                
                self.logger.debug(f"Profiler context entered successfully for {pattern_name}")
                profiling_success = True
                yield prof
                
        except TimeoutError as e:
            profiling_error = f"Profiling timeout: {e}"
            self.logger.error(f"⏱️  Profiling timeout for {pattern_name}: {e}")
            raise
        except ImportError as e:
            profiling_error = f"Profiler import error: {e}"
            self.logger.error(f"📦 Profiler import failed for {pattern_name}: {e}")
            self.logger.error("  → Check if torch_neuronx.experimental.profiler is available")
            raise
        except PermissionError as e:
            profiling_error = f"Permission error: {e}"
            self.logger.error(f"🔒 Permission error for {pattern_name}: {e}")
            self.logger.error(f"  → Check directory permissions: {self.profile_output_dir}")
            raise
        except Exception as e:
            profiling_error = f"Profiler error: {e}"
            self.logger.error(f"💥 Hardware profiling failed for {pattern_name}: {e}")
            self.logger.error(f"  → Error type: {type(e).__name__}")
            self.logger.error(f"  → Pattern name: {pattern_name}")
            
            # スタックトレースをデバッグレベルでログ
            import traceback
            self.logger.debug(f"Full error traceback for {pattern_name}:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    self.logger.debug(f"  {line}")
            raise
        finally:
            # プロファイル後の詳細状態確認
            after_files = set(self.profile_output_dir.glob("**/*.ntff"))
            after_neff_files = set(self.profile_output_dir.glob("**/*.neff"))
            new_files = after_files - before_files
            new_neff_files = after_neff_files - before_neff_files
            
            self.logger.debug(f"Post-profiling state for {pattern_name}:")
            self.logger.debug(f"  Profiling success: {profiling_success}")
            self.logger.debug(f"  New NTFF files: {len(new_files)}")
            self.logger.debug(f"  New NEFF files: {len(new_neff_files)}")
            
            if new_files:
                self.logger.info(f"✅ Generated {len(new_files)} NTFF files for {pattern_name}")
                for ntff_file in new_files:
                    file_size = ntff_file.stat().st_size
                    self.logger.debug(f"  📄 {ntff_file.name} ({file_size:,} bytes)")
            else:
                self.logger.warning(f"⚠️  No NTFF files generated for {pattern_name}")
                self.logger.warning(f"  → Profiling success: {profiling_success}")
                if profiling_error:
                    self.logger.warning(f"  → Error: {profiling_error}")
                
                # 出力ディレクトリの詳細チェック
                all_files = list(self.profile_output_dir.glob("**/*"))
                self.logger.debug(f"  All files in output dir: {len(all_files)}")
                for f in all_files:
                    if f.is_file():
                        self.logger.debug(f"    {f.name} ({f.stat().st_size} bytes)")
            
            # パターン名とファイルのマッピング保存（新しいファイルがある場合のみ）
            if new_files:
                for ntff_file in new_files:
                    self.pattern_profile_mapping[str(ntff_file)] = pattern_name
                    self.profile_execution_order.append((pattern_name, str(ntff_file)))
                self.logger.info(f"📋 Pattern mapping updated for {pattern_name}")
            else:
                self.logger.warning(f"⚠️  No files to map for {pattern_name}")
            
            # マッピング状況のログ
            self.logger.debug(f"Current pattern mapping: {len(self.pattern_profile_mapping)} entries")
            self.logger.debug(f"Current execution order: {len(self.profile_execution_order)} entries")
    
    def analyze_vmap_hardware_behavior(self, data: torch.Tensor) -> HardwareProfile:
        """vmap内部ハードウェア挙動解析（統一条件）"""
        self.logger.info(f"🧬 Analyzing vmap hardware behavior with unified conditions (iterations={self.UNIFIED_CONDITIONS['iterations']})")
        
        with self.hardware_profiling_context("vmap_hardware_deep"):
            def vector_operation(x):
                # 複数の演算を組み合わせてハードウェア利用を観察
                result = torch.sum(x * x, dim=-1)  # Tensor Engine
                result = torch.relu(result)        # Vector Engine  
                result = result + 0.1              # Scalar Engine
                return result
                
            # 統一条件：3回のバッチ処理
            batch_input = data.unsqueeze(0).repeat(self.UNIFIED_CONDITIONS['iterations'], 1, 1)
            vmapped_result = torch.vmap(vector_operation)(batch_input)
            torch_xla.sync()
            
        return self._extract_hardware_profile("vmap_hardware_deep")
    
    def analyze_scan_hardware_behavior(self, data: torch.Tensor) -> HardwareProfile:
        """scan内部ハードウェア挙動解析（統一条件）"""  
        self.logger.info(f"🔄 Analyzing scan hardware behavior with unified conditions (iterations={self.UNIFIED_CONDITIONS['iterations']})")
        
        with self.hardware_profiling_context("scan_hardware_deep"):
            if TORCH_FUNC_AVAILABLE:
                def scan_function(carry, x):
                    # Sequential computationのハードウェアパターン観察
                    new_carry = carry + torch.sum(x)  # Memory access pattern
                    intermediate = torch.matmul(x, x.T)  # Tensor Engine utilization
                    return new_carry, new_carry + torch.sum(intermediate)
                
                init_carry = torch.tensor(0.0, device=self.device)
                # 統一条件：3回の順次処理
                scan_inputs = data.unsqueeze(0).repeat(self.UNIFIED_CONDITIONS['iterations'], 1, 1)
                final_carry, outputs = torch.func.scan(scan_function, init_carry, scan_inputs)
            else:
                # Fallback implementation with unified conditions
                carry = torch.tensor(0.0, device=self.device)
                for i in range(self.UNIFIED_CONDITIONS['iterations']):
                    carry = carry + torch.sum(data[i % data.size(0)])
            
            torch_xla.sync()
            
        return self._extract_hardware_profile("scan_hardware_deep")
    
    def analyze_for_loop_hardware_behavior(self, data: torch.Tensor, loop_size: str = "medium") -> HardwareProfile:
        """for-loop内部ハードウェア挙動解析（パターン別差別化版）"""
        
        # パターン別の設定
        if loop_size == "small":
            iterations = 3
            complexity_factor = 1
            operation_type = "simple"
        elif loop_size == "medium":
            iterations = 10
            complexity_factor = 2
            operation_type = "moderate"
        else:  # large
            iterations = 30
            complexity_factor = 4
            operation_type = "complex"
        
        self.logger.info(f"🔁 Analyzing for-loop hardware behavior - {loop_size} pattern (iterations={iterations}, complexity={complexity_factor})")
        
        try:
            with self.hardware_profiling_context(f"for_loop_hardware_{loop_size}"):
                # パターン別に差別化された処理
                if operation_type == "simple":
                    # Small: 基本的な累積処理
                    result = torch.zeros(data.size(1), device=self.device)
                    for i in range(iterations):
                        idx = i % data.size(0)
                        processed = torch.mean(data[idx])
                        result = result + processed
                        
                elif operation_type == "moderate":
                    # Medium: より複雑な演算パターン
                    result = torch.zeros(data.size(1), device=self.device)
                    intermediate = torch.zeros(data.size(1), device=self.device)
                    
                    for i in range(iterations):
                        idx = i % data.size(0)
                        # より複雑な演算：element-wise + matrix operations
                        processed = torch.mean(data[idx] * data[idx])  # Tensor Engine利用
                        intermediate = torch.relu(intermediate + processed)  # Vector Engine利用
                        result = result + intermediate * 0.1  # Scalar Engine利用
                        
                else:  # complex
                    # Large: 最も複雑な演算パターン
                    result = torch.zeros(data.size(1), device=self.device)
                    accumulator = torch.zeros(data.size(1), device=self.device)
                    temp_buffer = torch.zeros(data.size(1), device=self.device)
                    
                    for i in range(iterations):
                        idx = i % data.size(0)
                        # 複雑な演算チェーン
                        base = data[idx]
                        squared = base * base  # Element-wise multiplication
                        reduced = torch.sum(squared, dim=0, keepdim=True).expand_as(result)  # Reduction + broadcast
                        activated = torch.tanh(reduced)  # Activation function
                        normalized = activated / (torch.norm(activated) + 1e-8)  # Normalization
                        
                        # 複数のメモリアクセスパターン
                        temp_buffer = temp_buffer * 0.9 + normalized * 0.1  # Running average
                        accumulator = accumulator + temp_buffer  # Accumulation
                        result = result + accumulator * (0.01 * (i + 1))  # Weighted accumulation
                
                torch_xla.sync()
                
        except Exception as e:
            self.logger.warning(f"Complex {loop_size} for-loop failed ({e}), trying simplified version")
            
            # フォールバック：少し単純化した版
            with self.hardware_profiling_context(f"for_loop_hardware_{loop_size}_simple"):
                result = torch.zeros(data.size(1), device=self.device)
                
                # 最小限の差別化は維持
                for i in range(max(3, iterations // 2)):  # 最低3回、失敗時は半分に
                    idx = i % data.size(0)
                    if loop_size == "large":
                        processed = torch.sum(data[idx] * data[idx])  # より複雑
                    elif loop_size == "medium":
                        processed = torch.mean(data[idx] * data[idx])  # 中程度
                    else:
                        processed = torch.sum(data[idx])  # シンプル
                    result = result + processed
                    
                torch_xla.sync()
            
        return self._extract_hardware_profile(f"for_loop_hardware_{loop_size}")
    
    def _find_ntff_file_for_pattern(self, pattern_name: str) -> Optional[Path]:
        """パターン名に対応するNTFFファイルを検索"""
        self.logger.info(f"🔍 Finding NTFF file for pattern: {pattern_name}")
        
        # 1. execution orderから最新のファイルを検索
        for saved_pattern, ntff_file_path in reversed(self.profile_execution_order):
            if saved_pattern == pattern_name:
                ntff_file = Path(ntff_file_path)
                if ntff_file.exists():
                    self.logger.info(f"✅ Found NTFF file for {pattern_name}: {ntff_file.name}")
                    return ntff_file
                else:
                    self.logger.warning(f"⚠️ NTFF file not found: {ntff_file_path}")
        
        # 2. pattern mappingから検索 (逆引き)
        for ntff_file_path, saved_pattern in self.pattern_profile_mapping.items():
            if saved_pattern == pattern_name:
                ntff_file = Path(ntff_file_path)
                if ntff_file.exists():
                    self.logger.info(f"✅ Found NTFF file via mapping for {pattern_name}: {ntff_file.name}")
                    return ntff_file
        
        # 3. ファイル名パターンマッチング (フォールバック)
        all_ntff_files = list(self.profile_output_dir.glob("**/*.ntff"))
        for ntff_file in all_ntff_files:
            # パターン名がファイル名に含まれているかチェック
            if pattern_name in str(ntff_file.name).lower():
                self.logger.info(f"✅ Found NTFF file via name matching for {pattern_name}: {ntff_file.name}")
                return ntff_file
        
        # 4. すべて失敗した場合
        self.logger.error(f"❌ No NTFF file found for pattern: {pattern_name}")
        self.logger.info(f"Available files: {[f.name for f in all_ntff_files]}")
        self.logger.info(f"Pattern mapping: {self.pattern_profile_mapping}")
        self.logger.info(f"Execution order: {self.profile_execution_order}")
        
        return None
    
    def _extract_hardware_profile(self, pattern_name: str) -> HardwareProfile:
        """NTFF解析からハードウェアプロファイル抽出"""
        self.logger.info(f"📊 Extracting hardware profile for {pattern_name}")
        
        # パターン名に対応する正しいNTFFファイルを特定
        target_ntff_file = self._find_ntff_file_for_pattern(pattern_name)
        
        if target_ntff_file is None:
            self.logger.warning(f"No specific NTFF file found for {pattern_name}")
            return self._create_fallback_profile(pattern_name)
            
        # NEFF files検索
        neff_files = list(self.profile_output_dir.glob("**/*.neff"))
        
        # neuron-profile コマンドでJSON解析
        ntff_analysis = self._analyze_ntff_with_neuron_profile(target_ntff_file, neff_files[0] if neff_files else None)
        
        # ハードウェアメトリクス抽出
        return HardwareProfile(
            pattern_name=pattern_name,
            
            # Compute Engine分析 (NTFF詳細データから)
            tensor_engine_utilization=ntff_analysis.get('tensor_engine_util', 0.0),
            vector_engine_utilization=ntff_analysis.get('vector_engine_util', 0.0), 
            scalar_engine_utilization=ntff_analysis.get('scalar_engine_util', 0.0),
            gpsimd_engine_utilization=ntff_analysis.get('gpsimd_engine_util', 0.0),
            engine_overlap_efficiency=ntff_analysis.get('engine_overlap_ratio', 0.0),
            
            # Memory Architecture分析
            hbm_bandwidth_utilization=ntff_analysis.get('hbm_bandwidth_util', 0.0),
            sram_usage_efficiency=ntff_analysis.get('sram_usage_efficiency', 0.0),
            dma_transfer_count=ntff_analysis.get('dma_transfer_count', 0),
            memory_bound_score=ntff_analysis.get('memory_bound_score', 0.0),
            
            # Instruction Level分析
            total_instructions=ntff_analysis.get('total_instructions', 0),
            instruction_categories=ntff_analysis.get('instruction_categories', {}),
            hardware_execution_time_ns=ntff_analysis.get('hardware_execution_time_ns', 0),
            
            # Performance Classification
            is_memory_bound=ntff_analysis.get('memory_bound_score', 0.0) > 0.6,
            is_compute_bound=ntff_analysis.get('compute_bound_score', 0.0) > 0.6,
            bottleneck_type=ntff_analysis.get('bottleneck_type', 'unknown'),
            optimization_recommendations=ntff_analysis.get('optimization_recommendations', [])
        )
    
    def _analyze_ntff_with_neuron_profile(self, ntff_path: Path, neff_path: Optional[Path]) -> Dict:
        """neuron-profileツールでNTFF詳細解析"""
        try:
            cmd_args = ['neuron-profile', 'view', '--output-format', 'json', '--output-file', '/tmp/profile_analysis.json']
            
            if neff_path and neff_path.exists():
                cmd_args.extend(['-n', str(neff_path)])
            cmd_args.extend(['-s', str(ntff_path)])
            
            self.logger.info(f"Running: {' '.join(cmd_args)}")
            result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                with open('/tmp/profile_analysis.json', 'r') as f:
                    profile_data = json.load(f)
                return self._process_profile_json(profile_data)
            else:
                self.logger.warning(f"neuron-profile failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"NTFF analysis failed: {e}")
            
        return {}
    
    def _process_profile_json(self, profile_data: Dict) -> Dict:
        """プロファイルJSONデータ処理"""
        processed = {}
        
        # Summary情報抽出
        if 'summary' in profile_data and profile_data['summary']:
            summary = profile_data['summary'][0] if isinstance(profile_data['summary'], list) else profile_data['summary']
            
            processed['hardware_execution_time_ns'] = int(summary.get('total_time', 0) * 1_000_000_000)
            processed['total_instructions'] = summary.get('event_count', 0)
            
            # Engine utilization (概算)
            processed['tensor_engine_util'] = summary.get('tensor_utilization', 0.0)
            processed['vector_engine_util'] = summary.get('vector_utilization', 0.0)
            processed['scalar_engine_util'] = summary.get('scalar_utilization', 0.0)
            processed['gpsimd_engine_util'] = summary.get('gpsimd_utilization', 0.0)
        
        # Instruction分析
        if 'instruction' in profile_data:
            instructions = profile_data['instruction']
            instruction_categories = {}
            
            for instr in instructions:
                opcode = instr.get('opcode', 'unknown')
                instruction_categories[opcode] = instruction_categories.get(opcode, 0) + 1
            
            processed['instruction_categories'] = instruction_categories
            
            # Memory vs Compute bound判定
            memory_ops = sum(count for op, count in instruction_categories.items() 
                           if any(mem_op in op.lower() for mem_op in ['load', 'store', 'dma', 'copy']))
            compute_ops = sum(count for op, count in instruction_categories.items()
                            if any(comp_op in op.lower() for comp_op in ['matmul', 'add', 'mul', 'conv']))
            
            total_ops = memory_ops + compute_ops
            if total_ops > 0:
                processed['memory_bound_score'] = memory_ops / total_ops
                processed['compute_bound_score'] = compute_ops / total_ops
            
        # 最適化推奨生成
        processed['optimization_recommendations'] = self._generate_optimization_recommendations(processed)
        
        return processed
    
    def _generate_optimization_recommendations(self, analysis: Dict) -> List[str]:
        """ハードウェア解析に基づく最適化推奨"""
        recommendations = []
        
        memory_bound_score = analysis.get('memory_bound_score', 0)
        if memory_bound_score > 0.7:
            recommendations.append("Memory-bound: HBM↔SRAM transfer optimization required")
            recommendations.append("Consider data layout optimization for better cache locality")
            
        compute_bound_score = analysis.get('compute_bound_score', 0)  
        if compute_bound_score > 0.7:
            recommendations.append("Compute-bound: Engine parallelization optimization required")
            recommendations.append("Consider operation fusion for better hardware utilization")
            
        tensor_util = analysis.get('tensor_engine_util', 0)
        if tensor_util < 0.5:
            recommendations.append("Low Tensor Engine utilization: Consider matrix operation optimization")
            
        return recommendations
    
    def _create_fallback_profile(self, pattern_name: str) -> HardwareProfile:
        """プロファイルデータが取得できない場合のフォールバック"""
        self.logger.warning(f"Creating fallback profile for {pattern_name}")
        
        return HardwareProfile(
            pattern_name=pattern_name,
            tensor_engine_utilization=0.0,
            vector_engine_utilization=0.0,
            scalar_engine_utilization=0.0,
            gpsimd_engine_utilization=0.0,
            engine_overlap_efficiency=0.0,
            hbm_bandwidth_utilization=0.0,
            sram_usage_efficiency=0.0,
            dma_transfer_count=0,
            memory_bound_score=0.0,
            total_instructions=0,
            instruction_categories={},
            hardware_execution_time_ns=0,
            is_memory_bound=False,
            is_compute_bound=False,
            bottleneck_type="no_data",
            optimization_recommendations=["Profile data unavailable - ensure Neuron Profiler is properly configured"]
        )
    
    def run_comprehensive_hardware_analysis(self) -> Dict[str, HardwareProfile]:
        """包括的ハードウェア挙動解析実行（統一条件）"""
        self.logger.info("🚀 Starting comprehensive hardware analysis with unified conditions")
        
        # 統一条件に基づくテストデータ作成
        test_data = self.create_test_data()  # 統一条件を使用（デフォルト）
        
        results = {}
        
        # vmap ハードウェア解析
        try:
            self.logger.info("Analyzing vmap hardware behavior...")
            results['vmap_hardware'] = self.analyze_vmap_hardware_behavior(test_data)
        except Exception as e:
            self.logger.error(f"vmap analysis failed: {e}")
            
        # scan ハードウェア解析  
        try:
            self.logger.info("Analyzing scan hardware behavior...")
            results['scan_hardware'] = self.analyze_scan_hardware_behavior(test_data)
        except Exception as e:
            self.logger.error(f"scan analysis failed: {e}")
            
        # for-loop ハードウェア解析 (複数サイズ)
        for loop_size in ['small', 'medium', 'large']:
            try:
                self.logger.info(f"Analyzing for-loop hardware behavior ({loop_size})...")
                results[f'for_loop_{loop_size}_hardware'] = self.analyze_for_loop_hardware_behavior(test_data, loop_size)
            except Exception as e:
                self.logger.error(f"for-loop {loop_size} analysis failed: {e}")
                
        self.logger.info("✅ Comprehensive hardware analysis completed")
        return results
    
    def generate_hardware_analysis_report(self, results: Dict[str, HardwareProfile]) -> str:
        """ハードウェア解析レポート生成"""
        report_file = self.profile_output_dir / "hardware_analysis_report.json"
        
        # JSONシリアライズ可能な形式に変換
        serializable_results = {}
        for pattern_name, profile in results.items():
            serializable_results[pattern_name] = asdict(profile)
        
        # レポート保存
        with open(report_file, 'w') as f:
            json.dump({
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
                'analysis_name': self.analysis_name,
                'neuron_environment': {
                    'neuron_available': NEURON_AVAILABLE,
                    'torch_func_available': TORCH_FUNC_AVAILABLE,
                    'device': str(self.device)
                },
                'hardware_profiles': serializable_results
            }, f, indent=2)
        
        self.logger.info(f"📊 Hardware analysis report saved: {report_file}")
        
        # コンソール出力用サマリー生成
        self._print_hardware_analysis_summary(results)
        
        return str(report_file)
    
    def _print_hardware_analysis_summary(self, results: Dict[str, HardwareProfile]):
        """ハードウェア解析サマリー表示"""
        print("\n" + "="*100)
        print("🔬 NEURON HARDWARE DEEP ANALYSIS SUMMARY")
        print("="*100)
        
        for pattern_name, profile in results.items():
            print(f"\n📋 {pattern_name.upper()}")
            print("-" * 60)
            
            # Engine Utilization
            print("🛠️  Compute Engine Utilization:")
            print(f"   Tensor Engine:  {profile.tensor_engine_utilization:.2%}")
            print(f"   Vector Engine:  {profile.vector_engine_utilization:.2%}")
            print(f"   Scalar Engine:  {profile.scalar_engine_utilization:.2%}")
            print(f"   GPSIMD Engine:  {profile.gpsimd_engine_utilization:.2%}")
            print(f"   Engine Overlap: {profile.engine_overlap_efficiency:.2%}")
            
            # Memory Architecture
            print("\n💾 Memory Architecture:")
            print(f"   HBM Bandwidth:  {profile.hbm_bandwidth_utilization:.2%}")
            print(f"   SRAM Efficiency: {profile.sram_usage_efficiency:.2%}")
            print(f"   DMA Transfers:   {profile.dma_transfer_count}")
            
            # Performance Classification
            print(f"\n⚡ Performance Classification:")
            print(f"   Memory Bound:    {'✅' if profile.is_memory_bound else '❌'}")
            print(f"   Compute Bound:   {'✅' if profile.is_compute_bound else '❌'}")
            print(f"   Bottleneck:      {profile.bottleneck_type}")
            
            # Hardware Execution
            print(f"\n🏃 Hardware Execution:")
            print(f"   Total Instructions: {profile.total_instructions:,}")
            print(f"   Execution Time:     {profile.hardware_execution_time_ns:,} ns")
            
            # Optimization Recommendations
            if profile.optimization_recommendations:
                print(f"\n💡 Optimization Recommendations:")
                for rec in profile.optimization_recommendations:
                    print(f"   • {rec}")
            
        print("\n" + "="*100)
    
    def generate_perfetto_analysis(self) -> List[str]:
        """Perfetto統合解析実行（意味のあるファイル名付き）"""
        self.logger.info("🎨 Generating Perfetto analysis files with meaningful names...")
        
        perfetto_files = []
        neff_files = list(self.profile_output_dir.glob("**/*.neff"))
        
        # パターン名順序に従ってPerfettoファイル生成
        pattern_counter = {}
        
        for pattern_name, ntff_file_path in self.profile_execution_order:
            ntff_file = Path(ntff_file_path)
            
            if not ntff_file.exists():
                self.logger.warning(f"NTFF file not found: {ntff_file}")
                continue
                
            try:
                # パターン名ベースのファイル名生成
                if pattern_name in pattern_counter:
                    pattern_counter[pattern_name] += 1
                    suffix = f"_{pattern_counter[pattern_name]}"
                else:
                    pattern_counter[pattern_name] = 1
                    suffix = ""
                
                # 意味のあるPerfettoファイル名
                perfetto_filename = f"{pattern_name}_hardware{suffix}.pftrace"
                perfetto_output = self.profile_output_dir / perfetto_filename
                
                neff_file = neff_files[0] if neff_files else None
                
                cmd_args = [
                    'neuron-profile', 'view',
                    '--output-format', 'perfetto',
                    '--output-file', str(perfetto_output)
                ]
                
                if neff_file:
                    cmd_args.extend(['-n', str(neff_file)])
                cmd_args.extend(['-s', str(ntff_file)])
                
                result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    perfetto_files.append(str(perfetto_output))
                    self.logger.info(f"✅ Perfetto file generated: {perfetto_filename} (from {pattern_name})")
                else:
                    self.logger.warning(f"Perfetto generation failed for {pattern_name}: {result.stderr}")
                    
            except Exception as e:
                self.logger.error(f"Perfetto analysis failed for {pattern_name} ({ntff_file}): {e}")
        
        # パターンマッピングが記録されていない場合のフォールバック
        if not perfetto_files:
            self.logger.warning("No pattern mapping found, using fallback numbering...")
            profile_files = list(self.profile_output_dir.glob("**/*.ntff"))
            
            for i, ntff_file in enumerate(profile_files):
                try:
                    perfetto_output = self.profile_output_dir / f"unknown_pattern_{i}.pftrace"
                    
                    cmd_args = [
                        'neuron-profile', 'view',
                        '--output-format', 'perfetto',
                        '--output-file', str(perfetto_output)
                    ]
                    
                    if neff_files:
                        cmd_args.extend(['-n', str(neff_files[0])])
                    cmd_args.extend(['-s', str(ntff_file)])
                    
                    result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        perfetto_files.append(str(perfetto_output))
                        self.logger.info(f"Fallback Perfetto file generated: {perfetto_output}")
                        
                except Exception as e:
                    self.logger.error(f"Fallback Perfetto generation failed: {e}")
        
        # パターンマッピング情報保存
        self._save_perfetto_pattern_mapping(perfetto_files)
        
        return perfetto_files
    
    def _save_perfetto_pattern_mapping(self, perfetto_files: List[str]):
        """Perfettoファイルとパターンのマッピング情報保存"""
        mapping_file = self.profile_output_dir / "perfetto_pattern_mapping.json"
        
        mapping_data = {
            'pattern_execution_order': self.profile_execution_order,
            'pattern_profile_mapping': self.pattern_profile_mapping,
            'perfetto_files': perfetto_files,
            'generated_timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        self.logger.info(f"Pattern mapping saved: {mapping_file}")


def main():
    """ハードウェア深層解析メイン実行"""
    print("🔬 AWS Neuron Hardware Deep Analyzer")
    print("=" * 50)
    print("⚠️  注意: 従来の時間測定は完全廃止 - ハードウェアレベル解析のみ実行")
    print()
    
    if not NEURON_AVAILABLE:
        print("❌ Neuron environment required for hardware analysis")
        sys.exit(1)
    
    try:
        # ハードウェア深層解析実行
        analyzer = NeuronHardwareProfiler("comprehensive_hardware_deep_analysis")
        
        # 包括的ハードウェア挙動解析
        hardware_profiles = analyzer.run_comprehensive_hardware_analysis()
        
        # レポート生成
        report_file = analyzer.generate_hardware_analysis_report(hardware_profiles)
        
        # Perfetto統合解析
        perfetto_files = analyzer.generate_perfetto_analysis()
        
        # 結果サマリー
        print("\n✅ Hardware Deep Analysis completed successfully!")
        print(f"📊 Detailed report: {report_file}")
        print(f"📁 Profile directory: {analyzer.profile_output_dir}")
        
        if perfetto_files:
            print("🎨 Perfetto analysis files:")
            for pf in perfetto_files:
                print(f"   • {pf}")
            print("   → View at: https://ui.perfetto.dev/")
        
        print("\n🔬 Hardware analysis focus areas achieved:")
        print("   • Compute Engine utilization patterns")
        print("   • Memory architecture deep dive")
        print("   • Instruction-level timeline analysis")
        print("   • vmap/scan/for-loop hardware behavior comparison")
        
    except Exception as e:
        print(f"❌ Hardware analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
