#!/bin/bash
# AWS Neuron JAX-0.6 完全検証環境セットアップスクリプト
# 公式ドキュメントに基づく網羅的な環境確認と設定

set -e

# 使用方法の表示
show_usage() {
    echo "使用方法: $0 [オプション]"
    echo ""
    echo "オプション:"
    echo "  --show-warnings    JAX/NCCL/OFI警告メッセージを表示する"
    echo "  --hide-warnings    JAX/NCCL/OFI警告メッセージを抑制する (デフォルト)"
    echo "  -h, --help         このヘルプを表示"
    echo ""
    echo "警告抑制について:"
    echo "  デフォルトでは以下の無害な警告メッセージを抑制します:"
    echo "  - NCCL NET/OFI Failed to initialize sendrecv protocol"
    echo "  - aws-ofi-nccl initialization failed"
    echo "  - OFI plugin initNet() failed is EFA enabled?"
    echo ""
    echo "  これらの警告はJAXがマルチノード分散処理環境を探索する際に"
    echo "  発生するもので、単一ノードでのNeuron処理には影響しません。"
}

# デフォルト設定: 警告を抑制
SUPPRESS_WARNINGS=true

# 引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --show-warnings)
            SUPPRESS_WARNINGS=false
            shift
            ;;
        --hide-warnings)
            SUPPRESS_WARNINGS=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 色付きの出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログファイルの設定（カレントディレクトリ）
LOG_FILE="./neuron-setup.log"

# ログ関数
log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# JSON出力用の変数
VERIFICATION_RESULTS="{}"

# JSON結果を更新する関数
update_json() {
    local key="$1"
    local value="$2"
    VERIFICATION_RESULTS=$(echo "$VERIFICATION_RESULTS" | jq --arg k "$key" --arg v "$value" '. + {($k): $v}')
}

# JSON結果をオブジェクトで更新する関数
update_json_object() {
    local key="$1"
    local object="$2"
    VERIFICATION_RESULTS=$(echo "$VERIFICATION_RESULTS" | jq --arg k "$key" --argjson obj "$object" '. + {($k): $obj}')
}

# 警告抑制の設定
if [ "$SUPPRESS_WARNINGS" = true ]; then
    log_info "JAX/NCCL/OFI警告メッセージを抑制します"
    # NCCL関連の警告を抑制
    export NCCL_DEBUG=ERROR
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=1
    
    # OFI関連の警告を抑制
    export OFI_NCCL_DISABLE_WARN=1
    export FI_EFA_USE_DEVICE_RDMA=0
    
    # JAX関連の警告を抑制
    export JAX_PLATFORMS=neuron
    export JAX_NEURON_EXPERIMENTAL_PYTHON_CACHE=1
    
    # Neuron Runtime の警告レベルを調整
    export NEURON_RT_LOG_LEVEL=ERROR
    
    log_success "以下の警告メッセージが抑制されます:"
    log_info "  - NCCL NET/OFI Failed to initialize sendrecv protocol"
    log_info "  - aws-ofi-nccl initialization failed"
    log_info "  - OFI plugin initNet() failed is EFA enabled?"
    log_info "  - その他のNCCL/OFI関連警告"
else
    log_info "JAX/NCCL/OFI警告メッセージを表示します"
    # 基本的なJAX設定のみ
    export JAX_PLATFORMS=neuron
    export JAX_NEURON_EXPERIMENTAL_PYTHON_CACHE=1
fi

log_info "=== AWS Neuron JAX-0.6 完全検証環境セットアップ開始 ==="
log_info "実行時刻: $(date)"
log_info "実行ユーザー: $(whoami)"
log_info "作業ディレクトリ: $(pwd)"
log_info "警告抑制モード: $([ "$SUPPRESS_WARNINGS" = true ] && echo "有効" || echo "無効")"

# 1. システム情報の収集
log_info "1. システム情報の収集"
# システム情報を個別に取得
HOSTNAME=$(hostname)
KERNEL=$(uname -r)
OS_RELEASE=$(cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '"')
ARCHITECTURE=$(uname -m)
CPU_INFO=$(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)
MEMORY_TOTAL=$(free -h | grep Mem | awk '{print $2}')
DISK_USAGE=$(df -h / | tail -1 | awk '{print $3"/"$2" ("$5" used)"}')

# JSON形式で組み立て
SYSTEM_INFO=$(cat << EOF
{
    "hostname": "$HOSTNAME",
    "kernel": "$KERNEL",
    "os_release": "$OS_RELEASE",
    "architecture": "$ARCHITECTURE",
    "cpu_info": "$CPU_INFO",
    "memory_total": "$MEMORY_TOTAL",
    "disk_usage": "$DISK_USAGE"
}
EOF
)
update_json_object "system_info" "$SYSTEM_INFO"
log_success "システム情報を収集しました"

# 2. Neuronドライバーの確認
log_info "2. Neuronドライバーの確認"
if lsmod | grep -q neuron; then
    log_success "Neuronドライバーが正常にロードされています"
    DRIVER_INFO=$(lsmod | grep neuron | head -1)
    update_json "neuron_driver_status" "loaded"
    update_json "neuron_driver_info" "$DRIVER_INFO"
else
    log_error "Neuronドライバーがロードされていません"
    update_json "neuron_driver_status" "not_loaded"
    update_json "neuron_driver_info" "N/A"
fi

# 3. neuron-lsコマンドの確認
log_info "3. neuron-lsコマンドの確認"
if command -v neuron-ls &> /dev/null; then
    log_success "neuron-lsコマンドが見つかりました"
    NEURON_LS_OUTPUT=$(neuron-ls 2>/dev/null || echo "実行エラー")
    update_json "neuron_ls_available" "true"
    update_json "neuron_ls_output" "$NEURON_LS_OUTPUT"
    log_info "Neuronデバイス情報:"
    echo "$NEURON_LS_OUTPUT" | tee -a "$LOG_FILE"
else
    log_success "neuron-lsコマンドをフルパスで検索中..."
    NEURON_LS_OUTPUT=""
    for path in "/opt/aws/neuron/bin/neuron-ls" "/usr/local/bin/neuron-ls" "/usr/bin/neuron-ls"; do
        if [ -f "$path" ]; then
            log_success "neuron-lsを発見: $path"
            NEURON_LS_OUTPUT=$($path 2>/dev/null || echo "実行エラー")
            update_json "neuron_ls_available" "true"
            update_json "neuron_ls_output" "$NEURON_LS_OUTPUT"
            log_info "Neuronデバイス情報:"
            echo "$NEURON_LS_OUTPUT" | tee -a "$LOG_FILE"
            break
        fi
    done
    
    if [ -z "$NEURON_LS_OUTPUT" ]; then
        log_error "neuron-lsコマンドが見つかりません"
        update_json "neuron_ls_available" "false"
        update_json "neuron_ls_output" "Command not found"
    fi
fi

# 4. Neuron仮想環境の確認
log_info "4. Neuron仮想環境の確認"
VENV_INFO="{}"
for venv_path in /opt/aws_neuron* /opt/aws_neuronx*; do
    if [ -d "$venv_path" ]; then
        venv_name=$(basename "$venv_path")
        if [ -f "$venv_path/bin/activate" ]; then
            log_success "仮想環境が見つかりました: $venv_name"
            # Python バージョンの確認
            if [ -f "$venv_path/bin/python" ]; then
                python_version=$($venv_path/bin/python --version 2>&1)
                VENV_INFO=$(echo "$VENV_INFO" | jq --arg name "$venv_name" --arg path "$venv_path" --arg python "$python_version" '. + {($name): {"path": $path, "python_version": $python, "status": "available"}}')
            else
                VENV_INFO=$(echo "$VENV_INFO" | jq --arg name "$venv_name" --arg path "$venv_path" '. + {($name): {"path": $path, "python_version": "N/A", "status": "no_python"}}')
            fi
        else
            log_warning "activate スクリプトが見つかりません: $venv_path"
            VENV_INFO=$(echo "$VENV_INFO" | jq --arg name "$venv_name" --arg path "$venv_path" '. + {($name): {"path": $path, "status": "no_activate"}}')
        fi
    fi
done

if [ "$VENV_INFO" = "{}" ]; then
    log_error "Neuron仮想環境が見つかりません"
    update_json "virtual_environments" "{}"
else
    update_json_object "virtual_environments" "$VENV_INFO"
    log_success "仮想環境情報を収集しました"
fi

# 5. JAX-0.6環境の詳細確認
log_info "5. JAX-0.6環境の詳細確認"
JAX_ENV_PATH="/opt/aws_neuronx_venv_jax_0_6"
JAX_INFO="{}"

if [ -d "$JAX_ENV_PATH" ] && [ -f "$JAX_ENV_PATH/bin/activate" ]; then
    log_success "JAX-0.6仮想環境が見つかりました: $JAX_ENV_PATH"
    
    # JAX環境をアクティベートして詳細情報を取得
    (
        source "$JAX_ENV_PATH/bin/activate"
        
        # Pythonスクリプトで環境情報を収集するため、個別の変数取得は不要
        
        # PythonスクリプトでJSONを安全に生成
        python "$(dirname "$0")/generate_jax_info.py" > /tmp/jax_env_info.json
    )
    
    if [ -f "/tmp/jax_env_info.json" ]; then
        JAX_INFO=$(cat /tmp/jax_env_info.json)
        rm -f /tmp/jax_env_info.json
        log_success "JAX-0.6環境の詳細情報を取得しました"
    else
        log_error "JAX-0.6環境の情報取得に失敗しました"
        JAX_INFO='{"error": "Failed to get environment info"}'
    fi
else
    log_error "JAX-0.6仮想環境が見つかりません"
    JAX_INFO='{"available": false, "error": "Environment not found"}'
fi

update_json_object "jax_environment" "$JAX_INFO"

# 6. 環境変数の設定と確認
log_info "6. 環境変数の設定と確認"
log_info "Neuron環境変数を設定中..."

# 重要な環境変数の設定（警告抑制設定は既に適用済み）
if [ "$SUPPRESS_WARNINGS" = false ]; then
    # 警告表示モードの場合、ログレベルをINFOに設定
    export NEURON_RT_LOG_LEVEL=INFO
fi
export NEURON_CC_FLAGS="--model-type=transformer"

# NeuronツールのPATH設定
if [ -d "/opt/aws/neuron/bin" ]; then
    export PATH="/opt/aws/neuron/bin:$PATH"
    log_success "Neuron tools added to PATH"
fi

# 環境変数の確認
PATH_CONTAINS_NEURON=$(echo $PATH | grep -q neuron && echo 'true' || echo 'false')
ENV_VARS=$(cat << EOF
{
    "warning_suppression_enabled": "$SUPPRESS_WARNINGS",
    "NEURON_RT_LOG_LEVEL": "${NEURON_RT_LOG_LEVEL:-not_set}",
    "NEURON_CC_FLAGS": "${NEURON_CC_FLAGS:-not_set}",
    "JAX_PLATFORMS": "${JAX_PLATFORMS:-not_set}",
    "JAX_NEURON_EXPERIMENTAL_PYTHON_CACHE": "${JAX_NEURON_EXPERIMENTAL_PYTHON_CACHE:-not_set}",
    "NCCL_DEBUG": "${NCCL_DEBUG:-not_set}",
    "OFI_NCCL_DISABLE_WARN": "${OFI_NCCL_DISABLE_WARN:-not_set}",
    "PATH_contains_neuron": "$PATH_CONTAINS_NEURON",
    "VIRTUAL_ENV": "${VIRTUAL_ENV:-not_set}"
}
EOF
)
update_json_object "environment_variables" "$ENV_VARS"
log_success "環境変数を設定しました"

# 7. JAX環境のアクティベート
log_info "7. JAX-0.6環境のアクティベート"
if [ -d "$JAX_ENV_PATH" ] && [ -f "$JAX_ENV_PATH/bin/activate" ]; then
    source "$JAX_ENV_PATH/bin/activate"
    log_success "JAX-0.6環境をアクティベートしました"
    log_info "Python interpreter: $(which python)"
    log_info "Python version: $(python --version)"
    update_json "active_environment" "jax-0.6"
    update_json "active_python_path" "$(which python)"
    update_json "active_python_version" "$(python --version)"
else
    log_error "JAX-0.6環境のアクティベートに失敗しました"
    update_json "active_environment" "none"
fi

# 8. 権限問題の確認と修正
log_info "8. 権限問題の確認と修正"
PERMISSION_RESULTS="{}"

# JAX環境での権限確認
if [ -d "$JAX_ENV_PATH" ] && [ -f "$JAX_ENV_PATH/bin/activate" ]; then
    log_info "JAX環境での権限確認中..."
    
    # 仮想環境をアクティベート
    source "$JAX_ENV_PATH/bin/activate"
    
    # pip install テスト（軽量パッケージで確認）
    log_info "pip install 権限テスト中..."
    if pip install --dry-run setuptools &>/dev/null; then
        log_success "pip install 権限は正常です"
        PERMISSION_RESULTS=$(echo "$PERMISSION_RESULTS" | jq '. + {"pip_permissions": "ok"}')
    else
        log_warning "pip install 権限に問題があります。修正を試行します..."
        PERMISSION_RESULTS=$(echo "$PERMISSION_RESULTS" | jq '. + {"pip_permissions": "error"}')
        
        # 権限修正の試行
        PERMISSION_FIXED=false
        
        # 方法1: site-packages の所有者を変更
        log_info "方法1: site-packages の所有者変更を試行中..."
        SITE_PACKAGES_DIR="$JAX_ENV_PATH/lib/python3.10/site-packages"
        if [ -d "$SITE_PACKAGES_DIR" ]; then
            if sudo chown -R $(whoami):$(whoami) "$SITE_PACKAGES_DIR" 2>/dev/null; then
                log_success "site-packages の所有者を変更しました"
                # 再テスト
                if pip install --dry-run setuptools &>/dev/null; then
                    log_success "権限修正が成功しました（方法1）"
                    PERMISSION_FIXED=true
                    PERMISSION_RESULTS=$(echo "$PERMISSION_RESULTS" | jq '. + {"fix_method": "chown_site_packages"}')
                fi
            else
                log_warning "sudo 権限がないため、所有者変更に失敗しました"
            fi
        fi
        
        # 方法2: 仮想環境全体の所有者を変更
        if [ "$PERMISSION_FIXED" = false ]; then
            log_info "方法2: 仮想環境全体の所有者変更を試行中..."
            if sudo chown -R $(whoami):$(whoami) "$JAX_ENV_PATH" 2>/dev/null; then
                log_success "仮想環境全体の所有者を変更しました"
                # 再テスト
                if pip install --dry-run setuptools &>/dev/null; then
                    log_success "権限修正が成功しました（方法2）"
                    PERMISSION_FIXED=true
                    PERMISSION_RESULTS=$(echo "$PERMISSION_RESULTS" | jq '. + {"fix_method": "chown_full_venv"}')
                fi
            else
                log_warning "sudo 権限がないため、所有者変更に失敗しました"
            fi
        fi
        
        # 方法3: 新しい仮想環境の作成を提案
        if [ "$PERMISSION_FIXED" = false ]; then
            log_info "方法3: 新しい仮想環境作成の準備..."
            NEW_VENV_PATH="$HOME/neuron_jax_env"
            
            if command -v python3 &> /dev/null; then
                log_info "新しい仮想環境を作成中: $NEW_VENV_PATH"
                if python3 -m venv "$NEW_VENV_PATH" 2>/dev/null; then
                    log_success "新しい仮想環境を作成しました: $NEW_VENV_PATH"
                    
                    # 新しい環境をアクティベート
                    source "$NEW_VENV_PATH/bin/activate"
                    
                    # 基本パッケージのインストールテスト
                    if pip install --upgrade pip setuptools wheel &>/dev/null; then
                        log_success "新しい環境でのpip動作を確認しました"
                        PERMISSION_FIXED=true
                        PERMISSION_RESULTS=$(echo "$PERMISSION_RESULTS" | jq --arg path "$NEW_VENV_PATH" '. + {"fix_method": "new_venv", "new_venv_path": $path}')
                        
                        # JAX環境のパッケージ情報を取得して後でインストール指示
                        log_info "必要なパッケージリストを準備中..."
                        REQUIRED_PACKAGES="jax jaxlib equinox optax torch torchvision jaxtyping"
                        PERMISSION_RESULTS=$(echo "$PERMISSION_RESULTS" | jq --arg pkgs "$REQUIRED_PACKAGES" '. + {"required_packages": $pkgs}')
                    else
                        log_error "新しい環境でもpipに問題があります"
                    fi
                else
                    log_error "新しい仮想環境の作成に失敗しました"
                fi
            else
                log_error "python3 コマンドが見つかりません"
            fi
        fi
        
        # 最終結果
        if [ "$PERMISSION_FIXED" = true ]; then
            log_success "権限問題を解決しました"
            PERMISSION_RESULTS=$(echo "$PERMISSION_RESULTS" | jq '. + {"status": "fixed"}')
        else
            log_error "権限問題を自動解決できませんでした"
            PERMISSION_RESULTS=$(echo "$PERMISSION_RESULTS" | jq '. + {"status": "failed"}')
            
            # 手動解決方法の提示
            log_info "=== 手動解決方法 ==="
            echo "以下のコマンドを試してください:"
            echo "1. sudo chown -R \$(whoami):\$(whoami) $JAX_ENV_PATH"
            echo "2. または新しい環境: python3 -m venv ~/my_neuron_env && source ~/my_neuron_env/bin/activate"
            echo "3. 管理者に権限変更を依頼してください"
        fi
    fi
else
    log_error "JAX環境が見つからないため、権限確認をスキップします"
    PERMISSION_RESULTS=$(echo "$PERMISSION_RESULTS" | jq '. + {"status": "skipped", "reason": "jax_env_not_found"}')
fi

update_json_object "permission_check" "$PERMISSION_RESULTS"

# 9. 基本的な動作テスト
log_info "9. 基本的な動作テスト"
TEST_RESULTS="{}"

# JAXの基本テスト
log_info "JAXの基本テストを実行中..."
if python -c "import jax; import jax.numpy as jnp; result = jnp.multiply(1, 1); print('JAX test passed:', result)" 2>/dev/null; then
    log_success "JAXの基本テストが成功しました"
    TEST_RESULTS=$(echo "$TEST_RESULTS" | jq '. + {"jax_basic_test": "passed"}')
else
    log_error "JAXの基本テストが失敗しました"
    TEST_RESULTS=$(echo "$TEST_RESULTS" | jq '. + {"jax_basic_test": "failed"}')
fi

# JAXデバイステスト
log_info "JAXデバイステストを実行中..."
# JAXデバイステスト（シンプルなカウントベース）
log_info "JAXデバイステストを実行中..."
JAX_DEVICE_COUNT=$(python -c "import jax; print(len(jax.devices()))" 2>/dev/null || echo "0")
JAX_NEURON_DEVICE_COUNT=$(python -c "import jax; devices = jax.devices(); print(len([d for d in devices if 'neuron' in str(d).lower()]))" 2>/dev/null || echo "0")

if [ "$JAX_DEVICE_COUNT" -gt 0 ]; then
    log_success "JAXデバイステストが成功しました"
    JAX_TEST_RESULT="Found $JAX_DEVICE_COUNT JAX devices, $JAX_NEURON_DEVICE_COUNT are Neuron devices"
    log_info "$JAX_TEST_RESULT"
    TEST_RESULTS=$(echo "$TEST_RESULTS" | jq --arg result "$JAX_TEST_RESULT" '. + {"jax_devices_test": "passed", "jax_devices_output": $result, "device_count": '"$JAX_DEVICE_COUNT"', "neuron_device_count": '"$JAX_NEURON_DEVICE_COUNT"'}')
else
    log_error "JAXデバイステストが失敗しました"
    JAX_TEST_RESULT="No JAX devices found"
    TEST_RESULTS=$(echo "$TEST_RESULTS" | jq --arg result "$JAX_TEST_RESULT" '. + {"jax_devices_test": "failed", "jax_devices_output": $result, "device_count": 0, "neuron_device_count": 0}')
fi

update_json_object "test_results" "$TEST_RESULTS"

# 10. 最終結果の出力
log_info "10. 検証結果の出力"

# JSON結果をファイルに保存（カレントディレクトリ）
RESULT_FILE="./neuron_verification_results.json"
echo "$VERIFICATION_RESULTS" | jq '.' > "$RESULT_FILE"
log_success "検証結果を $RESULT_FILE に保存しました"

# 結果の表示
log_info "=== 検証結果サマリー ==="
echo "$VERIFICATION_RESULTS" | jq -r '
"システム情報: " + .system_info.hostname + " (" + .system_info.os_release + ")",
"Neuronドライバー: " + .neuron_driver_status,
"neuron-ls利用可能: " + .neuron_ls_available,
"JAX環境: " + ((.jax_environment.jax.available // false) | tostring),
"JAX version: " + (.jax_environment.jax.version // "N/A"),
"JAX-NeuronX: " + ((.jax_environment.jax_neuronx.available // false) | tostring),
"アクティブ環境: " + (.active_environment // "none"),
"pip権限: " + (.permission_check.pip_permissions // "not_checked"),
"権限修正: " + (.permission_check.status // "not_checked"),
"基本テスト: " + (.test_results.jax_basic_test // "not_run")
'

log_info "=== セットアップ完了 ==="
log_success "AWS Neuron JAX-0.6環境のセットアップが完了しました"
log_info "詳細な結果は $RESULT_FILE を確認してください"
log_info "ログファイル: $LOG_FILE"

echo ""
echo "=== 使用可能なコマンド ==="
echo "  neuron-ls                    - Neuronデバイスの一覧表示"
echo "  neuron-top                   - Neuronデバイスの使用状況監視"
echo "  python -c 'import jax; print(jax.devices())'  - JAXデバイス確認"
echo ""
echo "=== 仮想環境の切り替え ==="
echo "  source /opt/aws_neuronx_venv_jax_0_6/bin/activate      - JAX-0.6環境"
echo "  source /opt/aws_neuronx_venv_pytorch_2_7/bin/activate  - PyTorch-2.7環境"
echo "  source /opt/aws_neuronx_venv_tensorflow_2_10/bin/activate - TensorFlow-2.10環境"
echo ""
echo "=== 権限問題がある場合 ==="
echo "  このスクリプトが自動的に修正を試行しますが、失敗した場合:"
echo "  1. sudo chown -R \$(whoami):\$(whoami) /opt/aws_neuronx_venv_jax_0_6"
echo "  2. python3 -m venv ~/my_neuron_env && source ~/my_neuron_env/bin/activate"
echo "  3. pip install jax equinox optax torch torchvision jaxtyping"
echo ""
echo "=== 警告制御 ==="
echo "  $0 --show-warnings           - 警告メッセージを表示して再実行"
echo "  $0 --hide-warnings           - 警告メッセージを抑制して再実行 (デフォルト)"
echo ""
echo "=== 検証結果の確認 ==="
echo "  cat $RESULT_FILE | jq '.'     - JSON形式の詳細結果"
echo "  tail -f $LOG_FILE                      - セットアップログ"