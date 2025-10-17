#!/bin/bash
#
# TRN1 Neuron PyTorch 環境チェックスクリプト（ラッパー）
# 仮想環境を有効化してからPythonスクリプトを実行します
#

set -e

VENV_PATH="/opt/aws_neuronx_venv_pytorch_2_8"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/check_neuron_env.py"

echo "========================================================================"
echo "  TRN1 Neuron PyTorch 環境チェック（ラッパー）"
echo "========================================================================"
echo ""

# 仮想環境の存在確認
if [ ! -d "${VENV_PATH}" ]; then
    echo "❌ エラー: 仮想環境が見つかりません"
    echo "   パス: ${VENV_PATH}"
    echo ""
    echo "   仮想環境を手動でインストールするか、正しいパスを指定してください。"
    exit 1
fi

echo "✅ 仮想環境を検出しました: ${VENV_PATH}"
echo ""

# 仮想環境のactivateスクリプトを確認
ACTIVATE_SCRIPT="${VENV_PATH}/bin/activate"
if [ ! -f "${ACTIVATE_SCRIPT}" ]; then
    echo "❌ エラー: activateスクリプトが見つかりません"
    echo "   パス: ${ACTIVATE_SCRIPT}"
    exit 1
fi

# Pythonスクリプトの存在確認
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "❌ エラー: Pythonスクリプトが見つかりません"
    echo "   パス: ${PYTHON_SCRIPT}"
    exit 1
fi

echo "仮想環境を有効化中..."
echo "  source ${ACTIVATE_SCRIPT}"
echo ""

# 仮想環境を有効化してPythonスクリプトを実行
source "${ACTIVATE_SCRIPT}"

# Pythonのパスを表示
echo "使用するPython: $(which python3)"
echo ""

# Pythonスクリプトを実行
python3 "${PYTHON_SCRIPT}"

# 仮想環境を無効化
deactivate

exit $?
