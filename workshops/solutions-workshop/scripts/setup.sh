#!/bin/bash

# =============================================================================
# 統合セットアップスクリプト
# - Cline拡張機能のインストール
# =============================================================================

set -e  # エラー時に停止

# 色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# エラーハンドリング
error_exit() {
    log_error "$1"
    exit 1
}

# 環境変数の設定
setup_environment() {
    log_info "環境変数を設定中..."
    
    # デフォルトユーザーの設定
    if [ -z "$CODE_SERVER_USER" ]; then
        CODE_SERVER_USER="coder"
        log_info "CODE_SERVER_USER が設定されていません。デフォルト値 'coder' を使用します"
    fi
    
    # 作業ディレクトリの確認
    WORK_DIR="/work/coding-agents/scripts"
    if [ ! -d "$WORK_DIR" ]; then
        error_exit "作業ディレクトリ $WORK_DIR が存在しません"
    fi
    
    # ホームディレクトリの設定
    HOME_DIR="/home/$CODE_SERVER_USER"
    if [ ! -d "$HOME_DIR" ]; then
        error_exit "ホームディレクトリ $HOME_DIR が存在しません"
    fi
    
    log_success "環境変数設定完了 (USER: $CODE_SERVER_USER, HOME: $HOME_DIR)"
}

# 前提条件の確認
check_prerequisites() {
    log_info "前提条件を確認中..."
    
    # Python3の確認
    if ! command -v python3 &> /dev/null; then
        error_exit "Python3 がインストールされていません"
    fi
    
    # code-serverの確認
    if ! command -v code-server &> /dev/null; then
        error_exit "code-server がインストールされていません"
    fi
    
    log_success "前提条件確認完了"
}

# Cline拡張機能のインストール
install_extension() {
    log_info "拡張機能をインストール中..."
    
    # 既存の拡張機能確認
    if code-server --list-extensions | grep -q "saoudrizwan.claude-dev"; then
        log_warning "拡張機能は既にインストールされています。強制再インストールします..."
    fi
    
    # 拡張機能のインストール
    if code-server --install-extension saoudrizwan.claude-dev --force; then
        log_success "Cline拡張機能のインストール完了"
    else
        error_exit "Cline拡張機能のインストールに失敗しました"
    fi
        if code-server --install-extension amazonwebservices.amazon-q-vscode --force; then
        log_success "Amazon Q Developer拡張機能のインストール完了"
    else
        error_exit "Amazon Q Developer拡張機能のインストールに失敗しました"
    fi
}



# uvxの確認とインストール
check_install_uvx() {
    log_info "uvx の確認中..."
    
    if command -v uvx &> /dev/null; then
        log_success "uvx は既にインストールされています"
        return 0
    fi
    
    log_info "uvx をインストール中..."
    
    # pipxの確認
    if ! command -v pipx &> /dev/null; then
        log_info "pipx をインストール中..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y pipx || {
                log_warning "pipx のインストールに失敗しました。手動でインストールしてください"
                return 1
            }
        else
            log_warning "apt-get が利用できません。pipx を手動でインストールしてください"
            return 1
        fi
    fi
    
    # uvの確認とインストール
    if ! command -v uv &> /dev/null; then
        log_info "uv をインストール中..."
        if ! pipx install uv; then
            log_warning "uv のインストールに失敗しました。手動でインストールしてください"
            return 1
        fi
    fi
    
    # PATHの更新
    export PATH="$HOME_DIR/.local/bin:$PATH"
    
    # uvx の確認
    if command -v uvx &> /dev/null; then
        log_success "uvx のインストール完了"
        return 0
    else
        log_warning "uvx のインストールに失敗しました"
        return 1
    fi
}


# インストール状況の表示
show_installation_summary() {
    echo ""
    echo "🎉 統合セットアップが完了しました！"
    echo ""
    log_info "インストール済みコンポーネント:"
    echo "  ✅ Cline拡張機能"
    echo "  ✅ Amazon Q Developer拡張機能"
    echo ""
    
    log_success "code-server を再起動して新しい機能をお試しください！"
}

# メイン実行関数
main() {
    echo "� 統合セットアップスクリプトを開始します..."
    echo ""
    
    # 1. 環境設定
    setup_environment
    
    # 2. 前提条件確認
    check_prerequisites
    
    # 3. 拡張機能インストール
    install_extension
    
    # 4. uvx の確認とインストール
    check_install_uvx
    
    # 5. インストール完了サマリー
    show_installation_summary
}

# スクリプトの実行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
