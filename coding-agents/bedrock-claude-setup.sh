#!/bin/bash

# Claude Code (CLI) + Bedrock Setup Script
# Claude Code CLIツールをBedrockと連携させるセットアップスクリプト

set -e

# 設定ファイル
CONFIG_FILE="config.json"

# 色付きメッセージ
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_claude() {
    echo -e "${PURPLE}[CLAUDE]${NC} $1"
}

# ヘルプ表示
show_help() {
    cat << EOF
🤖 Claude Code (CLI) + Bedrock Setup Script

使用方法:
    $0 [options]

オプション:
    -r, --region REGION     AWSリージョン (デフォルト: us-east-1)
    -u, --user USER         ユーザー名 (デフォルト: coder)
    -h, --help              このヘルプを表示

機能:
    📦 Node.js環境の確認
    🔧 Claude Code CLIのインストール
    ⚙️  Bedrock設定の自動構成
    🔑 AWS認証情報の確認
    🌍 環境変数の設定
    🧪 接続テスト

使用例:
    # 基本セットアップ
    $0

    # カスタムリージョン
    $0 -r us-west-2

    # 特定ユーザー向けセットアップ
    $0 -u developer
EOF
}

# パラメータ解析
parse_args() {
    REGION="us-east-1"
    USER="coder"
    # 記事に基づく正しいモデルID
    SONNET_MODEL="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    HAIKU_MODEL="us.anthropic.claude-3-5-haiku-20241022-v1:0"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -u|--user)
                USER="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "不明なオプション: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 設定ファイル読み込み
load_config() {
    if [[ -f "$CONFIG_FILE" ]]; then
        log_info "設定ファイルを読み込み中: $CONFIG_FILE"
        
        if command -v jq &> /dev/null; then
            local config_region=$(jq -r '.bedrock.region // empty' "$CONFIG_FILE" 2>/dev/null)
            local config_model=$(jq -r '.bedrock.modelId // empty' "$CONFIG_FILE" 2>/dev/null)
            local config_user=$(jq -r '.codeServer.user // empty' "$CONFIG_FILE" 2>/dev/null)
            
            if [[ -n "$config_region" && "$REGION" == "us-east-1" ]]; then
                REGION="$config_region"
            fi
            
            if [[ -n "$config_model" && "$MODEL_ID" == "anthropic.claude-3-5-sonnet-20241022-v2:0" ]]; then
                MODEL_ID="$config_model"
            fi
            
            if [[ -n "$config_user" && "$USER" == "coder" ]]; then
                USER="$config_user"
                VSCODE_SETTINGS_DIR="/home/$USER/.local/share/code-server/User"
                VSCODE_EXTENSIONS_DIR="/home/$USER/.local/share/code-server/extensions"
            fi
            
            log_success "設定ファイルを読み込みました"
        else
            log_warning "jqがインストールされていません。設定ファイルをスキップします"
        fi
    fi
}

# AWS CLI確認
check_aws_cli() {
    log_info "AWS CLI設定を確認中..."
    
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLIがインストールされていません"
        exit 1
    fi

    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS認証が設定されていません"
        exit 1
    fi

    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local current_region=$(aws configure get region || echo "未設定")
    
    log_success "AWS認証確認完了"
    log_info "アカウントID: $account_id"
    log_info "現在のリージョン: $current_region"
    log_info "使用するリージョン: $REGION"
}

# Node.js環境確認
check_nodejs() {
    log_info "Node.js環境を確認中..."
    
    if ! command -v node &> /dev/null; then
        log_error "Node.jsがインストールされていません"
        log_info "Node.jsをインストール中..."
        
        # NodeSourceリポジトリを追加してNode.js 18をインストール
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
        
        if ! command -v node &> /dev/null; then
            log_error "Node.jsのインストールに失敗しました"
            exit 1
        fi
    fi
    
    if ! command -v npm &> /dev/null; then
        log_error "npmがインストールされていません"
        exit 1
    fi
    
    local node_version=$(node --version)
    local npm_version=$(npm --version)
    
    log_success "Node.js環境確認完了"
    log_info "Node.js バージョン: $node_version"
    log_info "npm バージョン: $npm_version"
}

# Bedrock利用可能性確認
check_bedrock_availability() {
    log_info "Bedrock利用可能性を確認中..."
    
    # Claude 3.7 Sonnet の確認
    if aws bedrock list-foundation-models --region "$REGION" --query "modelSummaries[?modelId=='$SONNET_MODEL']" --output text &> /dev/null; then
        log_success "Claude 3.7 Sonnet が利用可能です: $SONNET_MODEL"
    else
        log_warning "Claude 3.7 Sonnet が利用できません: $SONNET_MODEL"
    fi
    
    # Claude 3.5 Haiku の確認
    if aws bedrock list-foundation-models --region "$REGION" --query "modelSummaries[?modelId=='$HAIKU_MODEL']" --output text &> /dev/null; then
        log_success "Claude 3.5 Haiku が利用可能です: $HAIKU_MODEL"
    else
        log_warning "Claude 3.5 Haiku が利用できません: $HAIKU_MODEL"
    fi
    
    # 利用可能なClaude モデルを表示
    log_info "利用可能なClaude モデルを確認中..."
    local available_models=$(aws bedrock list-foundation-models --region "$REGION" --query "modelSummaries[?contains(modelId, 'claude')].modelId" --output text 2>/dev/null || echo "")
    
    if [[ -n "$available_models" ]]; then
        log_info "利用可能なClaude モデル:"
        echo "$available_models" | tr '\t' '\n' | sed 's/^/  - /'
    else
        log_warning "Claudeモデルが見つかりません。リージョンを確認してください"
    fi
}

# Claude Code CLIインストール
install_claude_code_cli() {
    log_claude "Claude Code CLIをインストール中..."
    
    # グローバルインストール（sudo権限が必要）
    if npm install -g @anthropic-ai/claude-code; then
        log_success "Claude Code CLIのインストールが完了しました"
        
        # バージョン確認
        local claude_version=$(claude --version 2>/dev/null || echo "バージョン取得失敗")
        log_info "Claude Code バージョン: $claude_version"
    else
        log_error "Claude Code CLIのインストールに失敗しました"
        return 1
    fi
}

# Chrome/Chromiumのインストール
install_chrome_browser() {
    log_info "Chrome/Chromiumのインストール確認中..."
    
    if ! command -v google-chrome &> /dev/null && ! command -v chromium-browser &> /dev/null; then
        log_info "Chrome/Chromiumをインストール中..."
        
        # ディスク容量チェック
        sudo apt-get update
        sudo apt-get install -y chromium-browser
        
        if command -v chromium-browser &> /dev/null; then
            log_success "Chromiumのインストールが完了しました"
        else
            log_warning "Chromiumのインストールに失敗しました"
        fi
    else
        log_success "Chrome/Chromiumが既にインストールされています"
    fi
}

# Chrome DevTools MCP サーバーセットアップ（環境変数設定後に実行）
setup_chrome_devtools_mcp() {
    log_info "Chrome DevTools MCP サーバーをセットアップ中..."
    
    # 環境変数を現在のセッションに読み込み
    source "/home/$USER/.bashrc"
    
    # Chrome DevTools MCP サーバーを追加
    if claude mcp add chrome-devtools npx chrome-devtools-mcp@latest; then
        log_success "Chrome DevTools MCP サーバーが追加されました"
        
        # MCP設定確認
        local mcp_config_file="/home/$USER/.config/claude/mcp_servers.json"
        if [[ -f "$mcp_config_file" ]]; then
            log_info "MCP設定ファイル: $mcp_config_file"
            if command -v jq &> /dev/null; then
                local chrome_devtools_config=$(jq '.["chrome-devtools"]' "$mcp_config_file" 2>/dev/null)
                if [[ "$chrome_devtools_config" != "null" ]]; then
                    log_success "Chrome DevTools MCP設定が確認されました"
                else
                    log_warning "Chrome DevTools MCP設定が見つかりません"
                fi
            fi
        fi
        
        # MCP サーバーリストを表示
        log_info "設定されたMCPサーバー:"
        claude mcp list || log_warning "MCPサーバーリストの取得に失敗しました"
        
    else
        log_warning "Chrome DevTools MCP サーバーの追加に失敗しました"
        log_info "環境変数を確認してください:"
        echo "  CLAUDE_CODE_USE_BEDROCK=$CLAUDE_CODE_USE_BEDROCK"
        echo "  ANTHROPIC_MODEL=$ANTHROPIC_MODEL"
        log_info "手動で追加してください:"
        echo "  source ~/.bashrc"
        echo "  claude mcp add chrome-devtools npx chrome-devtools-mcp@latest"
    fi
}

# 環境変数設定
setup_environment_variables() {
    log_info "Claude Code用環境変数を設定中..."
    
    local bashrc_file="/home/$USER/.bashrc"
    
    # 既存の設定を削除（重複回避）
    sed -i '/# Claude Code + Bedrock設定/,/^$/d' "$bashrc_file" 2>/dev/null || true
    
    # 記事に基づく正しい環境変数設定
    local env_vars="
# Claude Code + Bedrock設定
export AWS_REGION=$REGION
export CLAUDE_CODE_USE_BEDROCK=1
export ANTHROPIC_MODEL='$SONNET_MODEL'
export ANTHROPIC_SMALL_FAST_MODEL='$HAIKU_MODEL'
export DISABLE_PROMPT_CACHING=1
"

    # 新しい設定を追加
    echo "$env_vars" >> "$bashrc_file"
    
    chown "$USER:$USER" "$bashrc_file"
    log_success "環境変数を設定しました"
    
    # 現在のセッションにも適用
    export AWS_REGION="$REGION"
    export CLAUDE_CODE_USE_BEDROCK=1
    export ANTHROPIC_MODEL="$SONNET_MODEL"
    export ANTHROPIC_SMALL_FAST_MODEL="$HAIKU_MODEL"
    export DISABLE_PROMPT_CACHING=1
}

# 接続テスト
test_bedrock_connection() {
    log_info "Bedrock接続テストを実行中..."
    
    # Claude 3.7 Sonnet でテスト
    local test_body=$(cat << EOF
{
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 100,
    "messages": [
        {
            "role": "user",
            "content": "Hello, this is a test message."
        }
    ]
}
EOF
)

    if aws bedrock-runtime invoke-model \
        --region "$REGION" \
        --model-id "$SONNET_MODEL" \
        --body "$test_body" \
        --cli-binary-format raw-in-base64-out \
        /tmp/bedrock_test_response.json &> /dev/null; then
        
        log_success "Bedrock接続テスト成功 (Claude 3.7 Sonnet)"
        
        # レスポンスの一部を表示
        if command -v jq &> /dev/null && [[ -f /tmp/bedrock_test_response.json ]]; then
            local response_text=$(jq -r '.content[0].text // "レスポンス解析エラー"' /tmp/bedrock_test_response.json 2>/dev/null)
            log_info "テストレスポンス: ${response_text:0:100}..."
        fi
        
        rm -f /tmp/bedrock_test_response.json
    else
        log_error "Bedrock接続テスト失敗"
        log_warning "IAM権限またはモデルアクセス権限を確認してください"
        return 1
    fi
}

# Claude Code使用方法の説明
show_claude_code_usage() {
    log_claude "Claude Code の使用方法:"
    echo ""
    echo "  基本的な使用方法:"
    echo "    claude                    # インタラクティブセッション開始"
    echo "    claude -p \"質問内容\"      # 1回限りの実行"
    echo "    echo \"データ\" | claude -p \"処理内容\"  # パイプ処理"
    echo ""
    echo "  スラッシュコマンド:"
    echo "    /cost                     # 使用コストを確認"
    echo "    /exit                     # セッション終了"
    echo "    /init                     # CLAUDE.mdファイル生成"
    echo ""
    echo "  Chrome DevTools MCP 使用例:"
    echo "    claude -p \"ウェブサイトのスクリーンショットを撮って\""
    echo "    claude -p \"ページのパフォーマンスを分析して\""
    echo "    claude -p \"JavaScriptエラーをチェックして\""
    echo "    claude -p \"ページの読み込み時間を測定して\""
    echo ""
    echo "  環境変数確認:"
    echo "    echo \$CLAUDE_CODE_USE_BEDROCK"
    echo "    echo \$ANTHROPIC_MODEL"
    echo ""
}

# セットアップ完了メッセージ
show_completion_message() {
    echo ""
    log_success "🎉 Claude Code (CLI) + Bedrock + Chrome DevTools MCP セットアップ完了!"
    echo ""
    log_claude "📋 設定情報:"
    echo "   • リージョン: $REGION"
    echo "   • Sonnet モデル: $SONNET_MODEL"
    echo "   • Haiku モデル: $HAIKU_MODEL"
    echo "   • ユーザー: $USER"
    echo "   • Chrome DevTools MCP: 有効"
    echo ""
    log_info "💡 次のステップ:"
    echo "   1. 新しいターミナルセッションを開始するか、以下を実行:"
    echo "      source ~/.bashrc"
    echo "   2. Claude Code を試してみてください:"
    echo "      claude --version"
    echo "      claude -p \"Hello, Claude!\""
    echo "   3. Chrome DevTools MCP デモを実行:"
    echo "      ./chrome-devtools-demo.sh"
    echo ""
    show_claude_code_usage
    log_info "🔧 トラブルシューティング:"
    echo "   • 認証エラーの場合: AWS認証情報を確認"
    echo "   • モデルエラーの場合: Bedrockモデルアクセス権限を確認"
    echo "   • 環境変数が反映されない場合: 新しいターミナルを開く"
    echo "   • Chrome DevTools MCP エラーの場合: Chromiumが正しくインストールされているか確認"
    echo ""
}

# メイン処理
main() {
    echo "🤖 Claude Code (CLI) + Bedrock + Chrome DevTools MCP セットアップを開始します..."
    echo ""
    
    parse_args "$@"
    load_config
    check_aws_cli
    check_nodejs
    check_bedrock_availability
    install_claude_code_cli
    install_chrome_browser
    setup_environment_variables
    setup_chrome_devtools_mcp
    test_bedrock_connection
    show_completion_message
}

# スクリプト実行
main "$@"
