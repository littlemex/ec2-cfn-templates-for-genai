#!/bin/bash

# Chrome DevTools MCP Demo Script
# Chrome DevTools MCPサーバーのデモンストレーション

set -e

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

log_demo() {
    echo -e "${PURPLE}[DEMO]${NC} $1"
}

# ヘルプ表示
show_help() {
    cat << EOF
🌐 Chrome DevTools MCP Demo Script

使用方法:
    $0 [options]

オプション:
    -d, --demo DEMO_NAME    実行するデモ (all, screenshot, performance, console, network)
    -u, --url URL           テスト対象のURL (デフォルト: https://example.com)
    -h, --help              このヘルプを表示

利用可能なデモ:
    screenshot              ウェブサイトのスクリーンショット撮影
    performance             ページパフォーマンス分析
    console                 JavaScriptコンソールログ取得
    network                 ネットワーク通信の監視
    all                     全てのデモを実行

使用例:
    # 全デモ実行
    $0

    # スクリーンショットのみ
    $0 -d screenshot

    # カスタムURL
    $0 -u https://github.com

    # 特定デモ + カスタムURL
    $0 -d performance -u https://www.google.com
EOF
}

# パラメータ解析
parse_args() {
    DEMO_TYPE="all"
    TARGET_URL="https://example.com"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--demo)
                DEMO_TYPE="$2"
                shift 2
                ;;
            -u|--url)
                TARGET_URL="$2"
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

# 前提条件チェック
check_prerequisites() {
    log_info "前提条件をチェック中..."

    # Claude Code CLI確認
    if ! command -v claude &> /dev/null; then
        log_error "Claude Code CLIがインストールされていません"
        log_info "bedrock-claude-setup.sh を実行してください"
        exit 1
    fi

    # Chrome/Chromium確認
    if ! command -v google-chrome &> /dev/null && ! command -v chromium-browser &> /dev/null; then
        log_error "Chrome/Chromiumがインストールされていません"
        log_info "bedrock-claude-setup.sh を実行してください"
        exit 1
    fi

    # 環境変数確認
    if [[ -z "$CLAUDE_CODE_USE_BEDROCK" ]]; then
        log_warning "CLAUDE_CODE_USE_BEDROCK環境変数が設定されていません"
        log_info "source ~/.bashrc を実行してください"
    fi

    log_success "前提条件チェック完了"
}

# スクリーンショットデモ
demo_screenshot() {
    log_demo "📸 スクリーンショットデモを開始..."
    
    local prompt="以下のURLのスクリーンショットを撮影してください: $TARGET_URL
    
スクリーンショットを撮影した後、以下の情報を教えてください:
- ページのタイトル
- 主要な要素の配置
- 色合いやデザインの特徴
- 気づいた点があれば"

    echo "実行するプロンプト:"
    echo "----------------------------------------"
    echo "$prompt"
    echo "----------------------------------------"
    echo ""
    
    read -p "このデモを実行しますか？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        claude -p "$prompt"
    else
        log_info "スクリーンショットデモをスキップしました"
    fi
}

# パフォーマンスデモ
demo_performance() {
    log_demo "⚡ パフォーマンス分析デモを開始..."
    
    local prompt="以下のURLのページパフォーマンスを分析してください: $TARGET_URL

以下の項目について詳細に分析してください:
- ページの読み込み時間
- リソースの読み込み状況
- Core Web Vitals (LCP, FID, CLS)
- パフォーマンスのボトルネック
- 改善提案があれば教えてください"

    echo "実行するプロンプト:"
    echo "----------------------------------------"
    echo "$prompt"
    echo "----------------------------------------"
    echo ""
    
    read -p "このデモを実行しますか？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        claude -p "$prompt"
    else
        log_info "パフォーマンス分析デモをスキップしました"
    fi
}

# コンソールログデモ
demo_console() {
    log_demo "🖥️ コンソールログデモを開始..."
    
    local prompt="以下のURLにアクセスして、JavaScriptコンソールのログを確認してください: $TARGET_URL

以下の情報を取得してください:
- エラーメッセージがあるか
- 警告メッセージの内容
- 一般的なログメッセージ
- パフォーマンスに関する情報
- セキュリティに関する警告があるか"

    echo "実行するプロンプト:"
    echo "----------------------------------------"
    echo "$prompt"
    echo "----------------------------------------"
    echo ""
    
    read -p "このデモを実行しますか？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        claude -p "$prompt"
    else
        log_info "コンソールログデモをスキップしました"
    fi
}

# ネットワーク監視デモ
demo_network() {
    log_demo "🌐 ネットワーク監視デモを開始..."
    
    local prompt="以下のURLにアクセスして、ネットワーク通信を監視してください: $TARGET_URL

以下の情報を分析してください:
- 読み込まれるリソースの一覧
- 各リソースの読み込み時間
- 失敗したリクエストがあるか
- 外部ドメインへのリクエスト
- 大きなファイルサイズのリソース
- CDNの使用状況"

    echo "実行するプロンプト:"
    echo "----------------------------------------"
    echo "$prompt"
    echo "----------------------------------------"
    echo ""
    
    read -p "このデモを実行しますか？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        claude -p "$prompt"
    else
        log_info "ネットワーク監視デモをスキップしました"
    fi
}

# 全デモ実行
run_all_demos() {
    log_demo "🚀 全デモを順次実行します..."
    echo ""
    
    demo_screenshot
    echo ""
    
    demo_performance
    echo ""
    
    demo_console
    echo ""
    
    demo_network
    echo ""
    
    log_success "全デモが完了しました！"
}

# デモ実行
run_demo() {
    case $DEMO_TYPE in
        "screenshot")
            demo_screenshot
            ;;
        "performance")
            demo_performance
            ;;
        "console")
            demo_console
            ;;
        "network")
            demo_network
            ;;
        "all")
            run_all_demos
            ;;
        *)
            log_error "不明なデモタイプ: $DEMO_TYPE"
            show_help
            exit 1
            ;;
    esac
}

# 使用方法の説明
show_usage_tips() {
    echo ""
    log_info "💡 Chrome DevTools MCP の使用方法:"
    echo ""
    echo "  インタラクティブセッション:"
    echo "    claude"
    echo "    > ウェブサイトのスクリーンショットを撮って"
    echo "    > ページのパフォーマンスを分析して"
    echo "    > JavaScriptエラーをチェックして"
    echo ""
    echo "  ワンライナー:"
    echo "    claude -p \"https://example.com のスクリーンショットを撮影して\""
    echo "    claude -p \"https://github.com のパフォーマンスを分析して\""
    echo ""
    echo "  高度な使用例:"
    echo "    claude -p \"複数のページを比較してパフォーマンスの違いを分析して\""
    echo "    claude -p \"モバイル表示でのスクリーンショットを撮影して\""
    echo "    claude -p \"ページの読み込み時間を最適化する提案をして\""
    echo ""
}

# メイン処理
main() {
    echo "🌐 Chrome DevTools MCP デモを開始します..."
    echo ""
    
    parse_args "$@"
    
    log_info "設定情報:"
    echo "  • デモタイプ: $DEMO_TYPE"
    echo "  • 対象URL: $TARGET_URL"
    echo ""
    
    check_prerequisites
    run_demo
    show_usage_tips
}

# スクリプト実行
main "$@"
