#!/bin/bash

# Disk Cleanup Analysis Script
# ディスク使用量分析とクリーンアップスクリプト

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

log_cleanup() {
    echo -e "${PURPLE}[CLEANUP]${NC} $1"
}

# ヘルプ表示
show_help() {
    cat << EOF
🧹 Disk Cleanup Analysis Script

使用方法:
    $0 [options]

オプション:
    -a, --analyze           ディスク使用量分析のみ実行
    -c, --cleanup           クリーンアップを実行
    -f, --force             確認なしでクリーンアップ実行
    -h, --help              このヘルプを表示

機能:
    📊 ディスク使用量分析
    🗑️  一時ファイルのクリーンアップ
    📦 パッケージキャッシュのクリーンアップ
    🔍 大きなファイル・ディレクトリの特定
    📋 クリーンアップ推奨事項の表示

使用例:
    # 分析のみ実行
    $0 -a

    # インタラクティブクリーンアップ
    $0 -c

    # 強制クリーンアップ
    $0 -c -f
EOF
}

# パラメータ解析
parse_args() {
    ANALYZE_ONLY=false
    CLEANUP=false
    FORCE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--analyze)
                ANALYZE_ONLY=true
                shift
                ;;
            -c|--cleanup)
                CLEANUP=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
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

# ディスク使用量の概要表示
show_disk_overview() {
    log_info "📊 ディスク使用量概要"
    echo ""
    
    # 全体のディスク使用量
    df -h / | awk 'NR==2 {printf "  💾 ルートパーティション: %s / %s 使用 (%s)\n", $3, $2, $5}'
    
    # 利用可能容量
    local available=$(df / | awk 'NR==2 {print $4}')
    local available_gb=$((available / 1024 / 1024))
    echo "  🆓 利用可能容量: ${available_gb}GB"
    
    # 警告レベルチェック
    local usage_percent=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $usage_percent -gt 90 ]]; then
        log_error "⚠️  ディスク使用量が90%を超えています！"
    elif [[ $usage_percent -gt 80 ]]; then
        log_warning "⚠️  ディスク使用量が80%を超えています"
    fi
    
    echo ""
}

# 大きなディレクトリの特定
find_large_directories() {
    log_info "📁 大きなディレクトリ (上位10位)"
    echo ""
    
    # ルートディレクトリ以下の大きなディレクトリを検索
    du -h --max-depth=2 / 2>/dev/null | sort -hr | head -10 | while read size dir; do
        echo "  📂 $size - $dir"
    done
    
    echo ""
    
    # /workディレクトリの詳細分析
    if [[ -d "/work" ]]; then
        log_info "📁 /work ディレクトリの詳細"
        echo ""
        du -h --max-depth=2 /work 2>/dev/null | sort -hr | head -10 | while read size dir; do
            echo "  📂 $size - $dir"
        done
        echo ""
    fi
}

# 一時ファイルとキャッシュの確認
check_temp_files() {
    log_info "🗂️  一時ファイルとキャッシュの確認"
    echo ""
    
    # /tmp ディレクトリ
    if [[ -d "/tmp" ]]; then
        local tmp_size=$(du -sh /tmp 2>/dev/null | cut -f1)
        echo "  🗑️  /tmp: $tmp_size"
    fi
    
    # APTキャッシュ
    if [[ -d "/var/cache/apt" ]]; then
        local apt_cache_size=$(du -sh /var/cache/apt 2>/dev/null | cut -f1)
        echo "  📦 APTキャッシュ: $apt_cache_size"
    fi
    
    # Snapキャッシュ
    if [[ -d "/var/lib/snapd" ]]; then
        local snap_size=$(du -sh /var/lib/snapd 2>/dev/null | cut -f1)
        echo "  📦 Snapキャッシュ: $snap_size"
    fi
    
    # ログファイル
    if [[ -d "/var/log" ]]; then
        local log_size=$(du -sh /var/log 2>/dev/null | cut -f1)
        echo "  📋 ログファイル: $log_size"
    fi
    
    # npmキャッシュ
    if [[ -d "$HOME/.npm" ]]; then
        local npm_cache_size=$(du -sh $HOME/.npm 2>/dev/null | cut -f1)
        echo "  📦 npmキャッシュ: $npm_cache_size"
    fi
    
    echo ""
}

# 大きなファイルの検索
find_large_files() {
    log_info "📄 大きなファイル (100MB以上)"
    echo ""
    
    find / -type f -size +100M 2>/dev/null | head -10 | while read file; do
        if [[ -f "$file" ]]; then
            local size=$(du -h "$file" 2>/dev/null | cut -f1)
            echo "  📄 $size - $file"
        fi
    done
    
    echo ""
}

# クリーンアップ推奨事項
show_cleanup_recommendations() {
    log_info "💡 クリーンアップ推奨事項"
    echo ""
    
    echo "  🧹 安全にクリーンアップできる項目:"
    echo "    • APTパッケージキャッシュ (apt clean)"
    echo "    • 一時ファイル (/tmp/*)"
    echo "    • ログファイルの圧縮・削除"
    echo "    • npmキャッシュ (npm cache clean --force)"
    echo ""
    
    echo "  ⚠️  注意が必要な項目:"
    echo "    • Snapパッケージ (使用中の可能性)"
    echo "    • システムログ (デバッグに必要な場合)"
    echo "    • 開発環境のnode_modules"
    echo ""
}

# APTキャッシュクリーンアップ
cleanup_apt_cache() {
    log_cleanup "APTキャッシュをクリーンアップ中..."
    
    local before_size=$(du -sh /var/cache/apt 2>/dev/null | cut -f1)
    
    sudo apt-get clean
    sudo apt-get autoclean
    sudo apt-get autoremove -y
    
    local after_size=$(du -sh /var/cache/apt 2>/dev/null | cut -f1)
    
    log_success "APTキャッシュクリーンアップ完了: $before_size → $after_size"
}

# 一時ファイルクリーンアップ
cleanup_temp_files() {
    log_cleanup "一時ファイルをクリーンアップ中..."
    
    # /tmp の古いファイルを削除（7日以上古い）
    sudo find /tmp -type f -atime +7 -delete 2>/dev/null || true
    
    # ユーザーの一時ファイル
    find $HOME/.cache -type f -atime +30 -delete 2>/dev/null || true
    
    log_success "一時ファイルクリーンアップ完了"
}

# npmキャッシュクリーンアップ
cleanup_npm_cache() {
    if command -v npm &> /dev/null; then
        log_cleanup "npmキャッシュをクリーンアップ中..."
        
        local before_size=$(du -sh $HOME/.npm 2>/dev/null | cut -f1 || echo "0")
        
        npm cache clean --force 2>/dev/null || true
        
        local after_size=$(du -sh $HOME/.npm 2>/dev/null | cut -f1 || echo "0")
        
        log_success "npmキャッシュクリーンアップ完了: $before_size → $after_size"
    fi
}

# ログファイルクリーンアップ
cleanup_logs() {
    log_cleanup "古いログファイルをクリーンアップ中..."
    
    # journalログの削除（30日以上古い）
    sudo journalctl --vacuum-time=30d 2>/dev/null || true
    
    # 古いログファイルの圧縮
    sudo find /var/log -name "*.log" -type f -size +10M -exec gzip {} \; 2>/dev/null || true
    
    log_success "ログファイルクリーンアップ完了"
}

# インタラクティブクリーンアップ
interactive_cleanup() {
    log_cleanup "🧹 インタラクティブクリーンアップを開始します"
    echo ""
    
    # APTキャッシュ
    read -p "APTキャッシュをクリーンアップしますか？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup_apt_cache
    fi
    
    # 一時ファイル
    read -p "一時ファイルをクリーンアップしますか？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup_temp_files
    fi
    
    # npmキャッシュ
    if command -v npm &> /dev/null; then
        read -p "npmキャッシュをクリーンアップしますか？ (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cleanup_npm_cache
        fi
    fi
    
    # ログファイル
    read -p "古いログファイルをクリーンアップしますか？ (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup_logs
    fi
}

# 強制クリーンアップ
force_cleanup() {
    log_cleanup "🧹 強制クリーンアップを実行中..."
    echo ""
    
    cleanup_apt_cache
    cleanup_temp_files
    cleanup_npm_cache
    cleanup_logs
    
    log_success "🎉 強制クリーンアップが完了しました！"
}

# メイン処理
main() {
    echo "🧹 ディスク使用量分析とクリーンアップツール"
    echo ""
    
    parse_args "$@"
    
    # 分析実行
    show_disk_overview
    find_large_directories
    check_temp_files
    find_large_files
    show_cleanup_recommendations
    
    # クリーンアップ実行
    if [[ "$CLEANUP" == true ]]; then
        if [[ "$FORCE" == true ]]; then
            force_cleanup
        else
            interactive_cleanup
        fi
        
        echo ""
        log_info "📊 クリーンアップ後のディスク使用量"
        show_disk_overview
    elif [[ "$ANALYZE_ONLY" == false ]]; then
        echo ""
        log_info "💡 クリーンアップを実行するには -c オプションを使用してください"
        echo "   例: $0 -c"
    fi
}

# スクリプト実行
main "$@"
