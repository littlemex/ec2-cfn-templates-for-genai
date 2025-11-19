#!/bin/bash

# Chrome DevTools MCP インストールスクリプト

set -e

echo "==================================="
echo "Chrome DevTools MCP セットアップ"
echo "==================================="
echo ""

# Google Chromeのインストール確認
echo "Step 1: Google Chromeのインストール確認..."
if command -v google-chrome &> /dev/null; then
    echo "✅ Google Chromeは既にインストールされています"
    google-chrome --version
else
    echo "⚠️  Google Chromeがインストールされていません。インストールを開始します..."
    
    # 依存関係のインストール
    echo "依存関係をインストールしています..."
    sudo apt-get update
    sudo apt-get install -y wget gnupg
    
    # Google Chromeのダウンロードとインストール
    echo "Google Chrome Stableをダウンロードしています..."
    wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
    
    echo "Google Chromeをインストールしています..."
    sudo dpkg -i google-chrome-stable_current_amd64.deb || true
    
    # 不足している依存関係を修正
    sudo apt-get install -f -y
    
    # ダウンロードしたdebファイルを削除
    rm -f google-chrome-stable_current_amd64.deb
    
    echo "✅ Google Chromeのインストールが完了しました"
    google-chrome --version
fi

echo ""
echo "Step 2: Chrome DevTools MCPのインストール..."

# mcp-serversディレクトリに移動
cd /work/mcp-servers

# 既存のディレクトリがあれば削除
if [ -d "chrome-devtools-mcp" ]; then
    echo "既存のchrome-devtools-mcpディレクトリを削除しています..."
    rm -rf chrome-devtools-mcp
fi

# Chrome DevTools MCPをクローン
echo "GitHubからクローンしています..."
git clone https://github.com/ChromeDevTools/chrome-devtools-mcp.git

# ディレクトリに移動
cd chrome-devtools-mcp

# 依存関係をインストール
echo "依存関係をインストールしています..."
npm install

# ビルド
echo "ビルドしています..."
npm run build

echo ""
echo "Step 3: Cline MCP設定の更新..."

# Cline設定ファイルのパス
CLINE_CONFIG="$HOME/.local/share/code-server/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"
CONFIG_DIR=$(dirname "$CLINE_CONFIG")

# 設定ディレクトリが存在しない場合は作成
if [ ! -d "$CONFIG_DIR" ]; then
    echo "設定ディレクトリを作成しています..."
    mkdir -p "$CONFIG_DIR"
fi

# 設定ファイルが存在しない場合は新規作成
if [ ! -f "$CLINE_CONFIG" ]; then
    echo "新しい設定ファイルを作成しています..."
    cat > "$CLINE_CONFIG" << 'EOF'
{
  "mcpServers": {}
}
EOF
fi

# jqを使用してchrome-devtoolsの設定を追加/更新
echo "MCP設定にchrome-devtoolsを追加しています..."
if command -v jq &> /dev/null; then
    # jqが利用可能な場合
    TMP_FILE=$(mktemp)
    jq '.mcpServers["chrome-devtools"] = {
      "type": "stdio",
      "command": "node",
      "args": ["/work/mcp-servers/chrome-devtools-mcp/build/src/index.js", "--headless=true"],
      "env": {}
    }' "$CLINE_CONFIG" > "$TMP_FILE"
    mv "$TMP_FILE" "$CLINE_CONFIG"
    echo "✅ MCP設定を更新しました（headlessモード有効）"
else
    echo "⚠️  jqがインストールされていません。手動で設定ファイルを更新してください。"
    echo "設定内容は chrome-devtools-mcp-config.json を参照してください"
fi

echo ""
echo "==================================="
echo "✅ セットアップ完了！"
echo "==================================="
echo ""
echo "インストールされたコンポーネント:"
echo "  - Google Chrome: $(google-chrome --version)"
echo "  - Chrome DevTools MCP: /work/mcp-servers/chrome-devtools-mcp"
echo "  - Cline MCP設定: $CLINE_CONFIG"
echo ""
echo "次のステップ:"
echo "1. Clineを再起動してMCPサーバーを認識させる"
echo "2. MCPツールが利用可能になります"
echo ""
echo "詳細は CHROME_DEVTOOLS_MCP_SETUP.md を参照してください"
