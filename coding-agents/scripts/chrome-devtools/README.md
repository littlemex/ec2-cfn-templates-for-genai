# MCP Servers セットアップガイド

このディレクトリには、Clineで使用できるModel Context Protocol (MCP)サーバーのセットアップスクリプトと設定ファイルが含まれています。

## 📁 ディレクトリ構成

```
./
├── README.md                          # このファイル
├── chrome-devtools-mcp/               # Chrome DevTools MCPサーバー
├── install-chrome-devtools-mcp.sh     # Chrome DevTools MCP自動インストールスクリプト
└── chrome-devtools-mcp-config.json    # Chrome DevTools MCP設定サンプル
```

## Chrome DevTools MCP

Puppeteerを使用してChromeブラウザを制御し、DevToolsの機能を提供するMCPサーバー。

**主な機能:**
- **ナビゲーション**: `new_page`, `navigate_page`, `close_page`, `list_pages`
- **操作**: `click`, `fill`, `fill_form`, `hover`, `press_key`, `drag`
- **デバッグ**: `take_snapshot`, `take_screenshot`, `list_console_messages`, `get_console_message`
- **ネットワーク**: `list_network_requests`, `get_network_request`
- **パフォーマンス**: `performance_start_trace`, `performance_stop_trace`, `performance_analyze_insight`
- **エミュレーション**: `emulate`, `resize_page`

**設定例:**
```json
{
  "mcpServers": {
    "chrome-devtools": {
      "type": "stdio",
      "command": "node",
      "args": [
        "/work/mcp-servers/chrome-devtools-mcp/build/src/index.js",
        "--headless=true"
      ],
      "env": {}
    }
  }
}
```

## 📦 Chrome DevTools MCPのインストール

### 自動インストール（推奨）

```bash
cd /work/mcp-servers
bash install-chrome-devtools-mcp.sh
```

このスクリプトは以下を自動的に実行します:
1. Google Chrome Stableのインストール（未インストールの場合）
2. Chrome DevTools MCPのGitHubリポジトリからのクローン
3. 依存関係のインストールとビルド
4. Cline MCP設定ファイルの自動更新（headlessモード有効）

### 手動インストール

1. **Google Chromeのインストール**
   ```bash
   wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
   sudo dpkg -i google-chrome-stable_current_amd64.deb
   sudo apt-get install -f -y
   ```

2. **Chrome DevTools MCPのクローン**
   ```bash
   cd /work/mcp-servers
   git clone https://github.com/ChromeDevTools/chrome-devtools-mcp.git
   cd chrome-devtools-mcp
   ```

3. **依存関係のインストールとビルド**
   ```bash
   npm install
   npm run build
   ```

4. **Cline設定の更新**
   
   `~/.local/share/code-server/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`に以下を追加:
   ```json
   {
     "mcpServers": {
       "chrome-devtools": {
         "type": "stdio",
         "command": "node",
         "args": [
           "/work/mcp-servers/chrome-devtools-mcp/build/src/index.js",
           "--headless=true"
         ],
         "env": {}
       }
     }
   }
   ```

5. **Clineの再起動**
   
   VSCodeウィンドウをリロード（Ctrl+Shift+P → "Developer: Reload Window"）

## 🔧 設定オプション

### Chrome DevTools MCPの主要オプション

| オプション | 説明 | デフォルト |
|----------|------|-----------|
| `--headless` | ヘッドレスモードの有効化 | `false` |
| `--isolated` | 一時的なユーザーデータディレクトリを使用 | `false` |
| `--viewport` | ビューポートサイズ（例: `1280x720`） | - |
| `--executablePath` | カスタムChrome実行ファイルパス | - |
| `--browserUrl` | 実行中のChromeインスタンスに接続 | - |

### 重要な注意事項

**X Serverについて:**
- code-server環境にはX Serverがないため、必ず`--headless=true`を指定してください
- headlessモードなしで起動すると「Missing X server」エラーが発生します

**環境変数方式は使用しない:**
- `PUPPETEER_HEADLESS`環境変数では動作しません
- 必ずコマンドライン引数`--headless=true`を使用してください

## 💡 使用例

### Chrome DevTools MCPでWebページを開く

```
プロンプト例:
"https://google.com を開いてスナップショットを取得してください"
```

これにより、Clineは以下のツールを使用します:
1. `new_page` - ページを開く
2. `take_snapshot` - ページのDOM構造を取得
3. `close_page` - ページを閉じる

### パフォーマンス分析

```
プロンプト例:
"https://example.com のパフォーマンスを分析してください"
```

### フォーム操作

```
プロンプト例:
"ログインフォームに入力してください"
```

## 🐛 トラブルシューティング

### 問題1: "Missing X server" エラー

**原因:** headlessモードが有効になっていない

**解決方法:**
1. `cline_mcp_settings.json`に`--headless=true`が含まれているか確認
2. Clineを再起動

### 問題2: MCPツールが認識されない

**原因:** 設定変更後にClineが再起動されていない

**解決方法:**
1. VSCodeウィンドウをリロード（Ctrl+Shift+P → "Developer: Reload Window"）
2. または、Clineのチャット画面右上のメニューから「Restart Cline」を選択

### 問題3: Chrome DevTools MCPのビルドエラー

**原因:** Node.jsのバージョンが古い、または依存関係の問題

**解決方法:**
```bash
cd /work/mcp-servers/chrome-devtools-mcp
rm -rf node_modules package-lock.json
npm install
npm run build
```

### 問題4: jqコマンドが見つからない

**原因:** jqがインストールされていない

**解決方法:**
```bash
sudo apt-get update
sudo apt-get install -y jq
```

その後、インストールスクリプトを再実行してください。

## 📚 関連ドキュメント

- [Chrome DevTools MCP公式ドキュメント](https://github.com/ChromeDevTools/chrome-devtools-mcp)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)