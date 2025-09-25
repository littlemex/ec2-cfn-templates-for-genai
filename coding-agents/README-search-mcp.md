# Web Search MCP Server インストールガイド

このドキュメントでは、web-search MCP サーバーのインストール方法と使用方法について説明します。

## 概要

web-search MCP サーバーは、複数の検索エンジン（Bing、Brave、DuckDuckGo）を使用してウェブ検索を行い、ページの完全なコンテンツを抽出する機能を提供する MCP サーバーです。

## 利用可能なツール

インストール後、以下の3つのツールが利用可能になります：

1. **`full-web-search`**: 包括的なウェブ検索（完全なページコンテンツ抽出付き）
2. **`get-web-search-summaries`**: 軽量な検索結果（要約のみ）
3. **`get-single-web-page-content`**: 特定のウェブページのコンテンツ抽出

## インストール方法

### 自動インストール（推奨）

Pythonスクリプトを使用した自動インストール：

```bash
# 基本インストール
python3 /work/install-search-mcp.py

# システム依存関係をスキップしてインストール
python3 /work/install-search-mcp.py --skip-system-deps

# 前回の状態から再開
python3 /work/install-search-mcp.py --resume

# インストール状態をクリアして最初から開始
python3 /work/install-search-mcp.py --clean
```

### インストールオプション

- `--skip-system-deps`: システム依存関係（libgbm1, libasound2）のインストールをスキップ
- `--resume`: 前回の状態から再開（中断されたインストールを継続）
- `--clean`: インストール状態をクリアして最初から開始

### 重要な注意事項

⚠️ **システム依存関係のインストールについて**

システム依存関係（`libgbm1`、`libasound2`）のインストール中に、システムの再起動が発生する可能性があります。

再起動が発生した場合は、再起動後に以下のコマンドでインストールを継続してください：

```bash
python3 /work/install-web-search-mcp.py --skip-system-deps
```

## 手動インストール手順

自動インストールが利用できない場合の手動手順：

### 1. 前提条件の確認

```bash
# Node.js と npm のバージョン確認
node --version  # v18.0.0 以上が必要
npm --version   # v8.0.0 以上が必要
```

### 2. リポジトリのクローン

```bash
mkdir -p /work/mcp-servers
cd /work/mcp-servers
git clone https://github.com/mrkrsl/web-search-mcp.git
cd web-search-mcp
```

### 3. 依存関係のインストール

```bash
# Node.js 依存関係のインストール
npm install

# Playwright ブラウザのインストール
npx playwright install

# システム依存関係のインストール（オプション）
sudo apt-get update
sudo apt-get install -y libgbm1 libasound2
```

### 4. プロジェクトのビルド

```bash
npm run build
```

### 5. MCP 設定ファイルの更新

設定ファイル（`/home/coder/.local/share/code-server/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`）に以下を追加：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "node",
      "args": ["/work/mcp-servers/web-search-mcp/dist/index.js"],
      "env": {
        "MAX_CONTENT_LENGTH": "10000",
        "BROWSER_HEADLESS": "true",
        "MAX_BROWSERS": "3",
        "BROWSER_FALLBACK_THRESHOLD": "3"
      }
    }
  }
}
```

## 使用方法

### 基本的な使用例

#### 1. 軽量検索（要約のみ）

```python
use_mcp_tool:
  server_name: web-search
  tool_name: get-web-search-summaries
  arguments:
    query: "TypeScript MCP server"
    limit: 5
```

#### 2. 包括的検索（完全なコンテンツ抽出）

```python
use_mcp_tool:
  server_name: web-search
  tool_name: full-web-search
  arguments:
    query: "MCP Model Context Protocol"
    limit: 3
    includeContent: true
```

#### 3. 特定ページのコンテンツ抽出

```python
use_mcp_tool:
  server_name: web-search
  tool_name: get-single-web-page-content
  arguments:
    url: "https://example.com/article"
    maxContentLength: 5000
```

## 環境変数

以下の環境変数で動作を調整できます：

- `MAX_CONTENT_LENGTH`: コンテンツの最大長（デフォルト: 10000 文字）
- `BROWSER_HEADLESS`: ヘッドレスモード（デフォルト: true）
- `MAX_BROWSERS`: 最大ブラウザインスタンス数（デフォルト: 3）
- `BROWSER_FALLBACK_THRESHOLD`: フォールバック閾値（デフォルト: 3）
- `DEFAULT_TIMEOUT`: リクエストタイムアウト（デフォルト: 6000ms）
- `ENABLE_RELEVANCE_CHECKING`: 検索結果品質チェック（デフォルト: true）

## トラブルシューティング

### よくある問題と解決方法

#### 1. Node.js/npm が見つからない

```bash
# Node.js のインストール（Ubuntu/Debian）
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### 2. Playwright ブラウザの問題

```bash
# ブラウザの再インストール
cd /work/mcp-servers/web-search-mcp
npx playwright install
```

#### 3. システム依存関係の問題

```bash
# 手動でシステム依存関係をインストール
sudo apt-get update
sudo apt-get install -y libgbm1 libasound2
```

#### 4. 権限エラー

```bash
# スクリプトに実行権限を付与
chmod +x /work/install-search-mcp.py
```

### ログの確認

インストール中に問題が発生した場合は、以下を確認してください：

1. Node.js と npm のバージョン
2. ネットワーク接続
3. ディスク容量
4. 権限設定

## ファイル構成

インストール後のファイル構成：

```
/work/
├── install-search-mcp.py          # インストールスクリプト
├── README-web-search-mcp.md           # このファイル
├── mcp-servers/
│   └── web-search-mcp/
│       ├── dist/
│       │   └── index.js               # ビルド済み MCP サーバー
│       ├── src/                       # ソースコード
│       ├── package.json               # Node.js 設定
│       └── ...
└── .web-search-mcp-install-state.json # インストール状態（一時的）
```

## サポート

問題が発生した場合は、以下を確認してください：

1. [公式リポジトリ](https://github.com/mrkrsl/web-search-mcp)
2. インストールログ
3. MCP サーバーの起動ログ

## ライセンス

この MCP サーバーは MIT ライセンスの下で提供されています。
