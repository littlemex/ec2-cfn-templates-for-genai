# Web Search MCP Server 使用ガイド

このドキュメントでは、web-search MCP サーバーの使用方法について説明します。

## 概要

web-search MCP サーバーは、複数の検索エンジン（Bing、Brave、DuckDuckGo）を使用してウェブ検索を行い、ページの完全なコンテンツを抽出する機能を提供する MCP サーバーです。

**注意**: このMCPサーバーのセットアップは `scripts/setup.sh` によって自動的に処理されるため、手動での設定は不要です。

## 利用可能なツール

以下の3つのツールが利用可能です：

1. **`full-web-search`**: 包括的なウェブ検索（完全なページコンテンツ抽出付き）
2. **`get-web-search-summaries`**: 軽量な検索結果（要約のみ）
3. **`get-single-web-page-content`**: 特定のウェブページのコンテンツ抽出

## 使用方法

### 基本的な使用例

#### 1. 包括的なウェブ検索

```python
use_mcp_tool:
  server_name: web-search
  tool_name: full-web-search
  arguments:
    query: "TypeScript MCP server"
    limit: 5
    includeContent: true
```

**パラメータ説明:**
- `query`: 検索クエリ（必須）
- `limit`: 取得する結果数（デフォルト: 10）
- `includeContent`: 完全なページコンテンツを含めるか（デフォルト: false）

#### 2. 軽量な検索結果

```python
use_mcp_tool:
  server_name: web-search
  tool_name: get-web-search-summaries
  arguments:
    query: "TypeScript MCP server"
    limit: 5
```

**パラメータ説明:**
- `query`: 検索クエリ（必須）
- `limit`: 取得する結果数（デフォルト: 10）

#### 3. 特定ページのコンテンツ抽出

```python
use_mcp_tool:
  server_name: web-search
  tool_name: get-single-web-page-content
  arguments:
    url: "https://example.com/article"
    maxContentLength: 5000
```

**パラメータ説明:**
- `url`: 読み込むURL（必須）
- `maxContentLength`: 最大コンテンツ長（デフォルト: 10000文字）

### 実用的な使用例

#### プログラミング関連の検索

```python
use_mcp_tool:
  server_name: web-search
  tool_name: get-web-search-summaries
  arguments:
    query: "Python async await best practices"
    limit: 3
```

#### 技術文書の検索

```python
use_mcp_tool:
  server_name: web-search
  tool_name: full-web-search
  arguments:
    query: "AWS Lambda cold start optimization"
    limit: 5
    includeContent: true
```

#### 特定サイトからの情報抽出

```python
# まず検索で関連ページを見つける
use_mcp_tool:
  server_name: web-search
  tool_name: get-web-search-summaries
  arguments:
    query: "site:docs.aws.amazon.com Lambda layers"
    limit: 3

# 次に特定のページの詳細を取得
use_mcp_tool:
  server_name: web-search
  tool_name: get-single-web-page-content
  arguments:
    url: "https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html"
    maxContentLength: 8000
```

## 検索のコツ

### 効果的な検索クエリの作成

1. **具体的なキーワードを使用**
   ```
   "React hooks useEffect cleanup" 
   ```

2. **サイト指定検索**
   ```
   "site:stackoverflow.com TypeScript generic constraints"
   ```

3. **除外検索**
   ```
   "JavaScript promises -jQuery"
   ```

4. **フレーズ検索**
   ```
   "exact phrase search"
   ```

### 結果の最適化

- `limit` パラメータで結果数を調整（1-20推奨）
- 完全なコンテンツが必要な場合は `full-web-search` を使用
- 特定のページの詳細が必要な場合は `get-single-web-page-content` を使用

## トラブルシューティング

### よくある問題

#### 1. 検索結果が取得できない
- ネットワーク接続を確認
- クエリを簡潔にして再試行

#### 2. ページコンテンツが読み込めない
- URLが正しいか確認
- `maxContentLength` 値を調整して再試行

#### 3. 検索結果の品質が低い
- より具体的なキーワードを使用
- サイト指定検索を活用
- 除外キーワードを追加

## 制限事項

- 検索結果は各検索エンジン（Bing、Brave、DuckDuckGo）の制限に依存
- 一部のサイトはコンテンツ抽出をブロックする場合がある
- 大量のリクエストは制限される可能性がある

## サポート

問題が発生した場合は、以下を確認してください：

1. ネットワーク接続
2. 検索クエリの妥当性
3. MCP サーバーの起動状態

設定に関する問題は `scripts/setup.sh` の実行ログを確認してください。
