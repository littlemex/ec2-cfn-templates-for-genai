# VS Code Server + Claude Code (CLI) + Bedrock 統合環境

このディレクトリには、AWS Bedrock上のClaude モデルと連携するVS Code Server環境とClaude Code CLIツールを構築するためのスクリプトとテンプレートが含まれています。

## 🚀 新機能

### 改良されたcfn_manager.sh
- **設定ファイル対応**: `config.json`で設定を管理
- **g6e.2xlarge対応**: GPU最適化インスタンス
- **200GBボリューム**: 大容量ストレージ
- **Bedrock権限**: Claude Code用のIAM権限を自動設定

### bedrock-claude-setup.sh
- **Claude Code CLI**: npmパッケージの自動インストール
- **Bedrock統合**: AWS Bedrock経由でClaude APIを利用
- **Chrome DevTools MCP**: ブラウザ自動化とウェブ分析機能
- **環境変数設定**: 正しいBedrock設定の自動構成
- **接続テスト**: セットアップ後の動作確認

### chrome-devtools-demo.sh
- **スクリーンショット撮影**: ウェブページの画面キャプチャ
- **パフォーマンス分析**: ページ読み込み時間、Core Web Vitals測定
- **コンソールログ取得**: JavaScriptエラーや警告の確認
- **ネットワーク監視**: リソース読み込み状況の分析

## 📋 前提条件

- AWS CLI v2がインストール済み
- AWS認証情報が設定済み
- jqがインストール済み（推奨）
- 適切なIAM権限（EC2、CloudFormation、Bedrock）

## 🛠️ セットアップ手順

### 1. 設定ファイルの確認・編集

```bash
# config.jsonを確認
cat config.json
```

```json
{
  "instance": {
    "type": "g6e.2xlarge",
    "volumeSize": 200,
    "operatingSystem": "Ubuntu-22"
  },
  "codeServer": {
    "user": "coder",
    "homeFolder": "/work"
  },
  "bedrock": {
    "region": "us-east-1",
    "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "enabled": true
  },
  "aws": {
    "defaultRegion": "us-east-1",
    "s3BucketPrefix": "vscode-cfn-templates"
  },
  "mcp": {
    "chromeDevTools": {
      "enabled": true,
      "browserPath": "/usr/bin/chromium-browser",
      "headless": true,
      "defaultTimeout": 30000
    }
  }
}
```

### 2. VS Code Serverスタックの作成

```bash
# 基本的な作成（config.jsonの設定を使用）
./cfn_manager.sh create

# カスタム設定で作成
./cfn_manager.sh create -n my-claude-env -t g6e.xlarge -r us-west-2

# 進捗監視
./cfn_manager.sh monitor -n my-claude-env
```

### 3. Claude Code + Bedrockのセットアップ

スタック作成完了後、EC2インスタンスに接続してセットアップを実行：

```bash
# EC2インスタンスに接続
./cfn_manager.sh connect -n my-claude-env

# インスタンス内でClaude Codeセットアップを実行
sudo ./bedrock-claude-setup.sh

# カスタム設定でセットアップ
sudo ./bedrock-claude-setup.sh -r us-west-2 -m anthropic.claude-3-haiku-20240307-v1:0

# Chrome DevTools MCP デモの実行
./chrome-devtools-demo.sh
```

### 4. VS Code Serverへのアクセス

```bash
# ブラウザでVS Code Serverを開く
./cfn_manager.sh open -n my-claude-env

# 接続情報を確認
./cfn_manager.sh outputs -n my-claude-env
```

## 🔧 利用可能なコマンド

### cfn_manager.sh

```bash
# スタック管理
./cfn_manager.sh create          # スタック作成
./cfn_manager.sh status          # 状態確認
./cfn_manager.sh monitor         # 進捗監視
./cfn_manager.sh delete          # スタック削除

# 接続・アクセス
./cfn_manager.sh connect         # SSM接続
./cfn_manager.sh open           # ブラウザでオープン
./cfn_manager.sh outputs        # 出力値表示

# テンプレート管理
./cfn_manager.sh validate       # テンプレート検証
./cfn_manager.sh upload         # S3アップロード
./cfn_manager.sh list          # スタック一覧
```

### bedrock-claude-setup.sh

```bash
# 基本セットアップ
./bedrock-claude-setup.sh

# カスタム設定
./bedrock-claude-setup.sh -r us-west-2 -m anthropic.claude-3-haiku-20240307-v1:0 -u developer

# ヘルプ表示
./bedrock-claude-setup.sh --help
```

### chrome-devtools-demo.sh

```bash
# 全デモ実行
./chrome-devtools-demo.sh

# 特定デモ実行
./chrome-devtools-demo.sh -d screenshot
./chrome-devtools-demo.sh -d performance

# カスタムURL
./chrome-devtools-demo.sh -u https://github.com

# ヘルプ表示
./chrome-devtools-demo.sh --help
```

## 📊 インスタンスタイプとコスト

### 推奨インスタンスタイプ

| インスタンス | vCPU | メモリ | GPU | 用途 | 時間単価（概算） |
|-------------|------|--------|-----|------|-----------------|
| g6e.xlarge  | 4    | 16GB   | 1   | 軽量開発 | ~$0.67 |
| g6e.2xlarge | 8    | 32GB   | 1   | 標準開発 | ~$1.34 |
| g6e.4xlarge | 16   | 64GB   | 1   | 重い処理 | ~$2.68 |

### Bedrockコスト

| モデル | 入力トークン | 出力トークン |
|--------|-------------|-------------|
| Claude 3.5 Sonnet | $3.00/1M | $15.00/1M |
| Claude 3 Haiku | $0.25/1M | $1.25/1M |

## 🔍 トラブルシューティング

### よくある問題

1. **スタック作成失敗**
   ```bash
   # エラーログを確認
   ./cfn_manager.sh logs -n my-claude-env
   
   # テンプレート検証
   ./cfn_manager.sh validate
   ```

2. **Claude Code拡張機能が動作しない**
   ```bash
   # VS Code Serverを再起動
   sudo systemctl restart code-server
   
   # 設定ファイルを確認
   cat ~/.local/share/code-server/User/settings.json
   ```

3. **Bedrock接続エラー**
   ```bash
   # AWS認証情報を確認
   aws sts get-caller-identity
   
   # Bedrockモデル一覧を確認
   aws bedrock list-foundation-models --region us-east-1
   ```

4. **権限エラー**
   ```bash
   # IAM権限を確認
   aws iam get-role --role-name CodeServerInstanceBootstrapRole
   ```

5. **Chrome DevTools MCP エラー**
   ```bash
   # Chromiumのインストール確認
   which chromium-browser
   
   # MCP設定確認
   cat ~/.config/claude/mcp_servers.json
   
   # 手動でMCPサーバー追加
   claude mcp add chrome-devtools npx chrome-devtools-mcp@latest
   ```

### ログファイル

- CloudFormation: AWS Console > CloudFormation > Events
- VS Code Server: `/var/log/code-server.log`
- Claude Code: VS Code Server > Output > Claude Code
- Chrome DevTools MCP: `~/.config/claude/logs/`

## 🔐 セキュリティ考慮事項

### IAM権限の最小化

現在のテンプレートは`AdministratorAccess`を使用していますが、本番環境では以下の権限に制限することを推奨：

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:ListFoundationModels",
        "bedrock:GetFoundationModel"
      ],
      "Resource": "*"
    }
  ]
}
```

### ネットワークセキュリティ

- CloudFront経由でのみアクセス可能
- SSM Session Manager経由でのSSH接続
- セキュリティグループで必要最小限のポート開放

## 📚 参考資料

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Claude Code Extension](https://marketplace.visualstudio.com/items?itemName=Anthropic.claude-dev)
- [VS Code Server](https://github.com/coder/code-server)
- [AWS CloudFormation](https://docs.aws.amazon.com/cloudformation/)

## 🤝 サポート

問題が発生した場合は、以下の情報を含めてお問い合わせください：

1. 実行したコマンド
2. エラーメッセージ
3. AWS リージョン
4. インスタンスタイプ
5. ログファイルの内容

---

**注意**: このセットアップはGPUインスタンスを使用するため、コストが発生します。使用後は必ずスタックを削除してください。

```bash
./cfn_manager.sh delete -n my-claude-env
