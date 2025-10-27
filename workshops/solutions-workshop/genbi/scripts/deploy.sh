#!/bin/bash

# 色の定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# スクリプトのディレクトリを取得
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ステップ実行関数
run_step() {
  echo -e "${GREEN}実行: $1${NC}"
  eval $1
  if [ $? -ne 0 ]; then
    echo -e "${RED}エラー: $1 の実行に失敗しました${NC}"
    return 1
  fi
  return 0
}

# 実行権限の付与
chmod +x "$SCRIPT_DIR/verify-env.sh"

echo -e "${GREEN}=== OSS + Amazon Bedrock 統合テスト ===${NC}"
echo "テスト開始時刻: $(date)"
echo "プロジェクトディレクトリ: $PROJECT_DIR"
echo

# ステップ 1: 環境変数検証
echo -e "${GREEN}ステップ 1: 環境変数検証${NC}"
if ! run_step "$SCRIPT_DIR/verify-env.sh"; then
  echo -e "${RED}環境変数検証に失敗しました。テストを中止します。${NC}"
  exit 1
fi
echo

# ステップ 2: Docker 環境起動
echo -e "${GREEN}ステップ 2: Docker 環境起動${NC}"
echo "既存の Docker コンテナを停止中..."
run_step "cd $PROJECT_DIR && docker compose down -v"

echo "Docker コンテナを起動中..."
run_step "cd $PROJECT_DIR && docker compose up -d"

sleep 10

# ステップ 3: 接続テスト
echo -e "${GREEN}ステップ 7: 接続テスト${NC}"
echo "API エンドポイントをテスト中..."
curl -s http://localhost:8000/health > /dev/null
if [ $? -ne 0 ]; then
  echo -e "${RED}API への接続に失敗しました${NC}"
  echo "コンテナのログを確認してください: docker logs wren-ai"
else
  echo -e "${GREEN}API への接続に成功しました${NC}"
fi
echo

CFN_URL="${VSCODE_PROXY_URI%%/proxy/*}"

# テスト結果サマリー
echo -e "${GREEN}=== テスト結果サマリー ===${NC}"
echo "テスト完了時刻: $(date)"
echo -e "${GREEN}統合テストが完了しました${NC}"
echo
echo "UI: https://${CFN_URL}/absproxy/3000/ をオープンしてください。"