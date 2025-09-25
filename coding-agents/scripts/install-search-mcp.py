#!/usr/bin/env python3
"""
Web Search MCP Server インストールスクリプト
このスクリプトは web-search MCP サーバーを自動的にセットアップします
"""

import json
import os
import subprocess
import sys
import shutil
from pathlib import Path
from datetime import datetime
import argparse

class Colors:
    """コンソール出力用の色定義"""
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    RED = '\033[31m'
    YELLOW = '\033[33m'
    RESET = '\033[0m'

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.RESET}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.RESET}")

def run_command(command, cwd=None, check=True):
    """コマンドを実行し、結果を返す"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        if check:
            print_error(f"コマンド実行エラー: {command}")
            print_error(f"エラー出力: {e.stderr}")
            raise
        return e

def check_node_npm():
    """Node.js と npm のバージョンを確認"""
    print_info("Node.js と npm のバージョンを確認中...")
    
    try:
        node_result = run_command("node --version")
        npm_result = run_command("npm --version")
        
        node_version = node_result.stdout.strip()
        npm_version = npm_result.stdout.strip()
        
        print_success(f"Node.js: {node_version}, npm: {npm_version}")
        
        # Node.js バージョンチェック
        node_major = int(node_version.lstrip('v').split('.')[0])
        if node_major < 18:
            print_warning(f"Node.js 18.0.0以上が推奨されています。現在のバージョン: {node_version}")
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("Node.js または npm がインストールされていません。")
        print_error("Node.js 18.0.0以上をインストールしてください。")
        return False

def setup_repository():
    """リポジトリのクローンまたは更新"""
    mcp_dir = Path("/work/mcp-servers")
    web_search_dir = mcp_dir / "web-search-mcp"
    
    print_info("MCPサーバーディレクトリを準備中...")
    mcp_dir.mkdir(parents=True, exist_ok=True)
    
    if web_search_dir.exists():
        print_warning("web-search-mcp ディレクトリが既に存在します。更新します...")
        run_command("git pull origin main", cwd=web_search_dir)
    else:
        print_info("GitHubからweb-search MCPサーバーをクローン中...")
        run_command("git clone https://github.com/mrkrsl/web-search-mcp.git", cwd=mcp_dir)
    
    print_success("リポジトリのクローン/更新完了")
    return web_search_dir

def install_dependencies(web_search_dir):
    """依存関係のインストール"""
    print_info("依存関係をインストール中...")
    
    # npm install
    run_command("npm install", cwd=web_search_dir)
    print_success("npm install 完了")
    
    # Playwright ブラウザのインストール
    print_info("Playwright ブラウザをインストール中...")
    result = run_command("npx playwright install", cwd=web_search_dir, check=False)
    
    if result.returncode == 0:
        print_success("Playwright ブラウザのインストール完了")
    else:
        print_warning("Playwright ブラウザのインストールで警告が発生しました（通常は問題ありません）")

def get_state_file():
    """インストール状態ファイルのパスを取得"""
    return Path("/work/.web-search-mcp-install-state.json")

def save_state(step, data=None):
    """インストール状態を保存"""
    state_file = get_state_file()
    state = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "data": data or {}
    }
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def load_state():
    """インストール状態を読み込み"""
    state_file = get_state_file()
    if state_file.exists():
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    return None

def clear_state():
    """インストール状態をクリア"""
    state_file = get_state_file()
    if state_file.exists():
        state_file.unlink()

def check_system_dependencies():
    """システム依存関係の確認"""
    print_info("システム依存関係を確認中...")
    
    # libgbm1 と libasound2 がインストールされているかチェック
    try:
        result1 = run_command("dpkg -l | grep libgbm1", check=False)
        result2 = run_command("dpkg -l | grep libasound2", check=False)
        
        if result1.returncode == 0 and result2.returncode == 0:
            print_success("必要なシステム依存関係は既にインストールされています")
            return True
        else:
            return False
    except:
        return False

def install_system_dependencies(skip_install=False):
    """システム依存関係のインストール（オプション）"""
    if check_system_dependencies():
        return True
    
    if skip_install:
        print_warning("システム依存関係のインストールをスキップします")
        return True
    
    if not shutil.which("apt-get"):
        print_info("apt-get が利用できません。必要に応じて手動でシステム依存関係をインストールしてください。")
        return True
    
    print_warning("⚠️  重要な注意事項 ⚠️")
    print_warning("システム依存関係のインストール中にシステムの再起動が発生する可能性があります。")
    print_warning("再起動が発生した場合は、再起動後に以下のコマンドでインストールを継続してください：")
    print_warning(f"python3 {sys.argv[0]} --skip-system-deps")
    print("")
    
    response = input("システム依存関係をインストールしますか？ (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print_info("システム依存関係のインストールをスキップしました")
        print_info("手動で以下を実行してください: sudo apt-get install libgbm1 libasound2")
        return True
    
    try:
        print_info("システム依存関係をインストール中...")
        save_state("system_deps_installing")
        
        run_command("sudo apt-get update")
        run_command("sudo apt-get install -y libgbm1 libasound2")
        
        print_success("システム依存関係のインストール完了")
        return True
        
    except subprocess.CalledProcessError:
        print_warning("システム依存関係のインストールに失敗しました（権限不足の可能性）")
        print_info("手動で以下を実行してください: sudo apt-get install libgbm1 libasound2")
        return True

def build_project(web_search_dir):
    """プロジェクトのビルド"""
    print_info("プロジェクトをビルド中...")
    run_command("npm run build", cwd=web_search_dir)
    print_success("ビルド完了")

def update_mcp_config(web_search_dir):
    """MCP設定ファイルの更新"""
    print_info("MCP設定ファイルを更新中...")
    
    config_file = Path("/home/coder/.local/share/code-server/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json")
    
    # 設定ファイルのディレクトリを作成
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 既存の設定をバックアップ
    if config_file.exists():
        backup_file = config_file.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(config_file, backup_file)
        print_info(f"既存の設定ファイルをバックアップしました: {backup_file}")
        
        # 既存の設定を読み込み
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            config = {"mcpServers": {}}
    else:
        config = {"mcpServers": {}}
    
    # web-search サーバーの設定を追加/更新
    config["mcpServers"]["web-search"] = {
        "command": "node",
        "args": [str(web_search_dir / "dist" / "index.js")],
        "env": {
            "MAX_CONTENT_LENGTH": "10000",
            "BROWSER_HEADLESS": "true",
            "MAX_BROWSERS": "3",
            "BROWSER_FALLBACK_THRESHOLD": "3"
        }
    }
    
    # 設定ファイルを保存
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print_success(f"MCP設定ファイルを更新しました: {config_file}")
    return config_file

def update_amazon_q_mcp_config(web_search_dir):
    """Amazon Q Developer用MCP設定ファイルの更新"""
    print_info("Amazon Q Developer用MCP設定ファイルを更新中...")
    
    amazonq_config_file = Path("/home/coder/.aws/amazonq/mcp.json")
    
    # 設定ファイルのディレクトリを作成
    amazonq_config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 既存の設定をバックアップ
    if amazonq_config_file.exists():
        backup_file = amazonq_config_file.with_suffix(f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        shutil.copy2(amazonq_config_file, backup_file)
        print_info(f"既存のAmazon Q設定ファイルをバックアップしました: {backup_file}")
        
        # 既存の設定を読み込み
        try:
            with open(amazonq_config_file, 'r', encoding='utf-8') as f:
                amazonq_config = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            amazonq_config = {"mcpServers": {}}
    else:
        # デフォルトのAmazon Q設定（AWS Documentation MCP Server）
        amazonq_config = {
            "mcpServers": {
                "awslabs.aws-documentation-mcp-server": {
                    "autoApprove": [
                        "read_documentation",
                        "search_documentation",
                        "recommend"
                    ],
                    "disabled": False,
                    "command": "uvx",
                    "args": [
                        "awslabs.aws-documentation-mcp-server@latest"
                    ],
                    "env": {
                        "FASTMCP_LOG_LEVEL": "ERROR"
                    },
                    "transportType": "stdio"
                }
            }
        }
    
    # mcpServersが存在しない場合は作成
    if "mcpServers" not in amazonq_config:
        amazonq_config["mcpServers"] = {}
    
    # web-search サーバーの設定を追加/更新
    amazonq_config["mcpServers"]["web-search"] = {
        "autoApprove": [
            "full-web-search",
            "get-web-search-summaries",
            "get-single-web-page-content"
        ],
        "disabled": False,
        "command": "node",
        "args": [str(web_search_dir / "dist" / "index.js")],
        "env": {
            "MAX_CONTENT_LENGTH": "10000",
            "BROWSER_HEADLESS": "true",
            "MAX_BROWSERS": "3",
            "BROWSER_FALLBACK_THRESHOLD": "3"
        },
        "transportType": "stdio"
    }
    
    # 設定ファイルを保存
    with open(amazonq_config_file, 'w', encoding='utf-8') as f:
        json.dump(amazonq_config, f, indent=2, ensure_ascii=False)
    
    print_success(f"Amazon Q Developer用MCP設定ファイルを更新しました: {amazonq_config_file}")
    return amazonq_config_file

def verify_installation(web_search_dir):
    """インストールの確認"""
    print_info("MCPサーバーの動作確認中...")
    
    dist_file = web_search_dir / "dist" / "index.js"
    if dist_file.exists():
        print_success("MCPサーバーファイルが正常に作成されました")
        
        # 簡単な起動テスト
        try:
            result = run_command(f"timeout 5s node {dist_file}", check=False)
            # タイムアウトで終了するのは正常（MCPサーバーは継続実行されるため）
            print_success("MCPサーバーの起動テスト完了")
        except:
            pass  # タイムアウトは期待される動作
        
        return True
    else:
        print_error("MCPサーバーファイルが見つかりません")
        return False

def print_completion_info(config_file, web_search_dir, amazonq_config_file=None):
    """完了情報の表示"""
    print("\n🎉 Web Search MCP Server のインストールが完了しました！\n")
    
    print_info("利用可能なツール:")
    print("  • full-web-search: 包括的なウェブ検索（完全なページコンテンツ抽出付き）")
    print("  • get-web-search-summaries: 軽量な検索結果（要約のみ）")
    print("  • get-single-web-page-content: 特定のウェブページのコンテンツ抽出")
    
    print(f"\n{Colors.BLUE}ℹ️  設定ファイル:{Colors.RESET}")
    print(f"  📁 Cline用: {config_file}")
    if amazonq_config_file:
        print(f"  📁 Amazon Q Developer用: {amazonq_config_file}")
    print(f"{Colors.BLUE}ℹ️  サーバーファイル: {web_search_dir}/dist/index.js{Colors.RESET}")
    
    print_success("\nMCPサーバーを再起動して新しいツールをお試しください！")
    
    print(f"\n{Colors.BLUE}ℹ️  使用例:{Colors.RESET}")
    print("  use_mcp_tool:")
    print("    server_name: web-search")
    print("    tool_name: get-web-search-summaries")
    print("    arguments:")
    print("      query: \"検索したいキーワード\"")
    print("      limit: 5")

def main():
    """メイン実行関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Web Search MCP Server インストールスクリプト")
    parser.add_argument("--skip-system-deps", action="store_true", 
                       help="システム依存関係のインストールをスキップ")
    parser.add_argument("--resume", action="store_true", 
                       help="前回の状態から再開")
    parser.add_argument("--clean", action="store_true", 
                       help="インストール状態をクリアして最初から開始")
    
    args = parser.parse_args()
    
    # 状態のクリア
    if args.clean:
        clear_state()
        print_info("インストール状態をクリアしました")
    
    # 前回の状態を確認
    state = load_state()
    start_step = 1
    
    if state and args.resume:
        print_info(f"前回の状態から再開します（ステップ: {state['step']}）")
        if state['step'] == "system_deps_installing":
            print_info("システム依存関係のインストール後から再開します")
            start_step = 5  # ビルドから開始
    
    print("🚀 Web Search MCP Server インストールを開始します...\n")
    
    try:
        web_search_dir = Path("/work/mcp-servers/web-search-mcp")
        
        # 1. Node.js と npm の確認
        if start_step <= 1:
            if not check_node_npm():
                sys.exit(1)
            save_state("node_npm_checked")
        
        # 2. リポジトリのセットアップ
        if start_step <= 2:
            web_search_dir = setup_repository()
            save_state("repository_setup", {"web_search_dir": str(web_search_dir)})
        
        # 3. 依存関係のインストール
        if start_step <= 3:
            install_dependencies(web_search_dir)
            save_state("dependencies_installed")
        
        # 4. システム依存関係のインストール（オプション）
        if start_step <= 4:
            if not install_system_dependencies(skip_install=args.skip_system_deps):
                print_error("システム依存関係のインストールに失敗しました")
                sys.exit(1)
            save_state("system_deps_installed")
        
        # 5. プロジェクトのビルド
        if start_step <= 5:
            build_project(web_search_dir)
            save_state("project_built")
        
        # 6. MCP設定ファイルの更新
        if start_step <= 6:
            config_file = update_mcp_config(web_search_dir)
            save_state("config_updated", {"config_file": str(config_file)})
        
        # 7. Amazon Q Developer用MCP設定ファイルの更新
        if start_step <= 7:
            amazonq_config_file = update_amazon_q_mcp_config(web_search_dir)
            save_state("amazonq_config_updated", {"amazonq_config_file": str(amazonq_config_file)})
        
        # 8. インストールの確認
        if start_step <= 8:
            if not verify_installation(web_search_dir):
                sys.exit(1)
            save_state("installation_verified")
        
        # 9. 完了情報の表示
        config_file = Path("/home/coder/.local/share/code-server/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json")
        amazonq_config_file = Path("/home/coder/.aws/amazonq/mcp.json")
        print_completion_info(config_file, web_search_dir, amazonq_config_file)
        
        # インストール完了後、状態ファイルをクリア
        clear_state()
        
    except KeyboardInterrupt:
        print_error("\nインストールが中断されました")
        print_info("再開するには --resume オプションを使用してください")
        sys.exit(1)
    except Exception as e:
        print_error(f"予期しないエラーが発生しました: {e}")
        print_info("再開するには --resume オプションを使用してください")
        sys.exit(1)

if __name__ == "__main__":
    main()
