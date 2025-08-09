#!/bin/bash
# VS Code Workshop Stack Manager
# CloudFormation + Cognito + VS Code Server の全操作を一つのスクリプトで実行

set -e

# デフォルト設定
DEFAULT_REGION="us-east-1"
DEFAULT_INSTANCE_TYPE="c7i.4xlarge"
TEMPLATE_FILE="ec2-cf-vscode.yml"

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

log_cognito() {
    echo -e "${PURPLE}[COGNITO]${NC} $1"
}

log_vscode() {
    echo -e "${CYAN}[VSCODE]${NC} $1"
}

# ヘルプ表示
show_help() {
    cat << EOF
🚀 VS Code Workshop Stack Manager

使用方法:
    $0 <command> [options]

コマンド:
    create      - ワークショップスタックを作成
    status      - スタック状態を確認
    monitor     - スタック作成/削除の進捗を監視
    outputs     - スタック出力値を表示
    login       - ログイン情報を表示
    open        - VS Code Serverをブラウザでオープン
    cognito     - Cognito詳細情報を表示
    fix-oauth   - OAuth設定を修正
    direct-login - 直接CognitoログインURLを表示
    logs        - CloudFormationイベントログを表示
    delete      - スタックを削除
    list        - 全スタック一覧
    validate    - テンプレート検証

オプション:
    -n, --name NAME         スタック名 (デフォルト: vscode-workshop-USERNAME)
    -r, --region REGION     AWSリージョン (デフォルト: $DEFAULT_REGION)
    -t, --type TYPE         インスタンスタイプ (デフォルト: $DEFAULT_INSTANCE_TYPE)
    -e, --email EMAIL       管理者メールアドレス (必須)
    -p, --password PASS     管理者パスワード (必須)
    -h, --help              このヘルプを表示

インスタンスタイプ:
    • c7i.large, c7i.xlarge, c7i.2xlarge, c7i.4xlarge
    • m5.large, m5.xlarge, m5.2xlarge, m5.4xlarge
    • t3.medium, t3.large, t3.xlarge

使用例:
    # 基本的な作成
    $0 create -e admin@example.com -p MyPassword123

    # カスタム設定で作成
    $0 create -n my-workshop -e admin@example.com -p MyPassword123 -t c7i.2xlarge

    # 進捗監視
    $0 monitor -n my-workshop

    # ログイン情報確認
    $0 login -n my-workshop

    # ブラウザでオープン
    $0 open -n my-workshop

    # スタック削除
    $0 delete -n my-workshop

認証について:
    🔐 Cognito User Poolが自動作成されます
    👤 指定したメール/パスワードで管理者ユーザーが作成されます
    🌐 CloudFront経由でアクセス可能になります

EOF
}

# パラメータ解析
parse_args() {
    COMMAND=""
    STACK_NAME=""
    REGION="$DEFAULT_REGION"
    INSTANCE_TYPE="$DEFAULT_INSTANCE_TYPE"
    ADMIN_EMAIL=""
    ADMIN_PASSWORD=""
    USER_NAME=$(whoami)

    while [[ $# -gt 0 ]]; do
        case $1 in
            create|status|monitor|outputs|login|open|cognito|fix-oauth|direct-login|logs|delete|list|validate)
                COMMAND="$1"
                shift
                ;;
            -n|--name)
                STACK_NAME="$2"
                shift 2
                ;;
            -r|--region)
                REGION="$2"
                shift 2
                ;;
            -t|--type)
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            -e|--email)
                ADMIN_EMAIL="$2"
                shift 2
                ;;
            -p|--password)
                ADMIN_PASSWORD="$2"
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

    # デフォルトスタック名
    if [[ -z "$STACK_NAME" ]]; then
        STACK_NAME="vscode-workshop-$USER_NAME"
    fi

    if [[ -z "$COMMAND" ]]; then
        log_error "コマンドが指定されていません"
        show_help
        exit 1
    fi
}

# AWS CLI確認
check_aws_cli() {
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLIがインストールされていません"
        exit 1
    fi

    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS認証が設定されていません"
        exit 1
    fi
}

# テンプレートファイル確認
check_template() {
    if [[ ! -f "$TEMPLATE_FILE" ]]; then
        log_error "テンプレートファイルが見つかりません: $TEMPLATE_FILE"
        exit 1
    fi
}

# メールアドレス検証
validate_email() {
    local email="$1"
    if [[ ! "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        log_error "無効なメールアドレス形式: $email"
        return 1
    fi
    return 0
}

# パスワード検証
validate_password() {
    local password="$1"
    if [[ ${#password} -lt 8 ]]; then
        log_error "パスワードは8文字以上である必要があります"
        return 1
    fi
    if [[ ! "$password" =~ [A-Z] ]]; then
        log_error "パスワードに大文字が含まれている必要があります"
        return 1
    fi
    if [[ ! "$password" =~ [a-z] ]]; then
        log_error "パスワードに小文字が含まれている必要があります"
        return 1
    fi
    if [[ ! "$password" =~ [0-9] ]]; then
        log_error "パスワードに数字が含まれている必要があります"
        return 1
    fi
    return 0
}

# スタック作成
create_stack() {
    if [[ -z "$ADMIN_EMAIL" ]]; then
        log_error "管理者メールアドレスが指定されていません (-e オプション)"
        exit 1
    fi

    if [[ -z "$ADMIN_PASSWORD" ]]; then
        log_error "管理者パスワードが指定されていません (-p オプション)"
        exit 1
    fi

    # 入力検証
    if ! validate_email "$ADMIN_EMAIL"; then
        exit 1
    fi

    if ! validate_password "$ADMIN_PASSWORD"; then
        exit 1
    fi

    log_info "🚀 VS Code Workshopスタックを作成中..."
    log_info "スタック名: $STACK_NAME"
    log_info "リージョン: $REGION"
    log_info "インスタンスタイプ: $INSTANCE_TYPE"
    log_cognito "管理者メール: $ADMIN_EMAIL"
    log_info "パスワード: [HIDDEN]"

    check_template

    aws cloudformation create-stack \
        --stack-name "$STACK_NAME" \
        --template-body "file://$TEMPLATE_FILE" \
        --parameters \
            "ParameterKey=AdminEmail,ParameterValue=$ADMIN_EMAIL" \
            "ParameterKey=AdminPassword,ParameterValue=$ADMIN_PASSWORD" \
            "ParameterKey=InstanceType,ParameterValue=$INSTANCE_TYPE" \
        --capabilities CAPABILITY_IAM \
        --region "$REGION"

    log_success "スタック作成を開始しました"
    log_info "📊 進捗を監視するには: $0 monitor -n $STACK_NAME -r $REGION"
    log_info "⏱️  作成完了まで約5-10分かかります"
}

# スタック状態確認
check_status() {
    local status
    status=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].StackStatus' \
        --output text 2>/dev/null || echo "NOT_FOUND")

    if [[ "$status" == "NOT_FOUND" ]]; then
        log_error "スタックが見つかりません: $STACK_NAME"
        return 1
    fi

    log_info "📊 スタック状態: $status"

    case "$status" in
        CREATE_COMPLETE)
            log_success "✅ スタック作成完了"
            show_quick_info
            ;;
        CREATE_IN_PROGRESS)
            log_info "🔄 スタック作成中..."
            show_creation_progress
            ;;
        CREATE_FAILED)
            log_error "❌ スタック作成失敗"
            show_errors
            ;;
        DELETE_IN_PROGRESS)
            log_info "🗑️  スタック削除中..."
            ;;
        DELETE_COMPLETE)
            log_success "✅ スタック削除完了"
            ;;
        *)
            log_warning "⚠️  スタック状態: $status"
            ;;
    esac

    return 0
}

# 作成進捗表示
show_creation_progress() {
    log_info "📈 作成進捗の詳細:"
    aws cloudformation describe-stack-events \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'StackEvents[?ResourceStatus==`CREATE_IN_PROGRESS`] | [0:5].[Timestamp,LogicalResourceId,ResourceStatus]' \
        --output table 2>/dev/null || log_warning "進捗情報を取得できませんでした"
}

# 進捗監視
monitor_stack() {
    log_info "📊 スタック進捗を監視中: $STACK_NAME"
    log_info "Ctrl+C で監視を終了"
    
    local start_time=$(date +%s)
    
    while true; do
        local status
        status=$(aws cloudformation describe-stacks \
            --stack-name "$STACK_NAME" \
            --region "$REGION" \
            --query 'Stacks[0].StackStatus' \
            --output text 2>/dev/null || echo "NOT_FOUND")

        if [[ "$status" == "NOT_FOUND" ]]; then
            log_error "スタックが見つかりません"
            break
        fi

        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        local elapsed_min=$((elapsed / 60))
        local elapsed_sec=$((elapsed % 60))

        echo -ne "\r$(date '+%H:%M:%S') - Status: $status (経過時間: ${elapsed_min}m${elapsed_sec}s)"

        case "$status" in
            CREATE_COMPLETE)
                echo ""
                log_success "🎉 スタック作成完了!"
                show_quick_info
                break
                ;;
            DELETE_COMPLETE)
                echo ""
                log_success "✅ スタック削除完了"
                break
                ;;
            CREATE_FAILED|DELETE_FAILED|ROLLBACK_COMPLETE)
                echo ""
                log_error "❌ 操作失敗: $status"
                show_errors
                break
                ;;
        esac

        sleep 5
    done
}

# クイック情報表示
show_quick_info() {
    echo ""
    log_success "🎯 ワークショップ準備完了!"
    echo ""
    
    local workshop_url
    workshop_url=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`WorkshopURL`].OutputValue' \
        --output text 2>/dev/null)

    if [[ -n "$workshop_url" && "$workshop_url" != "None" ]]; then
        log_vscode "🌐 VS Code Server URL:"
        echo "   $workshop_url"
        echo ""
    fi

    log_info "📋 次のステップ:"
    echo "   1. $0 login -n $STACK_NAME     # ログイン情報を確認"
    echo "   2. $0 open -n $STACK_NAME      # ブラウザでオープン"
    echo "   3. 上記URLにアクセスしてログイン"
    echo ""
}

# 出力値表示
show_outputs() {
    log_info "📋 スタック出力値:"
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table 2>/dev/null || log_warning "出力値を取得できませんでした"
}

# ログイン情報表示
show_login_info() {
    log_cognito "🔑 ログイン情報:"
    
    local login_info
    login_info=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`LoginCredentials`].OutputValue' \
        --output text 2>/dev/null)

    if [[ -n "$login_info" && "$login_info" != "None" ]]; then
        echo "$login_info"
    else
        log_warning "ログイン情報を取得できませんでした"
    fi

    echo ""
    
    local workshop_url
    workshop_url=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`WorkshopURL`].OutputValue' \
        --output text 2>/dev/null)

    if [[ -n "$workshop_url" && "$workshop_url" != "None" ]]; then
        log_vscode "🌐 アクセスURL:"
        echo "   $workshop_url"
    fi
    
    echo ""
    local cognito_login_url
    cognito_login_url=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`CognitoLoginURL`].OutputValue' \
        --output text 2>/dev/null)
    
    if [[ -n "$cognito_login_url" && "$cognito_login_url" != "None" ]]; then
        log_cognito "🔗 直接CognitoログインURL:"
        echo "   $cognito_login_url"
    fi
    
    echo ""
    log_info "💡 ブラウザでオープンするには: $0 open -n $STACK_NAME"
}

# ブラウザでオープン
open_browser() {
    local workshop_url
    workshop_url=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`WorkshopURL`].OutputValue' \
        --output text 2>/dev/null)

    if [[ -z "$workshop_url" || "$workshop_url" == "None" ]]; then
        log_error "Workshop URLを取得できませんでした"
        return 1
    fi

    log_vscode "🌐 VS Code Serverをブラウザでオープン中..."
    log_info "URL: $workshop_url"

    # OS判定してブラウザオープン
    case "$(uname -s)" in
        Darwin)
            open "$workshop_url"
            ;;
        Linux)
            if command -v xdg-open > /dev/null; then
                xdg-open "$workshop_url"
            else
                log_warning "ブラウザを自動オープンできません。手動で以下のURLにアクセスしてください:"
                echo "$workshop_url"
            fi
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
            start "$workshop_url"
            ;;
        *)
            log_warning "ブラウザを自動オープンできません。手動で以下のURLにアクセスしてください:"
            echo "$workshop_url"
            ;;
    esac
}

# Cognito詳細情報表示
show_cognito_details() {
    log_cognito "🔐 Cognito詳細情報:"
    
    local cognito_details
    cognito_details=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`CognitoDetails`].OutputValue' \
        --output text 2>/dev/null)

    if [[ -n "$cognito_details" && "$cognito_details" != "None" ]]; then
        echo "$cognito_details"
        
        # OAuth設定を確認
        check_oauth_settings
    else
        log_warning "Cognito詳細情報を取得できませんでした"
    fi
}

# OAuth設定確認と修正
check_oauth_settings() {
    log_cognito "🔍 OAuth設定を確認中..."
    
    # User Pool IDとClient IDを取得
    local user_pool_id client_id
    user_pool_id=$(aws cloudformation describe-stack-resources \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --logical-resource-id CognitoUserPool \
        --query 'StackResources[0].PhysicalResourceId' \
        --output text 2>/dev/null)
    
    client_id=$(aws cloudformation describe-stack-resources \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --logical-resource-id CognitoUserPoolClient \
        --query 'StackResources[0].PhysicalResourceId' \
        --output text 2>/dev/null)
    
    if [[ -z "$user_pool_id" || -z "$client_id" ]]; then
        log_warning "CognitoリソースIDを取得できませんでした"
        return 1
    fi
    
    # OAuth設定を確認
    local oauth_flows oauth_scopes
    oauth_flows=$(aws cognito-idp describe-user-pool-client \
        --user-pool-id "$user_pool_id" \
        --client-id "$client_id" \
        --region "$REGION" \
        --query 'UserPoolClient.AllowedOAuthFlows' \
        --output text 2>/dev/null)
    
    oauth_scopes=$(aws cognito-idp describe-user-pool-client \
        --user-pool-id "$user_pool_id" \
        --client-id "$client_id" \
        --region "$REGION" \
        --query 'UserPoolClient.AllowedOAuthScopes' \
        --output text 2>/dev/null)
    
    if [[ "$oauth_flows" == "None" || "$oauth_scopes" == "None" ]]; then
        log_warning "⚠️  OAuth設定が不完全です"
        echo "   OAuth Flows: $oauth_flows"
        echo "   OAuth Scopes: $oauth_scopes"
        
        read -p "🔧 OAuth設定を修正しますか？ (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            fix_oauth_settings "$user_pool_id" "$client_id"
        fi
    else
        log_success "✅ OAuth設定は正常です"
        echo "   OAuth Flows: $oauth_flows"
        echo "   OAuth Scopes: $oauth_scopes"
    fi
}

# OAuth設定修正
fix_oauth_settings() {
    local user_pool_id="$1"
    local client_id="$2"
    
    log_cognito "🔧 OAuth設定を修正中..."
    
    # CloudFrontドメインを取得
    local cloudfront_domain
    cloudfront_domain=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`Configuration`].OutputValue' \
        --output text | grep "CloudFront Domain" | cut -d: -f2 | xargs)
    
    if [[ -z "$cloudfront_domain" ]]; then
        log_error "CloudFrontドメインを取得できませんでした"
        return 1
    fi
    
    # OAuth設定を更新
    if aws cognito-idp update-user-pool-client \
        --user-pool-id "$user_pool_id" \
        --client-id "$client_id" \
        --region "$REGION" \
        --callback-urls "https://$cloudfront_domain/oauth/callback" \
        --logout-urls "https://$cloudfront_domain/" \
        --allowed-o-auth-flows "code" \
        --allowed-o-auth-scopes "openid" "email" "profile" \
        --allowed-o-auth-flows-user-pool-client \
        --supported-identity-providers "COGNITO" >/dev/null 2>&1; then
        
        log_success "✅ OAuth設定を修正しました"
        echo "   Callback URL: https://$cloudfront_domain/oauth/callback"
        echo "   OAuth Flows: code"
        echo "   OAuth Scopes: openid, email, profile"
    else
        log_error "❌ OAuth設定の修正に失敗しました"
        return 1
    fi
}

# 直接CognitoログインURL表示
show_direct_login() {
    log_cognito "🔗 直接CognitoログインURL:"
    
    local cognito_login_url
    cognito_login_url=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`CognitoLoginURL`].OutputValue' \
        --output text 2>/dev/null)
    
    if [[ -n "$cognito_login_url" && "$cognito_login_url" != "None" ]]; then
        echo "$cognito_login_url"
        echo ""
        log_info "💡 このURLをブラウザで開いてログインしてください"
    else
        log_warning "CognitoログインURLを取得できませんでした"
    fi
}

# エラー表示
show_errors() {
    log_error "❌ 最新のエラーイベント:"
    aws cloudformation describe-stack-events \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`] | [0:5].[Timestamp,LogicalResourceId,ResourceStatusReason]' \
        --output table 2>/dev/null || log_warning "エラー情報を取得できませんでした"
}

# ログ表示
show_logs() {
    log_info "📜 CloudFormationイベントログ (最新10件):"
    aws cloudformation describe-stack-events \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'StackEvents[0:10].[Timestamp,LogicalResourceId,ResourceStatus,ResourceStatusReason]' \
        --output table 2>/dev/null || log_warning "ログを取得できませんでした"
}

# スタック削除
delete_stack() {
    log_warning "⚠️  スタックを削除します: $STACK_NAME"
    log_warning "これにより以下が削除されます:"
    echo "   • EC2インスタンス"
    echo "   • Cognito User Pool"
    echo "   • CloudFront Distribution"
    echo "   • 全ての関連リソース"
    echo ""
    
    read -p "本当に削除しますか? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "削除をキャンセルしました"
        return 0
    fi

    aws cloudformation delete-stack \
        --stack-name "$STACK_NAME" \
        --region "$REGION"

    log_success "🗑️  スタック削除を開始しました"
    log_info "📊 進捗を監視するには: $0 monitor -n $STACK_NAME -r $REGION"
}

# スタック一覧
list_stacks() {
    log_info "📋 VS Code Workshopスタック一覧 (リージョン: $REGION):"
    aws cloudformation list-stacks \
        --region "$REGION" \
        --stack-status-filter CREATE_COMPLETE CREATE_IN_PROGRESS UPDATE_COMPLETE DELETE_IN_PROGRESS \
        --query 'StackSummaries[?contains(StackName, `vscode`) || contains(StackName, `workshop`)].[StackName,StackStatus,CreationTime]' \
        --output table
}

# テンプレート検証
validate_template() {
    check_template
    log_info "🔍 テンプレートを検証中: $TEMPLATE_FILE"
    
    if aws cloudformation validate-template \
        --template-body "file://$TEMPLATE_FILE" \
        --region "$REGION" > /dev/null; then
        log_success "✅ テンプレート検証成功"
    else
        log_error "❌ テンプレート検証失敗"
        return 1
    fi
}

# メイン処理
main() {
    parse_args "$@"
    check_aws_cli

    case "$COMMAND" in
        create)
            create_stack
            ;;
        status)
            check_status
            ;;
        monitor)
            monitor_stack
            ;;
        outputs)
            show_outputs
            ;;
        login)
            show_login_info
            ;;
        open)
            open_browser
            ;;
        cognito)
            show_cognito_details
            ;;
        fix-oauth)
            check_oauth_settings
            ;;
        direct-login)
            show_direct_login
            ;;
        logs)
            show_logs
            ;;
        delete)
            delete_stack
            ;;
        list)
            list_stacks
            ;;
        validate)
            validate_template
            ;;
        *)
            log_error "不明なコマンド: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# スクリプト実行
main "$@"