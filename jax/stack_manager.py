#!/bin/bash

# AWS Neuron DLAMI Stack Manager
# 全ての操作を一つのスクリプトで実行

set -e

# デフォルト設定
DEFAULT_REGION="us-east-1"
DEFAULT_INSTANCE_TYPE="trn1.2xlarge"
DEFAULT_DLAMI_TYPE="jax-0.6"
TEMPLATE_FILE="ec2-ssm.yml"

# 色付きメッセージ
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# ヘルプ表示
show_help() {
    cat << EOF
AWS Neuron DLAMI Stack Manager

使用方法:
  $0 <command> [options]

コマンド:
  create      - スタックを作成
  status      - スタック状態を確認
  monitor     - スタック作成/削除の進捗を監視
  connect     - インスタンスにSSH接続
  jupyter     - Jupyterポートフォワーディング
  vscode      - VS Codeポートフォワーディング
  ports       - 両方のポートフォワーディング
  delete      - スタックを削除
  list        - 全スタック一覧
  validate    - テンプレート検証

オプション:
  -n, --name NAME         スタック名 (デフォルト: neuron-dev-USERNAME)
  -r, --region REGION     AWSリージョン (デフォルト: $DEFAULT_REGION)
  -t, --type TYPE         インスタンスタイプ (デフォルト: $DEFAULT_INSTANCE_TYPE)
  -d, --dlami TYPE        DLAMIタイプ (デフォルト: $DEFAULT_DLAMI_TYPE)
  -u, --user USER         ユーザー名 (デフォルト: 現在のユーザー)
  -h, --help              このヘルプを表示

インスタンスタイプ:
  Trn1: trn1.2xlarge, trn1.32xlarge
  Inf2: inf2.xlarge, inf2.8xlarge, inf2.24xlarge, inf2.48xlarge

DLAMIタイプ:
  multi-framework, jax-0.6, pytorch-2.7, tensorflow-2.10

例:
  $0 create -n my-jax-dev -t trn1.2xlarge -d jax-0.6
  $0 status -n my-jax-dev
  $0 ports -n my-jax-dev
  $0 delete -n my-jax-dev

EOF
}

# パラメータ解析
parse_args() {
    COMMAND=""
    STACK_NAME=""
    REGION="$DEFAULT_REGION"
    INSTANCE_TYPE="$DEFAULT_INSTANCE_TYPE"
    DLAMI_TYPE="$DEFAULT_DLAMI_TYPE"
    USER_NAME=$(whoami)
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            create|status|monitor|connect|jupyter|vscode|ports|delete|list|validate)
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
            -d|--dlami)
                DLAMI_TYPE="$2"
                shift 2
                ;;
            -u|--user)
                USER_NAME="$2"
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
        STACK_NAME="neuron-dev-$USER_NAME"
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

# スタック作成
create_stack() {
    log_info "スタックを作成中: $STACK_NAME"
    log_info "リージョン: $REGION"
    log_info "インスタンスタイプ: $INSTANCE_TYPE"
    log_info "DLAMIタイプ: $DLAMI_TYPE"
    log_info "ユーザー名: $USER_NAME"
    
    check_template
    
    aws cloudformation create-stack \
        --stack-name "$STACK_NAME" \
        --template-body "file://$TEMPLATE_FILE" \
        --parameters \
            "ParameterKey=UserName,ParameterValue=$USER_NAME" \
            "ParameterKey=InstanceType,ParameterValue=$INSTANCE_TYPE" \
            "ParameterKey=DLAMIType,ParameterValue=$DLAMI_TYPE" \
            "ParameterKey=Region,ParameterValue=$REGION" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "$REGION"
    
    log_success "スタック作成を開始しました"
    log_info "進捗を監視するには: $0 monitor -n $STACK_NAME -r $REGION"
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
    
    log_info "スタック状態: $status"
    
    case "$status" in
        CREATE_COMPLETE)
            log_success "スタック作成完了"
            show_outputs
            ;;
        CREATE_IN_PROGRESS)
            log_info "スタック作成中..."
            ;;
        CREATE_FAILED)
            log_error "スタック作成失敗"
            show_errors
            ;;
        DELETE_IN_PROGRESS)
            log_info "スタック削除中..."
            ;;
        DELETE_COMPLETE)
            log_success "スタック削除完了"
            ;;
        *)
            log_warning "スタック状態: $status"
            ;;
    esac
    
    return 0
}

# 進捗監視
monitor_stack() {
    log_info "スタック進捗を監視中: $STACK_NAME"
    
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
        
        echo -ne "\r$(date '+%H:%M:%S') - Status: $status"
        
        case "$status" in
            CREATE_COMPLETE|DELETE_COMPLETE)
                echo ""
                log_success "操作完了: $status"
                if [[ "$status" == "CREATE_COMPLETE" ]]; then
                    show_outputs
                fi
                break
                ;;
            CREATE_FAILED|DELETE_FAILED|ROLLBACK_COMPLETE)
                echo ""
                log_error "操作失敗: $status"
                show_errors
                break
                ;;
        esac
        
        sleep 5
    done
}

# 出力値表示
show_outputs() {
    log_info "スタック出力値:"
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table 2>/dev/null || log_warning "出力値を取得できませんでした"
}

# エラー表示
show_errors() {
    log_error "最新のエラーイベント:"
    aws cloudformation describe-stack-events \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`] | [0:3].[Timestamp,LogicalResourceId,ResourceStatusReason]' \
        --output table 2>/dev/null || log_warning "エラー情報を取得できませんでした"
}

# インスタンスID取得
get_instance_id() {
    local instance_id
    instance_id=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`InstanceId`].OutputValue' \
        --output text 2>/dev/null)
    
    if [[ -z "$instance_id" || "$instance_id" == "None" ]]; then
        log_error "インスタンスIDを取得できませんでした"
        return 1
    fi
    
    echo "$instance_id"
}

# SSH接続
connect_ssh() {
    local instance_id
    instance_id=$(get_instance_id)
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    log_info "インスタンスに接続中: $instance_id"
    aws ssm start-session \
        --target "$instance_id" \
        --region "$REGION"
}

# Jupyterポートフォワーディング
forward_jupyter() {
    local instance_id
    instance_id=$(get_instance_id)
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    log_info "Jupyterポートフォワーディング開始: $instance_id"
    log_success "アクセスURL: http://localhost:18888"
    
    aws ssm start-session \
        --target "$instance_id" \
        --region "$REGION" \
        --document-name AWS-StartPortForwardingSession \
        --parameters '{"portNumber":["8888"],"localPortNumber":["18888"]}'
}

# VS Codeポートフォワーディング
forward_vscode() {
    local instance_id
    instance_id=$(get_instance_id)
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    log_info "VS Codeポートフォワーディング開始: $instance_id"
    log_success "アクセスURL: http://localhost:18080"
    
    aws ssm start-session \
        --target "$instance_id" \
        --region "$REGION" \
        --document-name AWS-StartPortForwardingSession \
        --parameters '{"portNumber":["8080"],"localPortNumber":["18080"]}'
}

# 複数ポートフォワーディング
forward_ports() {
    local instance_id
    instance_id=$(get_instance_id)
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    log_info "複数ポートフォワーディング開始: $instance_id"
    log_success "アクセスURL:"
    log_success "  Jupyter: http://localhost:18888"
    log_success "  VS Code: http://localhost:18080"
    log_info "Ctrl+Cで終了"
    
    # バックグラウンドでポートフォワーディング開始
    aws ssm start-session \
        --target "$instance_id" \
        --region "$REGION" \
        --document-name AWS-StartPortForwardingSession \
        --parameters '{"portNumber":["8888"],"localPortNumber":["18888"]}' &
    
    local jupyter_pid=$!
    
    aws ssm start-session \
        --target "$instance_id" \
        --region "$REGION" \
        --document-name AWS-StartPortForwardingSession \
        --parameters '{"portNumber":["8080"],"localPortNumber":["18080"]}' &
    
    local vscode_pid=$!
    
    # シグナルハンドラー設定
    trap "kill $jupyter_pid $vscode_pid 2>/dev/null; exit" INT TERM
    
    # 両方のプロセスを待機
    wait $jupyter_pid $vscode_pid
}

# スタック削除
delete_stack() {
    log_warning "スタックを削除します: $STACK_NAME"
    read -p "続行しますか? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "削除をキャンセルしました"
        return 0
    fi
    
    aws cloudformation delete-stack \
        --stack-name "$STACK_NAME" \
        --region "$REGION"
    
    log_success "スタック削除を開始しました"
    log_info "進捗を監視するには: $0 monitor -n $STACK_NAME -r $REGION"
}

# スタック一覧
list_stacks() {
    log_info "スタック一覧 (リージョン: $REGION):"
    aws cloudformation list-stacks \
        --region "$REGION" \
        --stack-status-filter CREATE_COMPLETE CREATE_IN_PROGRESS UPDATE_COMPLETE DELETE_IN_PROGRESS \
        --query 'StackSummaries[*].[StackName,StackStatus,CreationTime]' \
        --output table
}

# テンプレート検証
validate_template() {
    check_template
    
    log_info "テンプレートを検証中: $TEMPLATE_FILE"
    
    if aws cloudformation validate-template \
        --template-body "file://$TEMPLATE_FILE" \
        --region "$REGION" > /dev/null; then
        log_success "テンプレート検証成功"
    else
        log_error "テンプレート検証失敗"
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
        connect)
            connect_ssh
            ;;
        jupyter)
            forward_jupyter
            ;;
        vscode)
            forward_vscode
            ;;
        ports)
            forward_ports
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
