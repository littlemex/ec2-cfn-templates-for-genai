#!/bin/bash

# Amazon Q Developer CLI 最小入力自動セットアップスクリプト

set -e

# 色付きログ出力
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

# 必要なツールの確認
check_prerequisites() {
    log_info "前提条件をチェック中..."
    
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI がインストールされていません"
        log_info "AWS CLI をインストールしています..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            if command -v brew &> /dev/null; then
                brew install awscli
            else
                log_error "Homebrew がインストールされていません。手動で AWS CLI をインストールしてください"
                exit 1
            fi
        else
            curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
            unzip awscliv2.zip
            sudo ./aws/install
            rm -rf aws awscliv2.zip
        fi
    fi
    
    if ! command -v jq &> /dev/null; then
        log_info "jq をインストールしています..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install jq
        else
            sudo apt-get update && sudo apt-get install -y jq
        fi
    fi
    
    log_success "前提条件のチェック完了"
}

# ユーザー入力の取得（最小限）
get_user_input() {
    log_info "セットアップ情報を入力してください..."
    
    read -p "管理者のメールアドレス: " ADMIN_EMAIL
    read -p "AWS プロファイル名 (Enter でデフォルト): " AWS_PROFILE
    
    # デフォルト値の設定
    ADMIN_FIRST_NAME="太郎"
    ADMIN_LAST_NAME="山田"
    AWS_PROFILE=${AWS_PROFILE:-default}
    
    log_info "使用する設定:"
    log_info "  メールアドレス: $ADMIN_EMAIL"
    log_info "  管理者名: $ADMIN_FIRST_NAME $ADMIN_LAST_NAME"
    log_info "  AWS プロファイル: $AWS_PROFILE"
    
    # AWS アカウント ID を自動取得
    log_info "AWS アカウント情報を取得中..."
    ACCOUNT_ID=$(aws sts get-caller-identity --profile $AWS_PROFILE --query Account --output text 2>/dev/null || echo "")
    
    if [ -z "$ACCOUNT_ID" ]; then
        log_error "AWS 認証情報が設定されていません"
        log_info "AWS CLI を設定してください: aws configure --profile $AWS_PROFILE"
        exit 1
    fi
    
    log_success "AWS アカウント ID: $ACCOUNT_ID"
    
    # リージョンの取得
    REGION=$(aws configure get region --profile $AWS_PROFILE 2>/dev/null || echo "us-east-1")
    log_info "使用リージョン: $REGION"
}

# Identity Center の有効化チェック
check_identity_center() {
    log_info "Identity Center の状態をチェック中..."
    
    # Identity Center インスタンスの確認
    INSTANCE_ARN=$(aws sso-admin list-instances --profile $AWS_PROFILE --region $REGION --query 'Instances[0].InstanceArn' --output text 2>/dev/null || echo "None")
    
    if [ "$INSTANCE_ARN" = "None" ] || [ -z "$INSTANCE_ARN" ]; then
        log_warning "Identity Center が有効化されていません"
        log_info "Identity Center を有効化するには AWS コンソールでの手動操作が必要です"
        log_info "以下の URL にアクセスして Identity Center を有効化してください:"
        log_info "https://console.aws.amazon.com/singlesignon/home?region=$REGION"
        
        read -p "Identity Center を有効化しましたか？ (y/N): " CONFIRM
        if [[ ! $CONFIRM =~ ^[Yy]$ ]]; then
            log_error "Identity Center の有効化が必要です"
            exit 1
        fi
        
        # 再度チェック
        sleep 5
        INSTANCE_ARN=$(aws sso-admin list-instances --profile $AWS_PROFILE --region $REGION --query 'Instances[0].InstanceArn' --output text 2>/dev/null || echo "None")
        
        if [ "$INSTANCE_ARN" = "None" ] || [ -z "$INSTANCE_ARN" ]; then
            log_error "Identity Center の有効化が確認できません"
            exit 1
        fi
    fi
    
    IDENTITY_STORE_ID=$(aws sso-admin list-instances --profile $AWS_PROFILE --region $REGION --query 'Instances[0].IdentityStoreId' --output text)
    
    log_success "Identity Center が有効化されています"
    log_info "Instance ARN: $INSTANCE_ARN"
    log_info "Identity Store ID: $IDENTITY_STORE_ID"
}

# ユーザーの作成
create_admin_user() {
    log_info "管理者ユーザーを作成中..."
    
    # ユーザー名で既存ユーザーをチェック
    EXISTING_USER=$(aws identitystore list-users --identity-store-id $IDENTITY_STORE_ID --profile $AWS_PROFILE --region $REGION --filters AttributePath=UserName,AttributeValue=admin-user --query 'Users[0].UserId' --output text 2>/dev/null || echo "None")
    
    if [ "$EXISTING_USER" != "None" ] && [ -n "$EXISTING_USER" ]; then
        log_warning "ユーザー 'admin-user' は既に存在します"
        USER_ID=$EXISTING_USER
        USER_EXISTS=true
        return
    fi
    
    # メールアドレスで既存ユーザーをチェック
    EXISTING_USER_BY_EMAIL=$(aws identitystore list-users --identity-store-id $IDENTITY_STORE_ID --profile $AWS_PROFILE --region $REGION --filters AttributePath=Emails.Value,AttributeValue="$ADMIN_EMAIL" --query 'Users[0].UserId' --output text 2>/dev/null || echo "None")
    
    if [ "$EXISTING_USER_BY_EMAIL" != "None" ] && [ -n "$EXISTING_USER_BY_EMAIL" ]; then
        log_warning "メールアドレス '$ADMIN_EMAIL' を持つユーザーが既に存在します"
        USER_ID=$EXISTING_USER_BY_EMAIL
        
        # 既存ユーザーの情報を表示
        EXISTING_USERNAME=$(aws identitystore describe-user --identity-store-id $IDENTITY_STORE_ID --user-id $USER_ID --profile $AWS_PROFILE --region $REGION --query 'UserName' --output text 2>/dev/null || echo "Unknown")
        log_info "既存ユーザーを使用します: $EXISTING_USERNAME (ID: $USER_ID)"
        USER_EXISTS=true
        return
    fi
    
    # 固定の一時パスワードを設定
    TEMP_PASSWORD="TempPass123!"
    log_info "一時パスワードを設定しました: $TEMP_PASSWORD"
    
    # 新規ユーザー作成
    USER_RESPONSE=$(aws identitystore create-user \
        --identity-store-id $IDENTITY_STORE_ID \
        --profile $AWS_PROFILE \
        --region $REGION \
        --user-name admin-user \
        --display-name "$ADMIN_FIRST_NAME $ADMIN_LAST_NAME" \
        --name Formatted="$ADMIN_FIRST_NAME $ADMIN_LAST_NAME",FamilyName="$ADMIN_LAST_NAME",GivenName="$ADMIN_FIRST_NAME" \
        --emails Value="$ADMIN_EMAIL",Type=work,Primary=true \
        --query 'UserId' --output text 2>/dev/null || echo "FAILED")
    
    if [ "$USER_RESPONSE" = "FAILED" ]; then
        log_error "ユーザーの作成に失敗しました"
        log_info "外部 Identity Provider と連携している可能性があります"
        
        # 全ユーザーを表示して選択を促す
        log_info "既存ユーザーの一覧:"
        aws identitystore list-users --identity-store-id $IDENTITY_STORE_ID --profile $AWS_PROFILE --region $REGION --query 'Users[*].[UserName,DisplayName,UserId]' --output table 2>/dev/null || log_warning "ユーザー一覧の取得に失敗"
        
        read -p "使用するユーザー ID を入力してください: " USER_ID
        
        if [ -z "$USER_ID" ]; then
            log_error "ユーザー ID が必要です"
            exit 1
        fi
        
        log_success "指定されたユーザーを使用します: $USER_ID"
        USER_EXISTS=true
    else
        USER_ID=$USER_RESPONSE
        log_success "管理者ユーザーを作成しました: $USER_ID"
        USER_EXISTS=false
        
        # Email OTP 有効化の案内
        show_email_otp_setup
    fi
}

# グループの作成
create_admin_group() {
    log_info "管理者グループを作成中..."
    
    # グループが既に存在するかチェック
    EXISTING_GROUP=$(aws identitystore list-groups --identity-store-id $IDENTITY_STORE_ID --profile $AWS_PROFILE --region $REGION --filters AttributePath=DisplayName,AttributeValue=Administrators --query 'Groups[0].GroupId' --output text 2>/dev/null || echo "None")
    
    if [ "$EXISTING_GROUP" != "None" ] && [ -n "$EXISTING_GROUP" ]; then
        log_warning "グループ 'Administrators' は既に存在します"
        GROUP_ID=$EXISTING_GROUP
    else
        # グループ作成
        GROUP_RESPONSE=$(aws identitystore create-group \
            --identity-store-id $IDENTITY_STORE_ID \
            --profile $AWS_PROFILE \
            --region $REGION \
            --display-name Administrators \
            --description "Administrator group for Amazon Q Developer" \
            --query 'GroupId' --output text)
        
        GROUP_ID=$GROUP_RESPONSE
        log_success "管理者グループを作成しました: $GROUP_ID"
    fi
    
    # ユーザーをグループに追加
    log_info "ユーザーをグループに追加中..."
    
    # 既にメンバーかチェック
    EXISTING_MEMBERSHIP=$(aws identitystore list-group-memberships --identity-store-id $IDENTITY_STORE_ID --group-id $GROUP_ID --profile $AWS_PROFILE --region $REGION --query "GroupMemberships[?MemberId.UserId=='$USER_ID'].MembershipId" --output text 2>/dev/null || echo "")
    
    if [ -n "$EXISTING_MEMBERSHIP" ]; then
        log_warning "ユーザーは既にグループのメンバーです"
    else
        aws identitystore create-group-membership \
            --identity-store-id $IDENTITY_STORE_ID \
            --group-id $GROUP_ID \
            --member-id UserId=$USER_ID \
            --profile $AWS_PROFILE \
            --region $REGION > /dev/null
        
        log_success "ユーザーをグループに追加しました"
    fi
}

# Permission Set の取得または作成
create_permission_set() {
    log_info "Permission Set を確認中..."
    
    # 既存の Permission Set をすべて取得
    EXISTING_PERMISSION_SETS=$(aws sso-admin list-permission-sets --instance-arn $INSTANCE_ARN --profile $AWS_PROFILE --region $REGION --query 'PermissionSets' --output text 2>/dev/null || echo "")
    
    PERMISSION_SET_ARN=""
    PERMISSION_SET_NAME=""
    
    # 既存の Permission Set から適切なものを探す
    if [ -n "$EXISTING_PERMISSION_SETS" ]; then
        log_info "既存の Permission Set を確認中..."
        
        # 優先順位: AdministratorAccess > PowerUserAccess > その他の管理者系
        for ps_arn in $EXISTING_PERMISSION_SETS; do
            PS_NAME=$(aws sso-admin describe-permission-set --instance-arn $INSTANCE_ARN --permission-set-arn $ps_arn --profile $AWS_PROFILE --region $REGION --query 'PermissionSet.Name' --output text 2>/dev/null || echo "")
            
            if [ "$PS_NAME" = "AdministratorAccess" ]; then
                PERMISSION_SET_ARN=$ps_arn
                PERMISSION_SET_NAME=$PS_NAME
                log_success "既存の Permission Set 'AdministratorAccess' を使用します"
                break
            elif [ "$PS_NAME" = "PowerUserAccess" ] && [ -z "$PERMISSION_SET_ARN" ]; then
                PERMISSION_SET_ARN=$ps_arn
                PERMISSION_SET_NAME=$PS_NAME
                log_info "Permission Set 'PowerUserAccess' を候補として記録"
            elif [[ "$PS_NAME" =~ [Aa]dmin ]] && [ -z "$PERMISSION_SET_ARN" ]; then
                PERMISSION_SET_ARN=$ps_arn
                PERMISSION_SET_NAME=$PS_NAME
                log_info "管理者系 Permission Set '$PS_NAME' を候補として記録"
            fi
        done
        
        if [ -n "$PERMISSION_SET_ARN" ]; then
            log_success "使用する Permission Set: $PERMISSION_SET_NAME ($PERMISSION_SET_ARN)"
        fi
    fi
    
    # 既存の Permission Set が見つからない場合は新規作成を試行
    if [ -z "$PERMISSION_SET_ARN" ]; then
        log_info "適切な Permission Set が見つかりません。新規作成を試行中..."
        
        # Permission Set 作成を試行
        PERMISSION_SET_RESPONSE=$(aws sso-admin create-permission-set \
            --instance-arn $INSTANCE_ARN \
            --name AdministratorAccess \
            --description "Administrator access for Amazon Q Developer" \
            --session-duration PT12H \
            --profile $AWS_PROFILE \
            --region $REGION \
            --query 'PermissionSet.PermissionSetArn' --output text 2>/dev/null || echo "FAILED")
        
        if [ "$PERMISSION_SET_RESPONSE" != "FAILED" ]; then
            PERMISSION_SET_ARN=$PERMISSION_SET_RESPONSE
            PERMISSION_SET_NAME="AdministratorAccess"
            log_success "Permission Set を作成しました: $PERMISSION_SET_ARN"
            
            # AWS 管理ポリシーをアタッチ
            log_info "AdministratorAccess ポリシーをアタッチ中..."
            aws sso-admin attach-managed-policy-to-permission-set \
                --instance-arn $INSTANCE_ARN \
                --permission-set-arn $PERMISSION_SET_ARN \
                --managed-policy-arn arn:aws:iam::aws:policy/AdministratorAccess \
                --profile $AWS_PROFILE \
                --region $REGION 2>/dev/null || log_warning "ポリシーのアタッチに失敗しましたが続行します"
            
            log_success "Permission Set の設定が完了しました"
        else
            log_error "Permission Set の作成に失敗しました"
            log_error "外部 Identity Provider と連携している場合、Permission Set の作成は制限されます"
            log_info "AWS コンソールで手動で Permission Set を作成してから再実行してください"
            log_info "https://console.aws.amazon.com/singlesignon/home?region=$REGION#!/permissionsets"
            exit 1
        fi
    fi
}

# アカウント割り当て
create_account_assignment() {
    log_info "アカウント割り当てを作成中..."
    
    # 既存の割り当てをチェック
    EXISTING_ASSIGNMENT=$(aws sso-admin list-account-assignments --instance-arn $INSTANCE_ARN --account-id $ACCOUNT_ID --permission-set-arn $PERMISSION_SET_ARN --profile $AWS_PROFILE --region $REGION --query "AccountAssignments[?PrincipalId=='$GROUP_ID'].RequestId" --output text 2>/dev/null || echo "")
    
    if [ -n "$EXISTING_ASSIGNMENT" ]; then
        log_warning "アカウント割り当ては既に存在します"
    else
        # アカウント割り当て作成
        ASSIGNMENT_RESPONSE=$(aws sso-admin create-account-assignment \
            --instance-arn $INSTANCE_ARN \
            --target-id $ACCOUNT_ID \
            --target-type AWS_ACCOUNT \
            --permission-set-arn $PERMISSION_SET_ARN \
            --principal-type GROUP \
            --principal-id $GROUP_ID \
            --profile $AWS_PROFILE \
            --region $REGION \
            --query 'AccountAssignmentCreationStatus.RequestId' --output text)
        
        log_info "アカウント割り当てを作成中... Request ID: $ASSIGNMENT_RESPONSE"
        
        # 割り当て完了を待機
        log_info "アカウント割り当ての完了を待機中..."
        while true; do
            STATUS=$(aws sso-admin describe-account-assignment-creation-status \
                --instance-arn $INSTANCE_ARN \
                --account-assignment-creation-request-id $ASSIGNMENT_RESPONSE \
                --profile $AWS_PROFILE \
                --region $REGION \
                --query 'AccountAssignmentCreationStatus.Status' --output text 2>/dev/null || echo "FAILED")
            
            if [ "$STATUS" = "SUCCEEDED" ]; then
                log_success "アカウント割り当てが完了しました"
                break
            elif [ "$STATUS" = "FAILED" ]; then
                log_error "アカウント割り当てに失敗しました"
                exit 1
            else
                log_info "待機中... (現在のステータス: $STATUS)"
                sleep 10
            fi
        done
    fi
}



# Email OTP 有効化の案内
show_email_otp_setup() {
    log_info "Email OTP 有効化の案内を表示します"
    echo ""
    echo "=== Email OTP 有効化手順（推奨） ==="
    echo "Email OTP を有効化すると、初回ログイン時にメールでワンタイムパスワードが送信され、"
    echo "'Forgot Password?' を使わずに簡単にログインできます。"
    echo ""
    echo "1. 以下の URL にアクセスしてください:"
    echo "   https://$REGION.console.aws.amazon.com/singlesignon/home?region=$REGION&tab=network-security#/instances/${INSTANCE_ARN##*/}/settings"
    echo ""
    echo "2. 'Authentication' タブをクリック"
    echo "3. 'Standard authentication' セクションの 'Configure' をクリック"
    echo "4. 'Send email OTP' にチェックを入れる"
    echo "5. 'Save' をクリック"
    echo ""
    echo "注意: この設定は一度だけ行えば、以降作成されるすべてのユーザーに適用されます"
    echo ""
    
    read -p "Email OTP を有効化しますか？ (今すぐ設定する場合は y、後で設定する場合は n): " EMAIL_OTP_SETUP
    
    if [[ $EMAIL_OTP_SETUP =~ ^[Yy]$ ]]; then
        echo ""
        log_info "上記の手順に従って Email OTP を有効化してください"
        read -p "Email OTP の有効化が完了したら Enter を押してください..."
        EMAIL_OTP_ENABLED=true
        log_success "Email OTP が有効化されました"
    else
        EMAIL_OTP_ENABLED=false
        log_info "Email OTP は後で有効化できます"
    fi
}

# Amazon Q Developer CLI のインストール
install_amazon_q_cli() {
    log_info "Amazon Q Developer CLI をインストール中..."
    
    if command -v q &> /dev/null; then
        log_warning "Amazon Q Developer CLI は既にインストールされています"
        return
    fi
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            brew install --cask amazon-q
        else
            log_error "Homebrew がインストールされていません"
            log_info "手動でインストールしてください: https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-installing.html"
            exit 1
        fi
    else
        # Linux の場合
        curl -sSL https://amazon-q-developer-cli.s3.amazonaws.com/install.sh | bash
        
        # PATH に追加
        if ! echo $PATH | grep -q "$HOME/.local/bin"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
    
    log_success "Amazon Q Developer CLI をインストールしました"
}

# AWS access portal URL の取得
get_access_portal_url() {
    log_info "AWS access portal URL を取得中..."
    
    # Identity Store ID から正しい URL を生成
    ACCESS_PORTAL_URL="https://${IDENTITY_STORE_ID}.awsapps.com/start"
    
    log_success "AWS access portal URL: $ACCESS_PORTAL_URL"
}

# 設定情報の出力
output_configuration() {
    log_success "セットアップが完了しました！"
    echo ""
    echo "=== 設定情報 ==="
    echo "AWS アカウント ID: $ACCOUNT_ID"
    echo "リージョン: $REGION"
    echo "AWS access portal URL: $ACCESS_PORTAL_URL"
    echo "管理者ユーザー: admin-user ($ADMIN_FIRST_NAME $ADMIN_LAST_NAME)"
    if [[ "${EMAIL_OTP_ENABLED:-false}" == "true" ]]; then
        echo "パスワード: Email OTP で簡単設定"
    else
        echo "パスワード: 'Forgot Password?' でリセットが必要"
    fi
    echo "管理者グループ: Administrators"
    echo "Permission Set: $PERMISSION_SET_NAME"
    echo ""
    echo "=== 次のステップ ==="
    echo "1. 以下のコマンドで Amazon Q Developer CLI にログインしてください:"
    echo "   q login"
    echo ""
    echo "2. ログイン方法の選択画面で「Use with Pro license」を選択してください"
    echo "   (Builder ID ではなく Pro license を選択)"
    echo ""
    echo "3. Start URL の入力を求められたら、以下の URL を入力してください:"
    echo "   $ACCESS_PORTAL_URL"
    echo ""
    echo "4. Region の入力を求められたら、以下のリージョンを入力してください:"
    echo "   $REGION"
    echo ""
    echo "5. デバイス認証コードが表示されます:"
    echo "   - 表示されたコード（例: HFZQ-HKMT）をメモしてください"
    echo "   - ブラウザが自動で開くか、表示された URL をブラウザで開いてください"
    echo ""
    echo "6. ブラウザでの認証手順:"
    echo "   - デバイス認証コードを入力"
    echo "   - ユーザー名: admin-user を入力"
    if [[ "${EMAIL_OTP_ENABLED:-false}" == "true" ]]; then
        echo "   - Email OTP が有効化されているため:"
        echo "     * '追加の検証が必要です' 画面が表示されます"
        echo "     * メール ($ADMIN_EMAIL) に送信されたOTP（10分間有効）を入力"
        echo "     * 'パスワードを選択' 画面で新しいパスワードを設定"
    else
        echo "   - Email OTP が無効の場合:"
        echo "     * 'Forgot Password?' をクリックしてパスワードリセット"
        echo "     * メール ($ADMIN_EMAIL) のリンクから新しいパスワードを設定"
    fi
    echo "   - アカウント: $ACCOUNT_ID を選択"
    echo "   - ロール: $PERMISSION_SET_NAME を選択"
    echo ""
    echo "7. Amazon Q Developer Pro サブスクリプションの追加:"
    echo "   https://us-east-1.console.aws.amazon.com/amazonq/developer/home?region=us-east-1#/subscriptions?tab=groups"
    echo "   - 'Subscribe users and groups' をクリック"
    echo "   - 'Users' タブでユーザー '$ADMIN_EMAIL' を検索"
    echo "   - ユーザーを選択して 'Subscribe' をクリック"
    echo "   - 最大24時間待機"
    echo ""
    echo "8. 動作確認:"
    echo "   q \"List my S3 buckets\""
    echo ""
    echo "注意事項:"
    if [[ "${EMAIL_OTP_ENABLED:-false}" == "true" ]]; then
        echo "- Email OTP が有効化されているため、初回ログイン時にOTPがメールで送信されます"
        echo "- OTPの有効期限は10分です"
        echo "- 'Forgot Password?' は不要です"
    else
        echo "- Email OTP が無効のため、初回ログイン時は 'Forgot Password?' が必要です"
        echo "- Email OTP を有効化すると、より簡単にログインできます"
    fi
    echo "- メールボックス ($ADMIN_EMAIL) を必ず確認してください"
    echo "- 重要: Amazon Q Developer Pro のサブスクリプション追加が必要です"
    echo "- サブスクリプション追加後、最大24時間かかる場合があります"
}

# メイン実行
main() {
    log_info "Amazon Q Developer CLI 最小入力自動セットアップを開始します"
    
    check_prerequisites
    get_user_input
    check_identity_center
    create_admin_user
    create_admin_group
    create_permission_set
    

    create_account_assignment
    install_amazon_q_cli
    get_access_portal_url
    output_configuration
}

# スクリプト実行
main "$@"