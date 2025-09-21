#!/bin/bash

# Amazon Q Developer CLI 最小入力自動セットアップスクリプト
# 使用方法: ./setup-amazon-q-developer-minimal.sh

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
    read -p "AWS プロファイル名 (Instance Profile 使用時は Enter): " AWS_PROFILE
    
    # デフォルト値の設定
    ADMIN_FIRST_NAME="太郎"
    ADMIN_LAST_NAME="山田"
    
    # AWS プロファイルの設定（空の場合はプロファイルなしで実行）
    if [ -z "$AWS_PROFILE" ]; then
        AWS_PROFILE_OPTION=""
        log_info "Instance Profile または環境変数を使用します"
    else
        AWS_PROFILE_OPTION="$AWS_PROFILE_OPTION"
        log_info "AWS プロファイル: $AWS_PROFILE"
    fi
    
    log_info "使用する設定:"
    log_info "  メールアドレス: $ADMIN_EMAIL"
    log_info "  管理者名: $ADMIN_FIRST_NAME $ADMIN_LAST_NAME"

    
    # AWS アカウント ID を自動取得
    log_info "AWS アカウント情報を取得中..."
    ACCOUNT_ID=$(aws sts get-caller-identity $AWS_PROFILE_OPTION --query Account --output text 2>/dev/null || echo "")
    
    if [ -z "$ACCOUNT_ID" ]; then
        log_error "AWS 認証情報が設定されていません"
        if [ -z "$AWS_PROFILE" ]; then
            log_info "AWS CLI を設定してください: aws configure"
            log_info "または EC2 Instance Profile を設定してください"
        else
            log_info "AWS CLI を設定してください: aws configure $AWS_PROFILE_OPTION"
        fi
        exit 1
    fi
    
    log_success "AWS アカウント ID: $ACCOUNT_ID"
    
    # リージョンの取得
    REGION=$(aws configure get region $AWS_PROFILE_OPTION 2>/dev/null || echo "us-east-1")
    log_info "使用リージョン: $REGION"
}

# Identity Center の有効化チェック
check_identity_center() {
    log_info "Identity Center の状態をチェック中..."
    
    # Identity Center インスタンスの確認
    INSTANCE_ARN=$(aws sso-admin list-instances $AWS_PROFILE_OPTION --region $REGION --query 'Instances[0].InstanceArn' --output text 2>/dev/null || echo "None")
    
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
        INSTANCE_ARN=$(aws sso-admin list-instances $AWS_PROFILE_OPTION --region $REGION --query 'Instances[0].InstanceArn' --output text 2>/dev/null || echo "None")
        
        if [ "$INSTANCE_ARN" = "None" ] || [ -z "$INSTANCE_ARN" ]; then
            log_error "Identity Center の有効化が確認できません"
            exit 1
        fi
    fi
    
    IDENTITY_STORE_ID=$(aws sso-admin list-instances $AWS_PROFILE_OPTION --region $REGION --query 'Instances[0].IdentityStoreId' --output text)
    
    log_success "Identity Center が有効化されています"
    log_info "Instance ARN: $INSTANCE_ARN"
    log_info "Identity Store ID: $IDENTITY_STORE_ID"
}

# ユーザーの作成
create_admin_user() {
    log_info "管理者ユーザーを作成中..."
    
    # ユーザー名で既存ユーザーをチェック
    EXISTING_USER=$(aws identitystore list-users --identity-store-id $IDENTITY_STORE_ID $AWS_PROFILE_OPTION --region $REGION --filters AttributePath=UserName,AttributeValue=admin-user --query 'Users[0].UserId' --output text 2>/dev/null || echo "None")
    
    if [ "$EXISTING_USER" != "None" ] && [ -n "$EXISTING_USER" ]; then
        log_warning "ユーザー 'admin-user' は既に存在します"
        USER_ID=$EXISTING_USER
        USER_EXISTS=true
        return
    fi
    
    # メールアドレスで既存ユーザーをチェック
    EXISTING_USER_BY_EMAIL=$(aws identitystore list-users --identity-store-id $IDENTITY_STORE_ID $AWS_PROFILE_OPTION --region $REGION --filters AttributePath=Emails.Value,AttributeValue="$ADMIN_EMAIL" --query 'Users[0].UserId' --output text 2>/dev/null || echo "None")
    
    if [ "$EXISTING_USER_BY_EMAIL" != "None" ] && [ -n "$EXISTING_USER_BY_EMAIL" ]; then
        log_warning "メールアドレス '$ADMIN_EMAIL' を持つユーザーが既に存在します"
        USER_ID=$EXISTING_USER_BY_EMAIL
        
        # 既存ユーザーの情報を表示
        EXISTING_USERNAME=$(aws identitystore describe-user --identity-store-id $IDENTITY_STORE_ID --user-id $USER_ID $AWS_PROFILE_OPTION --region $REGION --query 'UserName' --output text 2>/dev/null || echo "Unknown")
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
        $AWS_PROFILE_OPTION \
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
        aws identitystore list-users --identity-store-id $IDENTITY_STORE_ID $AWS_PROFILE_OPTION --region $REGION --query 'Users[*].[UserName,DisplayName,UserId]' --output table 2>/dev/null || log_warning "ユーザー一覧の取得に失敗"
        
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
    EXISTING_GROUP=$(aws identitystore list-groups --identity-store-id $IDENTITY_STORE_ID $AWS_PROFILE_OPTION --region $REGION --filters AttributePath=DisplayName,AttributeValue=Administrators --query 'Groups[0].GroupId' --output text 2>/dev/null || echo "None")
    
    if [ "$EXISTING_GROUP" != "None" ] && [ -n "$EXISTING_GROUP" ]; then
        log_warning "グループ 'Administrators' は既に存在します"
        GROUP_ID=$EXISTING_GROUP
    else
        # グループ作成
        GROUP_RESPONSE=$(aws identitystore create-group \
            --identity-store-id $IDENTITY_STORE_ID \
            $AWS_PROFILE_OPTION \
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
    EXISTING_MEMBERSHIP=$(aws identitystore list-group-memberships --identity-store-id $IDENTITY_STORE_ID --group-id $GROUP_ID $AWS_PROFILE_OPTION --region $REGION --query "GroupMemberships[?MemberId.UserId=='$USER_ID'].MembershipId" --output text 2>/dev/null || echo "")
    
    if [ -n "$EXISTING_MEMBERSHIP" ]; then
        log_warning "ユーザーは既にグループのメンバーです"
    else
        aws identitystore create-group-membership \
            --identity-store-id $IDENTITY_STORE_ID \
            --group-id $GROUP_ID \
            --member-id UserId=$USER_ID \
            $AWS_PROFILE_OPTION \
            --region $REGION > /dev/null
        
        log_success "ユーザーをグループに追加しました"
    fi
}

# Permission Set の取得または作成
create_permission_set() {
    log_info "Permission Set を確認中..."
    
    # 既存の Permission Set をすべて取得
    EXISTING_PERMISSION_SETS=$(aws sso-admin list-permission-sets --instance-arn $INSTANCE_ARN $AWS_PROFILE_OPTION --region $REGION --query 'PermissionSets' --output text 2>/dev/null || echo "")
    
    PERMISSION_SET_ARN=""
    PERMISSION_SET_NAME=""
    
    # 既存の Permission Set から適切なものを探す
    if [ -n "$EXISTING_PERMISSION_SETS" ]; then
        log_info "既存の Permission Set を確認中..."
        
        # 優先順位: AdministratorAccess > PowerUserAccess > その他の管理者系
        for ps_arn in $EXISTING_PERMISSION_SETS; do
            PS_NAME=$(aws sso-admin describe-permission-set --instance-arn $INSTANCE_ARN --permission-set-arn $ps_arn $AWS_PROFILE_OPTION --region $REGION --query 'PermissionSet.Name' --output text 2>/dev/null || echo "")
            
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
        
        # Permission Set 作成を試行（詳細エラー情報付き）
        log_info "Permission Set 'AdministratorAccess' を作成中..."
        
        # デバッグ情報を表示
        log_info "実行コマンド: aws sso-admin create-permission-set --instance-arn $INSTANCE_ARN --name AdministratorAccess $AWS_PROFILE_OPTION --region $REGION"
        
        # タイムアウト付きでコマンド実行
        PERMISSION_SET_RESPONSE=$(timeout 30 aws sso-admin create-permission-set \
            --instance-arn $INSTANCE_ARN \
            --name AdministratorAccess \
            --description "Administrator access for Amazon Q Developer" \
            --session-duration PT12H \
            $AWS_PROFILE_OPTION \
            --region $REGION \
            --query 'PermissionSet.PermissionSetArn' --output text 2>&1)
        
        COMMAND_EXIT_CODE=$?
        
        # タイムアウトまたはコマンド失敗をチェック
        if [ $COMMAND_EXIT_CODE -eq 124 ]; then
            log_error "Permission Set 作成がタイムアウトしました（30秒）"
            PERMISSION_SET_RESPONSE="TIMEOUT"
        elif [ $COMMAND_EXIT_CODE -ne 0 ]; then
            log_error "Permission Set 作成コマンドが失敗しました（終了コード: $COMMAND_EXIT_CODE）"
        fi
        
        if [[ $PERMISSION_SET_RESPONSE == arn:aws:sso* ]]; then
            # 成功した場合
            PERMISSION_SET_ARN=$PERMISSION_SET_RESPONSE
            PERMISSION_SET_NAME="AdministratorAccess"
            log_success "Permission Set を作成しました: $PERMISSION_SET_ARN"
        elif [ "$PERMISSION_SET_RESPONSE" = "TIMEOUT" ]; then
            # タイムアウトの場合
            log_error "Permission Set の作成がタイムアウトしました"
            log_info "ネットワークまたは AWS API の問題の可能性があります"
        else
            # その他の失敗の場合、詳細エラーを表示
            log_error "Permission Set の作成に失敗しました"
            log_error "エラー詳細: $PERMISSION_SET_RESPONSE"
            
            # 既存の Permission Set を再確認
            log_info "既存の Permission Set を再確認中..."
            ALL_PERMISSION_SETS=$(aws sso-admin list-permission-sets --instance-arn $INSTANCE_ARN $AWS_PROFILE_OPTION --region $REGION --query 'PermissionSets' --output text 2>/dev/null || echo "")
            
            if [ -n "$ALL_PERMISSION_SETS" ]; then
                log_info "既存の Permission Set 一覧:"
                for ps_arn in $ALL_PERMISSION_SETS; do
                    PS_NAME=$(aws sso-admin describe-permission-set --instance-arn $INSTANCE_ARN --permission-set-arn $ps_arn $AWS_PROFILE_OPTION --region $REGION --query 'PermissionSet.Name' --output text 2>/dev/null || echo "Unknown")
                    echo "  - $PS_NAME ($ps_arn)"
                done
                
                # ユーザーに選択を促す
                echo ""
                read -p "使用する Permission Set 名を入力してください (例: PowerUserAccess): " SELECTED_PS_NAME
                
                if [ -n "$SELECTED_PS_NAME" ]; then
                    for ps_arn in $ALL_PERMISSION_SETS; do
                        PS_NAME=$(aws sso-admin describe-permission-set --instance-arn $INSTANCE_ARN --permission-set-arn $ps_arn $AWS_PROFILE_OPTION --region $REGION --query 'PermissionSet.Name' --output text 2>/dev/null || echo "")
                        if [ "$PS_NAME" = "$SELECTED_PS_NAME" ]; then
                            PERMISSION_SET_ARN=$ps_arn
                            PERMISSION_SET_NAME=$PS_NAME
                            log_success "既存の Permission Set '$PERMISSION_SET_NAME' を使用します"
                            break
                        fi
                    done
                fi
            fi
            
            if [ -z "$PERMISSION_SET_ARN" ]; then
                log_error "使用可能な Permission Set が見つかりません"
                log_info "手動でのテスト用コマンド:"
                echo "  aws sso-admin list-permission-sets --instance-arn $INSTANCE_ARN $AWS_PROFILE_OPTION --region $REGION"
                echo "  aws sso-admin create-permission-set --instance-arn $INSTANCE_ARN --name TestPermissionSet $AWS_PROFILE_OPTION --region $REGION"
                log_info "AWS コンソールで手動で Permission Set を作成してから再実行してください"
                log_info "https://console.aws.amazon.com/singlesignon/home?region=$REGION#!/permissionsets"
                exit 1
            fi
        fi
        
        # Permission Set が正常に作成または選択された場合、ポリシーをアタッチ
        if [ "$PERMISSION_SET_NAME" = "AdministratorAccess" ] && [[ $PERMISSION_SET_RESPONSE == arn:aws:sso* ]]; then
            # 新規作成した AdministratorAccess の場合のみポリシーをアタッチ
            log_info "AdministratorAccess ポリシーをアタッチ中..."
            aws sso-admin attach-managed-policy-to-permission-set \
                --instance-arn $INSTANCE_ARN \
                --permission-set-arn $PERMISSION_SET_ARN \
                --managed-policy-arn arn:aws:iam::aws:policy/AdministratorAccess \
                $AWS_PROFILE_OPTION \
                --region $REGION 2>/dev/null || log_warning "ポリシーのアタッチに失敗しましたが続行します"
            
            log_success "Permission Set の設定が完了しました"
        fi
    fi
}

# アカウント割り当て
create_account_assignment() {
    log_info "アカウント割り当てを作成中..."
    
    # 既存の割り当てをチェック
    EXISTING_ASSIGNMENT=$(aws sso-admin list-account-assignments --instance-arn $INSTANCE_ARN --account-id $ACCOUNT_ID --permission-set-arn $PERMISSION_SET_ARN $AWS_PROFILE_OPTION --region $REGION --query "AccountAssignments[?PrincipalId=='$GROUP_ID'].RequestId" --output text 2>/dev/null || echo "")
    
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
            $AWS_PROFILE_OPTION \
            --region $REGION \
            --query 'AccountAssignmentCreationStatus.RequestId' --output text)
        
        log_info "アカウント割り当てを作成中... Request ID: $ASSIGNMENT_RESPONSE"
        
        # 割り当て完了を待機
        log_info "アカウント割り当ての完了を待機中..."
        while true; do
            STATUS=$(aws sso-admin describe-account-assignment-creation-status \
                --instance-arn $INSTANCE_ARN \
                --account-assignment-creation-request-id $ASSIGNMENT_RESPONSE \
                $AWS_PROFILE_OPTION \
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



# Email OTP 有効化と MFA 設定の案内
show_email_otp_setup() {
    log_info "Email OTP 有効化と MFA 設定の案内を表示します"
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
    echo "=== MFA 設定（検証環境では無効化推奨） ==="
    echo "同じページで MFA（多要素認証）の設定も行えます。検証環境では無効化することを推奨します。"
    echo ""
    echo "6. 同じ 'Authentication' タブの 'Multi-factor authentication' セクション"
    echo "7. 'Configure' をクリック"
    echo "8. 'Prompt users for MFA' で 'Never (disabled)' を選択"
    echo "9. 'Save' をクリック"
    echo ""
    echo "注意: これらの設定は一度だけ行えば、以降作成されるすべてのユーザーに適用されます"
    echo ""
    
    read -p "Email OTP 有効化と MFA 無効化を行いますか？ (今すぐ設定する場合は y、後で設定する場合は n): " EMAIL_OTP_SETUP
    
    if [[ $EMAIL_OTP_SETUP =~ ^[Yy]$ ]]; then
        echo ""
        log_info "上記の手順に従って Email OTP 有効化と MFA 無効化を行ってください"
        echo ""
        echo "設定確認:"
        echo "✓ Email OTP: 有効化"
        echo "✓ MFA: 無効化（検証環境用）"
        echo ""
        read -p "Email OTP 有効化と MFA 無効化が完了したら Enter を押してください..."
        EMAIL_OTP_ENABLED=true
        MFA_DISABLED=true
        log_success "Email OTP が有効化され、MFA が無効化されました"
    else
        EMAIL_OTP_ENABLED=false
        MFA_DISABLED=false
        log_info "Email OTP と MFA 設定は後で変更できます"
    fi
}

# Amazon Q Developer アプリケーションの作成
create_amazon_q_application() {
    log_info "Amazon Q Developer アプリケーションを確認中..."
    
    # 既存のアプリケーションをチェック
    EXISTING_APPS=$(aws sso-admin list-applications --instance-arn $INSTANCE_ARN $AWS_PROFILE_OPTION --region $REGION --query 'Applications[?contains(Name, `Amazon Q`) || contains(Name, `Q Developer`) || contains(Name, `amazon-q`)].ApplicationArn' --output text 2>/dev/null || echo "")
    
    if [ -n "$EXISTING_APPS" ]; then
        log_success "Amazon Q Developer アプリケーションは既に存在します"
        AMAZON_Q_APP_ARN=$EXISTING_APPS
        return
    fi
    
    log_warning "Amazon Q Developer アプリケーションが存在しません"
    log_info "Amazon Q Developer の Identity Center 統合を有効化する必要があります"
    echo ""
    echo "=========================================="
    echo "🚀 Amazon Q Developer 統合有効化手順"
    echo "=========================================="
    echo ""
    echo "📋 以下の手順を順番に実行してください："
    echo ""
    echo "【ステップ 1: Amazon Q Developer コンソールにアクセス】"
    echo "   以下の URL をブラウザで開いてください："
    echo "   👉 https://$REGION.console.aws.amazon.com/amazonq/developer/home?region=$REGION"
    echo ""
    echo "【ステップ 2: Identity Center 統合を有効化】"
    echo "   ✅ 「Get started with Amazon Q Developer」をクリック"
    echo "   ✅ または「Settings」→「Identity and access management」をクリック"
    echo "   ✅ 「Identity Center integration」セクションを探す"
    echo "   ✅ 「Enable Identity Center integration」をクリック"
    echo "   ✅ 既存の Identity Center インスタンスを選択："
    echo "      Instance ARN: $INSTANCE_ARN"
    echo "   ✅ 「Enable」をクリック"
    echo ""
    echo "【ステップ 3: ユーザーサブスクリプションの追加】"
    echo "   ✅ 「Subscriptions」タブをクリック"
    echo "   ✅ 「Subscribe users and groups」をクリック"
    echo "   ✅ 「Users」タブで以下のいずれかを検索："
    echo "      - ユーザー名: admin-user"
    echo "      - メールアドレス: $ADMIN_EMAIL"
    echo "   ✅ ユーザーを選択して「Subscribe」をクリック"
    echo ""
    echo "【ステップ 4: Application Assignment 設定（重要）】"
    echo "   ✅ Identity Center コンソールで以下の URL にアクセス："
    echo "      👉 https://$REGION.console.aws.amazon.com/singlesignon/applications/home?region=$REGION&tab=application-assignments#/instances/${INSTANCE_ARN##*/}/"
    echo "   ✅ 「QDevProfile-$REGION」アプリケーションを選択"
    echo "   ✅ 「Edit」をクリック"
    echo "   ✅ 「User and group assignment method」で以下のいずれかを選択："
    echo ""
    echo "      🔧 開発・テスト環境（推奨）："
    echo "         「Do not require assignments」を選択"
    echo "         → すべての Identity Center ユーザーがアクセス可能"
    echo ""
    echo "      🔒 本番環境（セキュア）："
    echo "         「Require assignments」を選択"
    echo "         → 「Assign users and groups」で明示的にユーザー/グループを割り当て"
    echo "         → 「Administrators」グループまたは「admin-user」を割り当て"
    echo ""
    echo "   ✅ 「Save changes」をクリック"
    echo ""
    echo "【ステップ 5: 統合完了の確認】"
    echo "   ✅ 「Identity Center integration」が「Enabled」になっていることを確認"
    echo "   ✅ 「Subscriptions」でユーザーが「Active」になっていることを確認"
    echo "   ✅ Application Assignment が適切に設定されていることを確認"
    echo ""
    echo "=========================================="
    echo "⚠️  重要な注意事項"
    echo "=========================================="
    echo "• Application Assignment の設定が最も重要です"
    echo "• 「Do not require assignments」= 簡単だが全ユーザーがアクセス可能"
    echo "• 「Require assignments」= セキュアだが明示的な割り当てが必要"
    echo "• サブスクリプション追加後、最大24時間かかる場合があります"
    echo "• 統合が完了するまで Amazon Q CLI ログインは失敗します"
    echo ""
    
    # ブラウザを自動で開く（可能な場合）
    AMAZON_Q_URL="https://$REGION.console.aws.amazon.com/amazonq/developer/home?region=$REGION"
    
    if command -v open &> /dev/null; then
        # macOS
        log_info "ブラウザを自動で開いています..."
        open "$AMAZON_Q_URL" 2>/dev/null || log_warning "ブラウザの自動起動に失敗しました"
    elif command -v xdg-open &> /dev/null; then
        # Linux
        log_info "ブラウザを自動で開いています..."
        xdg-open "$AMAZON_Q_URL" 2>/dev/null || log_warning "ブラウザの自動起動に失敗しました"
    elif command -v start &> /dev/null; then
        # Windows (WSL)
        log_info "ブラウザを自動で開いています..."
        start "$AMAZON_Q_URL" 2>/dev/null || log_warning "ブラウザの自動起動に失敗しました"
    else
        log_info "手動で以下の URL をブラウザで開いてください："
        echo "   $AMAZON_Q_URL"
    fi
    
    echo ""
    read -p "🔄 Amazon Q Developer の統合、サブスクリプション、Application Assignment をすべて完了しましたか？ (y/N): " INTEGRATION_COMPLETED
    
    if [[ ! $INTEGRATION_COMPLETED =~ ^[Yy]$ ]]; then
        log_error "Amazon Q Developer の統合が必要です"
        log_info "統合完了後、このスクリプトを再実行してください"
        echo ""
        echo "再実行コマンド:"
        echo "  ./$(basename "$0")"
        exit 1
    fi
    
    # 統合完了後の確認
    log_info "Amazon Q Developer アプリケーションを再確認中..."
    echo "統合の反映を待機しています（最大60秒）..."
    
    # 最大60秒間、5秒間隔で確認
    for i in {1..12}; do
        sleep 5
        EXISTING_APPS=$(aws sso-admin list-applications --instance-arn $INSTANCE_ARN $AWS_PROFILE_OPTION --region $REGION --query 'Applications[?contains(Name, `Amazon Q`) || contains(Name, `Q Developer`) || contains(Name, `amazon-q`)].ApplicationArn' --output text 2>/dev/null || echo "")
        
        if [ -n "$EXISTING_APPS" ]; then
            AMAZON_Q_APP_ARN=$EXISTING_APPS
            log_success "✅ Amazon Q Developer アプリケーションが確認できました！"
            
            # アプリケーション名も取得
            APP_NAME=$(aws sso-admin list-applications --instance-arn $INSTANCE_ARN $AWS_PROFILE_OPTION --region $REGION --query 'Applications[?contains(Name, `Amazon Q`) || contains(Name, `Q Developer`) || contains(Name, `amazon-q`)].Name' --output text 2>/dev/null || echo "Amazon Q Developer")
            log_info "アプリケーション名: $APP_NAME"
            log_info "アプリケーション ARN: $AMAZON_Q_APP_ARN"
            return
        fi
        
        echo "確認中... ($i/12) - $(($i * 5))秒経過"
    done
    
    # 60秒経っても見つからない場合
    log_warning "Amazon Q Developer アプリケーションがまだ確認できません"
    log_info "統合の反映に時間がかかっている可能性があります"
    echo ""
    echo "手動確認コマンド:"
    echo "  aws sso-admin list-applications --instance-arn $INSTANCE_ARN $AWS_PROFILE_OPTION --region $REGION"
    echo ""
    
    read -p "🔄 統合が完了していることを確認できましたか？続行しますか？ (y/N): " CONTINUE_ANYWAY
    
    if [[ $CONTINUE_ANYWAY =~ ^[Yy]$ ]]; then
        log_info "統合が完了していると仮定して続行します"
        AMAZON_Q_APP_ARN="pending"
    else
        log_error "Amazon Q Developer の統合確認が必要です"
        log_info "時間をおいてから再実行してください"
        exit 1
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
        # macOS の場合
        if command -v brew &> /dev/null; then
            log_info "Homebrew を使用してインストール中..."
            brew install --cask amazon-q
        else
            log_error "Homebrew がインストールされていません"
            log_info "手動でインストールしてください: https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-installing.html"
            exit 1
        fi
    elif [[ -f /etc/debian_version ]]; then
        # Ubuntu/Debian の場合
        log_info "Ubuntu/Debian 用 .deb パッケージをインストール中..."
        
        # 必要な依存関係を事前にインストール
        log_info "必要な依存関係をインストール中..."
        sudo apt-get update -qq
        sudo apt-get install -y \
            libayatana-appindicator3-1 \
            libwebkit2gtk-4.1-0 \
            wget \
            ca-certificates \
            gnupg \
            lsb-release
        
        # 一時ディレクトリでダウンロード
        TEMP_DIR=$(mktemp -d)
        cd "$TEMP_DIR"
        
        log_info ".deb パッケージをダウンロード中..."
        wget -q https://desktop-release.q.us-east-1.amazonaws.com/latest/amazon-q.deb
        
        if [ $? -ne 0 ]; then
            log_error ".deb パッケージのダウンロードに失敗しました"
            rm -rf "$TEMP_DIR"
            exit 1
        fi
        
        log_info "Amazon Q Developer CLI をインストール中..."
        sudo dpkg -i amazon-q.deb
        
        # 依存関係の問題があった場合の修正
        if [ $? -ne 0 ]; then
            log_warning "依存関係の問題を修正中..."
            sudo apt-get install -f -y
            sudo dpkg -i amazon-q.deb
            
            # それでも失敗した場合
            if [ $? -ne 0 ]; then
                log_error "Amazon Q Developer CLI のインストールに失敗しました"
                log_info "手動で以下のコマンドを実行してください:"
                echo "sudo apt-get install -y libayatana-appindicator3-1 libwebkit2gtk-4.1-0"
                echo "sudo dpkg -i $TEMP_DIR/amazon-q.deb"
                echo "sudo apt-get install -f"
                rm -rf "$TEMP_DIR"
                exit 1
            fi
        fi
        
        # クリーンアップ
        cd - > /dev/null
        rm -rf "$TEMP_DIR"
        
    elif [[ -f /etc/redhat-release ]]; then
        # RHEL/CentOS/Fedora の場合
        log_info "Red Hat 系 Linux 用インストール中..."
        curl -sSL https://amazon-q-developer-cli.s3.amazonaws.com/install.sh | bash
        
        # PATH に追加
        if ! echo $PATH | grep -q "$HOME/.local/bin"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
            export PATH="$HOME/.local/bin:$PATH"
        fi
    else
        # その他の Linux
        log_info "汎用 Linux インストールスクリプトを使用中..."
        curl -sSL https://amazon-q-developer-cli.s3.amazonaws.com/install.sh | bash
        
        # PATH に追加
        if ! echo $PATH | grep -q "$HOME/.local/bin"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
            export PATH="$HOME/.local/bin:$PATH"
        fi
    fi
    
    # インストール確認
    if command -v q &> /dev/null; then
        log_success "Amazon Q Developer CLI をインストールしました"
        log_info "バージョン: $(q --version 2>/dev/null || echo '確認できませんでした')"
    else
        log_error "Amazon Q Developer CLI のインストールに失敗しました"
        log_info "手動でインストールしてください: https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-installing.html"
        exit 1
    fi
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
    echo "7. Amazon Q Developer Pro サブスクリプション確認:"
    echo "   https://$REGION.console.aws.amazon.com/amazonq/developer/home?region=$REGION#/subscriptions"
    echo "   - ユーザー '$ADMIN_EMAIL' が 'Active' ステータスになっていることを確認"
    echo "   - まだ 'Pending' の場合は最大24時間待機"
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
    if [[ "${MFA_DISABLED:-false}" == "true" ]]; then
        echo "- MFA（多要素認証）が無効化されているため、追加の認証は不要です"
    else
        echo "- MFA が有効の場合、追加の認証デバイス設定が必要になる場合があります"
    fi
    echo "- メールボックス ($ADMIN_EMAIL) を必ず確認してください"
    echo "- 重要: Amazon Q Developer の Identity Center 統合とサブスクリプションが必要です"
    echo "- サブスクリプション追加後、最大24時間かかる場合があります"
    echo ""
    echo "🔧 トラブルシューティング:"
    echo "- InvalidGrantException が発生する場合:"
    echo "  1. Amazon Q Developer アプリケーションが作成されているか確認"
    echo "     aws sso-admin list-applications --instance-arn $INSTANCE_ARN"
    echo "  2. ユーザーサブスクリプションが Active になっているか確認"
    echo "     https://$REGION.console.aws.amazon.com/amazonq/developer/home?region=$REGION#/subscriptions"
    echo ""
    echo "- AccessDeniedException が発生する場合:"
    echo "  1. Application Assignment を確認・設定"
    echo "     https://$REGION.console.aws.amazon.com/singlesignon/applications/home?region=$REGION&tab=application-assignments#/instances/${INSTANCE_ARN##*/}/"
    echo "  2. 「Do not require assignments」に設定（開発・テスト環境推奨）"
    echo "  3. または「Require assignments」で明示的にユーザー/グループを割り当て"
    echo ""
    echo "- 共通の解決方法:"
    echo "  1. キャッシュをクリアして再ログイン"
    echo "     rm -rf ~/.aws/sso/cache/ && q login --use-device-flow"
    echo "  2. 設定変更後は数分待ってから再試行"
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
    create_amazon_q_application
    install_amazon_q_cli
    get_access_portal_url
    output_configuration
}

# スクリプト実行
main "$@"