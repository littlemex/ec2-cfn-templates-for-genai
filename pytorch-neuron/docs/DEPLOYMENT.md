# TRN1 (Neuron) CloudFormation デプロイメント検証ガイド

## デプロイ前の確認事項

### 1. SSMパラメータ検証
```bash
# Neuron AMI IDの取得確認
aws ssm get-parameter \
  --name "/aws/service/neuron/dlami/multi-framework/ubuntu-22.04/latest/image_id" \
  --region us-east-1 \
  --query "Parameter.Value" \
  --output text
```

### 2. TRN1インスタンス利用可能性確認
```bash
# TRN1インスタンスタイプの利用可能性確認
aws ec2 describe-instance-type-offerings \
  --location-type availability-zone \
  --filters Name=instance-type,Values=trn1.2xlarge \
  --region us-east-1
```

## デプロイ手順

### Step 1: CloudFormation スタックデプロイ
```bash
bash cfn_manager.sh create
```

### Step 2: デプロイ状況確認
```bash
# create 時に monitor 用のコマンドが表示されますのでそれを利用してください。
bash cfn_manager.sh monitor -n $STACK_NAME -r $REGION"
```

### Step 3: Code Server アクセス

Trn1 常に Code Server を起動して CloudFront からアクセスすることができます。セキュリティ上の懸念がある場合は追加のセキュリティ対策を実施して利用ください。
monitor コマンドで正常に作成が完了すると CloudFront URL とパスワードが表示されるため、それを用いて Code Server にアクセスします。

### Step 4: 環境チェック

以下のコマンドで正常に Neuron デバイス、ライブラリ等が導入されているかを確認することができます。

```bash
cd /work/pytorch-neuron
bash scripts/check_neuron_env.sh
```

### Step 5: 動作検証

```bash
# NeuronX Pytorch の仮想環境をアクティベートします
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate

# vmap に関する動作確認を実施します。
python test_neuron_vmap.py
```

## 安全な接続

### 1. SSM接続とNeuron環境確認
```bash
# SSM セッション開始
aws ssm start-session --target $INSTANCE_ID --region us-east-1

# インスタンス内で以下を実行:
sudo su - coder

# Neuronデバイス確認
ls -la /dev/neuron*

# Neuron SDK確認
python3 -c "import torch_neuron; print('PyTorch Neuron version:', torch_neuron.__version__)"

# 仮想環境確認
ls -la /opt/aws_neuronx_venv_pytorch*
```