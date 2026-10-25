#!/bin/bash
# Launch NETRA on a small Graviton (arm64) EC2 instance. UNTESTED: written without an AWS account.
# Usage: REPO_URL=https://github.com/<you>/netra.git MY_IP=$(curl -s https://checkip.amazonaws.com) ./aws_deploy.sh
set -euo pipefail
: "${REPO_URL:?}"; : "${MY_IP:?}"
REGION=${REGION:-us-east-1}
SG=$(aws ec2 create-security-group --region "$REGION" --group-name netra-sg --description "NETRA UI" --query GroupId --output text)
# Least privilege: the UI is reachable only from your own IP, nothing else is open.
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" --protocol tcp --port 8501 --cidr "$MY_IP/32"
AMI=$(aws ssm get-parameter --region "$REGION" --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 --query Parameter.Value --output text)
aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type t4g.small --security-group-ids "$SG" \
  --metadata-options HttpTokens=required --user-data "$(sed "s|\${REPO_URL:?set REPO_URL}|$REPO_URL|" "$(dirname "$0")/user_data.sh")" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=netra}]' --query 'Instances[0].InstanceId' --output text
echo "Open http://<public-ip>:8501 from $MY_IP once the instance is running. Terminate it when you are done (it bills hourly)."
