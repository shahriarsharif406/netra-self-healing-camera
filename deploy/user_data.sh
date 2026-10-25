#!/bin/bash
# EC2 cloud-init script (Amazon Linux 2023). UNTESTED: written without an AWS account.
set -euo pipefail
dnf install -y docker git
systemctl enable --now docker
git clone "${REPO_URL:?set REPO_URL}" /opt/netra
cd /opt/netra
docker build -t netra .
# Run as a non-root container, restart on failure, logs go to the instance's journald/CloudWatch agent.
docker run -d --restart unless-stopped --name netra -p 8501:8501 --read-only --tmpfs /tmp netra
