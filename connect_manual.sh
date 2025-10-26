#!/bin/bash

# 手动SSH连接到SeetaCloud服务器
# 使用方法: ./connect_manual.sh

SSH_HOST="connect.bjb1.seetacloud.com"
SSH_PORT="56495"
SSH_USER="root"
SSH_PASS="fbI6qJn2IJ8A"

echo "=== 连接到SeetaCloud服务器 ==="
echo "主机: $SSH_HOST:$SSH_PORT"
echo "用户: $SSH_USER"
echo "密码: $SSH_PASS"
echo ""

# 方法1: 使用sshpass自动输入密码
if command -v sshpass &> /dev/null; then
    echo "使用sshpass自动连接..."
    sshpass -p "$SSH_PASS" ssh -p $SSH_PORT -o StrictHostKeyChecking=no $SSH_USER@$SSH_HOST
else
    echo "sshpass未安装，请手动输入密码"
    echo "连接命令: ssh -p $SSH_PORT $SSH_USER@$SSH_HOST"
    echo "密码: $SSH_PASS"
    echo ""
    ssh -p $SSH_PORT $SSH_USER@$SSH_HOST
fi