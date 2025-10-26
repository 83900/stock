#!/bin/bash

echo "=== 从SeetaCloud下载训练结果 ==="

# SSH连接信息
SSH_HOST="root@connect.seetacloud.com"
SSH_PORT="36067"
REMOTE_DIR="/root/stock"
LOCAL_DIR="./remote_results"

# 创建本地目录
mkdir -p "$LOCAL_DIR"

echo "连接到远程服务器并下载文件..."

# 下载模型文件
echo "下载模型文件..."
scp -P $SSH_PORT "$SSH_HOST:$REMOTE_DIR/models/simple_model.pth" "$LOCAL_DIR/" 2>/dev/null

# 下载训练历史
echo "下载训练历史..."
scp -P $SSH_PORT "$SSH_HOST:$REMOTE_DIR/training_history.json" "$LOCAL_DIR/" 2>/dev/null
scp -P $SSH_PORT "$SSH_HOST:$REMOTE_DIR/training_history.png" "$LOCAL_DIR/" 2>/dev/null

# 下载日志文件
echo "下载日志文件..."
scp -P $SSH_PORT "$SSH_HOST:$REMOTE_DIR/training.log" "$LOCAL_DIR/" 2>/dev/null

# 下载简化训练脚本（用于参考）
echo "下载训练脚本..."
scp -P $SSH_PORT "$SSH_HOST:$REMOTE_DIR/simple_train.py" "$LOCAL_DIR/" 2>/dev/null

echo "检查下载的文件..."
ls -la "$LOCAL_DIR/"

echo "=== 下载完成 ==="

# 显示训练历史内容
if [ -f "$LOCAL_DIR/training_history.json" ]; then
    echo ""
    echo "=== 训练历史摘要 ==="
    cat "$LOCAL_DIR/training_history.json"
fi

# 显示日志内容
if [ -f "$LOCAL_DIR/training.log" ]; then
    echo ""
    echo "=== 训练日志 ==="
    cat "$LOCAL_DIR/training.log"
fi