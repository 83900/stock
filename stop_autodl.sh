#!/bin/bash

echo "=== 停止AutoDL股票预测服务 ==="

# 设置工作目录
cd /root/autodl-tmp/stock 2>/dev/null || cd $(dirname "$0")

# 检查PID文件
if [ -f "logs/service.pid" ]; then
    SERVICE_PID=$(cat logs/service.pid)
    echo "找到服务PID: $SERVICE_PID"
    
    # 检查进程是否存在
    if kill -0 $SERVICE_PID 2>/dev/null; then
        echo "正在停止服务..."
        kill $SERVICE_PID
        
        # 等待进程结束
        sleep 2
        
        # 强制结束（如果需要）
        if kill -0 $SERVICE_PID 2>/dev/null; then
            echo "强制结束进程..."
            kill -9 $SERVICE_PID
        fi
        
        echo "✓ 服务已停止"
        rm logs/service.pid
    else
        echo "服务进程不存在"
        rm logs/service.pid
    fi
else
    echo "未找到PID文件，尝试查找相关进程..."
    
    # 查找Python web服务进程
    PIDS=$(pgrep -f "python.*web_app.py")
    
    if [ -n "$PIDS" ]; then
        echo "找到相关进程: $PIDS"
        echo "正在停止..."
        echo $PIDS | xargs kill
        sleep 2
        
        # 检查是否还有残留进程
        REMAINING=$(pgrep -f "python.*web_app.py")
        if [ -n "$REMAINING" ]; then
            echo "强制结束残留进程: $REMAINING"
            echo $REMAINING | xargs kill -9
        fi
        
        echo "✓ 服务已停止"
    else
        echo "未找到运行中的服务"
    fi
fi

# 清理端口占用（如果需要）
PORT_PID=$(lsof -ti:6006 2>/dev/null)
if [ -n "$PORT_PID" ]; then
    echo "清理端口6006占用: $PORT_PID"
    kill $PORT_PID 2>/dev/null
fi

echo "=== 停止完成 ==="