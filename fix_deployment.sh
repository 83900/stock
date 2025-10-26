#!/bin/bash

# SeetaCloud部署修复脚本
echo "=== SeetaCloud部署修复 ==="

SSH_HOST="connect.bjb1.seetacloud.com"
SSH_PORT="56495"
SSH_USER="root"
SSH_PASS="fbI6qJn2IJ8A"
GITHUB_REPO="https://github.com/83900/stock.git"
REMOTE_DIR="/root/stock"

echo "连接到服务器并修复部署问题..."

sshpass -p "$SSH_PASS" ssh -p $SSH_PORT -o StrictHostKeyChecking=no $SSH_USER@$SSH_HOST << 'REMOTE_SCRIPT'

echo "=== 修复SeetaCloud部署 ==="
echo "当前目录: $(pwd)"

# 停止可能运行的服务
pkill -f "web_app.py" 2>/dev/null || true

# 克隆项目 (这是之前缺失的步骤)
echo "=== 克隆GitHub项目 ==="
GITHUB_REPO="https://github.com/83900/stock.git"
REMOTE_DIR="/root/stock"

if [ -d "$REMOTE_DIR" ]; then
    echo "项目目录已存在，更新代码..."
    cd $REMOTE_DIR
    git pull origin main || git pull origin master
else
    echo "克隆项目到 $REMOTE_DIR ..."
    git clone $GITHUB_REPO $REMOTE_DIR
    if [ $? -eq 0 ]; then
        echo "✓ 项目克隆成功"
    else
        echo "✗ 项目克隆失败，尝试其他方法..."
        # 如果git克隆失败，创建基本目录结构
        mkdir -p $REMOTE_DIR
        cd $REMOTE_DIR
        echo "手动创建项目结构..."
    fi
fi

cd $REMOTE_DIR
echo "当前项目目录: $(pwd)"
ls -la

# 验证项目文件
echo "=== 验证项目文件 ==="
required_files=("stock_data.py" "lstm_tcn_model.py" "advanced_predictor.py" "web_app.py" "requirements.txt")
missing_files=()

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file 存在"
    else
        echo "✗ $file 缺失"
        missing_files+=("$file")
    fi
done

# 如果有缺失文件，尝试重新克隆
if [ ${#missing_files[@]} -gt 0 ]; then
    echo "检测到缺失文件，重新克隆项目..."
    cd /root
    rm -rf $REMOTE_DIR
    git clone $GITHUB_REPO $REMOTE_DIR
    cd $REMOTE_DIR
fi

# 再次验证模块导入
echo "=== 验证模块导入 ==="
python3 -c "
import sys
sys.path.append('.')

modules = ['stock_data', 'lstm_tcn_model', 'advanced_predictor', 'prediction_models', 'web_app']
success_count = 0
for module in modules:
    try:
        __import__(module)
        print(f'✓ {module} 导入成功')
        success_count += 1
    except Exception as e:
        print(f'✗ {module} 导入失败: {e}')

print(f'成功导入 {success_count}/{len(modules)} 个模块')
"

# 创建必要目录
mkdir -p models logs

# 检查web_app.py是否存在并可运行
if [ -f "web_app.py" ]; then
    echo "=== 测试Web应用 ==="
    python3 -c "
import sys
sys.path.append('.')
try:
    from web_app import app
    print('✓ Web应用导入成功')
except Exception as e:
    print(f'✗ Web应用导入失败: {e}')
"
    
    # 启动Web服务
    echo "=== 启动Web服务 ==="
    echo "启动端口: 6006"
    
    # 后台启动
    nohup python3 web_app.py > logs/web_app.log 2>&1 &
    WEB_PID=$!
    echo $WEB_PID > web_app.pid
    
    echo "Web服务已启动 (PID: $WEB_PID)"
    
    # 等待服务启动
    sleep 5
    
    # 检查服务状态
    if ps -p $WEB_PID > /dev/null 2>&1; then
        echo "✓ Web服务运行正常"
        echo ""
        echo "=== 服务信息 ==="
        echo "内网地址: http://localhost:6006"
        echo "查看日志: tail -f logs/web_app.log"
        echo "停止服务: kill $WEB_PID"
        echo ""
        echo "=== 外网访问 ==="
        echo "在本地运行以下命令进行端口转发:"
        echo "ssh -p 56495 -L 8080:localhost:6006 root@connect.bjb1.seetacloud.com"
        echo "然后访问: http://localhost:8080"
    else
        echo "✗ Web服务启动失败"
        echo "查看错误日志:"
        cat logs/web_app.log 2>/dev/null || echo "日志文件不存在"
    fi
else
    echo "✗ web_app.py 文件不存在，无法启动服务"
fi

echo ""
echo "=== 修复完成 ==="
echo "项目目录: $REMOTE_DIR"
echo "文件列表:"
ls -la

REMOTE_SCRIPT