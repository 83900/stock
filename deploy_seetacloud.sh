#!/bin/bash

# SeetaCloud部署脚本
# 环境: PyTorch 2.7.0, Python 3.12, Ubuntu 22.04, CUDA 12.8

echo "=== SeetaCloud股票预测系统部署 ==="
echo "环境: PyTorch 2.7.0 + Python 3.12 + Ubuntu 22.04 + CUDA 12.8"

# SSH连接信息
SSH_HOST="connect.bjb1.seetacloud.com"
SSH_PORT="56495"
SSH_USER="root"
SSH_PASS="fbI6qJn2IJ8A"
GITHUB_REPO="https://github.com/83900/stock.git"

# 项目目录
REMOTE_DIR="/root/stock"

echo "=== 连接到SeetaCloud服务器 ==="
echo "主机: $SSH_HOST:$SSH_PORT"
echo "用户: $SSH_USER"

# 创建SSH连接脚本
cat > connect_seetacloud.sh << 'EOF'
#!/bin/bash

# 连接参数
SSH_HOST="connect.bjb1.seetacloud.com"
SSH_PORT="56495"
SSH_USER="root"
SSH_PASS="fbI6qJn2IJ8A"
GITHUB_REPO="https://github.com/83900/stock.git"
REMOTE_DIR="/root/stock"

echo "=== 连接到SeetaCloud服务器 ==="

# 使用sshpass自动输入密码连接
sshpass -p "$SSH_PASS" ssh -p $SSH_PORT -o StrictHostKeyChecking=no $SSH_USER@$SSH_HOST << 'REMOTE_SCRIPT'

echo "=== 已连接到SeetaCloud服务器 ==="
echo "当前用户: $(whoami)"
echo "当前目录: $(pwd)"
echo "系统信息: $(uname -a)"

# 检查环境
echo "=== 环境检查 ==="
python3 --version
echo "PyTorch版本: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"
echo "CUDA版本: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo '未检测到')"
echo "CUDA可用: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'False')"

if python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null | grep -q "True"; then
    echo "GPU信息: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
    echo "GPU内存: $(python3 -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB\")' 2>/dev/null)"
fi

# 安装必要工具
echo "=== 安装必要工具 ==="
apt-get update -qq
apt-get install -y git curl wget

# 克隆或更新项目
echo "=== 获取最新项目代码 ==="
if [ -d "$REMOTE_DIR" ]; then
    echo "项目目录已存在，更新代码..."
    cd $REMOTE_DIR
    git pull origin main
else
    echo "克隆项目..."
    git clone $GITHUB_REPO $REMOTE_DIR
    cd $REMOTE_DIR
fi

echo "当前项目目录: $(pwd)"
ls -la

# 安装Python依赖
echo "=== 安装Python依赖 ==="
if [ -f "requirements.txt" ]; then
    echo "安装依赖包..."
    pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
else
    echo "未找到requirements.txt，手动安装基础依赖..."
    pip3 install pandas numpy requests flask matplotlib scikit-learn xgboost -i https://pypi.tuna.tsinghua.edu.cn/simple/
fi

# 验证依赖
echo "=== 验证依赖安装 ==="
python3 -c "
try:
    import pandas, numpy, requests, flask, matplotlib, sklearn, xgboost, torch
    print('✓ 所有依赖已安装')
    print(f'PyTorch版本: {torch.__version__}')
    print(f'CUDA可用: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'✗ 依赖缺失: {e}')
"

# 创建必要目录
echo "=== 创建目录 ==="
mkdir -p models logs
echo "已创建: models/, logs/"

# 运行优化配置
echo "=== GPU优化配置 ==="
if [ -f "rtx4090_optimization.py" ]; then
    python3 -c "
try:
    from rtx4090_optimization import setup_rtx4090_optimization
    setup_rtx4090_optimization()
    print('✓ GPU优化配置完成')
except Exception as e:
    print(f'优化配置警告: {e}')
"
fi

# 测试模块导入
echo "=== 模块测试 ==="
python3 -c "
modules = ['stock_data', 'lstm_tcn_model', 'advanced_predictor', 'prediction_models', 'web_app']
for module in modules:
    try:
        __import__(module)
        print(f'✓ {module} 导入成功')
    except Exception as e:
        print(f'✗ {module} 导入失败: {e}')
"

# 启动Web服务
echo "=== 启动Web服务 ==="
echo "启动端口: 6006"

# 检查端口占用
if netstat -tuln | grep -q ":6006 "; then
    echo "端口6006已被占用，尝试停止..."
    pkill -f "web_app.py" || true
    sleep 2
fi

# 后台启动服务
nohup python3 web_app.py > logs/web_app.log 2>&1 &
WEB_PID=$!
echo $WEB_PID > web_app.pid

echo "Web服务已启动 (PID: $WEB_PID)"
echo "日志文件: logs/web_app.log"

# 等待服务启动
sleep 3

# 检查服务状态
if ps -p $WEB_PID > /dev/null 2>&1; then
    echo "✓ Web服务运行正常"
    echo ""
    echo "=== 服务信息 ==="
    echo "内网访问: http://localhost:6006"
    echo "外网访问: 需要配置端口转发"
    echo "查看日志: tail -f logs/web_app.log"
    echo "停止服务: kill $WEB_PID"
    echo ""
    echo "=== 端口转发设置 ==="
    echo "如需外网访问，请在本地运行："
    echo "ssh -p 56495 -L 8080:localhost:6006 root@connect.bjb1.seetacloud.com"
    echo "然后访问: http://localhost:8080"
else
    echo "✗ Web服务启动失败"
    echo "查看错误日志: cat logs/web_app.log"
fi

echo ""
echo "=== 部署完成 ==="
echo "项目目录: $REMOTE_DIR"
echo "服务状态: $(ps -p $WEB_PID > /dev/null 2>&1 && echo '运行中' || echo '已停止')"

REMOTE_SCRIPT

EOF

# 检查sshpass是否安装
if ! command -v sshpass &> /dev/null; then
    echo "正在安装sshpass..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install sshpass
        else
            echo "请先安装Homebrew，然后运行: brew install sshpass"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update && sudo apt-get install -y sshpass
    else
        echo "请手动安装sshpass"
        exit 1
    fi
fi

# 执行连接脚本
chmod +x connect_seetacloud.sh
echo "=== 开始部署 ==="
./connect_seetacloud.sh

echo "=== 部署脚本执行完成 ==="