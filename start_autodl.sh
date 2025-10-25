#!/bin/bash

# AutoDL股票预测系统启动脚本
# 适用于RTX 4090 + CUDA 13.0环境

echo "=== AutoDL股票预测系统启动 ==="
echo "环境: RTX 4090 + CUDA 13.0 + PyTorch 2.0.0"

# 设置工作目录
WORK_DIR="/root/stock"
cd $WORK_DIR || { echo "错误: 无法进入工作目录 $WORK_DIR"; exit 1; }

echo "当前工作目录: $(pwd)"

# 检查Python和PyTorch环境
echo "=== 环境检查 ==="
python3 --version
echo "PyTorch版本: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA版本: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "CUDA可用: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

if python3 -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "GPU信息: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "GPU内存: $(python3 -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB\")')"
    
    # RTX 4090优化设置
    echo "=== RTX 4090优化设置 ==="
    python3 -c "
import torch
if torch.cuda.is_available():
    # 启用TF32加速
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print('已启用: TF32加速, cuDNN优化')
    
    # 显示GPU内存
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU内存: {gpu_memory:.1f} GB')
"
fi

# 安装依赖
echo "=== 安装依赖 ==="
if [ -f "requirements_autodl.txt" ]; then
    echo "使用AutoDL专用依赖文件..."
    pip install -r requirements_autodl.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
elif [ -f "requirements.txt" ]; then
    echo "使用标准依赖文件 (跳过torch)..."
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --ignore-installed torch
else
    echo "警告: 未找到依赖文件"
fi

# 验证依赖安装
echo "=== 验证依赖 ==="
python3 -c "
try:
    import pandas, numpy, requests, flask, matplotlib, sklearn, xgboost
    print('✓ 所有依赖已安装')
except ImportError as e:
    print(f'✗ 依赖缺失: {e}')
"

# 创建必要目录
echo "=== 创建目录 ==="
mkdir -p models logs
echo "已创建: models/, logs/"

# 运行RTX 4090优化配置
echo "=== RTX 4090性能优化 ==="
if [ -f "rtx4090_optimization.py" ]; then
    python3 rtx4090_optimization.py
else
    echo "未找到RTX 4090优化模块，使用默认设置"
fi

# 快速系统检查
echo "=== 系统检查 ==="
python3 -c "
try:
    from stock_data import StockDataFetcher
    print('✓ 数据获取模块正常')
except Exception as e:
    print(f'✗ 数据模块错误: {e}')

try:
    from lstm_tcn_model import LSTMTCNPredictor
    print('✓ 模型模块正常')
except Exception as e:
    print(f'✗ 模型模块错误: {e}')
"

# 启动Web服务
echo "=== 启动Web服务 ==="
echo "启动端口: 6006"
echo "访问地址: http://localhost:6006"

# 后台运行并保存PID
nohup python3 web_app.py > logs/web_app.log 2>&1 &
WEB_PID=$!
echo $WEB_PID > web_app.pid

echo "Web服务已启动 (PID: $WEB_PID)"
echo "日志文件: logs/web_app.log"

# 等待服务启动
sleep 3

# 检查服务状态
if ps -p $WEB_PID > /dev/null; then
    echo "✓ Web服务运行正常"
    echo ""
    echo "=== 服务信息 ==="
    echo "访问地址: http://localhost:6006"
    echo "查看日志: tail -f logs/web_app.log"
    echo "停止服务: ./stop_autodl.sh"
    echo ""
    echo "=== AutoDL端口映射 ==="
    echo "请在AutoDL控制台设置端口映射:"
    echo "容器端口: 6006 -> 自定义端口"
    echo "然后通过 https://xxx.autodl.com:端口 访问"
else
    echo "✗ Web服务启动失败"
    echo "请查看日志: cat logs/web_app.log"
    exit 1
fi