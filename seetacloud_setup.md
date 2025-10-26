# SeetaCloud 部署指南

## 环境信息
- **平台**: SeetaCloud
- **PyTorch**: 2.7.0
- **Python**: 3.12
- **系统**: Ubuntu 22.04
- **CUDA**: 12.8
- **SSH**: ssh -p 56495 root@connect.bjb1.seetacloud.com
- **密码**: fbI6qJn2IJ8A
- **项目仓库**: https://github.com/83900/stock.git

## 快速部署

### 方法1: 自动部署脚本
```bash
# 在本地运行
chmod +x deploy_seetacloud.sh
./deploy_seetacloud.sh
```

### 方法2: 手动部署
```bash
# 1. SSH连接到服务器
ssh -p 56495 root@connect.bjb1.seetacloud.com
# 输入密码: fbI6qJn2IJ8A

# 2. 克隆项目
git clone https://github.com/83900/stock.git /root/stock
cd /root/stock

# 3. 安装依赖
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 4. 启动服务
python3 web_app.py
```

## 环境优化配置

### PyTorch 2.7.0 + CUDA 12.8 优化
```python
import torch

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # 启用优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # GPU信息
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
```

### 推荐配置
```python
# 针对PyTorch 2.7.0的优化设置
BATCH_SIZE = 64  # 根据GPU内存调整
LEARNING_RATE = 0.001
EPOCHS = 100

# 混合精度训练 (PyTorch 2.7.0原生支持)
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()
```

## 网络访问配置

### 内网访问
- 服务地址: `http://localhost:6006`
- 仅限服务器内部访问

### 外网访问 (端口转发)
```bash
# 在本地终端运行
ssh -p 56495 -L 8080:localhost:6006 root@connect.bjb1.seetacloud.com

# 然后在本地浏览器访问
http://localhost:8080
```

### SeetaCloud端口映射
1. 登录SeetaCloud控制台
2. 找到实例管理
3. 配置端口映射: 6006 -> 外部端口
4. 通过分配的外网地址访问

## 服务管理

### 启动服务
```bash
cd /root/stock
python3 web_app.py
```

### 后台运行
```bash
nohup python3 web_app.py > logs/web_app.log 2>&1 &
echo $! > web_app.pid
```

### 停止服务
```bash
# 方法1: 使用PID文件
kill $(cat web_app.pid)

# 方法2: 查找进程
pkill -f "web_app.py"

# 方法3: 停止端口占用
lsof -ti:6006 | xargs kill -9
```

### 查看日志
```bash
# 实时日志
tail -f logs/web_app.log

# 错误日志
grep -i error logs/web_app.log
```

## 性能监控

### GPU监控
```bash
# 安装nvidia-smi (如果未安装)
nvidia-smi

# 实时监控
watch -n 1 nvidia-smi
```

### 系统资源
```bash
# CPU和内存
htop

# 磁盘使用
df -h

# 网络连接
netstat -tuln | grep 6006
```

## 常见问题解决

### 1. 依赖安装失败
```bash
# 更新pip
pip3 install --upgrade pip

# 使用国内镜像
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 单独安装问题包
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. GPU不可用
```bash
# 检查CUDA
nvcc --version
nvidia-smi

# 重新安装PyTorch
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. 端口被占用
```bash
# 查看端口占用
lsof -i :6006

# 强制释放端口
sudo fuser -k 6006/tcp
```

### 4. 内存不足
```bash
# 检查内存使用
free -h

# 清理缓存
sync && echo 3 > /proc/sys/vm/drop_caches

# 调整批次大小
# 在代码中设置较小的batch_size
```

## 自动化脚本

### 启动脚本 (start_seetacloud.sh)
```bash
#!/bin/bash
cd /root/stock
source venv/bin/activate 2>/dev/null || true
python3 web_app.py
```

### 停止脚本 (stop_seetacloud.sh)
```bash
#!/bin/bash
pkill -f "web_app.py"
echo "服务已停止"
```

### 重启脚本 (restart_seetacloud.sh)
```bash
#!/bin/bash
./stop_seetacloud.sh
sleep 2
./start_seetacloud.sh
```

## 数据备份

### 模型文件备份
```bash
# 创建备份目录
mkdir -p /root/backup/models

# 备份模型
cp -r models/* /root/backup/models/

# 定时备份 (添加到crontab)
0 2 * * * cp -r /root/stock/models/* /root/backup/models/
```

### 日志轮转
```bash
# 创建日志轮转配置
cat > /etc/logrotate.d/stock << EOF
/root/stock/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF
```

## 性能基准

### 预期性能 (SeetaCloud环境)
- **数据获取**: 2-5秒
- **模型训练**: 30-60秒 (100 epochs)
- **预测速度**: <1秒
- **Web响应**: <500ms

### 优化建议
1. **使用SSD存储**: 提高I/O性能
2. **启用GPU加速**: 确保CUDA正常工作
3. **调整批次大小**: 根据GPU内存优化
4. **使用缓存**: 减少重复计算
5. **异步处理**: 提高并发性能

## 监控和告警

### 服务健康检查
```bash
#!/bin/bash
# health_check.sh
if ! curl -f http://localhost:6006/health 2>/dev/null; then
    echo "服务异常，尝试重启..."
    ./restart_seetacloud.sh
fi
```

### 定时任务
```bash
# 添加到crontab
crontab -e

# 每5分钟检查服务状态
*/5 * * * * /root/stock/health_check.sh

# 每天凌晨2点备份模型
0 2 * * * cp -r /root/stock/models/* /root/backup/models/
```

## 联系信息
- **项目仓库**: https://github.com/83900/stock.git
- **问题反馈**: 通过GitHub Issues
- **更新日志**: 查看CHANGELOG.md