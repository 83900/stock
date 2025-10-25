# AutoDL 部署指南

## 环境信息
- **平台**: AutoDL
- **GPU**: RTX 4090
- **CUDA版本**: 13.0
- **驱动版本**: 580.76.05
- **PyTorch版本**: 2.0.0
- **Python版本**: 3.8
- **操作系统**: Ubuntu 20.04

## 快速部署步骤

### 1. 环境准备
AutoDL已预装PyTorch 2.0.0和CUDA 13.0，无需额外安装PyTorch。

```bash
# 检查环境
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'当前GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

### 2. 安装项目依赖
```bash
# 安装基础依赖（排除torch，因为已预装）
pip install pandas>=1.3.0 requests>=2.25.0 adata>=1.0.0 flask>=2.0.0 matplotlib>=3.3.0 numpy>=1.20.0 scikit-learn>=1.0.0 xgboost>=1.6.0
```

### 3. 项目部署
```bash
# 克隆或上传项目文件到AutoDL
# 进入项目目录
cd /root/autodl-tmp/stock

# 运行环境检查
python quick_start.py

# 启动Web服务
python web_app.py
```

### 4. 端口配置
AutoDL需要配置端口映射：

```python
# 在web_app.py中确保使用正确的端口
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006, debug=False)  # AutoDL推荐端口
```

### 5. 访问应用
- **内网访问**: `http://localhost:6006`
- **外网访问**: 通过AutoDL提供的公网地址

## AutoDL优化建议

### RTX 4090 GPU优化
```python
# 针对RTX 4090的优化配置
import torch

# 启用TensorFloat-32 (TF32) 加速 - RTX 4090支持
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 启用Flash Attention (如果可用)
torch.backends.cuda.enable_flash_sdp(True)

# 设置最佳的cuDNN benchmark
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# GPU内存优化 - RTX 4090有24GB显存
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    # 设置内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%显存
```

### 高性能训练配置
```python
# 针对大显存的批次大小优化
batch_size = 128  # RTX 4090可以支持更大的批次
learning_rate = 0.001 * (batch_size / 32)  # 根据批次大小调整学习率

# 混合精度训练 - RTX 4090原生支持
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

# 训练循环示例
for batch in dataloader:
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 数据存储
```bash
# 使用AutoDL的持久化存储
# 数据文件存放在 /root/autodl-tmp/ 目录下
# 模型文件存放在 /root/autodl-tmp/models/ 目录下
mkdir -p /root/autodl-tmp/models
```

### 性能监控
```bash
# 监控GPU使用情况
nvidia-smi

# 监控系统资源
htop
```

## 常见问题解决

### 1. 依赖包安装失败
```bash
# 使用清华源加速
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ package_name
```

### 2. GPU内存不足
由于RTX 4090拥有24GB显存，内存不足问题较少见，但仍需注意：

```python
# 如果遇到内存问题，可以调整以下参数
batch_size = 64  # 从128减少到64
gradient_accumulation_steps = 2  # 使用梯度累积

# 启用内存优化
torch.cuda.empty_cache()
torch.cuda.synchronize()

# 检查内存使用
print(f"已分配内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"缓存内存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### 3. 网络连接问题
```python
# 在stock_data.py中增加重试机制
import time
import requests

def fetch_with_retry(url, max_retries=3):
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            return response
        except Exception as e:
            if i == max_retries - 1:
                raise e
            time.sleep(2 ** i)  # 指数退避
```

## 自动启动脚本

创建 `start_service.sh`:
```bash
#!/bin/bash
cd /root/autodl-tmp/stock

# 检查环境
echo "检查环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 启动服务
echo "启动股票预测服务..."
nohup python web_app.py > service.log 2>&1 &

echo "服务已启动，日志文件: service.log"
echo "访问地址: http://localhost:6006"
```

```bash
# 给脚本执行权限
chmod +x start_service.sh

# 运行脚本
./start_service.sh
```

## 注意事项

1. **数据持久化**: 重要数据和模型文件请存储在 `/root/autodl-tmp/` 目录
2. **端口选择**: 建议使用6006端口，AutoDL对此端口支持较好
3. **资源监控**: 定期检查GPU和内存使用情况
4. **网络稳定性**: 数据获取可能受网络影响，建议添加重试机制
5. **定期备份**: 重要的训练模型和数据请及时备份

## 性能基准测试

在AutoDL RTX 4090环境下的预期性能：
- **数据获取**: 3-5秒/100只股票 (网络优化)
- **模型训练**: 30-60秒/1000个epoch (大批次+混合精度)
- **预测速度**: <0.1秒/单只股票 (GPU加速)
- **Web响应**: <1秒/请求
- **批量预测**: 100只股票/秒 (并行处理)

### RTX 4090性能优势
- **24GB大显存**: 支持更大的模型和批次大小
- **TF32加速**: 自动加速深度学习计算
- **高带宽**: 1008 GB/s内存带宽
- **CUDA核心**: 16384个CUDA核心提供强大并行计算能力