# 智能股票交易分析系统 - 部署指南

## 📋 部署前检查列表

### 🔧 系统要求检查

#### 硬件要求
- [ ] **GPU**: NVIDIA RTX 4090 (推荐) 或 RTX 3080以上
- [ ] **显存**: ≥ 12GB VRAM
- [ ] **内存**: ≥ 16GB RAM  
- [ ] **存储**: ≥ 5GB 可用空间
- [ ] **网络**: 稳定的互联网连接

#### 软件要求
- [ ] **操作系统**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- [ ] **Python**: 3.8 - 3.11 (推荐 3.10)
- [ ] **CUDA**: 11.8+ (如使用GPU)
- [ ] **Git**: 最新版本

### 📦 环境准备

#### 1. 检查Python版本
```bash
python --version
# 应显示 Python 3.8.x 到 3.11.x
```

#### 2. 检查CUDA版本 (GPU用户)
```bash
nvidia-smi
# 检查CUDA版本和GPU状态
```

#### 3. 检查网络连接
```bash
ping finance.yahoo.com
ping qt.gtimg.cn
# 确保能访问数据源
```

## 🚀 快速部署步骤

### 步骤1: 获取代码
```bash
# 方法1: 从GitHub克隆 (如果已上传)
git clone https://github.com/your-username/smart-trading-analyzer.git
cd smart-trading-analyzer

# 方法2: 直接下载解压
# 下载项目压缩包并解压到目标目录
```

### 步骤2: 创建虚拟环境 (强烈推荐)
```bash
# 创建虚拟环境
python -m venv trading_env

# 激活虚拟环境
# Windows:
trading_env\Scripts\activate
# macOS/Linux:
source trading_env/bin/activate
```

### 步骤3: 安装依赖
```bash
# 基础依赖安装
pip install -r requirements.txt

# GPU用户额外安装CUDA版PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 步骤4: 验证安装
```bash
# 运行环境检查
python -c "
import torch
import pandas as pd
import numpy as np
print('✅ 基础包安装成功')
if torch.cuda.is_available():
    print(f'✅ GPU可用: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ 未检测到GPU，将使用CPU')
"
```

### 步骤5: 首次运行测试
```bash
# 快速测试 (推荐)
python run_trading_analysis.py
# 选择选项 1 - 快速分析模式
```

## 📊 详细部署配置

### 🎮 GPU优化配置

#### RTX 4090用户
```bash
# 设置环境变量优化GPU性能
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 内存优化
```bash
# 对于大内存系统
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### 🌐 网络配置

#### 代理设置 (如需要)
```bash
# 设置代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=https://your-proxy:port
```

#### 数据源备用配置
如果Yahoo Finance访问受限，系统会自动切换到腾讯财经API。

### 📁 目录结构验证

部署完成后，确认以下文件存在：
```
smart-trading-analyzer/
├── README.md                    # 项目说明
├── deploy.md                    # 部署指南 (本文件)
├── requirements.txt             # 依赖列表
├── smart_trading_analyzer.py    # 主分析器
├── run_trading_analysis.py      # 快速启动脚本
├── real_stock_data_fetcher.py   # 数据获取器
├── get_kedaxunfei_data.py       # 单股分析
├── improved_gpu_train.py        # GPU训练脚本
├── stock_data.py               # 数据管理
└── rtx4090_optimization.py     # GPU优化配置
```

## 🔍 部署验证测试

### 测试1: 环境检查
```bash
python run_trading_analysis.py
# 应显示环境检查通过
```

### 测试2: 数据获取测试
```bash
python get_kedaxunfei_data.py
# 应成功获取科大讯飞数据
```

### 测试3: GPU性能测试 (GPU用户)
```bash
python -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    print('✅ GPU计算测试通过')
else:
    print('⚠️ 使用CPU模式')
"
```

### 测试4: 完整分析测试
```bash
# 运行快速分析 (5-10分钟)
python run_trading_analysis.py
# 选择选项 1，等待完成
```

## 🚨 常见问题解决

### 问题1: CUDA版本不匹配
```bash
# 症状: RuntimeError: CUDA version mismatch
# 解决: 重新安装匹配的PyTorch版本
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题2: 内存不足
```bash
# 症状: CUDA out of memory
# 解决: 减少批处理大小
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

### 问题3: 网络连接失败
```bash
# 症状: 无法获取股票数据
# 解决: 检查网络连接和防火墙设置
ping finance.yahoo.com
```

### 问题4: 依赖包冲突
```bash
# 症状: 包版本冲突
# 解决: 使用虚拟环境重新安装
rm -rf trading_env
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 问题5: 权限问题
```bash
# 症状: Permission denied
# 解决: 检查文件权限
chmod +x *.py
```

## 📈 性能优化建议

### CPU优化
```bash
# 设置线程数
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### GPU优化
```bash
# 启用混合精度训练
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 内存优化
```bash
# 限制内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## 🔄 更新和维护

### 定期更新
```bash
# 更新代码 (如果使用Git)
git pull origin main

# 更新依赖
pip install -r requirements.txt --upgrade
```

### 数据清理
```bash
# 清理旧的数据文件 (可选)
find . -name "*.csv" -mtime +30 -delete
find . -name "*.json" -mtime +30 -delete
```

### 日志管理
```bash
# 清理日志文件 (如果有)
find . -name "*.log" -mtime +7 -delete
```

## 📊 监控和日志

### 系统监控
```bash
# 监控GPU使用率
nvidia-smi -l 1

# 监控内存使用
htop
```

### 性能基准
- **快速分析**: 5-10分钟 (10支股票)
- **完整分析**: 30-60分钟 (50支股票)
- **单股分析**: 1-2分钟

## 🛡️ 安全注意事项

### 数据安全
- [ ] 不要在公共网络运行
- [ ] 定期备份分析结果
- [ ] 不要泄露API密钥 (如果使用)

### 系统安全
- [ ] 保持系统和依赖更新
- [ ] 使用虚拟环境隔离
- [ ] 定期检查异常进程

## 📞 技术支持

### 自助诊断
1. 检查Python版本和依赖
2. 验证GPU驱动和CUDA
3. 测试网络连接
4. 查看错误日志

### 性能调优
1. 根据硬件调整批处理大小
2. 优化内存使用设置
3. 调整线程数配置

## ✅ 部署完成确认

部署成功的标志：
- [ ] 环境检查全部通过
- [ ] 能够成功获取股票数据
- [ ] GPU正常工作 (如适用)
- [ ] 快速分析能正常完成
- [ ] 生成分析报告

## 🎯 下一步

部署完成后，您可以：
1. **运行快速分析** - 熟悉系统功能
2. **查看分析报告** - 了解输出格式
3. **调整参数** - 根据需求优化
4. **定期运行** - 获取最新分析

---

**⚠️ 重要提醒**: 本系统仅供学习研究使用，不构成投资建议。股票投资有风险，请谨慎决策！