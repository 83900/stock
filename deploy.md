# 智能股票交易分析系统 - 部署指南

## 📋 部署前检查列表

### 🔧 系统要求检查

#### 硬件要求
- [ ] **GPU**: NVIDIA RTX 4090 (推荐) 或 RTX 3080以上
- [ ] **显存**: ≥ 12GB VRAM
- [ ] **内存**: ≥ 16GB RAM  
- [ ] **存储**: ≥ 5GB 可用空间
- [ ] **网络**: 稳定的互联网连接
- [ ] **代理工具**: 可选，用于访问Yahoo Finance

#### 软件要求
- [ ] **操作系统**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- [ ] **Python**: 3.8 - 3.11 (推荐 3.10)
- [ ] **CUDA**: 11.8+ (如使用GPU)
- [ ] **Git**: 最新版本
- [ ] **Tushare账号**: 推荐注册获取免费token

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
# 先安装基础依赖（不包含PyTorch）
pip install -r requirements.txt

# 单独安装PyTorch（推荐使用官方源或清华源）
# 方法1: 使用官方源（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 方法2: 使用清华源（国内用户推荐）
pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 方法3: 如果网络较慢，可以分别安装
pip install torch==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torchvision==0.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install torchaudio==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/
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

##### 本地环境代理
如果在本地运行，可以配置代理：
```bash
# 设置代理环境变量
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
```

##### SSH远程服务器代理配置
如果通过SSH控制云服务器，有以下几种方案：

**方案1: SSH隧道转发（推荐）**
```bash
# 在本地终端建立SSH隧道，将云服务器的7890端口转发到本地代理
ssh -L 7890:localhost:7890 user@your-server-ip

# 然后在云服务器上设置代理
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
```

**方案2: 云服务器安装代理工具**
```bash
# 在云服务器上安装代理工具（如v2ray、clash等）
# 然后设置相应的代理端口
export https_proxy=http://127.0.0.1:代理端口
export http_proxy=http://127.0.0.1:代理端口
```

**方案3: 使用公共代理服务**
```bash
# 使用免费或付费的HTTP代理服务
export https_proxy=http://proxy-server:port
export http_proxy=http://proxy-server:port
```

**方案4: 优先使用Tushare（推荐）**
由于已集成Tushare数据源，建议主要依赖Tushare获取数据，无需代理：
```bash
# 直接运行，Tushare不需要代理
python run_trading_analysis.py
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

### 问题1: PyTorch下载速度慢
**现象**: `pip install torch` 下载速度极慢或超时
**解决方案**:
1. **使用清华源**:
   ```bash
   pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

2. **使用中科大源**:
   ```bash
   pip install torch torchvision torchaudio -i https://pypi.mirrors.ustc.edu.cn/simple/
   ```

3. **使用官方CUDA源**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **手动下载安装**:
   - 访问 https://pytorch.org/get-started/locally/
   - 下载对应版本的whl文件
   - 使用 `pip install 文件名.whl` 安装

### 问题2: Yahoo Finance API访问被阻止 (403错误)
**现象**: 
```
Yahoo Finance获取 600588.SS 失败: 403 Client Error: Forbidden
```

**原因**: Yahoo Finance对频繁请求或某些地区的访问进行了限制

**解决方案**:
1. **使用VPN或代理**:
   ```bash
   # 设置HTTP代理
   export http_proxy=http://your-proxy:port
   export https_proxy=http://your-proxy:port
   
   # 运行程序
   python run_trading_analysis.py
   ```

2. **SSH隧道代理**（适用于云服务器）:
   ```bash
   # 在本地终端建立SSH隧道
   ssh -L 7890:localhost:7890 user@your-server-ip
   
   # 在云服务器上设置代理并运行
   export https_proxy=http://127.0.0.1:7890
   export http_proxy=http://127.0.0.1:7890
   python run_trading_analysis.py
   ```

3. **修改请求频率**:
   - 系统已自动增加请求间隔到1秒
   - 如仍有问题，可在 `real_stock_data_fetcher.py` 中增加 `time.sleep()` 时间

4. **使用备用数据源**:
   - 系统会自动切换到腾讯财经API
   - 如需添加更多数据源，可修改 `RealStockDataFetcher` 类

5. **网络环境检查**:
   ```bash
   # 检查网络连接
   ping finance.yahoo.com
   
   # 检查DNS解析
   nslookup finance.yahoo.com
   
   # 测试HTTPS连接
   curl -I https://finance.yahoo.com
   ```

### 问题3: CUDA版本不匹配
```bash
# 症状: RuntimeError: CUDA version mismatch
# 解决: 重新安装匹配的PyTorch版本
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题4: 内存不足
```bash
# 症状: CUDA out of memory
# 解决: 减少批处理大小
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

### 问题5: 网络连接失败
```bash
# 症状: 无法获取股票数据
# 解决: 检查网络连接和防火墙设置
ping finance.yahoo.com
```

### 问题6: 依赖包冲突
```bash
# 症状: 包版本冲突
# 解决: 使用虚拟环境重新安装
rm -rf trading_env
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 问题7: 权限问题
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