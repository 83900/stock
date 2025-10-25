# LSTM-TCN股票预测系统部署指南

## 📋 目录
- [系统概述](#系统概述)
- [算力需求](#算力需求)
- [环境配置](#环境配置)
- [部署步骤](#部署步骤)
- [使用指南](#使用指南)
- [性能优化](#性能优化)
- [故障排除](#故障排除)
- [API文档](#api文档)

## 🎯 系统概述

本系统是一个基于LSTM-TCN联合模型的高精度股票预测系统，专为短线交易设计。系统具有以下特点：

### 核心功能
- **实时数据获取**: 使用adata库获取中国股市实时数据
- **LSTM-TCN预测**: 结合长短期记忆网络和时间卷积网络的优势
- **多任务学习**: 同时预测价格、趋势和波动率
- **风险评估**: 提供置信度和风险等级评估
- **批量预测**: 支持多股票并行预测
- **Web界面**: 提供直观的可视化界面

### 技术架构
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   数据获取层     │    │    预测模型层     │    │   应用服务层     │
│                │    │                 │    │                │
│ • 实时股价      │───▶│ • LSTM网络      │───▶│ • Web API      │
│ • 技术指标      │    │ • TCN网络       │    │ • 批量预测      │
│ • 成交量        │    │ • 注意力机制     │    │ • 风险评估      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 💻 算力需求

### 推荐配置 (生产环境)
- **GPU**: NVIDIA RTX 4090 / A100 / V100
- **显存**: ≥ 16GB VRAM
- **内存**: ≥ 32GB RAM
- **CPU**: ≥ 8核心
- **存储**: ≥ 100GB SSD

### 最低配置 (开发/测试)
- **GPU**: NVIDIA GTX 1660 / RTX 3060
- **显存**: ≥ 6GB VRAM
- **内存**: ≥ 16GB RAM
- **CPU**: ≥ 4核心
- **存储**: ≥ 50GB SSD

### 算力平台推荐

#### 1. 阿里云PAI-EAS
```bash
# 推荐实例规格
实例类型: ecs.gn6i-c4g1.xlarge
GPU: NVIDIA V100 (16GB)
CPU: 4核心
内存: 15GB
价格: ~¥8-12/小时
```

#### 2. 腾讯云TI-ONE
```bash
# 推荐实例规格
实例类型: TI.GN10X.2XLARGE32
GPU: NVIDIA V100 (32GB)
CPU: 8核心
内存: 32GB
价格: ~¥15-20/小时
```

#### 3. 百度智能云
```bash
# 推荐实例规格
实例类型: GPU.V100-32G
GPU: NVIDIA V100 (32GB)
CPU: 12核心
内存: 92GB
价格: ~¥18-25/小时
```

#### 4. AutoDL (性价比推荐)
```bash
# 推荐实例规格
GPU: RTX 4090 (24GB)
CPU: 12核心
内存: 50GB
价格: ~¥2.5-4/小时
```

## 🔧 环境配置

### 1. Python环境
```bash
# 创建虚拟环境
conda create -n stock_prediction python=3.9
conda activate stock_prediction

# 或使用venv
python -m venv stock_prediction
source stock_prediction/bin/activate  # Linux/Mac
# stock_prediction\Scripts\activate  # Windows
```

### 2. CUDA环境 (GPU必需)
```bash
# 检查CUDA版本
nvidia-smi

# 安装CUDA 11.8 (推荐)
# 下载地址: https://developer.nvidia.com/cuda-11-8-0-download-archive

# 验证安装
nvcc --version
```

### 3. cuDNN安装
```bash
# 下载cuDNN 8.6 for CUDA 11.x
# 下载地址: https://developer.nvidia.com/cudnn

# 解压并复制文件到CUDA目录
# 详细步骤请参考NVIDIA官方文档
```

## 🚀 部署步骤

### 步骤1: 下载代码
```bash
# 如果使用Git
git clone <repository_url>
cd stock_prediction

# 或直接上传文件到服务器
```

### 步骤2: 安装依赖
```bash
# 基础依赖安装
pip install -r requirements.txt

# GPU版本PyTorch安装 (推荐，根据CUDA版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 验证GPU可用性
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count())"
```

### 步骤3: 验证环境
```bash
# 运行环境检查脚本
python -c "
import torch
import pandas as pd
import numpy as np
from adata import stock
print('✓ 所有依赖已正确安装')
print(f'✓ PyTorch版本: {torch.__version__}')
print(f'✓ GPU可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU设备: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA版本: {torch.version.cuda}')
"
```

### 步骤4: 测试基础功能
```bash
# 测试数据获取
python stock_data.py

# 测试预测模型 (小规模)
python -c "
from lstm_tcn_model import LSTMTCNPredictor, get_sample_data
model = LSTMTCNPredictor()
data = get_sample_data()
print('✓ 模型创建成功')
"
```

## 📖 使用指南

### 1. 快速开始

#### 训练模型
```python
from advanced_predictor import AdvancedStockPredictor

# 创建预测器
predictor = AdvancedStockPredictor()

# 训练模型 (建议在GPU上运行)
result = predictor.train_model(
    stock_code="000001",  # 平安银行
    days=500,            # 使用500天历史数据
    epochs=100,          # 训练100轮
    save_model=True      # 保存模型
)

print(f"训练完成，MAPE: {result['metrics']['mape']:.2f}%")
```

#### 预测股票
```python
# 预测单只股票
prediction = predictor.predict_stock("000001")

print(f"当前价格: {prediction['current_price']:.2f}")
print(f"预测价格: {prediction['prediction']['predicted_price']:.2f}")
print(f"趋势预测: {prediction['prediction']['trend_prediction']}")
print(f"建议操作: {prediction['analysis']['trading_action']}")
```

#### 批量预测
```python
# 批量预测热门股票
stocks = ["000001", "000002", "600036", "600519", "000858"]
results = predictor.batch_predict(stocks)

print(f"成功率: {results['success_rate']:.1f}%")
```

### 2. Web界面使用

#### 启动Web服务
```bash
# 启动原有的实时数据展示
python web_app.py

# 访问 http://localhost:8080
```

#### 集成预测功能
```python
# 修改web_app.py，添加预测接口
from advanced_predictor import AdvancedStockPredictor

predictor = AdvancedStockPredictor("best_model.pth")

@app.route('/api/predict/<stock_code>')
def predict_api(stock_code):
    result = predictor.predict_stock(stock_code)
    return jsonify(result)
```

### 3. 命令行工具

#### 创建训练脚本
```bash
# train.py
python -c "
import sys
from advanced_predictor import AdvancedStockPredictor

stock_code = sys.argv[1] if len(sys.argv) > 1 else '000001'
predictor = AdvancedStockPredictor()
result = predictor.train_model(stock_code, epochs=50)
print('训练完成!')
" 000001
```

#### 创建预测脚本
```bash
# predict.py
python -c "
import sys
from advanced_predictor import AdvancedStockPredictor

stock_code = sys.argv[1] if len(sys.argv) > 1 else '000001'
predictor = AdvancedStockPredictor('lstm_tcn_model_000001_*.pth')
result = predictor.predict_stock(stock_code)
print(f'预测结果: {result}')
" 000001
```

## ⚡ 性能优化

### 1. GPU优化
```python
# 在模型训练前添加GPU配置
import torch

# 设置GPU设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 启用混合精度训练 (提升速度，需要GPU支持)
if torch.cuda.is_available():
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
    
    # 在训练循环中使用
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 2. 数据加载优化
```python
# 使用PyTorch DataLoader优化数据管道
from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=32, num_workers=4):
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # 多进程加载
        pin_memory=True,          # 加速GPU传输
        persistent_workers=True   # 保持worker进程
    )
    return dataloader
```

### 3. 模型优化
```python
# 在LSTMTCNPredictor中添加优化选项
class LSTMTCNPredictor:
    def __init__(self, ..., use_mixed_precision=True, compile_model=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        
        if compile_model and hasattr(torch, 'compile'):
            # PyTorch 2.0+ 编译优化
            tf.config.optimizer.set_jit(True)
```

### 4. 批处理优化
```python
# 增加批处理大小 (如果GPU内存充足)
batch_size = 64  # 或更大

# 使用多GPU训练 (如果有多个GPU)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
```

## 🔍 故障排除

### 常见问题

#### 1. GPU内存不足
```bash
# 错误信息: ResourceExhaustedError: OOM when allocating tensor

# 解决方案:
# 1. 减少批处理大小
batch_size = 16  # 从32减少到16

# 2. 启用GPU内存增长
tf.config.experimental.set_memory_growth(gpu, True)

# 3. 使用梯度累积
# 在模型训练中实现梯度累积
```

#### 2. CUDA版本不匹配
```bash
# 错误信息: RuntimeError: CUDA error: no kernel image is available for execution on the device

# 解决方案:
# 1. 检查CUDA版本
nvidia-smi
nvcc --version

# 2. 安装匹配的PyTorch版本
# CUDA 11.8 -> torch>=2.0.0+cu118
# CUDA 12.1 -> torch>=2.0.0+cu121
# 或访问 https://pytorch.org/get-started/locally/ 获取最新安装命令
```

#### 3. 数据获取失败
```bash
# 错误信息: 无法获取股票数据

# 解决方案:
# 1. 检查网络连接
# 2. 验证股票代码格式
# 3. 检查adata库版本
pip install --upgrade adata
```

#### 4. 模型训练缓慢
```bash
# 可能原因和解决方案:

# 1. 使用CPU训练 -> 启用GPU
# 2. 数据加载慢 -> 使用tf.data
# 3. 模型过大 -> 减少参数数量
# 4. 批处理太小 -> 增加batch_size
```

### 性能监控
```python
# 添加训练监控
import time
import psutil
import GPUtil

def monitor_training():
    # CPU使用率
    cpu_percent = psutil.cpu_percent()
    
    # 内存使用率
    memory = psutil.virtual_memory()
    
    # GPU使用率
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_load = gpu.load * 100
        gpu_memory = gpu.memoryUtil * 100
        
        print(f"CPU: {cpu_percent}%, RAM: {memory.percent}%, "
              f"GPU: {gpu_load}%, VRAM: {gpu_memory}%")
```

## 📚 API文档

### AdvancedStockPredictor类

#### 初始化
```python
predictor = AdvancedStockPredictor(model_path=None)
```

#### 主要方法

##### train_model()
```python
result = predictor.train_model(
    stock_code="000001",    # 股票代码
    days=500,              # 历史数据天数
    epochs=100,            # 训练轮数
    save_model=True        # 是否保存模型
)
```

**返回值:**
```json
{
    "success": true,
    "stock_code": "000001",
    "training_samples": 500,
    "epochs_completed": 100,
    "final_loss": 0.001234,
    "final_val_loss": 0.001456,
    "metrics": {
        "mse": 0.001234,
        "mae": 0.023456,
        "rmse": 0.035123,
        "r2_score": 0.987654,
        "mape": 1.23,
        "trend_accuracy": 0.85
    },
    "model_path": "lstm_tcn_model_000001_20241025_143022.pth"
}
```

##### predict_stock()
```python
result = predictor.predict_stock(
    stock_code="000001",      # 股票代码
    return_analysis=True      # 是否返回分析
)
```

**返回值:**
```json
{
    "stock_code": "000001",
    "stock_name": "平安银行",
    "current_price": 12.34,
    "current_change": 0.12,
    "current_change_pct": 0.98,
    "volume": 12345678,
    "prediction": {
        "predicted_price": 12.56,
        "trend_prediction": "上涨",
        "trend_probabilities": {
            "上涨": 0.75,
            "下跌": 0.15,
            "横盘": 0.10
        },
        "predicted_volatility": 0.023,
        "confidence_score": 0.75,
        "risk_level": "中风险"
    },
    "analysis": {
        "expected_return_pct": 1.78,
        "trading_action": "买入",
        "confidence_level": "中",
        "risk_factors": [],
        "recommendation": [
            "可以考虑小仓位买入",
            "密切关注市场变化"
        ]
    },
    "timestamp": "2024-10-25 14:30:22"
}
```

##### batch_predict()
```python
results = predictor.batch_predict(
    stock_codes=["000001", "000002", "600036"],
    save_results=True
)
```

##### get_top_recommendations()
```python
recommendations = predictor.get_top_recommendations(
    stock_codes=["000001", "000002", "600036"],
    top_n=5
)
```

### LSTMTCNPredictor类

#### 核心参数
```python
model = LSTMTCNPredictor(
    sequence_length=60,     # 输入序列长度
    n_features=5,          # 特征数量
    lstm_units=128,        # LSTM单元数
    tcn_filters=64,        # TCN滤波器数
    dense_units=64,        # 全连接层单元数
    dropout_rate=0.2       # Dropout比率
)
```

## 📈 使用建议

### 1. 模型训练建议
- **数据量**: 建议使用至少300天的历史数据
- **训练轮数**: 初始训练50-100轮，根据验证损失调整
- **批处理大小**: GPU内存16GB建议使用32-64
- **学习率**: 默认0.001，可根据收敛情况调整

### 2. 预测使用建议
- **置信度阈值**: 建议只采纳置信度>0.7的预测
- **风险控制**: 高风险预测建议谨慎操作
- **时间窗口**: 预测适用于1-3个交易日的短线操作
- **止损设置**: 建议设置2-3%的止损位

### 3. 生产环境建议
- **模型更新**: 建议每周重新训练模型
- **数据备份**: 定期备份训练数据和模型文件
- **监控告警**: 设置预测准确率监控
- **负载均衡**: 高并发场景建议使用多实例部署

## ⚠️ 重要声明

1. **投资风险**: 本系统仅供参考，投资有风险，决策需谨慎
2. **数据准确性**: 请确保数据源的可靠性和时效性
3. **模型局限性**: 机器学习模型无法预测所有市场情况
4. **合规使用**: 请遵守相关法律法规和交易所规定

## 📞 技术支持

如遇到技术问题，请提供以下信息：
- 系统环境 (操作系统、Python版本、GPU型号)
- 错误日志
- 复现步骤
- 数据样本 (如涉及)

---

**版本**: v1.0  
**更新日期**: 2024-10-25  
**作者**: AI Assistant