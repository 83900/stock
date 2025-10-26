#!/bin/bash

# SeetaCloud远程训练脚本（包含数据获取）
# 连接到远程服务器，获取更多数据并启动模型训练

echo "=== SeetaCloud远程训练（含数据获取） ==="
echo "连接到服务器并启动训练..."

# SSH连接信息
SSH_HOST="connect.bjb1.seetacloud.com"
SSH_PORT="56495"
SSH_USER="root"
SSH_PASS="fbI6qJn2IJ8A"

# 使用sshpass进行自动登录并执行训练
sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no -p $SSH_PORT $SSH_USER@$SSH_HOST << 'EOF'

echo "=== 远程训练环境检查 ==="
echo "当前目录: $(pwd)"
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"

# 进入项目目录
cd /root/stock || { echo "项目目录不存在"; exit 1; }

echo "=== 获取更多股票数据 ==="
python stock_data.py

echo "=== 检查数据文件 ==="
latest_csv=$(ls -t stock_data_*.csv | head -1)
echo "最新数据文件: $latest_csv"
echo "数据行数: $(wc -l < $latest_csv)"
head -5 "$latest_csv"

echo "=== 安装必要的Python包 ==="
# 检查并安装缺失的包
python -c "import torch" 2>/dev/null || pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import sklearn" 2>/dev/null || pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import matplotlib" 2>/dev/null || pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "=== 验证包安装 ==="
python -c "
import torch
import sklearn
import matplotlib
import pandas as pd
import numpy as np
print('✓ 所有必要包已安装')
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU设备: {torch.cuda.get_device_name()}')
    print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo "=== 创建简化训练脚本 ==="
cat > simple_train.py << 'SIMPLE_TRAIN'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的股票预测模型训练脚本
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    # 启用优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# 简化的LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

def load_data():
    """加载最新的股票数据"""
    # 找到最新的CSV文件
    csv_files = [f for f in os.listdir('.') if f.startswith('stock_data_') and f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("未找到股票数据文件")
    
    latest_file = sorted(csv_files)[-1]
    print(f"使用数据文件: {latest_file}")
    
    # 读取数据
    df = pd.read_csv(latest_file)
    print(f"数据形状: {df.shape}")
    
    # 选择数值特征
    feature_columns = ['open', 'close', 'high', 'low', 'volume']
    available_columns = [col for col in feature_columns if col in df.columns]
    
    if not available_columns:
        raise ValueError("未找到所需的特征列")
    
    print(f"使用特征: {available_columns}")
    
    # 提取数据并处理
    data = df[available_columns].values.astype(np.float32)
    data = np.nan_to_num(data, nan=0.0)
    
    return data, available_columns

def create_sequences(data, seq_length=10):
    """创建时间序列"""
    X, y = [], []
    
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 1])  # 预测收盘价
    
    return np.array(X), np.array(y)

def main():
    print("=" * 50)
    print("简化股票预测模型训练")
    print("=" * 50)
    
    try:
        # 加载数据
        data, feature_names = load_data()
        print(f"原始数据形状: {data.shape}")
        
        # 数据标准化
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # 创建序列
        seq_length = min(10, len(data) // 3)  # 动态调整序列长度
        X, y = create_sequences(scaled_data, seq_length)
        
        if len(X) == 0:
            print("数据量不足，无法创建序列")
            return
        
        print(f"序列数据: X={X.shape}, y={y.shape}")
        
        # 划分数据
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 转换为张量
        X_train = torch.FloatTensor(X_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
        
        print(f"训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
        
        # 创建模型
        model = SimpleLSTM(
            input_size=len(feature_names),
            hidden_size=32,
            num_layers=2,
            output_size=1,
            dropout=0.2
        ).to(device)
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 训练
        epochs = 50
        train_losses = []
        test_losses = []
        
        print("开始训练...")
        
        for epoch in range(epochs):
            # 训练
            model.train()
            optimizer.zero_grad()
            
            train_pred = model(X_train)
            train_loss = criterion(train_pred.squeeze(), y_train)
            train_loss.backward()
            optimizer.step()
            
            # 测试
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = criterion(test_pred.squeeze(), y_test)
            
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}')
        
        # 保存模型
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'seq_length': seq_length,
            'feature_names': feature_names
        }, 'models/simple_model.pth')
        
        # 保存训练历史
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='训练损失')
        plt.plot(test_losses, label='测试损失')
        plt.title('训练历史')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存历史数据
        with open('training_history.json', 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'test_losses': test_losses,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print("\n" + "=" * 50)
        print("训练完成！")
        print("=" * 50)
        print(f"最终训练损失: {train_losses[-1]:.6f}")
        print(f"最终测试损失: {test_losses[-1]:.6f}")
        print("模型已保存到: models/simple_model.pth")
        print("训练历史: training_history.png")
        
    except Exception as e:
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
SIMPLE_TRAIN

echo "=== 开始简化模型训练 ==="
python simple_train.py

echo "=== 训练完成，检查结果 ==="
echo "模型文件:"
ls -la models/ 2>/dev/null || echo "无模型文件"
echo "训练历史:"
ls -la training_history.* 2>/dev/null || echo "无历史文件"
echo "日志文件:"
ls -la *.log 2>/dev/null || echo "无日志文件"

EOF

echo "=== 远程训练完成 ==="