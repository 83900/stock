#!/bin/bash

echo "=== SeetaCloud远程测试模型 ==="

# SSH连接信息
SSH_HOST="root@connect.seetacloud.com"
SSH_PORT="36067"

echo "连接到服务器并测试模型..."

ssh -p $SSH_PORT $SSH_HOST << 'EOF'
echo "=== 远程环境检查 ==="
echo "当前目录: $(pwd)"
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '未安装')"

cd /root/stock

echo ""
echo "=== 检查训练结果文件 ==="
echo "模型文件:"
ls -la models/ 2>/dev/null || echo "models目录不存在"

echo ""
echo "训练历史文件:"
ls -la training_history.* 2>/dev/null || echo "训练历史文件不存在"

echo ""
echo "=== 创建测试脚本 ==="
cat > test_model.py << 'PYTHON_EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在远程服务器上测试训练好的模型
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import os
from datetime import datetime

class SimpleStockLSTM(nn.Module):
    """简化的股票预测LSTM模型"""
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(SimpleStockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

def load_and_preprocess_data(csv_file):
    """加载和预处理数据"""
    print(f"加载数据文件: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"数据形状: {df.shape}")
    
    feature_columns = ['open', 'close', 'high', 'low', 'volume']
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"可用特征: {available_features}")
    
    if not available_features:
        raise ValueError("没有找到可用的特征列")
    
    data = df[available_features].values.astype(np.float32)
    
    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, scaler, available_features, df

def create_sequences(data, seq_length=3):
    """创建时间序列数据"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, 1])  # 预测收盘价
    return np.array(X), np.array(y)

def test_model():
    """测试训练好的模型"""
    print("=" * 50)
    print("测试训练好的股票预测模型")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 查找数据文件
    data_files = [f for f in os.listdir('.') if f.startswith('stock_data_') and f.endswith('.csv')]
    if not data_files:
        print("❌ 没有找到股票数据文件")
        return
    
    latest_file = sorted(data_files)[-1]
    print(f"使用数据文件: {latest_file}")
    
    try:
        # 加载数据
        data_scaled, scaler, features, df = load_and_preprocess_data(latest_file)
        
        # 创建序列
        seq_length = 3
        X, y = create_sequences(data_scaled, seq_length)
        print(f"序列数据: X={X.shape}, y={y.shape}")
        
        if len(X) == 0:
            print("❌ 数据不足，无法创建序列")
            return
        
        # 转换为张量
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        # 创建模型
        model = SimpleStockLSTM(
            input_size=len(features),
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.2
        ).to(device)
        
        # 加载训练好的模型
        model_path = 'models/simple_model.pth'
        if os.path.exists(model_path):
            print(f"✓ 加载模型: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("❌ 没有找到训练好的模型文件")
            return
        
        # 测试模型
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        # 反标准化
        temp_data = np.zeros((len(predictions), len(features)))
        temp_data[:, 1] = predictions
        predictions_original = scaler.inverse_transform(temp_data)[:, 1]
        
        temp_data_true = np.zeros((len(y), len(features)))
        temp_data_true[:, 1] = y.cpu().numpy()
        y_original = scaler.inverse_transform(temp_data_true)[:, 1]
        
        # 计算指标
        mse = np.mean((predictions_original - y_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_original - y_original))
        
        print("\n" + "=" * 30)
        print("模型评估结果")
        print("=" * 30)
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        
        print("\n预测结果对比:")
        print("序号\t真实值\t预测值\t误差")
        print("-" * 40)
        for i in range(len(predictions_original)):
            error = abs(predictions_original[i] - y_original[i])
            print(f"{i+1}\t{y_original[i]:.2f}\t{predictions_original[i]:.2f}\t{error:.2f}")
        
        # 保存测试结果
        test_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_file': latest_file,
            'model_file': model_path,
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'predictions': predictions_original.tolist(),
            'actual_values': y_original.tolist()
        }
        
        with open('remote_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 测试结果已保存: remote_test_results.json")
        print("\n" + "=" * 50)
        print("远程模型测试完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()
PYTHON_EOF

echo ""
echo "=== 运行模型测试 ==="
python test_model.py

echo ""
echo "=== 检查测试结果 ==="
if [ -f "remote_test_results.json" ]; then
    echo "测试结果文件:"
    cat remote_test_results.json
else
    echo "没有找到测试结果文件"
fi

echo ""
echo "=== 远程测试完成 ==="
EOF

echo "=== 远程测试脚本执行完成 ==="