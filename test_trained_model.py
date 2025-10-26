#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的股票预测模型
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
        # LSTM层
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.dropout(lstm_out[:, -1, :])
        # 全连接层
        out = self.fc(out)
        return out

def load_and_preprocess_data(csv_file):
    """加载和预处理数据"""
    print(f"加载数据文件: {csv_file}")
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 选择特征列
    feature_columns = ['open', 'close', 'high', 'low', 'volume']
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"可用特征: {available_features}")
    
    if not available_features:
        raise ValueError("没有找到可用的特征列")
    
    # 提取特征数据
    data = df[available_features].values.astype(np.float32)
    print(f"特征数据形状: {data.shape}")
    
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
        y.append(data[i + seq_length, 1])  # 预测收盘价 (close)
    return np.array(X), np.array(y)

def test_model():
    """测试训练好的模型"""
    print("=" * 50)
    print("测试训练好的股票预测模型")
    print("=" * 50)
    
    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 查找最新的数据文件
    data_files = [f for f in os.listdir('.') if f.startswith('stock_data_') and f.endswith('.csv')]
    if not data_files:
        print("❌ 没有找到股票数据文件")
        return
    
    latest_file = sorted(data_files)[-1]
    print(f"使用数据文件: {latest_file}")
    
    try:
        # 加载和预处理数据
        data_scaled, scaler, features, df = load_and_preprocess_data(latest_file)
        
        # 创建序列数据
        seq_length = 3
        X, y = create_sequences(data_scaled, seq_length)
        print(f"序列数据: X={X.shape}, y={y.shape}")
        
        if len(X) == 0:
            print("❌ 数据不足，无法创建序列")
            return
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.FloatTensor(y).to(device)
        
        # 创建模型实例
        model = SimpleStockLSTM(
            input_size=len(features),
            hidden_size=64,
            num_layers=2,
            output_size=1,
            dropout=0.2
        ).to(device)
        
        # 检查本地是否有训练好的模型
        local_model_path = 'simple_model.pth'
        if os.path.exists(local_model_path):
            print(f"✓ 找到本地模型文件: {local_model_path}")
            model.load_state_dict(torch.load(local_model_path, map_location=device))
        else:
            print("❌ 本地没有找到训练好的模型文件")
            print("提示: 模型文件应该在远程服务器上，需要先下载")
            return
        
        # 设置为评估模式
        model.eval()
        
        # 进行预测
        with torch.no_grad():
            predictions = model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        # 反标准化预测结果
        # 创建一个临时数组来反标准化
        temp_data = np.zeros((len(predictions), len(features)))
        temp_data[:, 1] = predictions  # 收盘价在第1列
        predictions_original = scaler.inverse_transform(temp_data)[:, 1]
        
        # 反标准化真实值
        temp_data_true = np.zeros((len(y), len(features)))
        temp_data_true[:, 1] = y.cpu().numpy()
        y_original = scaler.inverse_transform(temp_data_true)[:, 1]
        
        # 计算评估指标
        mse = np.mean((predictions_original - y_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_original - y_original))
        
        print("\n" + "=" * 30)
        print("模型评估结果")
        print("=" * 30)
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        
        # 显示预测结果
        print("\n预测结果对比:")
        print("序号\t真实值\t预测值\t误差")
        print("-" * 40)
        for i in range(len(predictions_original)):
            error = abs(predictions_original[i] - y_original[i])
            print(f"{i+1}\t{y_original[i]:.2f}\t{predictions_original[i]:.2f}\t{error:.2f}")
        
        # 绘制预测结果
        plt.figure(figsize=(12, 8))
        
        # 子图1: 预测vs真实值对比
        plt.subplot(2, 1, 1)
        x_axis = range(len(predictions_original))
        plt.plot(x_axis, y_original, 'bo-', label='真实值', markersize=8)
        plt.plot(x_axis, predictions_original, 'ro-', label='预测值', markersize=8)
        plt.title('股票价格预测结果对比', fontsize=14, fontweight='bold')
        plt.xlabel('样本序号')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2: 误差分布
        plt.subplot(2, 1, 2)
        errors = predictions_original - y_original
        plt.bar(x_axis, errors, alpha=0.7, color='green')
        plt.title('预测误差分布', fontsize=14, fontweight='bold')
        plt.xlabel('样本序号')
        plt.ylabel('误差 (预测值 - 真实值)')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_test_results.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ 测试结果图表已保存: model_test_results.png")
        
        # 保存测试结果
        test_results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_file': latest_file,
            'model_file': local_model_path,
            'data_shape': data_scaled.shape,
            'sequence_length': seq_length,
            'features': features,
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'predictions': predictions_original.tolist(),
            'actual_values': y_original.tolist(),
            'errors': errors.tolist()
        }
        
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 测试结果已保存: test_results.json")
        
        print("\n" + "=" * 50)
        print("模型测试完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()