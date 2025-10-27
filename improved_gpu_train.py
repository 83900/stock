#!/usr/bin/env python3
"""
改进版GPU优化股票预测模型训练脚本
基于数据分析结果的改进版本，解决标准化、数据不足和模型架构问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class ImprovedStockDataset(Dataset):
    """改进的股票数据集类，支持技术指标和更好的数据处理"""
    
    def __init__(self, data, sequence_length=60, prediction_days=1):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.scaler = RobustScaler()  # 使用RobustScaler处理异常值
        
        # 按股票分组处理
        self.sequences = []
        self.targets = []
        
        for stock_code in data['stock_code'].unique():
            stock_data = data[data['stock_code'] == stock_code].copy()
            
            if len(stock_data) < sequence_length + prediction_days:
                print(f"警告: 股票 {stock_code} 数据不足 ({len(stock_data)} < {sequence_length + prediction_days})")
                continue
            
            # 按时间排序
            stock_data = stock_data.sort_values('trade_date')
            
            # 计算技术指标
            stock_data = self._add_technical_indicators(stock_data)
            
            # 特征列
            feature_columns = ['close', 'volume', 'ma5', 'ma10', 'ma20', 'rsi', 'macd', 'price_change_pct']
            
            # 检查是否有足够的特征列
            available_features = [col for col in feature_columns if col in stock_data.columns]
            if not available_features:
                print(f"警告: 股票 {stock_code} 缺少必要的特征列")
                continue
            
            # 使用对数变换处理价格（解决右偏分布问题）
            if 'close' in stock_data.columns:
                stock_data['log_close'] = np.log(stock_data['close'] + 1e-8)
                available_features = ['log_close'] + [f for f in available_features if f != 'close']
            
            # 标准化（按股票分别标准化）
            features = stock_data[available_features].fillna(method='ffill').fillna(0)
            scaled_features = self.scaler.fit_transform(features)
            
            # 创建序列
            for i in range(len(scaled_features) - sequence_length - prediction_days + 1):
                sequence = scaled_features[i:i + sequence_length]
                target = scaled_features[i + sequence_length:i + sequence_length + prediction_days, 0]  # 预测log_close
                
                self.sequences.append(sequence)
                self.targets.append(target)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
        print(f"数据集创建完成: {len(self.sequences)} 个序列, 特征维度: {self.sequences.shape[-1]}")
    
    def _add_technical_indicators(self, data):
        """添加技术指标"""
        data = data.copy()
        
        if 'close' in data.columns:
            # 移动平均线
            data['ma5'] = data['close'].rolling(window=5).mean()
            data['ma10'] = data['close'].rolling(window=10).mean()
            data['ma20'] = data['close'].rolling(window=20).mean()
            
            # 价格变化百分比
            data['price_change_pct'] = data['close'].pct_change()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            data['macd'] = exp1 - exp2
        
        return data
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

class ImprovedStockLSTM(nn.Module):
    """改进的LSTM模型，增加正则化和更好的架构"""
    
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3, output_size=1):
        super(ImprovedStockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 批标准化
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步的输出
        last_output = attn_out[:, -1, :]
        
        # 批标准化
        if last_output.size(0) > 1:  # 批大小大于1时才应用批标准化
            last_output = self.batch_norm(last_output)
        
        # 全连接层
        output = self.fc_layers(last_output)
        
        return output

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def load_and_prepare_data():
    """加载和准备真实股票数据"""
    print("=== 加载真实股票数据 ===")
    
    # 首先尝试加载真实数据获取器生成的数据
    import glob
    real_data_files = glob.glob('real_tech_stocks_data_*.csv')
    
    data = None
    if real_data_files:
        # 使用最新的真实数据文件
        latest_file = max(real_data_files)
        try:
            data = pd.read_csv(latest_file)
            print(f"成功加载真实数据文件: {latest_file}")
            
            # 检查数据格式并标准化列名
            if 'date' in data.columns:
                data['trade_date'] = pd.to_datetime(data['date']).dt.strftime('%Y%m%d')
            elif 'trade_date' not in data.columns:
                print("错误: 数据文件缺少日期列")
                data = None
                
        except Exception as e:
            print(f"加载真实数据失败: {e}")
            data = None
    
    # 如果没有真实数据，尝试获取
    if data is None:
        print("未找到真实数据文件，尝试获取真实数据...")
        try:
            from real_stock_data_fetcher import RealStockDataFetcher
            fetcher = RealStockDataFetcher()
            data = fetcher.fetch_all_stocks()
            
            if not data.empty:
                # 保存获取的数据
                csv_file, json_file = fetcher.save_data(data)
                print(f"真实数据已保存: {csv_file}")
                
                # 标准化列名
                if 'date' in data.columns:
                    data['trade_date'] = pd.to_datetime(data['date']).dt.strftime('%Y%m%d')
            else:
                data = None
                
        except Exception as e:
            print(f"获取真实数据失败: {e}")
            data = None
    
    # 最后的备选方案：生成模拟数据（但会发出警告）
    if data is None:
        print("警告: 无法获取真实数据，使用模拟数据进行训练")
        print("注意: 模拟数据仅用于测试，不适合实际交易!")
        data = generate_enhanced_synthetic_data()
    
    print(f"数据形状: {data.shape}")
    print(f"股票数量: {data['stock_code'].nunique()}")
    print(f"日期范围: {data['trade_date'].min()} - {data['trade_date'].max()}")
    
    return data

def generate_enhanced_synthetic_data():
    """生成增强的模拟数据，包含更多历史数据"""
    np.random.seed(42)
    
    # 生成更长的时间序列
    dates = pd.date_range('2020-01-01', '2024-10-26', freq='D')
    stocks = [1, 2, 858, 2415, 2594]
    
    data_list = []
    
    for stock in stocks:
        n_days = len(dates)
        base_price = np.random.uniform(20, 200)  # 更合理的价格范围
        
        # 生成更真实的价格序列
        prices = []
        volumes = []
        current_price = base_price
        
        for i in range(n_days):
            # 添加趋势和季节性
            trend = 0.0001 * i  # 轻微上升趋势
            seasonal = 0.05 * np.sin(2 * np.pi * i / 252)  # 年度季节性
            
            # 价格变化
            daily_return = np.random.normal(trend + seasonal, 0.02)
            current_price = current_price * (1 + daily_return)
            current_price = max(1.0, current_price)  # 确保价格为正
            
            prices.append(current_price)
            
            # 成交量（与价格变化相关）
            volume_base = np.random.randint(1000000, 5000000)
            volume_multiplier = 1 + abs(daily_return) * 2  # 价格波动大时成交量增加
            volumes.append(int(volume_base * volume_multiplier))
        
        # 创建数据记录
        for i, (date, price, volume) in enumerate(zip(dates, prices, volumes)):
            high = price * np.random.uniform(1.0, 1.03)
            low = price * np.random.uniform(0.97, 1.0)
            open_price = price * np.random.uniform(0.99, 1.01)
            
            data_list.append({
                'stock_code': stock,
                'trade_date': date.strftime('%Y%m%d'),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': volume
            })
    
    return pd.DataFrame(data_list)

def train_improved_model():
    """训练改进的模型"""
    print("=== 开始训练改进的GPU模型 ===")
    
    # 检查GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data = load_and_prepare_data()
    
    # 创建数据集
    sequence_length = 60  # 增加序列长度
    dataset = ImprovedStockDataset(data, sequence_length=sequence_length)
    
    if len(dataset) == 0:
        print("错误: 数据集为空，无法训练模型")
        return
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # 创建模型
    input_size = dataset.sequences.shape[-1]
    model = ImprovedStockLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        output_size=1
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 早停
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    
    # 训练历史
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # 训练循环
    num_epochs = 100
    print(f"开始训练，最大轮数: {num_epochs}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets.squeeze())
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets.squeeze())
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'input_size': input_size,
                'sequence_length': sequence_length
            }, 'improved_gpu_model.pth')
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')
        
        # 早停检查
        if early_stopping(val_loss):
            print(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    # 保存训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'epochs_trained': len(train_losses),
        'model_config': {
            'input_size': input_size,
            'hidden_size': 128,
            'num_layers': 3,
            'sequence_length': sequence_length,
            'batch_size': batch_size
        }
    }
    
    with open('improved_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Validation Loss (Zoomed)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练完成！最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存为: improved_gpu_model.pth")
    print(f"训练历史已保存为: improved_training_history.json")
    print(f"训练曲线已保存为: improved_training_curve.png")

if __name__ == "__main__":
    train_improved_model()