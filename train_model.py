#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票预测模型训练脚本
使用LSTM-TCN混合模型进行股票价格预测训练
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# 导入自定义模块
try:
    from lstm_tcn_model import StockLSTMTCN
    from rtx4090_optimization import setup_rtx4090_optimization, get_optimal_batch_size
    print("✓ 成功导入RTX 4090优化模块")
except ImportError as e:
    print(f"⚠️ RTX 4090优化模块导入失败: {e}")
    print("使用基础优化设置...")
    # 基础优化设置
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockTrainer:
    def __init__(self, data_file=None, sequence_length=60, prediction_days=5):
        """
        初始化训练器
        
        Args:
            data_file: 数据文件路径
            sequence_length: 输入序列长度
            prediction_days: 预测天数
        """
        self.data_file = data_file or self._find_latest_data_file()
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # 设置GPU优化
        self._setup_optimization()
        
        logger.info(f"使用设备: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def _find_latest_data_file(self):
        """查找最新的数据文件"""
        csv_files = [f for f in os.listdir('.') if f.startswith('stock_data_') and f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("未找到股票数据文件")
        
        # 按文件名排序，获取最新的
        latest_file = sorted(csv_files)[-1]
        logger.info(f"使用数据文件: {latest_file}")
        return latest_file
    
    def _setup_optimization(self):
        """设置GPU优化"""
        try:
            setup_rtx4090_optimization()
            logger.info("✓ RTX 4090优化已启用")
        except:
            logger.info("使用基础GPU优化设置")
    
    def load_and_prepare_data(self):
        """加载和预处理数据"""
        logger.info("加载数据...")
        
        # 读取CSV数据
        df = pd.read_csv(self.data_file)
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"数据列: {df.columns.tolist()}")
        
        # 选择特征列
        feature_columns = ['open', 'close', 'high', 'low', 'volume', 'change_pct']
        
        # 检查列是否存在
        available_columns = [col for col in feature_columns if col in df.columns]
        if not available_columns:
            raise ValueError(f"数据文件中未找到所需的特征列: {feature_columns}")
        
        logger.info(f"使用特征列: {available_columns}")
        
        # 提取特征数据
        data = df[available_columns].values
        
        # 处理缺失值
        data = np.nan_to_num(data, nan=0.0)
        
        # 数据标准化
        scaled_data = self.scaler.fit_transform(data)
        
        # 创建序列数据
        X, y = self._create_sequences(scaled_data)
        
        logger.info(f"序列数据形状: X={X.shape}, y={y.shape}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # 转换为PyTorch张量
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_train = torch.FloatTensor(y_train).to(self.device)
        self.y_test = torch.FloatTensor(y_test).to(self.device)
        
        logger.info(f"训练集大小: {self.X_train.shape[0]}")
        logger.info(f"测试集大小: {self.X_test.shape[0]}")
        
        return len(available_columns)
    
    def _create_sequences(self, data):
        """创建时间序列数据"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.prediction_days + 1):
            # 输入序列
            X.append(data[i-self.sequence_length:i])
            # 目标值（预测未来几天的收盘价）
            y.append(data[i:i+self.prediction_days, 1])  # 假设收盘价在索引1
        
        return np.array(X), np.array(y)
    
    def create_model(self, input_size):
        """创建模型"""
        logger.info("创建LSTM-TCN模型...")
        
        # 获取最优批次大小
        try:
            batch_size = get_optimal_batch_size(input_size, self.sequence_length)
            logger.info(f"推荐批次大小: {batch_size}")
        except:
            batch_size = 32
            logger.info(f"使用默认批次大小: {batch_size}")
        
        self.batch_size = batch_size
        
        # 创建模型
        self.model = StockLSTMTCN(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            output_size=self.prediction_days,
            dropout=0.2,
            tcn_channels=[64, 128, 256],
            kernel_size=3
        ).to(self.device)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info("模型创建完成")
    
    def train(self, epochs=100, save_interval=10):
        """训练模型"""
        logger.info(f"开始训练，共 {epochs} 个epoch...")
        
        # 创建数据加载器
        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        test_dataset = TensorDataset(self.X_test, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 训练历史
        train_losses = []
        test_losses = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            self.model.eval()
            test_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    test_loss += loss.item()
            
            test_loss /= len(test_loader)
            test_losses.append(test_loss)
            
            # 学习率调度
            self.scheduler.step(test_loss)
            
            # 保存最佳模型
            if test_loss < best_loss:
                best_loss = test_loss
                self.save_model('best_model.pth')
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], '
                          f'Train Loss: {train_loss:.6f}, '
                          f'Test Loss: {test_loss:.6f}, '
                          f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 定期保存
            if (epoch + 1) % save_interval == 0:
                self.save_model(f'model_epoch_{epoch+1}.pth')
        
        # 保存训练历史
        self._save_training_history(train_losses, test_losses)
        
        logger.info("训练完成！")
        logger.info(f"最佳测试损失: {best_loss:.6f}")
        
        return train_losses, test_losses
    
    def save_model(self, filename):
        """保存模型"""
        os.makedirs('models', exist_ok=True)
        filepath = os.path.join('models', filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'prediction_days': self.prediction_days,
            'input_size': self.model.input_size if hasattr(self.model, 'input_size') else None
        }, filepath)
        
        logger.info(f"模型已保存到: {filepath}")
    
    def _save_training_history(self, train_losses, test_losses):
        """保存训练历史"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='训练损失')
        plt.plot(test_losses, label='测试损失')
        plt.title('训练历史')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_losses[-50:], label='训练损失 (最后50个epoch)')
        plt.plot(test_losses[-50:], label='测试损失 (最后50个epoch)')
        plt.title('训练历史 (最后50个epoch)')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数值数据
        history = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("训练历史已保存")

def main():
    """主函数"""
    print("=" * 60)
    print("股票预测模型训练")
    print("=" * 60)
    
    try:
        # 创建训练器
        trainer = StockTrainer(
            sequence_length=60,
            prediction_days=5
        )
        
        # 加载数据
        input_size = trainer.load_and_prepare_data()
        
        # 创建模型
        trainer.create_model(input_size)
        
        # 开始训练
        train_losses, test_losses = trainer.train(epochs=100)
        
        print("\n" + "=" * 60)
        print("训练完成！")
        print("=" * 60)
        print(f"最终训练损失: {train_losses[-1]:.6f}")
        print(f"最终测试损失: {test_losses[-1]:.6f}")
        print(f"最佳模型已保存到: models/best_model.pth")
        print(f"训练历史图表: training_history.png")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)