"""
LSTM-TCN联合股票预测模型 (PyTorch版本)
结合LSTM的时序记忆能力和TCN的并行计算优势，实现高精度股票价格预测
适用于短线交易策略
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 导入RTX 4090优化配置
try:
    from rtx4090_optimization import setup_rtx4090_optimization
    setup_rtx4090_optimization()
except ImportError:
    # 如果没有优化模块，使用基本设置
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class TemporalBlock(nn.Module):
    """TCN的时间块"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """移除填充"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    """时间卷积网络(TCN)"""
    
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AttentionLayer(nn.Module):
    """注意力机制层"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended = torch.sum(x * attention_weights, dim=1)
        return attended

class StockDataset(Dataset):
    """股票数据集"""
    
    def __init__(self, X, y_price, y_trend, y_volatility):
        self.X = torch.FloatTensor(X)
        self.y_price = torch.FloatTensor(y_price)
        self.y_trend = torch.FloatTensor(y_trend)
        self.y_volatility = torch.FloatTensor(y_volatility)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y_price[idx], self.y_trend[idx], self.y_volatility[idx])

class LSTMTCNPredictor(nn.Module):
    """LSTM-TCN联合预测模型"""
    
    def __init__(self, sequence_length=60, n_features=5, 
                 lstm_units=128, tcn_channels=[64, 64, 64], 
                 dense_units=64, dropout_rate=0.2):
        super(LSTMTCNPredictor, self).__init__()
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.tcn_channels = tcn_channels
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        # LSTM分支
        self.lstm = nn.LSTM(n_features, lstm_units, batch_first=True, dropout=dropout_rate)
        self.lstm_bn = nn.BatchNorm1d(lstm_units)
        
        # TCN分支
        self.tcn = TemporalConvNet(n_features, tcn_channels, dropout=dropout_rate)
        
        # 维度匹配层
        self.lstm_proj = nn.Linear(lstm_units, tcn_channels[-1])
        
        # 注意力机制
        self.attention = AttentionLayer(tcn_channels[-1])
        
        # 全连接层
        self.fc1 = nn.Linear(tcn_channels[-1], dense_units)
        self.bn1 = nn.BatchNorm1d(dense_units)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.bn2 = nn.BatchNorm1d(dense_units // 2)
        self.dropout2 = nn.Dropout(dropout_rate / 2)
        
        # 输出层
        self.price_head = nn.Linear(dense_units // 2, 1)
        self.trend_head = nn.Linear(dense_units // 2, 3)
        self.volatility_head = nn.Linear(dense_units // 2, 1)
        
        # 初始化权重
        self.init_weights()
        
        # 数据缩放器
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        self.history = None
        
    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM分支
        lstm_out, _ = self.lstm(x)  # (batch, seq, lstm_units)
        lstm_out = self.lstm_bn(lstm_out.transpose(1, 2)).transpose(1, 2)
        lstm_out = self.lstm_proj(lstm_out)  # 投影到TCN维度
        
        # TCN分支
        x_tcn = x.transpose(1, 2)  # (batch, features, seq)
        tcn_out = self.tcn(x_tcn)  # (batch, channels, seq)
        tcn_out = tcn_out.transpose(1, 2)  # (batch, seq, channels)
        
        # 特征融合 (加权平均)
        alpha, beta = 0.6, 0.4
        fused = alpha * lstm_out + beta * tcn_out
        
        # 注意力机制
        attended = self.attention(fused)  # (batch, channels)
        
        # 全连接层
        x = F.relu(self.fc1(attended))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # 多任务输出
        price_out = self.price_head(x)
        trend_out = F.softmax(self.trend_head(x), dim=1)
        volatility_out = torch.sigmoid(self.volatility_head(x))
        
        return price_out, trend_out, volatility_out
    
    def prepare_features(self, data):
        """准备技术指标特征"""
        df = data.copy()
        
        # 基础价格特征
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # 移动平均线
        for window in [5, 10, 20, 30]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ma_ratio_{window}'] = df['close'] / df[f'ma_{window}']
        
        # 技术指标
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # 波动率
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # 价格位置
        df['price_position'] = (df['close'] - df['close'].rolling(window=20).min()) / \
                              (df['close'].rolling(window=20).max() - df['close'].rolling(window=20).min())
        
        return df
    
    def create_sequences(self, data, target_col='close'):
        """创建时间序列数据"""
        # 选择特征
        feature_cols = [
            'close', 'volume', 'returns', 'rsi', 'macd', 
            'bb_position', 'volume_ratio', 'volatility', 'price_position',
            'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20'
        ]
        
        # 确保所有特征列都存在
        available_cols = [col for col in feature_cols if col in data.columns]
        
        if len(available_cols) < 5:
            raise ValueError(f"需要至少5个特征，但只找到 {len(available_cols)} 个: {available_cols}")
        
        # 删除包含NaN的行
        data_clean = data[available_cols + [target_col]].dropna()
        
        if len(data_clean) < self.sequence_length + 1:
            raise ValueError(f"数据长度不足，需要至少 {self.sequence_length + 1} 行数据")
        
        X, y_price, y_trend, y_volatility = [], [], [], []
        
        for i in range(self.sequence_length, len(data_clean)):
            # 输入特征
            X.append(data_clean[available_cols].iloc[i-self.sequence_length:i].values)
            
            # 价格目标
            current_price = data_clean[target_col].iloc[i-1]
            next_price = data_clean[target_col].iloc[i]
            y_price.append(next_price)
            
            # 趋势目标 (上涨/下跌/横盘)
            price_change = (next_price - current_price) / current_price
            if price_change > 0.01:  # 上涨超过1%
                trend = [1, 0, 0]
            elif price_change < -0.01:  # 下跌超过1%
                trend = [0, 1, 0]
            else:  # 横盘
                trend = [0, 0, 1]
            y_trend.append(trend)
            
            # 波动率目标
            volatility = abs(price_change)
            y_volatility.append(volatility)
        
        return (np.array(X), np.array(y_price), 
                np.array(y_trend), np.array(y_volatility))
    
    def train_model(self, data, epochs=100, batch_size=32, 
                   early_stopping_patience=15, learning_rate=0.001):
        """训练模型，使用过去1年作为训练集，前一周作为验证集"""
        print("准备数据...")
        
        # 准备特征
        data_with_features = self.prepare_features(data)
        
        # 假设数据按日期排序
        if 'date' not in data_with_features.columns:
            raise ValueError("数据必须包含 'date' 列")
        
        data_with_features = data_with_features.sort_values('date')
        
        # 计算分割点
        max_date = data_with_features['date'].max()
        val_start = max_date - timedelta(days=6)  # 前一周（7天）
        train_start = max_date - timedelta(days=365 + 6)  # 过去1年 + 前一周
        
        # 筛选数据
        train_data = data_with_features[
            (data_with_features['date'] >= train_start) & 
            (data_with_features['date'] < val_start)
        ]
        val_data = data_with_features[data_with_features['date'] >= val_start]
        
        if len(train_data) < self.sequence_length:
            raise ValueError("训练数据不足")
        if len(val_data) < 1:
            raise ValueError("验证数据不足")
        
        print(f"训练数据: {len(train_data)} 天 ({train_data['date'].min()} 到 {train_data['date'].max()})")
        print(f"验证数据: {len(val_data)} 天 ({val_data['date'].min()} 到 {val_data['date'].max()})")
        
        # 创建序列
        X_train, y_price_train, y_trend_train, y_vol_train = self.create_sequences(train_data)
        X_val, y_price_val, y_trend_val, y_vol_val = self.create_sequences(val_data)
        
        print(f"训练形状: X={X_train.shape}, y_price={y_price_train.shape}")
        print(f"验证形状: X={X_val.shape}, y_price={y_price_val.shape}")
        
        # 数据标准化（仅用训练数据拟合缩放器）
        X_train_scaled = self.scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        y_price_train_scaled = self.scaler_y.fit_transform(y_price_train.reshape(-1, 1)).flatten()
        
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        y_price_val_scaled = self.scaler_y.transform(y_price_val.reshape(-1, 1)).flatten()
        
        # 更新特征数量
        self.n_features = X_train.shape[-1]
        
        # 创建数据集和数据加载器
        train_dataset = StockDataset(X_train_scaled, y_price_train_scaled, y_trend_train, y_vol_train)
        val_dataset = StockDataset(X_val_scaled, y_price_val_scaled, y_trend_val, y_vol_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 移动模型到设备
        self.to(device)
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                        patience=10, min_lr=1e-7, verbose=True)
        
        # 损失函数
        price_criterion = nn.HuberLoss()
        trend_criterion = nn.CrossEntropyLoss()
        volatility_criterion = nn.MSELoss()
        
        # 训练历史
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("开始训练...")
        for epoch in range(epochs):
            # 训练阶段
            self.train()
            train_loss = 0.0
            for batch_X, batch_y_price, batch_y_trend, batch_y_vol in train_loader:
                batch_X = batch_X.to(device)
                batch_y_price = batch_y_price.to(device)
                batch_y_trend = batch_y_trend.to(device)
                batch_y_vol = batch_y_vol.to(device)
                
                optimizer.zero_grad()
                
                price_pred, trend_pred, vol_pred = self(batch_X)
                
                # 计算损失
                price_loss = price_criterion(price_pred.squeeze(), batch_y_price)
                trend_loss = trend_criterion(trend_pred, batch_y_trend.argmax(dim=1))
                vol_loss = volatility_criterion(vol_pred.squeeze(), batch_y_vol)
                
                # 加权总损失
                total_loss = price_loss + 0.3 * trend_loss + 0.2 * vol_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
            
            # 验证阶段
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y_price, batch_y_trend, batch_y_vol in val_loader:
                    batch_X = batch_X.to(device)
                    batch_y_price = batch_y_price.to(device)
                    batch_y_trend = batch_y_trend.to(device)
                    batch_y_vol = batch_y_vol.to(device)
                    
                    price_pred, trend_pred, vol_pred = self(batch_X)
                    
                    price_loss = price_criterion(price_pred.squeeze(), batch_y_price)
                    trend_loss = trend_criterion(trend_pred, batch_y_trend.argmax(dim=1))
                    vol_loss = volatility_criterion(vol_pred.squeeze(), batch_y_vol)
                    
                    total_loss = price_loss + 0.3 * trend_loss + 0.2 * vol_loss
                    val_loss += total_loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.state_dict(), 'best_lstm_tcn_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"早停在第 {epoch+1} 轮")
                    break
        
        # 加载最佳模型
        self.load_state_dict(torch.load('best_lstm_tcn_model.pth'))
        
        self.history = {'train_loss': train_losses, 'val_loss': val_losses}
        return self.history
    
    def predict_stock(self, data, return_confidence=True):
        """预测"""
        self.eval()
        
        # 准备特征
        data_with_features = self.prepare_features(data)
        
        # 获取最后一个序列
        feature_cols = [
            'close', 'volume', 'returns', 'rsi', 'macd', 
            'bb_position', 'volume_ratio', 'volatility', 'price_position',
            'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20'
        ]
        
        available_cols = [col for col in feature_cols if col in data_with_features.columns]
        
        if len(data_with_features) < self.sequence_length:
            raise ValueError(f"数据长度不足，需要至少 {self.sequence_length} 行数据")
        
        # 获取最后的序列
        last_sequence = data_with_features[available_cols].tail(self.sequence_length).values
        X = np.array([last_sequence])
        
        # 标准化
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        # 预测
        with torch.no_grad():
            price_pred, trend_pred, volatility_pred = self(X_tensor)
        
        # 转换回CPU并获取数值
        price_pred = price_pred.cpu().numpy()
        trend_pred = trend_pred.cpu().numpy()
        volatility_pred = volatility_pred.cpu().numpy()
        
        # 反标准化价格预测
        price_pred_original = self.scaler_y.inverse_transform(price_pred.reshape(-1, 1))[0, 0]
        
        # 趋势预测
        trend_probs = trend_pred[0]
        trend_labels = ['上涨', '下跌', '横盘']
        trend_prediction = trend_labels[np.argmax(trend_probs)]
        
        # 波动率预测
        volatility_prediction = volatility_pred[0, 0]
        
        result = {
            'predicted_price': float(price_pred_original),
            'trend_prediction': trend_prediction,
            'trend_probabilities': {
                '上涨': float(trend_probs[0]),
                '下跌': float(trend_probs[1]),
                '横盘': float(trend_probs[2])
            },
            'predicted_volatility': float(volatility_prediction),
            'confidence_score': float(np.max(trend_probs))
        }
        
        if return_confidence:
            # 计算置信度
            current_price = data_with_features['close'].iloc[-1]
            price_change_pct = (price_pred_original - current_price) / current_price * 100
            
            result.update({
                'current_price': float(current_price),
                'predicted_change_pct': float(price_change_pct),
                'risk_level': self._assess_risk(volatility_prediction, np.max(trend_probs))
            })
        
        return result
    
    def _assess_risk(self, volatility, confidence):
        """评估风险等级"""
        if volatility > 0.03 or confidence < 0.6:
            return "高风险"
        elif volatility > 0.015 or confidence < 0.75:
            return "中风险"
        else:
            return "低风险"
    
    def evaluate_model(self, data):
        """评估模型性能"""
        self.eval()
        
        # 准备数据
        data_with_features = self.prepare_features(data)
        X, y_price, y_trend, y_volatility = self.create_sequences(data_with_features)
        
        # 标准化
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_price_scaled = self.scaler_y.transform(y_price.reshape(-1, 1)).flatten()
        
        # 预测
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        with torch.no_grad():
            price_pred, trend_pred, volatility_pred = self(X_tensor)
        
        # 转换回CPU
        price_pred = price_pred.cpu().numpy()
        trend_pred = trend_pred.cpu().numpy()
        volatility_pred = volatility_pred.cpu().numpy()
        
        # 反标准化价格预测
        price_pred_original = self.scaler_y.inverse_transform(price_pred.reshape(-1, 1)).flatten()
        
        # 计算指标
        mse = mean_squared_error(y_price, price_pred_original)
        mae = mean_absolute_error(y_price, price_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_price, price_pred_original)
        
        # 计算MAPE
        mape = np.mean(np.abs((y_price - price_pred_original) / y_price)) * 100
        
        # 趋势准确率
        trend_accuracy = np.mean(np.argmax(y_trend, axis=1) == np.argmax(trend_pred, axis=1))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape),
            'trend_accuracy': float(trend_accuracy),
            'volatility_mae': float(mean_absolute_error(y_volatility, volatility_pred.flatten()))
        }
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'model_params': {
                'sequence_length': self.sequence_length,
                'n_features': self.n_features,
                'lstm_units': self.lstm_units,
                'tcn_channels': self.tcn_channels,
                'dense_units': self.dense_units,
                'dropout_rate': self.dropout_rate
            }
        }, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        print(f"模型已从 {filepath} 加载")

def get_sample_data():
    """生成示例数据用于测试"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    
    # 生成模拟股价数据
    price = 100
    prices = [price]
    volumes = []
    
    for i in range(999):
        # 随机游走 + 趋势
        change = np.random.normal(0, 0.02) + 0.0001 * np.sin(i / 50)
        price = price * (1 + change)
        prices.append(price)
        
        # 成交量与价格变化相关
        volume = np.random.normal(1000000, 200000) * (1 + abs(change) * 5)
        volumes.append(max(volume, 100000))
    
    volumes.append(volumes[-1])  # 补齐长度
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volumes
    })
    
    return data

if __name__ == "__main__":
    # 示例使用
    print("LSTM-TCN联合股票预测模型 (PyTorch版本)")
    print("=" * 50)
    
    # 生成示例数据
    print("生成示例数据...")
    sample_data = get_sample_data()
    
    # 创建模型
    print("创建LSTM-TCN模型...")
    model = LSTMTCNPredictor(
        sequence_length=60,
        lstm_units=128,
        tcn_channels=[64, 64, 64],
        dense_units=64,
        dropout_rate=0.2
    )
    
    # 训练模型
    print("训练模型...")
    history = model.train_model(sample_data, epochs=50, batch_size=32)
    
    # 评估模型
    print("评估模型...")
    metrics = model.evaluate_model(sample_data)
    print("模型性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 预测
    print("进行预测...")
    prediction = model.predict_stock(sample_data)
    print("预测结果:")
    for key, value in prediction.items():
        print(f"  {key}: {value}")
    
    # 保存模型
    model.save_model('lstm_tcn_stock_model.pth')
    
    print("\n模型训练和测试完成！")