"""
股票短线交易预测模型
支持多种机器学习算法：LSTM、XGBoost、随机森林、SVM
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StockDataset(Dataset):
    """PyTorch数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleLSTM(nn.Module):
    """简单的LSTM模型"""
    def __init__(self, input_size, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(SimpleLSTM, self).__init__()
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
        lstm_out = lstm_out[:, -1, :]
        # Dropout
        lstm_out = self.dropout(lstm_out)
        # 全连接层
        output = self.fc(lstm_out)
        return output

class StockPredictionModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def prepare_data(self, data, target_col='close', lookback_days=5):
        """
        准备训练数据
        """
        # 创建技术指标特征
        data = data.copy()
        
        # 移动平均线
        data['ma5'] = data[target_col].rolling(window=5).mean()
        data['ma10'] = data[target_col].rolling(window=10).mean()
        data['ma20'] = data[target_col].rolling(window=20).mean()
        
        # 价格变化率
        data['price_change'] = data[target_col].pct_change()
        data['price_change_5'] = data[target_col].pct_change(5)
        
        # 波动率
        data['volatility'] = data['price_change'].rolling(window=5).std()
        
        # RSI指标
        delta = data[target_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # 删除包含NaN的行
        data = data.dropna()
        
        # 创建特征和目标变量
        feature_cols = ['ma5', 'ma10', 'ma20', 'price_change', 'price_change_5', 'volatility', 'rsi']
        X = data[feature_cols].values
        
        # 创建目标变量（下一天的价格变化）
        y_price = data[target_col].shift(-1).dropna().values
        X = X[:-1]  # 对应调整X的长度
        
        # 创建分类目标（涨跌）
        y_class = (y_price > data[target_col].iloc[:-1].values).astype(int)
        
        return X, y_price, y_class
    
    def create_lstm_data(self, data, target_col='close', lookback_days=10):
        """
        为LSTM创建时间序列数据
        """
        # 准备基础数据
        X, y_price, y_class = self.prepare_data(data, target_col)
        
        # 创建时间序列数据
        X_lstm, y_lstm = [], []
        for i in range(lookback_days, len(X)):
            X_lstm.append(X[i-lookback_days:i])
            y_lstm.append(y_price[i])
        
        return np.array(X_lstm), np.array(y_lstm)
    
    def build_lstm_model(self, input_shape):
        """
        构建PyTorch LSTM模型
        """
        model = SimpleLSTM(
            input_size=input_shape[1],  # 特征数量
            hidden_size=50,
            num_layers=2,
            output_size=1,
            dropout=0.2
        )
        return model.to(device)
    
    def train_lstm(self, data, target_col='close', lookback_days=10):
        """
        训练LSTM模型
        """
        print("训练LSTM模型...")
        
        # 准备数据
        X, y = self.create_lstm_data(data, target_col, lookback_days)
        
        # 数据标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        # 创建数据集和数据加载器
        train_dataset = StockDataset(X_train, y_train)
        test_dataset = StockDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 构建模型
        model = self.build_lstm_model(X.shape)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        model.train()
        train_losses = []
        
        for epoch in range(50):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        # 评估模型
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 反标准化
        predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # 计算指标
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        
        # 保存模型和缩放器
        self.models['lstm'] = model
        self.scalers['lstm'] = {'X': scaler_X, 'y': scaler_y}
        
        return {
            'model': 'LSTM',
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'train_losses': train_losses
        }
    
    def train_xgboost(self, data, target_type='classification'):
        """
        训练XGBoost模型
        """
        print(f"训练XGBoost模型 ({target_type})...")
        
        X, y_price, y_class = self.prepare_data(data)
        
        if target_type == 'classification':
            y = y_class
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            y = y_price
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 保存模型
        self.models[f'xgboost_{target_type}'] = model
        
        if target_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            return {
                'model': f'XGBoost ({target_type})',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            return {
                'model': f'XGBoost ({target_type})',
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
    
    def train_random_forest(self, data, target_type='classification'):
        """
        训练随机森林模型
        """
        print(f"训练随机森林模型 ({target_type})...")
        
        X, y_price, y_class = self.prepare_data(data)
        
        if target_type == 'classification':
            y = y_class
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            y = y_price
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 保存模型
        self.models[f'rf_{target_type}'] = model
        
        if target_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            return {
                'model': f'Random Forest ({target_type})',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            return {
                'model': f'Random Forest ({target_type})',
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
    
    def train_svm(self, data, target_type='classification'):
        """
        训练SVM模型
        """
        print(f"训练SVM模型 ({target_type})...")
        
        X, y_price, y_class = self.prepare_data(data)
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if target_type == 'classification':
            y = y_class
            model = SVC(kernel='rbf', C=1.0, random_state=42)
        else:
            y = y_price
            model = SVR(kernel='rbf', C=1.0)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 保存模型和缩放器
        self.models[f'svm_{target_type}'] = model
        self.scalers[f'svm_{target_type}'] = scaler
        
        if target_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            return {
                'model': f'SVM ({target_type})',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            return {
                'model': f'SVM ({target_type})',
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
    
    def compare_models(self, data):
        """
        比较所有模型的性能
        """
        print("开始模型比较...")
        
        # 训练所有模型
        results = []
        
        # LSTM
        try:
            lstm_result = self.train_lstm(data)
            results.append(lstm_result)
            self.results['lstm'] = lstm_result
        except Exception as e:
            print(f"LSTM训练失败: {e}")
        
        # XGBoost
        try:
            xgb_class_result = self.train_xgboost(data, 'classification')
            xgb_reg_result = self.train_xgboost(data, 'regression')
            results.extend([xgb_class_result, xgb_reg_result])
            self.results['xgboost_classification'] = xgb_class_result
            self.results['xgboost_regression'] = xgb_reg_result
        except Exception as e:
            print(f"XGBoost训练失败: {e}")
        
        # Random Forest
        try:
            rf_class_result = self.train_random_forest(data, 'classification')
            rf_reg_result = self.train_random_forest(data, 'regression')
            results.extend([rf_class_result, rf_reg_result])
            self.results['rf_classification'] = rf_class_result
            self.results['rf_regression'] = rf_reg_result
        except Exception as e:
            print(f"Random Forest训练失败: {e}")
        
        # SVM
        try:
            svm_class_result = self.train_svm(data, 'classification')
            svm_reg_result = self.train_svm(data, 'regression')
            results.extend([svm_class_result, svm_reg_result])
            self.results['svm_classification'] = svm_class_result
            self.results['svm_regression'] = svm_reg_result
        except Exception as e:
            print(f"SVM训练失败: {e}")
        
        return results
    
    def print_results(self):
        """
        打印所有模型的结果
        """
        print("\n" + "="*60)
        print("模型性能比较结果")
        print("="*60)
        
        for model_name, result in self.results.items():
            print(f"\n{result['model']}:")
            for metric, value in result.items():
                if metric != 'model' and metric != 'train_losses':
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
    
    def get_recommendation(self):
        """
        基于模型结果给出推荐
        """
        print("\n" + "="*60)
        print("模型推荐")
        print("="*60)
        
        print("\n📊 分类任务推荐 (预测涨跌):")
        print("1. XGBoost: 在金融数据上表现优异，特征重要性清晰")
        print("2. Random Forest: 稳定性好，抗过拟合")
        print("3. SVM: 在小数据集上表现良好")
        
        print("\n📈 回归任务推荐 (预测价格):")
        print("1. LSTM: 擅长时间序列预测，能捕捉长期依赖")
        print("2. XGBoost: 非线性关系处理能力强")
        print("3. Random Forest: 稳定可靠的基准模型")
        
        print("\n💡 实际应用建议:")
        print("- 短线交易: 推荐XGBoost分类模型")
        print("- 价格预测: 推荐LSTM回归模型")
        print("- 风险控制: 推荐Random Forest（稳定性好）")
        print("- 模型集成: 结合多个模型的预测结果")

if __name__ == "__main__":
    # 生成示例数据
    print("股票预测模型测试")
    print("请确保已安装所需依赖：pip install torch xgboost scikit-learn")
    
    # 生成模拟股票数据
    import pandas as pd
    from datetime import datetime, timedelta
    
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    np.random.seed(42)
    
    # 生成模拟价格数据
    price = 100
    prices = [price]
    for i in range(499):
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        prices.append(price)
    
    data = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    # 创建模型比较器
    models = StockPredictionModels()
    
    # 比较所有模型
    results = models.compare_models(data)
    
    # 打印结果
    models.print_results()
    
    # 给出推荐
    models.get_recommendation()