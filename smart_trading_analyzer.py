#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能股票交易分析器
基于RTX 4090深度学习的短期交易策略系统
- 获取50支科技股票数据
- 识别低价股票机会
- 1-2天短期价格预测
- 考虑交易成本的最佳买卖点推荐
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from real_stock_data_fetcher import RealStockDataFetcher

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class TradingCostCalculator:
    """交易成本计算器"""
    
    def __init__(self):
        # A股交易费用标准（2024年）
        self.stamp_tax_rate = 0.001  # 印花税 0.1%（仅卖出收取）
        self.commission_rate = 0.0003  # 佣金 0.03%（买卖双向，最低5元）
        self.transfer_fee_rate = 0.00002  # 过户费 0.002%（买卖双向）
        self.min_commission = 5.0  # 最低佣金5元
    
    def calculate_buy_cost(self, price: float, shares: int) -> Dict[str, float]:
        """计算买入成本"""
        total_amount = price * shares
        
        # 佣金（最低5元）
        commission = max(total_amount * self.commission_rate, self.min_commission)
        
        # 过户费
        transfer_fee = total_amount * self.transfer_fee_rate
        
        # 总买入成本
        total_cost = total_amount + commission + transfer_fee
        
        return {
            'stock_amount': total_amount,
            'commission': commission,
            'transfer_fee': transfer_fee,
            'total_cost': total_cost,
            'cost_per_share': total_cost / shares
        }
    
    def calculate_sell_revenue(self, price: float, shares: int) -> Dict[str, float]:
        """计算卖出收入"""
        total_amount = price * shares
        
        # 印花税
        stamp_tax = total_amount * self.stamp_tax_rate
        
        # 佣金（最低5元）
        commission = max(total_amount * self.commission_rate, self.min_commission)
        
        # 过户费
        transfer_fee = total_amount * self.transfer_fee_rate
        
        # 总费用
        total_fees = stamp_tax + commission + transfer_fee
        
        # 实际收入
        net_revenue = total_amount - total_fees
        
        return {
            'stock_amount': total_amount,
            'stamp_tax': stamp_tax,
            'commission': commission,
            'transfer_fee': transfer_fee,
            'total_fees': total_fees,
            'net_revenue': net_revenue,
            'revenue_per_share': net_revenue / shares
        }
    
    def calculate_profit(self, buy_price: float, sell_price: float, shares: int) -> Dict[str, float]:
        """计算交易利润"""
        buy_info = self.calculate_buy_cost(buy_price, shares)
        sell_info = self.calculate_sell_revenue(sell_price, shares)
        
        profit = sell_info['net_revenue'] - buy_info['total_cost']
        profit_rate = (profit / buy_info['total_cost']) * 100
        
        return {
            'buy_cost': buy_info['total_cost'],
            'sell_revenue': sell_info['net_revenue'],
            'profit': profit,
            'profit_rate': profit_rate,
            'total_fees': buy_info['commission'] + buy_info['transfer_fee'] + sell_info['total_fees']
        }

class ShortTermPredictor(nn.Module):
    """短期价格预测模型（1-2天）"""
    
    def __init__(self, input_size=20, hidden_size=128, num_layers=3, dropout=0.2):
        super(ShortTermPredictor, self).__init__()
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 批标准化
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 预测1天和2天后的价格
        )
        
    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步
        last_hidden = attn_out[:, -1, :]
        
        # 批标准化
        normalized = self.batch_norm(last_hidden)
        
        # 全连接层
        output = self.fc_layers(normalized)
        
        return output

class LowPriceStockAnalyzer:
    """低价股票分析器"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # 成交量指标
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        
        # 价格位置
        df['price_position'] = (df['close'] - df['low'].rolling(window=60).min()) / \
                              (df['high'].rolling(window=60).max() - df['low'].rolling(window=60).min())
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """计算MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2):
        """计算布林带"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, ma, lower
    
    def identify_low_price_opportunities(self, df: pd.DataFrame) -> Dict[str, float]:
        """识别低价机会"""
        latest = df.iloc[-1]
        
        # 低价信号评分
        signals = {}
        
        # RSI超卖信号（RSI < 30）
        signals['rsi_oversold'] = 1.0 if latest['rsi'] < 30 else 0.0
        
        # 价格接近布林带下轨
        bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
        signals['bb_low'] = 1.0 if bb_position < 0.2 else 0.0
        
        # MACD金叉信号
        signals['macd_golden'] = 1.0 if (latest['macd'] > latest['macd_signal'] and 
                                        df.iloc[-2]['macd'] <= df.iloc[-2]['macd_signal']) else 0.0
        
        # 价格在历史低位
        signals['price_low'] = 1.0 if latest['price_position'] < 0.3 else 0.0
        
        # 成交量放大
        signals['volume_surge'] = 1.0 if latest['volume_ratio'] > 1.5 else 0.0
        
        # 短期均线支撑
        signals['ma_support'] = 1.0 if (latest['close'] > latest['ma5'] and 
                                       latest['ma5'] > latest['ma10']) else 0.0
        
        # 综合评分
        total_score = sum(signals.values())
        
        return {
            'signals': signals,
            'total_score': total_score,
            'max_score': len(signals),
            'opportunity_level': total_score / len(signals)
        }

class SmartTradingAnalyzer:
    """智能交易分析器主类"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 使用设备: {self.device}")
        
        self.data_fetcher = RealStockDataFetcher()
        self.cost_calculator = TradingCostCalculator()
        self.low_price_analyzer = LowPriceStockAnalyzer()
        self.scaler = MinMaxScaler()
        
        # 模型参数
        self.sequence_length = 30
        self.prediction_days = 2
        
    def fetch_all_tech_stocks_data(self) -> pd.DataFrame:
        """获取所有50支科技股票数据"""
        print("📊 正在获取50支科技股票的历史数据...")
        
        try:
            df = self.data_fetcher.fetch_all_stocks()
            if not df.empty:
                print(f"✅ 成功获取 {df['stock_code'].nunique()} 支股票的数据")
                print(f"📈 数据范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
                return df
            else:
                print("❌ 未能获取股票数据")
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ 获取数据时发生错误: {str(e)}")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """准备模型特征"""
        # 计算技术指标
        df = self.low_price_analyzer.calculate_technical_indicators(df)
        
        # 选择特征
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'ma5', 'ma10', 'ma20', 'ma60',
            'volume_ma5', 'volume_ratio', 'price_position'
        ]
        
        # 填充缺失值
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # 提取特征
        features = df[feature_columns].values
        
        # 标准化
        features = self.scaler.fit_transform(features)
        
        return features
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列"""
        X, y = [], []
        
        for i in range(self.sequence_length, len(data) - self.prediction_days + 1):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i:i+self.prediction_days])
        
        return np.array(X), np.array(y)
    
    def train_prediction_model(self, df: pd.DataFrame, stock_code: str) -> ShortTermPredictor:
        """训练短期预测模型"""
        print(f"🤖 正在为 {stock_code} 训练预测模型...")
        
        # 准备数据
        stock_data = df[df['stock_code'] == stock_code].copy()
        stock_data = stock_data.sort_values('date').reset_index(drop=True)
        
        if len(stock_data) < self.sequence_length + self.prediction_days:
            print(f"❌ {stock_code} 数据不足，跳过训练")
            return None
        
        # 准备特征和目标
        features = self.prepare_features(stock_data)
        targets = stock_data['close'].values
        
        # 创建序列
        X, y = self.create_sequences(features, targets)
        
        if len(X) == 0:
            print(f"❌ {stock_code} 无法创建有效序列")
            return None
        
        # 分割训练集和验证集
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # 创建模型
        model = ShortTermPredictor(
            input_size=features.shape[1],
            hidden_size=128,
            num_layers=3,
            dropout=0.2
        ).to(self.device)
        
        # 训练参数
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(200):
            # 训练
            model.train()
            train_loss = 0
            
            # 批量训练
            batch_size = 32
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            scheduler.step(val_loss)
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss/len(X_train):.6f}, Val Loss = {val_loss:.6f}")
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        print(f"✅ {stock_code} 模型训练完成，最佳验证损失: {best_val_loss:.6f}")
        
        return model
    
    def predict_short_term_prices(self, model: ShortTermPredictor, df: pd.DataFrame, stock_code: str) -> Dict[str, float]:
        """预测短期价格"""
        stock_data = df[df['stock_code'] == stock_code].copy()
        stock_data = stock_data.sort_values('date').reset_index(drop=True)
        
        # 准备最新数据
        features = self.prepare_features(stock_data)
        
        if len(features) < self.sequence_length:
            return None
        
        # 获取最新序列
        latest_sequence = features[-self.sequence_length:]
        latest_sequence = torch.FloatTensor(latest_sequence).unsqueeze(0).to(self.device)
        
        # 预测
        model.eval()
        with torch.no_grad():
            predictions = model(latest_sequence)
            predictions = predictions.cpu().numpy()[0]
        
        current_price = stock_data.iloc[-1]['close']
        
        return {
            'current_price': current_price,
            'predicted_1day': predictions[0],
            'predicted_2day': predictions[1],
            'change_1day': (predictions[0] - current_price) / current_price * 100,
            'change_2day': (predictions[1] - current_price) / current_price * 100
        }
    
    def calculate_optimal_trading_strategy(self, stock_code: str, current_price: float, 
                                         predicted_prices: Dict[str, float], 
                                         opportunity_score: float) -> Dict[str, any]:
        """计算最优交易策略"""
        
        # 基础投资金额（可调整）
        base_investment = 10000  # 1万元
        
        # 根据机会评分调整投资金额
        investment_amount = base_investment * (0.5 + opportunity_score * 0.5)
        
        # 计算可买股数（100股为一手）
        shares_per_lot = 100
        max_shares = int(investment_amount / current_price / shares_per_lot) * shares_per_lot
        
        if max_shares == 0:
            return None
        
        strategies = []
        
        # 策略1: 1天持有
        if predicted_prices['change_1day'] > 0:
            sell_price_1day = predicted_prices['predicted_1day']
            profit_info_1day = self.cost_calculator.calculate_profit(
                current_price, sell_price_1day, max_shares
            )
            
            if profit_info_1day['profit'] > 0:
                strategies.append({
                    'strategy': '1天持有',
                    'buy_price': current_price,
                    'sell_price': sell_price_1day,
                    'shares': max_shares,
                    'investment': profit_info_1day['buy_cost'],
                    'expected_profit': profit_info_1day['profit'],
                    'profit_rate': profit_info_1day['profit_rate'],
                    'total_fees': profit_info_1day['total_fees'],
                    'holding_days': 1
                })
        
        # 策略2: 2天持有
        if predicted_prices['change_2day'] > 0:
            sell_price_2day = predicted_prices['predicted_2day']
            profit_info_2day = self.cost_calculator.calculate_profit(
                current_price, sell_price_2day, max_shares
            )
            
            if profit_info_2day['profit'] > 0:
                strategies.append({
                    'strategy': '2天持有',
                    'buy_price': current_price,
                    'sell_price': sell_price_2day,
                    'shares': max_shares,
                    'investment': profit_info_2day['buy_cost'],
                    'expected_profit': profit_info_2day['profit'],
                    'profit_rate': profit_info_2day['profit_rate'],
                    'total_fees': profit_info_2day['total_fees'],
                    'holding_days': 2
                })
        
        # 选择最优策略（利润率最高）
        if strategies:
            best_strategy = max(strategies, key=lambda x: x['profit_rate'])
            return {
                'stock_code': stock_code,
                'opportunity_score': opportunity_score,
                'all_strategies': strategies,
                'recommended_strategy': best_strategy
            }
        
        return None
    
    def analyze_all_stocks(self) -> List[Dict[str, any]]:
        """分析所有股票"""
        print("🔍 开始全面分析50支科技股票...")
        
        # 获取数据
        all_data = self.fetch_all_tech_stocks_data()
        if all_data.empty:
            print("❌ 无法获取股票数据")
            return []
        
        results = []
        stock_codes = all_data['stock_code'].unique()
        
        for i, stock_code in enumerate(stock_codes, 1):
            print(f"\n📈 分析进度: {i}/{len(stock_codes)} - {stock_code}")
            
            try:
                # 获取单只股票数据
                stock_data = all_data[all_data['stock_code'] == stock_code].copy()
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                
                if len(stock_data) < 60:  # 至少需要60天数据
                    print(f"⚠️ {stock_code} 数据不足，跳过")
                    continue
                
                # 计算技术指标
                stock_data = self.low_price_analyzer.calculate_technical_indicators(stock_data)
                
                # 识别低价机会
                opportunity_analysis = self.low_price_analyzer.identify_low_price_opportunities(stock_data)
                
                # 如果机会评分太低，跳过
                if opportunity_analysis['opportunity_level'] < 0.3:
                    print(f"⚠️ {stock_code} 机会评分过低 ({opportunity_analysis['opportunity_level']:.2f})，跳过")
                    continue
                
                # 训练预测模型
                model = self.train_prediction_model(all_data, stock_code)
                if model is None:
                    continue
                
                # 预测价格
                predictions = self.predict_short_term_prices(model, all_data, stock_code)
                if predictions is None:
                    continue
                
                # 计算交易策略
                strategy = self.calculate_optimal_trading_strategy(
                    stock_code, predictions['current_price'], predictions, 
                    opportunity_analysis['opportunity_level']
                )
                
                if strategy is not None:
                    # 添加股票名称
                    stock_name = next((name for code, name in self.data_fetcher.tech_stocks if code == stock_code), stock_code)
                    strategy['stock_name'] = stock_name
                    strategy['predictions'] = predictions
                    strategy['opportunity_analysis'] = opportunity_analysis
                    
                    results.append(strategy)
                    print(f"✅ {stock_code} 分析完成，发现盈利机会!")
                
            except Exception as e:
                print(f"❌ {stock_code} 分析失败: {str(e)}")
                continue
        
        return results
    
    def generate_trading_report(self, results: List[Dict[str, any]]) -> str:
        """生成交易报告"""
        if not results:
            return "未发现任何盈利机会"
        
        # 按利润率排序
        results.sort(key=lambda x: x['recommended_strategy']['profit_rate'], reverse=True)
        
        report = []
        report.append("🎯 智能股票交易分析报告")
        report.append("=" * 60)
        report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"发现机会: {len(results)} 支股票")
        report.append("")
        
        # 总体统计
        total_investment = sum(r['recommended_strategy']['investment'] for r in results)
        total_profit = sum(r['recommended_strategy']['expected_profit'] for r in results)
        avg_profit_rate = np.mean([r['recommended_strategy']['profit_rate'] for r in results])
        
        report.append("📊 总体统计:")
        report.append(f"总投资金额: ¥{total_investment:,.2f}")
        report.append(f"预期总利润: ¥{total_profit:,.2f}")
        report.append(f"平均利润率: {avg_profit_rate:.2f}%")
        report.append("")
        
        # 详细推荐
        report.append("🏆 推荐交易机会 (按利润率排序):")
        report.append("-" * 60)
        
        for i, result in enumerate(results[:10], 1):  # 显示前10个机会
            strategy = result['recommended_strategy']
            predictions = result['predictions']
            
            report.append(f"{i}. {result['stock_name']} ({result['stock_code']})")
            report.append(f"   机会评分: {result['opportunity_score']:.2f}/1.0")
            report.append(f"   当前价格: ¥{predictions['current_price']:.2f}")
            report.append(f"   推荐策略: {strategy['strategy']}")
            report.append(f"   建议买入: ¥{strategy['buy_price']:.2f} × {strategy['shares']} 股")
            report.append(f"   目标卖出: ¥{strategy['sell_price']:.2f}")
            report.append(f"   投资金额: ¥{strategy['investment']:,.2f}")
            report.append(f"   预期利润: ¥{strategy['expected_profit']:,.2f}")
            report.append(f"   利润率: {strategy['profit_rate']:.2f}%")
            report.append(f"   交易费用: ¥{strategy['total_fees']:.2f}")
            report.append("")
        
        # 风险提示
        report.append("⚠️ 风险提示:")
        report.append("1. 本分析基于历史数据和AI预测，不构成投资建议")
        report.append("2. 股市有风险，投资需谨慎")
        report.append("3. 实际交易中可能存在滑点和流动性风险")
        report.append("4. 建议分散投资，控制单只股票仓位")
        
        return "\n".join(report)

def main():
    """主函数"""
    print("🚀 智能股票交易分析器")
    print("基于RTX 4090深度学习的短期交易策略系统")
    print("=" * 60)
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 检测到GPU: {gpu_name}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️ 未检测到CUDA GPU，将使用CPU运行")
    
    # 创建分析器
    analyzer = SmartTradingAnalyzer()
    
    # 开始分析
    start_time = time.time()
    results = analyzer.analyze_all_stocks()
    end_time = time.time()
    
    print(f"\n⏱️ 分析耗时: {end_time - start_time:.1f} 秒")
    
    # 生成报告
    report = analyzer.generate_trading_report(results)
    print("\n" + report)
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"trading_analysis_report_{timestamp}.txt"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存详细数据
    if results:
        results_filename = f"trading_analysis_data_{timestamp}.json"
        with open(results_filename, 'w', encoding='utf-8') as f:
            # 转换numpy类型为Python原生类型
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_numpy(results), f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 报告已保存:")
        print(f"📄 文本报告: {report_filename}")
        print(f"📊 详细数据: {results_filename}")

if __name__ == "__main__":
    main()