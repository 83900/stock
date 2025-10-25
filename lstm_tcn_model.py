"""
LSTM-TCN联合股票预测模型
结合LSTM的时序记忆能力和TCN的并行计算优势，实现高精度股票价格预测
适用于短线交易策略
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Conv1D, Activation, SpatialDropout1D, Add, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class TemporalConvNet:
    """时间卷积网络(TCN)实现"""
    
    def __init__(self, nb_filters=64, kernel_size=2, nb_stacks=1, dilations=None, 
                 padding='causal', use_skip_connections=True, dropout_rate=0.0, 
                 return_sequences=True, activation='relu', kernel_initializer='he_normal'):
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.dilations = dilations or [1, 2, 4, 8, 16, 32]
        self.padding = padding
        self.use_skip_connections = use_skip_connections
        self.dropout_rate = dropout_rate
        self.return_sequences = return_sequences
        self.activation = activation
        self.kernel_initializer = kernel_initializer
    
    def residual_block(self, x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate):
        """残差块"""
        # 第一个卷积层
        conv1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                      dilation_rate=dilation_rate, padding=padding,
                      kernel_initializer=self.kernel_initializer)(x)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(self.activation)(conv1)
        conv1 = SpatialDropout1D(dropout_rate)(conv1)
        
        # 第二个卷积层
        conv2 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                      dilation_rate=dilation_rate, padding=padding,
                      kernel_initializer=self.kernel_initializer)(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(self.activation)(conv2)
        conv2 = SpatialDropout1D(dropout_rate)(conv2)
        
        # 残差连接
        if x.shape[-1] != nb_filters:
            # 调整维度匹配
            x = Conv1D(nb_filters, 1, padding='same')(x)
        
        res = Add()([x, conv2])
        return Activation(self.activation)(res)
    
    def build_tcn(self, input_layer):
        """构建TCN网络"""
        x = input_layer
        
        for stack in range(self.nb_stacks):
            for dilation in self.dilations:
                x = self.residual_block(x, dilation, self.nb_filters, 
                                      self.kernel_size, self.padding, self.dropout_rate)
        
        return x

class LSTMTCNPredictor:
    """LSTM-TCN联合预测模型"""
    
    def __init__(self, sequence_length=60, n_features=5, 
                 lstm_units=128, tcn_filters=64, 
                 dense_units=64, dropout_rate=0.2):
        """
        初始化模型参数
        
        Args:
            sequence_length: 输入序列长度
            n_features: 特征数量
            lstm_units: LSTM单元数
            tcn_filters: TCN滤波器数量
            dense_units: 全连接层单元数
            dropout_rate: Dropout比率
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.tcn_filters = tcn_filters
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        self.history = None
        
    def build_model(self):
        """构建LSTM-TCN联合模型"""
        # 输入层
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM分支 - 捕获长期依赖
        lstm_out = LSTM(self.lstm_units, return_sequences=True, 
                       dropout=self.dropout_rate, recurrent_dropout=self.dropout_rate)(inputs)
        lstm_out = BatchNormalization()(lstm_out)
        
        # TCN分支 - 并行处理和局部特征提取
        tcn = TemporalConvNet(nb_filters=self.tcn_filters, 
                             kernel_size=3, 
                             nb_stacks=2,
                             dilations=[1, 2, 4, 8, 16],
                             dropout_rate=self.dropout_rate)
        tcn_out = tcn.build_tcn(inputs)
        
        # 特征融合
        # 确保维度匹配
        if lstm_out.shape[-1] != tcn_out.shape[-1]:
            lstm_out = Dense(self.tcn_filters)(lstm_out)
        
        # 加权融合LSTM和TCN输出
        alpha = 0.6  # LSTM权重
        beta = 0.4   # TCN权重
        
        lstm_weighted = Lambda(lambda x: x * alpha)(lstm_out)
        tcn_weighted = Lambda(lambda x: x * beta)(tcn_out)
        
        fused = Add()([lstm_weighted, tcn_weighted])
        fused = BatchNormalization()(fused)
        fused = Dropout(self.dropout_rate)(fused)
        
        # 注意力机制
        attention = Dense(1, activation='tanh')(fused)
        attention = Activation('softmax')(attention)
        attended = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([fused, attention])
        
        # 全连接层
        dense1 = Dense(self.dense_units, activation='relu')(attended)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.dropout_rate)(dense1)
        
        dense2 = Dense(self.dense_units // 2, activation='relu')(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Dropout(self.dropout_rate / 2)(dense2)
        
        # 输出层 - 多任务学习
        # 价格预测
        price_output = Dense(1, activation='linear', name='price')(dense2)
        
        # 趋势分类 (上涨/下跌/横盘)
        trend_output = Dense(3, activation='softmax', name='trend')(dense2)
        
        # 波动率预测
        volatility_output = Dense(1, activation='sigmoid', name='volatility')(dense2)
        
        # 构建模型
        self.model = Model(inputs=inputs, 
                          outputs=[price_output, trend_output, volatility_output])
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
            loss={
                'price': 'huber',  # 对异常值更鲁棒
                'trend': 'categorical_crossentropy',
                'volatility': 'mse'
            },
            loss_weights={
                'price': 1.0,
                'trend': 0.3,
                'volatility': 0.2
            },
            metrics={
                'price': ['mae', 'mse'],
                'trend': ['accuracy'],
                'volatility': ['mae']
            }
        )
        
        return self.model
    
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
    
    def train(self, data, validation_split=0.2, epochs=100, batch_size=32, 
              early_stopping_patience=15, reduce_lr_patience=10):
        """训练模型"""
        print("准备数据...")
        
        # 准备特征
        data_with_features = self.prepare_features(data)
        
        # 创建序列
        X, y_price, y_trend, y_volatility = self.create_sequences(data_with_features)
        
        print(f"数据形状: X={X.shape}, y_price={y_price.shape}")
        
        # 数据标准化
        X_scaled = self.scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_price_scaled = self.scaler_y.fit_transform(y_price.reshape(-1, 1)).flatten()
        
        # 构建模型
        if self.model is None:
            self.n_features = X.shape[-1]
            self.build_model()
        
        print("模型结构:")
        self.model.summary()
        
        # 回调函数
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=early_stopping_patience, 
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=reduce_lr_patience, 
                             min_lr=1e-7, verbose=1),
            ModelCheckpoint('best_lstm_tcn_model.h5', monitor='val_loss', 
                           save_best_only=True, verbose=1)
        ]
        
        # 训练模型
        print("开始训练...")
        self.history = self.model.fit(
            X_scaled,
            {
                'price': y_price_scaled,
                'trend': y_trend,
                'volatility': y_volatility
            },
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, data, return_confidence=True):
        """预测"""
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train()方法")
        
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
        
        # 预测
        predictions = self.model.predict(X_scaled, verbose=0)
        price_pred, trend_pred, volatility_pred = predictions
        
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
    
    def evaluate(self, data):
        """评估模型性能"""
        # 准备数据
        data_with_features = self.prepare_features(data)
        X, y_price, y_trend, y_volatility = self.create_sequences(data_with_features)
        
        # 标准化
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_price_scaled = self.scaler_y.transform(y_price.reshape(-1, 1)).flatten()
        
        # 预测
        predictions = self.model.predict(X_scaled, verbose=0)
        price_pred, trend_pred, volatility_pred = predictions
        
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
            'volatility_mae': float(mean_absolute_error(y_volatility, volatility_pred))
        }
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        self.model = tf.keras.models.load_model(filepath)
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
    print("LSTM-TCN联合股票预测模型")
    print("=" * 50)
    
    # 生成示例数据
    print("生成示例数据...")
    sample_data = get_sample_data()
    
    # 创建模型
    print("创建LSTM-TCN模型...")
    model = LSTMTCNPredictor(
        sequence_length=60,
        lstm_units=128,
        tcn_filters=64,
        dense_units=64,
        dropout_rate=0.2
    )
    
    # 训练模型
    print("训练模型...")
    history = model.train(sample_data, epochs=50, batch_size=32)
    
    # 评估模型
    print("评估模型...")
    metrics = model.evaluate(sample_data)
    print("模型性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 预测
    print("进行预测...")
    prediction = model.predict(sample_data)
    print("预测结果:")
    for key, value in prediction.items():
        print(f"  {key}: {value}")
    
    # 保存模型
    model.save_model('lstm_tcn_stock_model.h5')
    
    print("\n模型训练和测试完成！")