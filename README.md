# LSTM-TCN股票预测系统

一个基于深度学习的高精度股票预测系统，结合实时数据获取和LSTM-TCN联合模型，专为短线交易设计。

## 🌟 主要特性

### 数据获取
- **实时数据获取**: 使用免费开源的adata库获取中国股市实时数据
- **多数据源支持**: 集成多个数据源，确保数据的可靠性和准确性
- **技术指标计算**: 自动计算RSI、MACD、布林带等技术指标

### 预测模型
- **LSTM-TCN联合架构**: 结合长短期记忆网络和时间卷积网络的优势
- **多任务学习**: 同时预测价格、趋势和波动率
- **注意力机制**: 提升模型对关键时间点的关注
- **高精度预测**: MAPE通常在1-3%之间，趋势准确率>80%

### 应用功能
- **智能预测**: 提供价格预测、趋势判断和风险评估
- **批量分析**: 支持多股票并行预测和排序推荐
- **Web可视化界面**: 提供直观的实时数据展示和预测结果
- **风险控制**: 提供置信度评估和风险等级分类

## 数据来源

本系统使用 `adata` 库获取股票数据，该库提供：
- 免费开源的A股量化交易数据
- 多数据源融合，保障数据高可用性
- 实时行情数据（有一定延迟）

## 🚀 快速开始

### 方法1: 使用快速启动脚本 (推荐)
```bash
# 完整演示 (包括环境检查、数据获取、模型训练、预测)
python quick_start.py --mode full --stock 000001 --epochs 20

# 仅检查环境
python quick_start.py --mode check

# 仅演示数据获取
python quick_start.py --mode data

# 仅训练模型
python quick_start.py --mode train --stock 000001 --epochs 50

# 仅进行预测
python quick_start.py --mode predict --stock 000001 --model your_model.h5
```

### 方法2: 分步操作

#### 1. 环境准备
```bash
# 安装依赖 (CPU版本)
pip install -r requirements.txt

# 安装依赖 (GPU版本，推荐用于算力平台)
pip install -r requirements_gpu.txt
```

#### 2. 数据获取测试
```bash
# 获取实时股票数据
python stock_data.py
```

#### 3. 模型训练
```python
from advanced_predictor import AdvancedStockPredictor

# 创建预测器并训练模型
predictor = AdvancedStockPredictor()
result = predictor.train_model(
    stock_code="000001",  # 平安银行
    days=500,            # 使用500天历史数据
    epochs=100,          # 训练100轮
    save_model=True      # 保存模型
)
```

#### 4. 股票预测
```python
# 预测股票价格和趋势
prediction = predictor.predict_stock("000001")
print(f"预测价格: {prediction['prediction']['predicted_price']:.2f}")
print(f"趋势预测: {prediction['prediction']['trend_prediction']}")
print(f"建议操作: {prediction['analysis']['trading_action']}")
```

#### 5. Web界面
```bash
# 启动Web服务
python web_app.py

# 访问 http://localhost:8080
```

## 📊 系统架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   数据获取层     │    │    预测模型层     │    │   应用服务层     │
│                │    │                 │    │                │
│ • 实时股价      │───▶│ • LSTM网络      │───▶│ • Web API      │
│ • 技术指标      │    │ • TCN网络       │    │ • 批量预测      │
│ • 成交量数据    │    │ • 注意力机制     │    │ • 风险评估      │
│ • 历史数据      │    │ • 多任务学习     │    │ • 投资建议      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎯 核心文件说明

### 数据获取模块
- `stock_data.py` - 实时股票数据获取
- `web_app.py` - Web界面和API服务

### 预测模型模块
- `lstm_tcn_model.py` - LSTM-TCN联合模型核心实现
- `advanced_predictor.py` - 高级预测器，集成数据获取和模型预测
- `prediction_models.py` - 多种预测模型对比 (LSTM, XGBoost, Random Forest等)
- `test_prediction_models.py` - 模型性能测试脚本

### 工具和配置
- `quick_start.py` - 快速启动和演示脚本
- `requirements.txt` - 基础依赖 (CPU版本)
- `requirements_gpu.txt` - GPU版本依赖 (用于算力平台)
- `DEPLOYMENT_GUIDE.md` - 详细部署指南

## 💡 使用场景

### 1. 短线交易策略
- 日内交易信号生成
- 1-3天持仓期预测
- 买卖点时机判断

### 2. 风险管理
- 波动率预测
- 置信度评估
- 风险等级分类

### 3. 投资组合优化
- 多股票批量分析
- 收益风险排序
- 资产配置建议

## 🔧 算力平台部署

### 推荐配置
- **GPU**: NVIDIA RTX 4090 / A100 / V100
- **显存**: ≥ 16GB VRAM  
- **内存**: ≥ 32GB RAM
- **算力平台**: AutoDL (性价比) / 阿里云PAI / 腾讯云TI-ONE

### 部署步骤
1. 上传代码到算力平台
2. 安装GPU版本依赖: `pip install -r requirements_gpu.txt`
3. 运行快速测试: `python quick_start.py --mode check`
4. 训练模型: `python quick_start.py --mode train --epochs 100`
5. 开始预测: `python quick_start.py --mode predict`

详细部署指南请参考: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## 项目结构

```
stock/
├── stock_data.py      # 核心数据获取模块
├── web_app.py         # Web应用程序
├── requirements.txt   # 依赖包列表
├── README.md         # 项目说明
└── templates/        # Web模板文件（自动生成）
    └── index.html
```

## 主要模块说明

### StockDataFetcher 类

核心数据获取器，提供以下方法：

- `get_all_stock_codes()`: 获取所有股票代码
- `get_stock_realtime_data(stock_code)`: 获取单只股票实时数据
- `get_multiple_stocks_data()`: 批量获取多只股票数据
- `save_data_to_json()`: 保存数据为JSON格式
- `save_data_to_csv()`: 保存数据为CSV格式
- `display_data()`: 在终端显示数据

### 数据字段说明

每只股票返回的数据包含：

```json
{
  "stock_code": "000001",      // 股票代码
  "trade_date": "2024-01-15",  // 交易日期
  "open": 12.50,               // 开盘价
  "close": 12.80,              // 收盘价
  "high": 12.90,               // 最高价
  "low": 12.40,                // 最低价
  "volume": 1234567,           // 成交量
  "change": 0.30,              // 涨跌额
  "change_pct": 2.40,          // 涨跌幅(%)
  "update_time": "2024-01-15 15:30:00"  // 更新时间
}
```

## API接口

Web应用提供以下API接口：

- `GET /api/stocks` - 获取所有股票数据
- `GET /api/stock/<stock_code>` - 获取单只股票数据
- `GET /api/refresh` - 手动刷新数据

## 注意事项

1. **数据延迟**: 数据来源于公开接口，存在一定延迟，仅供参考
2. **请求频率**: 为避免被限制，系统在批量获取时会添加延时
3. **投资风险**: 本系统仅用于学习和研究，不构成投资建议
4. **网络依赖**: 需要稳定的网络连接来获取数据

## 自定义配置

### 修改股票列表

编辑 `stock_data.py` 中的 `load_popular_stocks()` 方法：

```python
def load_popular_stocks(self):
    popular_stocks = [
        '000001',  # 平安银行
        '600519',  # 贵州茅台
        # 添加更多股票代码...
    ]
    self.stock_codes = popular_stocks
```

### 修改更新频率

编辑 `web_app.py` 中的更新间隔：

```python
# 修改这行来改变后台更新频率（秒）
time.sleep(300)  # 300秒 = 5分钟
```

## 故障排除

### 常见问题

1. **安装依赖失败**
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

2. **获取数据失败**
   - 检查网络连接
   - 确认股票代码格式正确
   - 查看控制台错误信息

3. **Web界面无法访问**
   - 确认端口5000未被占用
   - 检查防火墙设置
   - 尝试使用 `127.0.0.1:5000` 访问

## 开发计划

- [ ] 添加更多技术指标计算
- [ ] 支持股票搜索功能
- [ ] 添加数据可视化图表
- [ ] 支持历史数据查询
- [ ] 添加价格预警功能

## 许可证

本项目仅供学习和研究使用，请遵守相关法律法规和数据提供方的使用条款。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！