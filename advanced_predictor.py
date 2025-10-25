import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 导入RTX 4090优化配置
try:
    from rtx4090_optimization import setup_rtx4090_optimization, get_optimal_batch_size
    setup_rtx4090_optimization()
except ImportError:
    # 如果没有优化模块，使用基本设置
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    def get_optimal_batch_size(model_size="medium"):
        return 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

"""
高级股票预测器
集成实时数据获取和LSTM-TCN预测模型
适用于短线交易策略
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from stock_data import StockDataFetcher
from lstm_tcn_model import LSTMTCNPredictor
import warnings
warnings.filterwarnings('ignore')

# 导入RTX 4090优化配置
try:
    from rtx4090_optimization import setup_rtx4090_optimization, get_optimal_batch_size
    setup_rtx4090_optimization()
except ImportError:
    # 如果没有优化模块，使用基本设置
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
    
    def get_optimal_batch_size(model_size="medium"):
        return 32

class AdvancedStockPredictor:
    """高级股票预测器"""
    
    def __init__(self, model_path=None):
        """
        初始化预测器
        
        Args:
            model_path: 预训练模型路径
        """
        self.data_fetcher = StockDataFetcher()
        self.predictor = LSTMTCNPredictor()
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            try:
                self.predictor.load_model(model_path)
                print(f"已加载预训练模型: {model_path}")
            except Exception as e:
                print(f"加载模型失败: {e}")
                print("将使用新模型进行训练")
    
    def prepare_historical_data(self, stock_code, days=500):
        """
        准备历史数据用于训练
        
        Args:
            stock_code: 股票代码
            days: 历史数据天数
            
        Returns:
            DataFrame: 历史数据
        """
        try:
            # 这里需要实际的历史数据API
            # 由于adata主要提供实时数据，这里生成模拟历史数据
            print(f"准备 {stock_code} 的历史数据...")
            
            # 获取当前实时数据作为参考
            current_data = self.data_fetcher.get_single_stock_data(stock_code)
            if not current_data:
                raise ValueError(f"无法获取股票 {stock_code} 的数据")
            
            current_price = float(current_data.get('close', 100))
            
            # 生成模拟历史数据
            dates = pd.date_range(
                end=datetime.now().date(), 
                periods=days, 
                freq='D'
            )
            
            # 使用随机游走生成价格序列
            np.random.seed(hash(stock_code) % 2**32)  # 基于股票代码的固定种子
            
            prices = []
            volumes = []
            price = current_price * 0.8  # 从较低价格开始
            
            for i in range(days):
                # 价格变化 (带趋势和周期性)
                trend = 0.0002  # 轻微上升趋势
                cycle = 0.001 * np.sin(2 * np.pi * i / 50)  # 50天周期
                noise = np.random.normal(0, 0.015)  # 随机噪声
                
                change = trend + cycle + noise
                price = price * (1 + change)
                prices.append(price)
                
                # 成交量 (与价格变化相关)
                base_volume = 1000000
                volume_change = abs(change) * 5 + np.random.normal(0, 0.3)
                volume = base_volume * (1 + volume_change)
                volumes.append(max(volume, 100000))
            
            # 调整最后的价格接近当前价格
            price_adjustment = current_price / prices[-1]
            prices = [p * price_adjustment for p in prices]
            
            historical_data = pd.DataFrame({
                'date': dates,
                'close': prices,
                'volume': volumes
            })
            
            print(f"生成了 {len(historical_data)} 天的历史数据")
            return historical_data
            
        except Exception as e:
            print(f"准备历史数据时出错: {e}")
            return None
    
    def train_model(self, stock_code, days=500, epochs=100, save_model=True):
        """
        训练预测模型
        
        Args:
            stock_code: 股票代码
            days: 训练数据天数
            epochs: 训练轮数
            save_model: 是否保存模型
            
        Returns:
            dict: 训练结果
        """
        print(f"开始训练股票 {stock_code} 的预测模型...")
        
        # 准备历史数据
        historical_data = self.prepare_historical_data(stock_code, days)
        if historical_data is None:
            return {"error": "无法获取历史数据"}
        
        try:
            # 训练模型 (PyTorch版本)
            train_losses, val_losses = self.predictor.train_model(
                historical_data, 
                epochs=epochs,
                batch_size=32,
                validation_split=0.2
            )
            
            # 评估模型
            metrics = self.predictor.evaluate_model(historical_data)
            
            # 保存模型
            if save_model:
                model_filename = f"lstm_tcn_model_{stock_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                self.predictor.save_model(model_filename)
                self.model_path = model_filename
            
            result = {
                "success": True,
                "stock_code": stock_code,
                "training_samples": len(historical_data),
                "epochs_completed": epochs,
                "final_loss": float(train_losses[-1]),
                "final_val_loss": float(val_losses[-1]),
                "metrics": metrics,
                "model_path": self.model_path if save_model else None
            }
            
            print("训练完成！")
            print(f"最终损失: {result['final_loss']:.6f}")
            print(f"验证损失: {result['final_val_loss']:.6f}")
            print(f"MAPE: {metrics['mape']:.2f}%")
            print(f"趋势准确率: {metrics['trend_accuracy']:.2f}")
            
            return result
            
        except Exception as e:
            print(f"训练模型时出错: {e}")
            return {"error": str(e)}
    
    def predict_stock(self, stock_code, return_analysis=True):
        """
        预测股票价格和趋势
        
        Args:
            stock_code: 股票代码
            return_analysis: 是否返回详细分析
            
        Returns:
            dict: 预测结果
        """
        try:
            print(f"预测股票 {stock_code}...")
            
            # 获取历史数据用于预测
            historical_data = self.prepare_historical_data(stock_code, days=100)
            if historical_data is None:
                return {"error": "无法获取历史数据"}
            
            # 获取当前实时数据
            current_data = self.data_fetcher.get_single_stock_data(stock_code)
            if not current_data:
                return {"error": f"无法获取股票 {stock_code} 的实时数据"}
            
            # 进行预测 (PyTorch版本)
            prediction = self.predictor.predict_stock(historical_data)
            
            # 整合结果
            result = {
                "stock_code": stock_code,
                "stock_name": current_data.get('name', '未知'),
                "current_price": float(current_data.get('close', 0)),
                "current_change": float(current_data.get('change', 0)),
                "current_change_pct": float(current_data.get('change_pct', 0)),
                "volume": int(current_data.get('volume', 0)),
                "prediction": prediction,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            if return_analysis:
                result["analysis"] = self._generate_analysis(current_data, prediction)
            
            return result
            
        except Exception as e:
            print(f"预测股票时出错: {e}")
            return {"error": str(e)}
    
    def _generate_analysis(self, current_data, prediction):
        """生成交易分析，包括税费计算"""
        current_price = float(current_data.get('close', 0))
        predicted_price = prediction['predicted_price']
        trend = prediction['trend_prediction']
        confidence = prediction['confidence_score']
        risk_level = prediction['risk_level']
        
        # 税费率（中国A股标准）
        buy_commission = 0.00025  # 买入佣金 0.025%
        sell_commission = 0.00025  # 卖出佣金 0.025%
        sell_stamp_tax = 0.001     # 卖出印花税 0.1%
        
        # 计算税费
        buy_cost = current_price * buy_commission
        sell_cost = predicted_price * (sell_commission + sell_stamp_tax)
        
        # 计算净预期收益
        net_profit = predicted_price - current_price - buy_cost - sell_cost
        expected_return = (net_profit / current_price) * 100
        
        # 生成交易建议（基于净收益）
        if trend == '上涨' and confidence > 0.7 and risk_level != '高风险':
            if expected_return > 2:
                action = "强烈买入"
            elif expected_return > 0.5:
                action = "买入"
            else:
                action = "观望"
        elif trend == '下跌' and confidence > 0.7:
            if expected_return < -2:
                action = "卖出"
            elif expected_return < -0.5:
                action = "减仓"
            else:
                action = "观望"
        else:
            action = "观望"
        
        # 风险评估
        risk_factors = []
        if risk_level == '高风险':
            risk_factors.append("高波动性")
        if confidence < 0.6:
            risk_factors.append("预测不确定性高")
        if abs(expected_return) > 5:
            risk_factors.append("预期变动幅度大")
        
        # 计算建议价格（示例：买入价为当前价，卖出价为预测价）
        suggested_buy_price = round(current_price, 2)
        suggested_sell_price = round(predicted_price, 2)
        
        return {
            "expected_return_pct": round(expected_return, 2),
            "net_profit": round(net_profit, 2),
            "buy_cost": round(buy_cost, 4),
            "sell_cost": round(sell_cost, 4),
            "trading_action": action,
            "confidence_level": "高" if confidence > 0.8 else "中" if confidence > 0.6 else "低",
            "risk_factors": risk_factors,
            "suggested_buy_price": suggested_buy_price,
            "suggested_sell_price": suggested_sell_price,
            "recommendation": self._get_recommendation(action, expected_return, risk_level)
        }
    
    def _get_recommendation(self, action, expected_return, risk_level):
        """获取详细建议"""
        recommendations = []
        
        if action == "强烈买入":
            recommendations.append("建议适量买入，设置止损位")
            recommendations.append(f"目标收益: +{abs(expected_return):.1f}%")
            recommendations.append("建议持仓时间: 1-3个交易日")
        elif action == "买入":
            recommendations.append("可以考虑小仓位买入")
            recommendations.append("密切关注市场变化")
        elif action == "卖出":
            recommendations.append("建议及时止损或减仓")
            recommendations.append("避免追跌")
        elif action == "减仓":
            recommendations.append("适当减少仓位")
            recommendations.append("等待更好的入场时机")
        else:
            recommendations.append("建议观望，等待明确信号")
            recommendations.append("可关注其他机会")
        
        if risk_level == "高风险":
            recommendations.append("⚠️ 高风险警告：谨慎操作")
        
        return recommendations
    
    def batch_predict(self, stock_codes, save_results=True):
        """
        批量预测多只股票
        
        Args:
            stock_codes: 股票代码列表
            save_results: 是否保存结果
            
        Returns:
            dict: 批量预测结果
        """
        print(f"开始批量预测 {len(stock_codes)} 只股票...")
        
        results = {}
        successful_predictions = 0
        
        for i, stock_code in enumerate(stock_codes, 1):
            print(f"[{i}/{len(stock_codes)}] 预测 {stock_code}...")
            
            result = self.predict_stock(stock_code)
            results[stock_code] = result
            
            if "error" not in result:
                successful_predictions += 1
                print(f"  ✓ 预测成功: {result['prediction']['trend_prediction']}")
            else:
                print(f"  ✗ 预测失败: {result['error']}")
        
        # 汇总结果
        summary = {
            "total_stocks": len(stock_codes),
            "successful_predictions": successful_predictions,
            "failed_predictions": len(stock_codes) - successful_predictions,
            "success_rate": successful_predictions / len(stock_codes) * 100,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "results": results
        }
        
        # 保存结果
        if save_results:
            filename = f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {filename}")
        
        print(f"批量预测完成！成功率: {summary['success_rate']:.1f}%")
        return summary
    
    def get_top_recommendations(self, stock_codes, top_n=5):
        """
        获取最佳投资建议
        
        Args:
            stock_codes: 股票代码列表
            top_n: 返回前N个推荐
            
        Returns:
            list: 排序后的推荐列表
        """
        print("分析最佳投资机会...")
        
        recommendations = []
        
        for stock_code in stock_codes:
            result = self.predict_stock(stock_code)
            
            if "error" not in result and "analysis" in result:
                analysis = result["analysis"]
                prediction = result["prediction"]
                
                # 计算综合评分
                score = 0
                
                # 预期收益权重
                expected_return = analysis["expected_return_pct"]
                if expected_return > 0:
                    score += expected_return * 2
                
                # 置信度权重
                confidence = prediction["confidence_score"]
                score += confidence * 50
                
                # 风险调整
                risk_level = prediction["risk_level"]
                if risk_level == "低风险":
                    score += 20
                elif risk_level == "中风险":
                    score += 10
                # 高风险不加分
                
                # 趋势权重
                if prediction["trend_prediction"] == "上涨":
                    score += 15
                
                recommendations.append({
                    "stock_code": stock_code,
                    "stock_name": result["stock_name"],
                    "score": score,
                    "expected_return": expected_return,
                    "confidence": confidence,
                    "risk_level": risk_level,
                    "action": analysis["trading_action"],
                    "current_price": result["current_price"],
                    "predicted_price": prediction["predicted_price"]
                })
        
        # 按评分排序
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        
        return recommendations[:top_n]

def main():
    """主函数 - 演示使用"""
    print("高级股票预测系统")
    print("=" * 50)
    
    # 创建预测器
    predictor = AdvancedStockPredictor()
    
    # 热门股票代码
    popular_stocks = [
        "000001",  # 平安银行
        "000002",  # 万科A
        "600036",  # 招商银行
        "600519",  # 贵州茅台
        "000858"   # 五粮液
    ]
    
    print("1. 训练模型示例...")
    # 训练一个股票的模型
    train_result = predictor.train_model("000001", days=300, epochs=20)
    if train_result.get("success"):
        print("✓ 模型训练成功")
    
    print("\n2. 单股预测示例...")
    # 预测单只股票
    prediction = predictor.predict_stock("000001")
    if "error" not in prediction:
        analysis = prediction['analysis']
        print(f"{prediction['stock_name']}股，买入价格输入：{analysis['suggested_buy_price']} 卖出价格输入：{analysis['suggested_sell_price']}")
    
    print("\n3. 批量预测示例...")
    # 批量预测
    batch_results = predictor.batch_predict(popular_stocks[:3])
    
    print("\n4. 投资建议示例...")
    # 获取最佳推荐
    top_recommendations = predictor.get_top_recommendations(popular_stocks[:3], top_n=3)
    
    print("最佳投资机会:")
    for i, rec in enumerate(top_recommendations, 1):
        print(f"{i}. {rec['stock_name']} ({rec['stock_code']})")
        print(f"   评分: {rec['score']:.1f}")
        print(f"   预期收益: {rec['expected_return']:+.2f}%")
        print(f"   建议: {rec['action']}")
        print()

if __name__ == "__main__":
    main()