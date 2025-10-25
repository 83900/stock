#!/usr/bin/env python3
"""
LSTM-TCN股票预测系统 - 快速启动脚本
用于快速测试和演示系统功能
"""

import os
import sys
import argparse
from datetime import datetime

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow版本: {tf.__version__}")
        
        # 检查GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ 检测到GPU: {len(gpus)}个")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
        
        import pandas as pd
        print(f"✓ Pandas版本: {pd.__version__}")
        
        import numpy as np
        print(f"✓ NumPy版本: {np.__version__}")
        
        from adata import stock
        print("✓ AData库可用")
        
        print("✅ 环境检查通过！")
        return True
        
    except ImportError as e:
        print(f"❌ 环境检查失败: {e}")
        print("请运行: pip install -r requirements_gpu.txt")
        return False

def demo_data_fetching():
    """演示数据获取功能"""
    print("\n📊 演示数据获取功能...")
    
    try:
        from stock_data import StockDataFetcher
        
        fetcher = StockDataFetcher()
        
        # 获取热门股票数据
        print("获取热门股票实时数据...")
        stocks_data = fetcher.get_multiple_stocks_data(limit=5)
        
        if stocks_data:
            print("✓ 成功获取股票数据:")
            for stock in stocks_data[:3]:  # 显示前3只
                print(f"  {stock.get('name', 'N/A')} ({stock.get('code', 'N/A')}): "
                      f"¥{stock.get('close', 0):.2f} "
                      f"({stock.get('change_pct', 0):+.2f}%)")
        else:
            print("❌ 获取股票数据失败")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ 数据获取演示失败: {e}")
        return False

def demo_model_training(stock_code="000001", epochs=10):
    """演示模型训练"""
    print(f"\n🤖 演示模型训练 (股票: {stock_code}, 轮数: {epochs})...")
    
    try:
        from advanced_predictor import AdvancedStockPredictor
        
        # 创建预测器
        predictor = AdvancedStockPredictor()
        
        # 快速训练演示
        print("开始训练模型...")
        result = predictor.train_model(
            stock_code=stock_code,
            days=200,  # 使用较少数据加快演示
            epochs=epochs,
            save_model=True
        )
        
        if result.get("success"):
            print("✓ 模型训练成功!")
            print(f"  最终损失: {result['final_loss']:.6f}")
            print(f"  验证损失: {result['final_val_loss']:.6f}")
            print(f"  MAPE: {result['metrics']['mape']:.2f}%")
            print(f"  趋势准确率: {result['metrics']['trend_accuracy']:.2f}")
            return result['model_path']
        else:
            print(f"❌ 模型训练失败: {result.get('error', '未知错误')}")
            return None
            
    except Exception as e:
        print(f"❌ 模型训练演示失败: {e}")
        return None

def demo_prediction(model_path=None, stock_code="000001"):
    """演示股票预测"""
    print(f"\n🔮 演示股票预测 (股票: {stock_code})...")
    
    try:
        from advanced_predictor import AdvancedStockPredictor
        
        # 创建预测器
        predictor = AdvancedStockPredictor(model_path)
        
        # 进行预测
        print("正在预测...")
        result = predictor.predict_stock(stock_code)
        
        if "error" not in result:
            print("✓ 预测成功!")
            print(f"  股票名称: {result['stock_name']}")
            print(f"  当前价格: ¥{result['current_price']:.2f}")
            print(f"  预测价格: ¥{result['prediction']['predicted_price']:.2f}")
            print(f"  趋势预测: {result['prediction']['trend_prediction']}")
            print(f"  置信度: {result['prediction']['confidence_score']:.2f}")
            print(f"  风险等级: {result['prediction']['risk_level']}")
            
            if "analysis" in result:
                analysis = result["analysis"]
                print(f"  预期收益: {analysis['expected_return_pct']:+.2f}%")
                print(f"  建议操作: {analysis['trading_action']}")
            
            return True
        else:
            print(f"❌ 预测失败: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ 预测演示失败: {e}")
        return False

def demo_batch_prediction():
    """演示批量预测"""
    print("\n📈 演示批量预测...")
    
    try:
        from advanced_predictor import AdvancedStockPredictor
        
        predictor = AdvancedStockPredictor()
        
        # 热门股票列表
        stocks = ["000001", "000002", "600036"]
        
        print(f"批量预测 {len(stocks)} 只股票...")
        results = predictor.batch_predict(stocks, save_results=False)
        
        print(f"✓ 批量预测完成!")
        print(f"  成功率: {results['success_rate']:.1f}%")
        print(f"  成功预测: {results['successful_predictions']} 只")
        print(f"  失败预测: {results['failed_predictions']} 只")
        
        # 显示成功的预测结果
        for stock_code, result in results['results'].items():
            if "error" not in result:
                trend = result['prediction']['trend_prediction']
                confidence = result['prediction']['confidence_score']
                print(f"  {result['stock_name']} ({stock_code}): {trend} (置信度: {confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ 批量预测演示失败: {e}")
        return False

def demo_web_interface():
    """演示Web界面"""
    print("\n🌐 启动Web界面...")
    
    try:
        print("Web界面将在后台启动...")
        print("请在浏览器中访问: http://localhost:8080")
        print("按 Ctrl+C 停止Web服务")
        
        # 这里不实际启动，只是提示
        print("💡 要启动Web界面，请运行: python web_app.py")
        return True
        
    except Exception as e:
        print(f"❌ Web界面演示失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LSTM-TCN股票预测系统快速启动")
    parser.add_argument("--mode", choices=["check", "data", "train", "predict", "batch", "web", "full"], 
                       default="full", help="运行模式")
    parser.add_argument("--stock", default="000001", help="股票代码 (默认: 000001)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数 (默认: 10)")
    parser.add_argument("--model", help="预训练模型路径")
    
    args = parser.parse_args()
    
    print("🚀 LSTM-TCN股票预测系统快速启动")
    print("=" * 50)
    print(f"运行模式: {args.mode}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success = True
    model_path = args.model
    
    if args.mode in ["check", "full"]:
        success &= check_environment()
    
    if success and args.mode in ["data", "full"]:
        success &= demo_data_fetching()
    
    if success and args.mode in ["train", "full"]:
        model_path = demo_model_training(args.stock, args.epochs)
        success &= model_path is not None
    
    if success and args.mode in ["predict", "full"]:
        success &= demo_prediction(model_path, args.stock)
    
    if success and args.mode in ["batch", "full"]:
        success &= demo_batch_prediction()
    
    if success and args.mode in ["web", "full"]:
        success &= demo_web_interface()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 所有演示完成!")
        print("\n📚 下一步:")
        print("1. 查看详细文档: DEPLOYMENT_GUIDE.md")
        print("2. 在算力平台部署: 参考部署指南")
        print("3. 自定义训练: 修改参数重新训练")
        print("4. 集成到交易系统: 使用API接口")
    else:
        print("❌ 演示过程中出现错误")
        print("请检查环境配置和依赖安装")
    
    print("\n⚠️  投资风险提示:")
    print("本系统仅供学习和研究使用，投资有风险，决策需谨慎！")

if __name__ == "__main__":
    main()