#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动智能交易分析
简化版启动脚本，用于快速运行交易分析
"""

import sys
import os
import time
from datetime import datetime

def check_requirements():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    required_packages = [
        'torch', 'pandas', 'numpy', 'sklearn', 
        'matplotlib', 'seaborn', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ 缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name} ({memory_gb:.1f} GB)")
        else:
            print("⚠️ 未检测到CUDA GPU，将使用CPU")
    except:
        print("⚠️ 无法检测GPU状态")
    
    return True

def run_quick_analysis():
    """运行快速分析（仅分析前10支股票）"""
    print("\n🚀 启动快速分析模式...")
    print("只分析前10支股票，用于快速验证")
    
    try:
        from smart_trading_analyzer import SmartTradingAnalyzer
        import torch
        
        # 创建分析器
        analyzer = SmartTradingAnalyzer()
        
        # 获取数据
        all_data = analyzer.fetch_all_tech_stocks_data()
        if all_data.empty:
            print("❌ 无法获取股票数据")
            return
        
        # 只分析前10支股票
        stock_codes = all_data['stock_code'].unique()[:10]
        print(f"📊 分析股票: {list(stock_codes)}")
        
        results = []
        
        for i, stock_code in enumerate(stock_codes, 1):
            print(f"\n📈 分析进度: {i}/{len(stock_codes)} - {stock_code}")
            
            try:
                # 获取单只股票数据
                stock_data = all_data[all_data['stock_code'] == stock_code].copy()
                stock_data = stock_data.sort_values('date').reset_index(drop=True)
                
                if len(stock_data) < 60:
                    print(f"⚠️ {stock_code} 数据不足，跳过")
                    continue
                
                # 计算技术指标
                stock_data = analyzer.low_price_analyzer.calculate_technical_indicators(stock_data)
                
                # 识别低价机会
                opportunity_analysis = analyzer.low_price_analyzer.identify_low_price_opportunities(stock_data)
                
                print(f"💡 机会评分: {opportunity_analysis['opportunity_level']:.2f}")
                
                # 如果机会评分太低，跳过训练
                if opportunity_analysis['opportunity_level'] < 0.2:
                    print(f"⚠️ {stock_code} 机会评分过低，跳过")
                    continue
                
                # 训练预测模型（减少训练轮数）
                print(f"🤖 训练预测模型...")
                model = analyzer.train_prediction_model(all_data, stock_code)
                if model is None:
                    continue
                
                # 预测价格
                predictions = analyzer.predict_short_term_prices(model, all_data, stock_code)
                if predictions is None:
                    continue
                
                # 计算交易策略
                strategy = analyzer.calculate_optimal_trading_strategy(
                    stock_code, predictions['current_price'], predictions, 
                    opportunity_analysis['opportunity_level']
                )
                
                if strategy is not None:
                    # 添加股票名称
                    stock_name = next((name for code, name in analyzer.data_fetcher.tech_stocks if code == stock_code), stock_code)
                    strategy['stock_name'] = stock_name
                    strategy['predictions'] = predictions
                    strategy['opportunity_analysis'] = opportunity_analysis
                    
                    results.append(strategy)
                    print(f"✅ {stock_code} 发现盈利机会!")
                    print(f"   预期利润率: {strategy['recommended_strategy']['profit_rate']:.2f}%")
                
            except Exception as e:
                print(f"❌ {stock_code} 分析失败: {str(e)}")
                continue
        
        # 生成报告
        if results:
            report = analyzer.generate_trading_report(results)
            print("\n" + "="*60)
            print(report)
            
            # 保存报告
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"quick_trading_report_{timestamp}.txt"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"\n💾 快速分析报告已保存: {report_filename}")
        else:
            print("\n❌ 未发现任何盈利机会")
    
    except Exception as e:
        print(f"❌ 分析过程中发生错误: {str(e)}")

def run_full_analysis():
    """运行完整分析"""
    print("\n🚀 启动完整分析模式...")
    print("将分析所有50支科技股票")
    
    try:
        from smart_trading_analyzer import main as run_full_main
        run_full_main()
    except Exception as e:
        print(f"❌ 完整分析失败: {str(e)}")

def main():
    """主函数"""
    print("🎯 智能股票交易分析系统")
    print("=" * 50)
    print("基于RTX 4090深度学习的短期交易策略")
    print("分析50支科技股票，寻找1-2天的盈利机会")
    print("=" * 50)
    
    # 检查环境
    if not check_requirements():
        print("\n❌ 环境检查失败，请先安装必要的依赖包")
        return
    
    print("\n✅ 环境检查通过")
    
    # 选择运行模式
    print("\n请选择运行模式:")
    print("1. 快速分析 (分析前10支股票，约5-10分钟)")
    print("2. 完整分析 (分析所有50支股票，约30-60分钟)")
    print("3. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1/2/3): ").strip()
            
            if choice == '1':
                start_time = time.time()
                run_quick_analysis()
                end_time = time.time()
                print(f"\n⏱️ 快速分析耗时: {end_time - start_time:.1f} 秒")
                break
            elif choice == '2':
                print("\n⚠️ 完整分析将需要较长时间，请确保:")
                print("1. 网络连接稳定")
                print("2. GPU有足够内存")
                print("3. 有足够的时间等待")
                
                confirm = input("\n确认开始完整分析? (y/n): ").strip().lower()
                if confirm == 'y':
                    start_time = time.time()
                    run_full_analysis()
                    end_time = time.time()
                    print(f"\n⏱️ 完整分析耗时: {end_time - start_time:.1f} 秒")
                break
            elif choice == '3':
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请输入 1、2 或 3")
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 输入错误: {str(e)}")

if __name__ == "__main__":
    main()