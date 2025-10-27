#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
科大讯飞股票数据获取器
专门获取科大讯飞(002230)的实时股票数据
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
from real_stock_data_fetcher import RealStockDataFetcher

def get_kedaxunfei_data():
    """获取科大讯飞的股票数据"""
    print("科大讯飞股票数据获取器")
    print("=" * 50)
    
    # 创建数据获取器
    fetcher = RealStockDataFetcher()
    
    # 科大讯飞股票信息
    stock_code = "002230.SZ"
    stock_name = "科大讯飞"
    
    print(f"正在获取 {stock_name}({stock_code}) 的数据...")
    
    try:
        # 获取股票数据
        df = fetcher.fetch_stock_data(stock_code, stock_name)
        
        if not df.empty:
            print(f"✅ 成功获取 {stock_name} 数据!")
            
            # 显示基本信息
            print(f"\n📊 数据概览:")
            print(f"股票代码: {stock_code}")
            print(f"股票名称: {stock_name}")
            print(f"数据记录数: {len(df):,} 条")
            print(f"日期范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
            print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f} 元")
            print(f"平均价格: {df['close'].mean():.2f} 元")
            
            # 显示最新数据
            latest_data = df.iloc[-1]
            print(f"\n📈 最新交易数据 ({latest_data['date'].date()}):")
            print(f"开盘价: {latest_data['open']:.2f} 元")
            print(f"收盘价: {latest_data['close']:.2f} 元")
            print(f"最高价: {latest_data['high']:.2f} 元")
            print(f"最低价: {latest_data['low']:.2f} 元")
            print(f"成交量: {latest_data['volume']:,.0f}")
            
            # 计算涨跌幅
            if len(df) >= 2:
                prev_close = df.iloc[-2]['close']
                change = latest_data['close'] - prev_close
                change_pct = (change / prev_close) * 100
                print(f"涨跌额: {change:+.2f} 元")
                print(f"涨跌幅: {change_pct:+.2f}%")
            
            # 显示近期数据趋势
            print(f"\n📊 近5个交易日数据:")
            recent_data = df.tail(5)[['date', 'open', 'close', 'high', 'low', 'volume']].copy()
            recent_data['date'] = recent_data['date'].dt.date
            recent_data['volume'] = recent_data['volume'].apply(lambda x: f"{x:,.0f}")
            print(recent_data.to_string(index=False))
            
            # 保存数据
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_filename = f"kedaxunfei_data_{timestamp}.csv"
            json_filename = f"kedaxunfei_data_{timestamp}.json"
            
            # 保存CSV
            df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            
            # 保存JSON统计信息
            stats = {
                "stock_info": {
                    "code": stock_code,
                    "name": stock_name,
                    "fetch_time": datetime.now().isoformat()
                },
                "data_summary": {
                    "total_records": len(df),
                    "date_range": {
                        "start": df['date'].min().isoformat(),
                        "end": df['date'].max().isoformat()
                    },
                    "price_stats": {
                        "min": float(df['close'].min()),
                        "max": float(df['close'].max()),
                        "mean": float(df['close'].mean()),
                        "current": float(latest_data['close'])
                    },
                    "volume_stats": {
                        "min": int(df['volume'].min()),
                        "max": int(df['volume'].max()),
                        "mean": int(df['volume'].mean()),
                        "current": int(latest_data['volume'])
                    }
                },
                "latest_data": {
                    "date": latest_data['date'].isoformat(),
                    "open": float(latest_data['open']),
                    "close": float(latest_data['close']),
                    "high": float(latest_data['high']),
                    "low": float(latest_data['low']),
                    "volume": int(latest_data['volume'])
                }
            }
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 数据已保存:")
            print(f"CSV文件: {csv_filename}")
            print(f"统计文件: {json_filename}")
            
            return df
            
        else:
            print(f"❌ 未能获取 {stock_name} 的数据")
            print("可能的原因:")
            print("1. 网络连接问题")
            print("2. API访问限制")
            print("3. 股票代码格式问题")
            return None
            
    except Exception as e:
        print(f"❌ 获取数据时发生错误: {str(e)}")
        return None

def get_realtime_price():
    """获取科大讯飞的实时价格（简化版）"""
    print("\n🔄 获取实时价格...")
    
    # 使用腾讯财经API获取实时价格
    url = "https://qt.gtimg.cn/q=sz002230"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.text
            if data and "sz002230" in data:
                # 解析数据
                parts = data.split('~')
                if len(parts) > 10:
                    name = parts[1]
                    current_price = float(parts[3])
                    prev_close = float(parts[4])
                    open_price = float(parts[5])
                    high_price = float(parts[33])
                    low_price = float(parts[34])
                    volume = int(parts[6])
                    
                    change = current_price - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    print(f"📈 {name} 实时行情:")
                    print(f"当前价格: {current_price:.2f} 元")
                    print(f"涨跌额: {change:+.2f} 元")
                    print(f"涨跌幅: {change_pct:+.2f}%")
                    print(f"开盘价: {open_price:.2f} 元")
                    print(f"最高价: {high_price:.2f} 元")
                    print(f"最低价: {low_price:.2f} 元")
                    print(f"成交量: {volume:,}")
                    
                    return {
                        "name": name,
                        "current_price": current_price,
                        "change": change,
                        "change_pct": change_pct,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "volume": volume
                    }
    except Exception as e:
        print(f"❌ 获取实时价格失败: {str(e)}")
    
    return None

def main():
    """主函数"""
    print("🚀 科大讯飞股票数据获取工具")
    print("=" * 60)
    
    # 获取历史数据
    historical_data = get_kedaxunfei_data()
    
    # 获取实时价格
    realtime_data = get_realtime_price()
    
    print("\n" + "=" * 60)
    print("✅ 数据获取完成!")
    
    if historical_data is not None:
        print(f"📊 历史数据: {len(historical_data)} 条记录")
    
    if realtime_data is not None:
        print(f"💰 实时价格: {realtime_data['current_price']:.2f} 元")

if __name__ == "__main__":
    main()