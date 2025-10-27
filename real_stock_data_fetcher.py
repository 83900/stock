#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实股票数据获取器
使用免费API获取真实的股票历史数据
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict, Tuple

class RealStockDataFetcher:
    def __init__(self):
        """初始化数据获取器"""
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # 50支科技股票（A股转换为Yahoo Finance格式）
        self.tech_stocks = [
            # 软件开发
            ("600588.SS", "用友网络"), ("002410.SZ", "广联达"), ("300033.SZ", "同花顺"), 
            ("002405.SZ", "四维图新"), ("300496.SZ", "中科创达"), ("300253.SZ", "卫宁健康"),
            ("300454.SZ", "深信服"), ("002230.SZ", "科大讯飞"),
            
            # 电子设备  
            ("002415.SZ", "海康威视"), ("000725.SZ", "京东方A"), ("002241.SZ", "歌尔股份"),
            ("000063.SZ", "中兴通讯"), ("002236.SZ", "大华股份"), ("300136.SZ", "信维通信"),
            ("002938.SZ", "鹏鼎控股"), ("300782.SZ", "卓胜微"),
            
            # 通信设备
            ("000050.SZ", "深天马A"), ("002049.SZ", "紫光国微"), ("300408.SZ", "三环集团"),
            ("002371.SZ", "北方华创"), ("300661.SZ", "圣邦股份"), ("300223.SZ", "北京君正"),
            ("300327.SZ", "中颖电子"), ("300373.SZ", "扬杰科技"),
            
            # 半导体
            ("300474.SZ", "景嘉微"), ("300458.SZ", "全志科技"), ("002185.SZ", "华天科技"),
            ("300671.SZ", "富满电子"), ("300456.SZ", "耐威科技"), ("300623.SZ", "捷捷微电"),
            
            # 人工智能
            ("300059.SZ", "东方财富"), ("300017.SZ", "网宿科技"), ("300168.SZ", "万达信息"),
            ("300188.SZ", "美亚柏科"), ("300245.SZ", "天玑科技"), ("300271.SZ", "华宇软件"),
            ("300297.SZ", "蓝盾股份"), ("300339.SZ", "润和软件"),
            
            # 云计算大数据
            ("300348.SZ", "长亮科技"), ("300365.SZ", "恒华科技"), ("300377.SZ", "赢时胜"),
            ("300379.SZ", "东土科技"), ("300383.SZ", "光环新网"), ("300386.SZ", "飞天诚信"),
            ("300418.SZ", "昆仑万维"), ("300431.SZ", "暴风集团"),
            
            # 物联网
            ("300449.SZ", "汉邦高科"), ("300467.SZ", "迅游科技"), ("300468.SZ", "四方精创"),
            ("300469.SZ", "信息发展")
        ]
        
        # 备用数据源 - 使用腾讯财经API
        self.tencent_base_url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        
    def fetch_yahoo_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """
        从Yahoo Finance获取股票数据
        """
        try:
            url = f"{self.base_url}{symbol}"
            params = {
                'period1': int((datetime.now() - timedelta(days=730)).timestamp()),
                'period2': int(datetime.now().timestamp()),
                'interval': '1d',
                'includePrePost': 'true',
                'events': 'div%2Csplit'
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'chart' not in data or not data['chart']['result']:
                return None
                
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            df = pd.DataFrame({
                'date': [datetime.fromtimestamp(ts) for ts in timestamps],
                'open': quotes['open'],
                'high': quotes['high'], 
                'low': quotes['low'],
                'close': quotes['close'],
                'volume': quotes['volume']
            })
            
            # 清理数据
            df = df.dropna()
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            return df
            
        except Exception as e:
            print(f"Yahoo Finance获取 {symbol} 失败: {e}")
            return None
    
    def fetch_tencent_data(self, symbol: str) -> pd.DataFrame:
        """
        从腾讯财经获取股票数据（备用方案）
        """
        try:
            # 转换股票代码格式
            if symbol.endswith('.SS'):
                code = 'sh' + symbol.replace('.SS', '')
            elif symbol.endswith('.SZ'):
                code = 'sz' + symbol.replace('.SZ', '')
            else:
                return None
                
            params = {
                '_var': 'kline_dayqfq',
                'param': f'{code},day,2020-01-01,2024-12-31,640,qfq',
                'r': str(int(time.time()))
            }
            
            response = requests.get(self.tencent_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            # 解析腾讯返回的数据
            text = response.text
            if 'kline_dayqfq=' in text:
                json_str = text.split('kline_dayqfq=')[1]
                data = json.loads(json_str)
                
                if 'data' in data and code in data['data']:
                    klines = data['data'][code]['day']
                    
                    df_data = []
                    for kline in klines:
                        df_data.append({
                            'date': datetime.strptime(kline[0], '%Y-%m-%d'),
                            'open': float(kline[1]),
                            'close': float(kline[2]),
                            'high': float(kline[3]),
                            'low': float(kline[4]),
                            'volume': int(kline[5])
                        })
                    
                    df = pd.DataFrame(df_data)
                    df = df.sort_values('date')
                    return df
                    
        except Exception as e:
            print(f"腾讯财经获取 {symbol} 失败: {e}")
            return None
    
    def fetch_stock_data(self, symbol: str, name: str) -> pd.DataFrame:
        """
        获取单只股票数据，优先使用Yahoo Finance，失败则使用腾讯财经
        """
        print(f"正在获取 {symbol} ({name}) 的真实数据...")
        
        # 首先尝试Yahoo Finance
        df = self.fetch_yahoo_data(symbol)
        
        # 如果失败，尝试腾讯财经
        if df is None or df.empty:
            print(f"  Yahoo Finance失败，尝试腾讯财经...")
            df = self.fetch_tencent_data(symbol)
        
        if df is not None and not df.empty:
            # 添加股票信息
            df['stock_code'] = symbol.split('.')[0]
            df['stock_name'] = name
            print(f"  成功获取 {len(df)} 条记录，价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
            return df
        else:
            print(f"  获取失败，跳过 {symbol}")
            return None
    
    def fetch_all_stocks(self) -> pd.DataFrame:
        """
        获取所有股票的真实数据
        """
        all_data = []
        success_count = 0
        
        print(f"开始获取 {len(self.tech_stocks)} 支科技股票的真实数据...")
        print("=" * 60)
        
        for i, (symbol, name) in enumerate(self.tech_stocks, 1):
            print(f"[{i}/{len(self.tech_stocks)}] ", end="")
            
            df = self.fetch_stock_data(symbol, name)
            
            if df is not None:
                all_data.append(df)
                success_count += 1
            
            # 避免请求过于频繁
            time.sleep(0.5)
        
        print("=" * 60)
        print(f"数据获取完成: 成功 {success_count}/{len(self.tech_stocks)} 支股票")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values(['stock_code', 'date'])
            return combined_df
        else:
            print("警告: 没有成功获取任何股票数据!")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        保存数据到CSV和JSON文件
        """
        if df.empty:
            print("没有数据可保存!")
            return None, None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"real_tech_stocks_data_{timestamp}.csv"
        json_filename = f"real_tech_stocks_data_{timestamp}.json"
        
        # 保存CSV
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        # 生成统计信息并保存JSON
        stats = {
            'generation_time': datetime.now().isoformat(),
            'data_source': 'Yahoo Finance & Tencent Finance',
            'total_records': len(df),
            'stock_count': df['stock_code'].nunique(),
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat()
            },
            'price_stats': {
                'min_price': float(df['close'].min()),
                'max_price': float(df['close'].max()),
                'avg_price': float(df['close'].mean())
            },
            'stocks_info': []
        }
        
        # 每只股票的统计信息
        for stock_code in df['stock_code'].unique():
            stock_data = df[df['stock_code'] == stock_code]
            stock_info = {
                'stock_code': stock_code,
                'stock_name': stock_data['stock_name'].iloc[0],
                'records_count': len(stock_data),
                'price_range': {
                    'min': float(stock_data['close'].min()),
                    'max': float(stock_data['close'].max()),
                    'avg': float(stock_data['close'].mean())
                },
                'latest_price': float(stock_data['close'].iloc[-1]),
                'latest_date': stock_data['date'].iloc[-1].isoformat()
            }
            stats['stocks_info'].append(stock_info)
        
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return csv_filename, json_filename

def main():
    """主函数"""
    print("真实股票数据获取器")
    print("=" * 50)
    
    fetcher = RealStockDataFetcher()
    
    # 获取所有股票数据
    df = fetcher.fetch_all_stocks()
    
    if not df.empty:
        # 保存数据
        csv_file, json_file = fetcher.save_data(df)
        
        print(f"\n数据保存完成:")
        print(f"CSV文件: {csv_file}")
        print(f"统计文件: {json_file}")
        print(f"\n数据概览:")
        print(f"总记录数: {len(df):,}")
        print(f"股票数量: {df['stock_code'].nunique()}")
        print(f"日期范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
        print(f"价格范围: {df['close'].min():.2f} - {df['close'].max():.2f} 元")
        
        # 显示部分数据
        print(f"\n数据样例:")
        print(df.head(10).to_string(index=False))
        
    else:
        print("错误: 未能获取任何真实股票数据!")
        print("可能的原因:")
        print("1. 网络连接问题")
        print("2. API访问限制")
        print("3. 股票代码格式问题")

if __name__ == "__main__":
    main()