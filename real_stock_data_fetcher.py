#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实股票数据获取器
优先使用Tushare获取A股数据，Yahoo Finance作为备用
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 尝试导入tushare
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("警告: 未安装tushare，将使用Yahoo Finance作为主要数据源")

class RealStockDataFetcher:
    def __init__(self, tushare_token: str = None):
        """初始化数据获取器"""
        self.tushare_token = tushare_token
        self.tushare_pro = None
        
        # 初始化Tushare
        if TUSHARE_AVAILABLE and tushare_token:
            try:
                ts.set_token(tushare_token)
                self.tushare_pro = ts.pro_api()
                print("✅ Tushare初始化成功，将优先使用Tushare数据源")
            except Exception as e:
                print(f"⚠️ Tushare初始化失败: {e}")
                self.tushare_pro = None
        
        # Yahoo Finance配置
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://finance.yahoo.com/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        # 配置代理
        self.proxies = self._setup_proxies()
        
        # 50支科技股票（A股代码）
        self.tech_stocks = [
            # 软件开发
            ("600588", "用友网络"), ("002410", "广联达"), ("300033", "同花顺"), 
            ("002405", "四维图新"), ("300496", "中科创达"), ("300253", "卫宁健康"),
            ("300454", "深信服"), ("002230", "科大讯飞"),
            
            # 电子设备  
            ("002415", "海康威视"), ("000725", "京东方A"), ("002241", "歌尔股份"),
            ("000063", "中兴通讯"), ("002236", "大华股份"), ("300136", "信维通信"),
            ("002938", "鹏鼎控股"), ("300782", "卓胜微"),
            
            # 通信设备
            ("000050", "深天马A"), ("002049", "紫光国微"), ("300408", "三环集团"),
            ("002371", "北方华创"), ("300661", "圣邦股份"), ("300223", "北京君正"),
            ("300327", "中颖电子"), ("300373", "扬杰科技"),
            
            # 半导体
            ("300474", "景嘉微"), ("300458", "全志科技"), ("002185", "华天科技"),
            ("300671", "富满电子"), ("300456", "耐威科技"), ("300623", "捷捷微电"),
            
            # 人工智能
            ("300059", "东方财富"), ("300017", "网宿科技"), ("300168", "万达信息"),
            ("300188", "美亚柏科"), ("300245", "天玑科技"), ("300271", "华宇软件"),
            ("300297", "蓝盾股份"), ("300339", "润和软件"),
            
            # 云计算大数据
            ("300348", "长亮科技"), ("300365", "恒华科技"), ("300377", "赢时胜"),
            ("300379", "东土科技"), ("300383", "光环新网"), ("300386", "飞天诚信"),
            ("300418", "昆仑万维"), ("300431", "暴风集团"),
            
            # 物联网
            ("300449", "汉邦高科"), ("300467", "迅游科技"), ("300468", "四方精创"),
            ("300469", "信息发展")
        ]
        
        # 备用数据源 - 使用腾讯财经API
        self.tencent_base_url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    
    def _setup_proxies(self):
        """设置代理配置"""
        proxies = {}
        
        # 检查环境变量中的代理设置
        http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
        https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
        
        if http_proxy:
            proxies['http'] = http_proxy
            print(f"✅ 检测到HTTP代理: {http_proxy}")
        
        if https_proxy:
            proxies['https'] = https_proxy
            print(f"✅ 检测到HTTPS代理: {https_proxy}")
        
        return proxies if proxies else None
    
    def fetch_tushare_data(self, symbol: str) -> pd.DataFrame:
        """
        从Tushare获取股票数据
        """
        if not self.tushare_pro:
            return None
            
        try:
            # 转换股票代码格式
            if symbol.startswith('6'):
                ts_symbol = f"{symbol}.SH"
            else:
                ts_symbol = f"{symbol}.SZ"
            
            # 获取2年历史数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
            
            df = self.tushare_pro.daily(ts_code=ts_symbol, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
                return None
            
            # 转换为标准格式
            df = df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume'
            })
            
            # 转换日期格式
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # 选择需要的列
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            return df
            
        except Exception as e:
            print(f"Tushare获取 {symbol} 失败: {str(e)}")
            return None
    
    def _convert_to_yahoo_symbol(self, symbol: str) -> str:
        """将A股代码转换为Yahoo Finance格式"""
        if symbol.startswith('6'):
            return f"{symbol}.SS"
        else:
            return f"{symbol}.SZ"
    
    def fetch_yahoo_data(self, symbol: str) -> pd.DataFrame:
        """
        从Yahoo Finance获取股票数据，带重试机制
        """
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}{symbol}"
                params = {
                    'period1': int((datetime.now() - timedelta(days=730)).timestamp()),
                    'period2': int(datetime.now().timestamp()),
                    'interval': '1d',
                    'includePrePost': 'true',
                    'events': 'div%2Csplit'
                }
                
                # 添加随机延迟避免被限制
                if attempt > 0:
                    time.sleep(retry_delay * attempt)
                
                response = requests.get(url, headers=self.headers, params=params, timeout=15, proxies=self.proxies)
                
                if response.status_code == 403:
                    print(f"Yahoo Finance获取 {symbol} 失败: {response.status_code} {response.reason} for url: {response.url}")
                    if attempt < max_retries - 1:
                        print(f"  第 {attempt + 1} 次重试...")
                        continue
                    else:
                        return None
                
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
                
            except requests.exceptions.RequestException as e:
                print(f"Yahoo Finance获取 {symbol} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"  第 {attempt + 1} 次重试...")
                    continue
                else:
                    return None
            except Exception as e:
                print(f"Yahoo Finance获取 {symbol} 数据解析失败: {str(e)}")
                return None
        
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
        获取单只股票数据，优先使用Tushare，失败时使用Yahoo Finance作为备用
        """
        print(f"📊 开始获取股票 {symbol} ({name}) 的数据...")
        
        # 优先使用Tushare
        if self.tushare_pro:
            print(f"🔄 尝试从Tushare获取 {symbol} 数据...")
            tushare_data = self.fetch_tushare_data(symbol)
            if tushare_data is not None and not tushare_data.empty:
                print(f"✅ Tushare获取 {symbol} 数据成功，共 {len(tushare_data)} 条记录")
                # 添加股票信息
                tushare_data['stock_code'] = symbol
                tushare_data['stock_name'] = name
                return tushare_data
            else:
                print(f"❌ Tushare获取 {symbol} 数据失败，尝试Yahoo Finance...")
        
        # 备用：使用Yahoo Finance
        yahoo_symbol = self._convert_to_yahoo_symbol(symbol)
        print(f"🔄 尝试从Yahoo Finance获取 {yahoo_symbol} 数据...")
        yahoo_data = self.fetch_yahoo_data(yahoo_symbol)
        
        if yahoo_data is not None and not yahoo_data.empty:
            print(f"✅ Yahoo Finance获取 {symbol} 数据成功，共 {len(yahoo_data)} 条记录")
            # 添加股票信息
            yahoo_data['stock_code'] = symbol
            yahoo_data['stock_name'] = name
            return yahoo_data
        
        # 如果Yahoo Finance也失败，尝试腾讯财经
        print(f"  Yahoo Finance失败，尝试腾讯财经...")
        df = self.fetch_tencent_data(yahoo_symbol)
        
        if df is not None and not df.empty:
            # 添加股票信息
            df['stock_code'] = symbol
            df['stock_name'] = name
            print(f"  成功获取 {len(df)} 条记录，价格范围: {df['close'].min():.2f} - {df['close'].max():.2f}")
            return df
        else:
            print(f"❌ 所有数据源都失败，跳过 {symbol}")
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
            time.sleep(1.0)  # 增加延迟到1秒
        
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
            'data_source': 'Tushare & Yahoo Finance & Tencent Finance',
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
    """
    主函数：演示如何使用RealStockDataFetcher
    """
    # 使用Tushare token初始化
    tushare_token = "7ab33ca92888ab9381e91389b091b970c768a7de8715fe7fd647c3c7"
    fetcher = RealStockDataFetcher(tushare_token=tushare_token)
    
    print("🚀 开始获取股票数据...")
    print(f"📡 数据源优先级: Tushare -> Yahoo Finance -> 腾讯财经")
    
    # 获取所有股票数据
    df = fetcher.fetch_all_stocks()
    
    if df is not None and not df.empty:
        # 保存数据
        csv_file, json_file = fetcher.save_data(df)
        print(f"\n✅ 数据获取完成！")
        print(f"📄 CSV文件: {csv_file}")
        print(f"📊 统计文件: {json_file}")
        print(f"📈 总共获取 {len(df)} 条记录，涵盖 {df['stock_code'].nunique()} 只股票")
    else:
        print("❌ 未能获取到任何股票数据")
        
    print("\n🎯 数据获取任务完成！")

if __name__ == "__main__":
    main()