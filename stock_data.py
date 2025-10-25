#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国股票实时数据获取系统
使用adata库获取A股实时行情数据
"""

import adata
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataFetcher:
    """股票数据获取器"""
    
    def __init__(self):
        """初始化"""
        self.stock_codes = []
        self.load_popular_stocks()
    
    def load_popular_stocks(self):
        """加载热门股票代码"""
        # 一些热门股票代码
        popular_stocks = [
            '000430',  # ST张家界
            '601868',  # 中国能建
            '002877',  # 智能自控
            '601012',  # 隆基绿能
            '000778',  # 新兴铸管
        ]
        self.stock_codes = popular_stocks
        logger.info(f"已加载 {len(self.stock_codes)} 只热门股票")
    
    def get_all_stock_codes(self):
        """获取所有股票代码"""
        try:
            logger.info("正在获取所有股票代码...")
            all_codes_df = adata.stock.info.all_code()
            if not all_codes_df.empty:
                self.stock_codes = all_codes_df['stock_code'].tolist()[:50]  # 限制前50只股票
                logger.info(f"成功获取 {len(self.stock_codes)} 只股票代码")
                return all_codes_df
            else:
                logger.warning("未获取到股票代码，使用默认热门股票")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"获取股票代码失败: {e}")
            return pd.DataFrame()
    
    def get_stock_realtime_data(self, stock_code):
        """获取单只股票的实时数据"""
        try:
            # 获取最新的日K线数据（最近1天）
            market_data = adata.stock.market.get_market(
                stock_code=stock_code, 
                k_type=1,  # 日K线
                start_date=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            )
            
            if not market_data.empty:
                latest_data = market_data.iloc[-1]  # 获取最新一条数据
                
                # 格式化数据
                stock_info = {
                    'stock_code': stock_code,
                    'trade_date': latest_data['trade_date'],
                    'open': float(latest_data['open']),
                    'close': float(latest_data['close']),
                    'high': float(latest_data['high']),
                    'low': float(latest_data['low']),
                    'volume': int(latest_data['volume']) if pd.notna(latest_data['volume']) else 0,
                    'change': float(latest_data['close'] - latest_data['pre_close']) if 'pre_close' in latest_data else 0,
                    'change_pct': float((latest_data['close'] - latest_data['pre_close']) / latest_data['pre_close'] * 100) if 'pre_close' in latest_data and latest_data['pre_close'] != 0 else 0,
                    'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                return stock_info
            else:
                logger.warning(f"股票 {stock_code} 未获取到数据")
                return None
                
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 数据失败: {e}")
            return None
    
    def get_multiple_stocks_data(self, stock_codes=None):
        """获取多只股票的实时数据"""
        if stock_codes is None:
            stock_codes = self.stock_codes
        
        stocks_data = []
        logger.info(f"开始获取 {len(stock_codes)} 只股票的实时数据...")
        
        for i, stock_code in enumerate(stock_codes):
            logger.info(f"正在获取第 {i+1}/{len(stock_codes)} 只股票: {stock_code}")
            
            stock_data = self.get_stock_realtime_data(stock_code)
            if stock_data:
                stocks_data.append(stock_data)
            
            # 添加延时避免请求过于频繁
            time.sleep(0.1)
        
        logger.info(f"成功获取 {len(stocks_data)} 只股票的数据")
        return stocks_data
    
    def save_data_to_json(self, data, filename=None):
        """保存数据到JSON文件"""
        if filename is None:
            filename = f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"数据已保存到 {filename}")
            return filename
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return None
    
    def save_data_to_csv(self, data, filename=None):
        """保存数据到CSV文件"""
        if filename is None:
            filename = f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"数据已保存到 {filename}")
            return filename
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return None
    
    def display_data(self, data):
        """显示股票数据"""
        if not data:
            print("没有数据可显示")
            return
        
        df = pd.DataFrame(data)
        print("\n=== 股票实时数据 ===")
        print(f"更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"股票数量: {len(data)} 只")
        print("-" * 80)
        
        # 设置显示格式
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        # 选择要显示的列
        display_columns = ['stock_code', 'close', 'change', 'change_pct', 'volume', 'trade_date']
        if all(col in df.columns for col in display_columns):
            display_df = df[display_columns].copy()
            display_df['change_pct'] = display_df['change_pct'].apply(lambda x: f"{x:.2f}%")
            display_df['volume'] = display_df['volume'].apply(lambda x: f"{x:,}")
            print(display_df.to_string(index=False))
        else:
            print(df.to_string(index=False))
        
        print("-" * 80)

def main():
    """主函数"""
    print("=== 中国股票实时数据获取系统 ===")
    
    # 创建数据获取器
    fetcher = StockDataFetcher()
    
    try:
        # 获取所有股票代码（可选）
        print("\n1. 获取股票代码列表...")
        all_codes = fetcher.get_all_stock_codes()
        
        # 获取实时数据
        print("\n2. 获取股票实时数据...")
        stocks_data = fetcher.get_multiple_stocks_data()
        
        if stocks_data:
            # 显示数据
            print("\n3. 显示数据...")
            fetcher.display_data(stocks_data)
            
            # 保存数据
            print("\n4. 保存数据...")
            json_file = fetcher.save_data_to_json(stocks_data)
            csv_file = fetcher.save_data_to_csv(stocks_data)
            
            print(f"\n数据获取完成！")
            print(f"JSON文件: {json_file}")
            print(f"CSV文件: {csv_file}")
        else:
            print("未获取到任何股票数据")
            
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()