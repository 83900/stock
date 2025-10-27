#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据获取系统
使用真实数据获取器获取股票数据
"""

import pandas as pd
import json
from datetime import datetime
import logging
from real_stock_data_fetcher import RealStockDataFetcher

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataManager:
    """股票数据管理器"""
    
    def __init__(self):
        """初始化"""
        self.fetcher = RealStockDataFetcher()
        self.data = None
    
    def fetch_real_data(self):
        """获取真实股票数据"""
        logger.info("开始获取真实股票数据...")
        self.data = self.fetcher.fetch_all_stocks()
        
        if not self.data.empty:
            logger.info(f"成功获取 {len(self.data)} 条记录")
            return True
        else:
            logger.error("获取数据失败")
            return False
    
    def save_data(self):
        """保存数据"""
        if self.data is not None and not self.data.empty:
            csv_file, json_file = self.fetcher.save_data(self.data)
            logger.info(f"数据已保存: {csv_file}, {json_file}")
            return csv_file, json_file
        else:
            logger.error("没有数据可保存")
            return None, None
    
    def get_data_summary(self):
        """获取数据摘要"""
        if self.data is None or self.data.empty:
            return None
            
        summary = {
            'total_records': len(self.data),
            'stock_count': self.data['stock_code'].nunique(),
            'date_range': {
                'start': self.data['date'].min(),
                'end': self.data['date'].max()
            },
            'price_stats': {
                'min': float(self.data['close'].min()),
                'max': float(self.data['close'].max()),
                'mean': float(self.data['close'].mean())
            }
        }
        return summary

def main():
    """主函数"""
    print("=== 股票数据获取系统 ===")
    
    manager = StockDataManager()
    
    try:
        print("\n1. 获取真实股票数据...")
        if manager.fetch_real_data():
            print("\n2. 保存数据...")
            csv_file, json_file = manager.save_data()
            
            if csv_file:
                print(f"\n数据获取完成！")
                print(f"CSV文件: {csv_file}")
                print(f"JSON文件: {json_file}")
                
                print("\n3. 数据摘要:")
                summary = manager.get_data_summary()
                if summary:
                    print(f"总记录数: {summary['total_records']}")
                    print(f"股票数量: {summary['stock_count']}")
                    print(f"日期范围: {summary['date_range']['start']} - {summary['date_range']['end']}")
                    print(f"价格范围: {summary['price_stats']['min']:.2f} - {summary['price_stats']['max']:.2f}")
            else:
                print("数据保存失败")
        else:
            print("数据获取失败")
            
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()