#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票交易盈亏计算器
考虑所有交易费用：佣金、印花税、过户费等
"""

class StockProfitCalculator:
    def __init__(self):
        # 交易费用设置（可根据券商调整）
        self.commission_rate = 0.0003  # 佣金费率 0.03%（双向收取）
        self.min_commission = 5.0      # 最低佣金 5元
        self.stamp_tax_rate = 0.001    # 印花税 0.1%（仅卖出收取）
        self.transfer_fee_rate = 0.00002  # 过户费 0.002%（双向收取）
        
    def calculate_commission(self, amount):
        """计算佣金"""
        commission = amount * self.commission_rate
        return max(commission, self.min_commission)
    
    def calculate_stamp_tax(self, sell_amount):
        """计算印花税（仅卖出时收取）"""
        return sell_amount * self.stamp_tax_rate
    
    def calculate_transfer_fee(self, amount):
        """计算过户费"""
        return amount * self.transfer_fee_rate
    
    def calculate_profit_per_share(self, buy_price, sell_price, shares=100):
        """
        计算每股盈亏
        
        Args:
            buy_price: 买入价格
            sell_price: 卖出价格  
            shares: 股数（默认100股，用于计算最低佣金影响）
        
        Returns:
            dict: 包含详细费用和每股盈亏的字典
        """
        # 计算交易金额
        buy_amount = buy_price * shares
        sell_amount = sell_price * shares
        
        # 买入费用
        buy_commission = self.calculate_commission(buy_amount)
        buy_transfer_fee = self.calculate_transfer_fee(buy_amount)
        buy_total_fee = buy_commission + buy_transfer_fee
        
        # 卖出费用
        sell_commission = self.calculate_commission(sell_amount)
        sell_stamp_tax = self.calculate_stamp_tax(sell_amount)
        sell_transfer_fee = self.calculate_transfer_fee(sell_amount)
        sell_total_fee = sell_commission + sell_stamp_tax + sell_transfer_fee
        
        # 总费用
        total_fee = buy_total_fee + sell_total_fee
        
        # 计算盈亏
        gross_profit = sell_amount - buy_amount  # 毛利润
        net_profit = gross_profit - total_fee    # 净利润
        profit_per_share = net_profit / shares   # 每股盈亏
        
        # 计算费用率
        fee_rate = total_fee / buy_amount * 100
        
        return {
            'buy_price': buy_price,
            'sell_price': sell_price,
            'shares': shares,
            'buy_amount': buy_amount,
            'sell_amount': sell_amount,
            'buy_fees': {
                'commission': buy_commission,
                'transfer_fee': buy_transfer_fee,
                'total': buy_total_fee
            },
            'sell_fees': {
                'commission': sell_commission,
                'stamp_tax': sell_stamp_tax,
                'transfer_fee': sell_transfer_fee,
                'total': sell_total_fee
            },
            'total_fee': total_fee,
            'gross_profit': gross_profit,
            'net_profit': net_profit,
            'profit_per_share': profit_per_share,
            'fee_rate': fee_rate,
            'return_rate': (profit_per_share / buy_price) * 100
        }
    
    def print_detailed_result(self, result):
        """打印详细的计算结果"""
        print("\n" + "="*60)
        print("📊 股票交易盈亏详细计算")
        print("="*60)
        
        print(f"💰 交易信息:")
        print(f"   买入价格: ¥{result['buy_price']:.3f}")
        print(f"   卖出价格: ¥{result['sell_price']:.3f}")
        print(f"   交易股数: {result['shares']} 股")
        print(f"   买入金额: ¥{result['buy_amount']:.2f}")
        print(f"   卖出金额: ¥{result['sell_amount']:.2f}")
        
        print(f"\n💸 买入费用:")
        print(f"   佣金: ¥{result['buy_fees']['commission']:.2f}")
        print(f"   过户费: ¥{result['buy_fees']['transfer_fee']:.2f}")
        print(f"   买入总费用: ¥{result['buy_fees']['total']:.2f}")
        
        print(f"\n💸 卖出费用:")
        print(f"   佣金: ¥{result['sell_fees']['commission']:.2f}")
        print(f"   印花税: ¥{result['sell_fees']['stamp_tax']:.2f}")
        print(f"   过户费: ¥{result['sell_fees']['transfer_fee']:.2f}")
        print(f"   卖出总费用: ¥{result['sell_fees']['total']:.2f}")
        
        print(f"\n📈 盈亏分析:")
        print(f"   总交易费用: ¥{result['total_fee']:.2f}")
        print(f"   费用率: {result['fee_rate']:.3f}%")
        print(f"   毛利润: ¥{result['gross_profit']:.2f}")
        print(f"   净利润: ¥{result['net_profit']:.2f}")
        
        # 每股盈亏用不同颜色显示
        profit_per_share = result['profit_per_share']
        return_rate = result['return_rate']
        
        if profit_per_share > 0:
            print(f"   ✅ 每股盈利: +¥{profit_per_share:.4f}")
            print(f"   ✅ 收益率: +{return_rate:.3f}%")
        elif profit_per_share < 0:
            print(f"   ❌ 每股亏损: ¥{profit_per_share:.4f}")
            print(f"   ❌ 亏损率: {return_rate:.3f}%")
        else:
            print(f"   ⚖️  每股盈亏: ¥{profit_per_share:.4f}")
            print(f"   ⚖️  收益率: {return_rate:.3f}%")
        
        print("="*60)

def main():
    """主函数"""
    calculator = StockProfitCalculator()
    
    print("🎯 股票交易盈亏计算器")
    print("📝 考虑佣金、印花税、过户费等所有交易成本")
    print("-" * 50)
    
    while True:
        try:
            print("\n请输入交易信息:")
            
            # 输入买入价格
            buy_price = float(input("💵 买入价格 (元): "))
            if buy_price <= 0:
                print("❌ 买入价格必须大于0")
                continue
            
            # 输入卖出价格
            sell_price = float(input("💵 卖出价格 (元): "))
            if sell_price <= 0:
                print("❌ 卖出价格必须大于0")
                continue
            
            # 输入股数（可选）
            shares_input = input("📊 交易股数 (默认100股，直接回车使用默认值): ").strip()
            if shares_input:
                shares = int(shares_input)
                if shares <= 0:
                    print("❌ 股数必须大于0")
                    continue
            else:
                shares = 100
            
            # 计算盈亏
            result = calculator.calculate_profit_per_share(buy_price, sell_price, shares)
            
            # 显示结果
            calculator.print_detailed_result(result)
            
            # 询问是否继续
            continue_calc = input("\n是否继续计算？(y/n): ").strip().lower()
            if continue_calc not in ['y', 'yes', '是', '']:
                break
                
        except ValueError:
            print("❌ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n\n👋 程序已退出")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")
    
    print("\n💡 费用说明:")
    print("   • 佣金: 0.03% (最低5元，买卖双向收取)")
    print("   • 印花税: 0.1% (仅卖出时收取)")
    print("   • 过户费: 0.002% (买卖双向收取)")
    print("   • 不同券商费率可能有差异，请根据实际情况调整")
    print("\n🎯 计算完成，感谢使用！")

if __name__ == "__main__":
    main()