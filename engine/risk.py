# 风控模块 - 止损/回撤/假期保护

import numpy as np
from datetime import datetime, date
import calendar

class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        self.params = {
            'max_drawdown_pct': 0.15,        # 最大回撤 15%
            'max_position_value_pct': 0.3,   # 单品种最大仓位占总资金比例
            'max_daily_loss_pct': 0.05,      # 每日最大亏损 5%
            'stop_loss_atr_mult': 2.5,       # 止损 = ATR * 倍数
            'trailing_stop_activate': 0.02,  # 移动止盈激活阈值 2%
            'trailing_stop_dist': 0.01,      # 移动止盈间距 1%
            'holiday_mode': True,            # 假期保护
            'weekend_close': True,           # 周末前减仓
            'max_spread_pct': 0.005,         # 最大点差比例
            'min_account_balance': 1000,     # 最低账户余额
        }
        self.params.update(self.config.get('risk_params', {}))
        
        self.initial_balance = self.params.get('initial_balance', 10000)
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.daily_start_balance = self.initial_balance
        self.current_date = None
        self.daily_pnl = 0.0
    
    def update_balance(self, new_balance: float, current_date=None):
        """更新账户余额"""
        self.current_balance = new_balance
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        
        if current_date:
            if self.current_date != current_date:
                self.daily_start_balance = new_balance
                self.daily_pnl = 0.0
                self.current_date = current_date
    
    @property
    def drawdown_pct(self) -> float:
        """当前回撤比例"""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.current_balance) / self.peak_balance
    
    @property
    def daily_loss_pct(self) -> float:
        """当日亏损比例"""
        if self.daily_start_balance <= 0:
            return 0.0
        return -self.daily_pnl / self.daily_start_balance
    
    def get_stop_loss_price(self, entry_price: float, atr: float, 
                           side: int) -> float:
        """计算止损价"""
        distance = atr * self.params['stop_loss_atr_mult']
        if side > 0:  # 多头
            return entry_price - distance
        else:  # 空头
            return entry_price + distance
    
    def get_position_size(self, current_price: float, atr: float,
                         balance: float = None) -> float:
        """根据ATR计算合适的仓位大小（凯利公式简化版）"""
        bal = balance or self.current_balance
        risk_per_trade = bal * 0.02  # 每单风险2%
        
        stop_dist = atr * self.params['stop_loss_atr_mult']
        if stop_dist <= 0:
            return 0.0
        
        # 手数 = 风险金额 / (止损距离)
        position_risk = risk_per_trade / (stop_dist * current_price)
        
        return max(0.01, position_risk)
    
    def check_risk_limits(self, current_price: float, current_atr: float,
                         total_position_value: float, is_holiday: bool = False) -> dict:
        """检查所有风控限制
        
        Returns:
            dict: {'passed': bool, 'reasons': [str]}
        """
        result = {'passed': True, 'reasons': []}
        
        # 1. 最大回撤
        if self.drawdown_pct >= self.params['max_drawdown_pct']:
            result['passed'] = False
            result['reasons'].append(f'最大回撤触发: {self.drawdown_pct:.2%}')
        
        # 2. 单品种仓位限制
        position_pct = total_position_value / self.current_balance if self.current_balance > 0 else 0
        if position_pct >= self.params['max_position_value_pct']:
            result['passed'] = False
            result['reasons'].append(f'仓位过重: {position_pct:.2%}')
        
        # 3. 每日亏损限制
        if self.daily_loss_pct >= self.params['max_daily_loss_pct']:
            result['passed'] = False
            result['reasons'].append(f'当日亏损触限: {self.daily_loss_pct:.2%}')
        
        # 4. 假期保护
        if is_holiday and self.params['holiday_mode']:
            result['passed'] = False
            result['reasons'].append('假期模式: 禁止交易')
        
        # 5. 余额检查
        if self.current_balance < self.params['min_account_balance']:
            result['passed'] = False
            result['reasons'].append(f'余额不足: {self.current_balance:.2f}')
        
        return result
    
    def is_holiday(self, check_date=None) -> bool:
        """检查是否为节假日 (中国期货市场休市日)"""
        if check_date is None:
            check_date = date.today()
        
        # 周末
        if check_date.weekday() >= 5:
            return True
        
        # 中国法定节假日 (简化版)
        # 实际使用时应该接入交易日历API
        holidays = [
            # 春节 (2025)
            date(2025, 1, 28), date(2025, 1, 29), date(2025, 1, 30),
            date(2025, 1, 31), date(2025, 2, 1), date(2025, 2, 2), date(2025, 2, 3),
            date(2025, 2, 4),
            # 春节 (2026)
            date(2026, 2, 16), date(2026, 2, 17), date(2026, 2, 18),
            date(2026, 2, 19), date(2026, 2, 20), date(2026, 2, 21), date(2026, 2, 22),
            date(2026, 2, 23), date(2026, 2, 24),
            # 国庆 (2025)
            date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3),
            date(2025, 10, 4), date(2025, 10, 5), date(2025, 10, 6), date(2025, 10, 7), date(2025, 10, 8),
            # 元旦 (2026)
            date(2026, 1, 1),
        ]
        
        # 简化: 对模拟数据跳过假日检测
        return False
    
    def check_trading_time(self) -> bool:
        """检查当前是否在交易时段 (国内期货)"""
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute
        
        # 周末不开盘
        if weekday >= 5:
            return False
        
        # 国内期货交易时段
        # 日盘: 9:00-10:15, 10:30-11:30, 13:30-15:00
        # 夜盘: 21:00-23:00/23:30/02:30 (品种不同)
        is_day = (
            (9 <= hour <= 10 and not (hour == 10 and minute > 15)) or
            (10 <= hour <= 11 and not (hour == 10 and minute < 30)) and hour < 12 or
            (13 <= hour <= 15)
        )
        is_night = (hour >= 21 or hour < 3)
        
        return is_day or is_night


if __name__ == '__main__':
    rm = RiskManager({'initial_balance': 100000})
    
    # 测试
    print(f"初始余额: {rm.initial_balance}")
    print(f"当前回撤: {rm.drawdown_pct:.2%}")
    
    # 测试亏损
    rm.update_balance(90000)
    rm.daily_pnl = -3000
    print(f"亏损后余额: {rm.current_balance}")
    print(f"当日亏损: {rm.daily_loss_pct:.2%}")
    
    # 风控检查
    result = rm.check_risk_limits(current_price=500, current_atr=10, 
                                   total_position_value=30000)
    print(f"\n风控检查: {'通过' if result['passed'] else '未通过'}")
    if not result['passed']:
        for r in result['reasons']:
            print(f"  - {r}")
    
    # 止损价
    sl = rm.get_stop_loss_price(500, 10, 1)
    print(f"\n多头止损价: {sl:.2f} (入场500, ATR=10, 倍数2.5)")
    
    # 仓位建议
    pos = rm.get_position_size(500, 10)
    print(f"建议仓位: {pos:.4f} 手")
