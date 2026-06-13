# 仓位/持仓管理 - 多订单管理、盈亏计算、对账单

import numpy as np
from enum import Enum

class PositionSide(Enum):
    LONG = 1
    SHORT = -1


class Position:
    """单个持仓"""
    def __init__(self, symbol: str, side: PositionSide,
                 entry_price: float, volume: float,
                 entry_time, order_id: int):
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.volume = volume
        self.entry_time = entry_time
        self.order_id = order_id
        self.stop_loss = None
        self.take_profit = None
    
    @property
    def value(self):
        return self.entry_price * self.volume
    
    def unrealized_pnl(self, current_price: float) -> float:
        """未实现盈亏"""
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) * self.volume
        else:
            return (self.entry_price - current_price) * self.volume
    
    def __repr__(self):
        side = "LONG" if self.side == PositionSide.LONG else "SHORT"
        return f"[{self.symbol}] {side} @{self.entry_price:.2f} x{self.volume:.4f}"


class Portfolio:
    """投资组合管理"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = []      # 当前持仓
        self.closed_trades = []  # 已平仓记录
        self.peak_balance = initial_balance
        
        # 统计
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
    
    @property
    def total_position_value(self) -> float:
        """总持仓市值"""
        return sum(p.value for p in self.positions)
    
    @property
    def total_volume(self) -> float:
        return sum(p.volume for p in self.positions)
    
    @property
    def equity(self) -> float:
        """总权益 = 余额 + 持仓价值"""
        return self.balance
    
    @property
    def drawdown(self) -> float:
        """当前回撤"""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance
    
    @property
    def leverage(self) -> float:
        """当前杠杆率"""
        if self.balance <= 0:
            return 0.0
        return self.total_position_value / self.balance
    
    def open_position(self, symbol: str, side: PositionSide,
                     price: float, volume: float, timestamp) -> Position:
        """开仓"""
        order_id = len(self.closed_trades) + len(self.positions) + 1
        pos = Position(symbol, side, price, volume, timestamp, order_id)
        self.positions.append(pos)
        self.total_trades += 1
        return pos
    
    def close_position(self, position: Position, close_price: float,
                      volume: float = None, timestamp=None) -> float:
        """平仓（可部分平仓）
        
        Returns: 盈亏
        """
        if volume is None:
            volume = position.volume
        
        volume = min(volume, position.volume)
        
        if position.side == PositionSide.LONG:
            pnl = (close_price - position.entry_price) * volume
        else:
            pnl = (position.entry_price - close_price) * volume
        
        # 记录
        self.closed_trades.append({
            'symbol': position.symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': close_price,
            'volume': volume,
            'pnl': pnl,
            'entry_time': position.entry_time,
            'exit_time': timestamp,
        })
        
        position.volume -= volume
        self.balance += pnl
        self.total_pnl += pnl
        
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        if pnl > 0:
            self.winning_trades += 1
        
        # 如果全平了，移除持仓
        if position.volume <= 0:
            self.positions.remove(position)
        
        return pnl
    
    def close_all(self, current_price: float, timestamp=None) -> float:
        """全平所有仓位"""
        total_pnl = 0.0
        for pos in list(self.positions):
            pnl = self.close_position(pos, current_price, timestamp=timestamp)
            total_pnl += pnl
        return total_pnl
    
    def get_summary(self) -> dict:
        """获取账户摘要"""
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.balance,
            'total_pnl': self.total_pnl,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'drawdown': self.drawdown * 100,
            'leverage': self.leverage,
            'open_positions': len(self.positions),
            'total_volume': self.total_volume,
        }


if __name__ == '__main__':
    from datetime import datetime
    
    pf = Portfolio(100000)
    
    # 模拟一笔多头交易
    pos = pf.open_position('AU', PositionSide.LONG, 580.5, 0.1, datetime.now())
    print(f"开仓: {pos}")
    
    # 价格上涨
    pnl = pf.close_position(pos, 585.0, timestamp=datetime.now())
    print(f"平仓盈亏: {pnl:.2f}")
    
    print(f"\n账户摘要: {pf.get_summary()}")
