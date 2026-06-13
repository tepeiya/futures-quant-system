# 网格加仓管理模块
# 实现类似量子女王的趋势跟随网格策略

import numpy as np
import pandas as pd
from datetime import datetime
from enum import Enum

class PositionSide(Enum):
    LONG = 1
    SHORT = -1

class GridOrder:
    """网格订单"""
    def __init__(self, order_id: int, side: PositionSide, 
                 entry_price: float, volume: float,
                 timestamp):
        self.order_id = order_id
        self.side = side
        self.entry_price = entry_price
        self.volume = volume
        self.timestamp = timestamp
        self.margin = 0
        self.unrealized_pnl = 0
        self.stop_loss = None
        self.take_profit = None
    
    @property
    def value(self):
        return self.entry_price * self.volume
    
    def __repr__(self):
        side_str = "LONG" if self.side == PositionSide.LONG else "SHORT"
        return f"[#{self.order_id}] {side_str} @{self.entry_price:.2f} x{self.volume:.2f}"


class GridManager:
    """网格管理器 - 管理分批加仓和部分平仓"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # 默认参数
        self.params = {
            'initial_volume': 0.01,          # 首单量
            'max_positions': 10,              # 最大持仓数
            'grid_spacing_atr_mult': 1.5,     # 网格间距 = ATR * 倍数
            'volume_mode': 'fixed',           # 'fixed': 固定手数, 'scaled': 递增
            'volume_scale_factor': 1.0,       # 递增系数 (scaled模式)
            'partial_take_profit': 0.5,       # 部分止盈比例 (盈利单平掉多少)
            'partial_tp_atr_mult': 1.0,       # 部分止盈触发 = ATR * 倍数
            'max_grid_levels': 5,             # 最大网格层数
            'min_grid_spacing_pct': 0.003,    # 最小网格间距(百分比)
        }
        self.params.update(self.config.get('grid_params', {}))
        
        self.orders = []        # 当前持仓
        self.order_counter = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
    
    @property
    def current_volume(self) -> float:
        """当前总持仓量"""
        return sum(o.volume for o in self.orders)
    
    @property
    def position_count(self) -> int:
        return len(self.orders)
    
    def avg_entry_price(self) -> float:
        """加权平均入场价"""
        if not self.orders:
            return 0.0
        total_value = sum(o.value for o in self.orders)
        total_vol = self.current_volume
        return total_value / total_vol if total_vol > 0 else 0.0
    
    def get_grid_spacing(self, current_atr: float, current_price: float) -> float:
        """计算网格间距"""
        spacing = current_atr * self.params['grid_spacing_atr_mult']
        min_spacing = current_price * self.params['min_grid_spacing_pct']
        return max(spacing, min_spacing)
    
    def should_open_first(self, signal_value: int, current_positions: int) -> bool:
        """是否开首单"""
        if current_positions >= self.params['max_positions']:
            return False
        if abs(signal_value) < 1:  # 只有 BUY(1) 或以上才开
            return False
        return True
    
    def should_add_grid(self, signal_value: int, current_price: float,
                       current_atr: float, side: PositionSide) -> bool:
        """是否应该加仓 (网格加仓逻辑)
        
        条件: 
        1. 首单已经有了
        2. 价格往反方向走了足够距离
        3. 还没达到最大仓位数
        """
        if not self.orders or self.position_count >= self.params['max_positions']:
            return False
        
        # 检查是否有同方向的仓位 (网格只在反方向加)
        orders_same_side = [o for o in self.orders if o.side == side]
        if not orders_same_side:
            return False
        
        # 计算当前价格距离最近同向单的距离
        last_order = max(orders_same_side, key=lambda o: o.entry_price if o.side == PositionSide.LONG else -o.entry_price)
        
        if side == PositionSide.LONG:
            # 对多头: 价格下跌时加仓
            price_diff = last_order.entry_price - current_price
            grid_spacing = self.get_grid_spacing(current_atr, current_price)
            return price_diff >= grid_spacing
        else:
            # 对空头: 价格上涨时加仓
            price_diff = current_price - last_order.entry_price
            grid_spacing = self.get_grid_spacing(current_atr, current_price)
            return price_diff >= grid_spacing
    
    def open_position(self, side: PositionSide, price: float, 
                     volume: float = None, timestamp=None) -> GridOrder:
        """开一个新仓位"""
        if volume is None:
            volume = self.params['initial_volume']
        
        # 如果是加仓，根据模式调整手数
        if len(self.orders) > 0 and self.params['volume_mode'] == 'scaled':
            grid_level = len(self.orders)
            volume *= (self.params['volume_scale_factor'] ** grid_level)
        
        # 不能超过最大持仓数
        total_vol = self.current_volume + volume
        # 简化: 不设最大仓位限制
        
        self.order_counter += 1
        order = GridOrder(
            order_id=self.order_counter,
            side=side,
            entry_price=price,
            volume=volume,
            timestamp=timestamp or datetime.now()
        )
        
        self.orders.append(order)
        self.total_trades += 1
        return order
    
    def should_partial_close(self, current_price: float, current_atr: float) -> list:
        """检查是否需要部分平仓 (盈利单止盈)
        
        量子女王的做法: 当网格中有盈利单时，部分平仓盈利单来补贴亏损单的浮亏
        
        Returns: 要平掉的订单列表
        """
        to_close = []
        
        for order in self.orders:
            if order.side == PositionSide.LONG:
                profit_pct = (current_price - order.entry_price) / order.entry_price
            else:
                profit_pct = (order.entry_price - current_price) / order.entry_price
            
            tp_threshold = self.params['partial_tp_atr_mult'] * current_atr / current_price
            
            if profit_pct >= tp_threshold:
                # 只平部分
                close_volume = order.volume * self.params['partial_take_profit']
                if close_volume > 0:
                    to_close.append((order, close_volume))
        
        return to_close
    
    def partial_close(self, order: GridOrder, close_volume: float, 
                     close_price: float) -> float:
        """部分平仓，返回盈亏"""
        if order.side == PositionSide.LONG:
            pnl = (close_price - order.entry_price) * close_volume
        else:
            pnl = (order.entry_price - close_price) * close_volume
        
        order.volume -= close_volume
        
        # 如果全平了，移除订单
        if order.volume <= 0:
            self.orders.remove(order)
        
        if pnl > 0:
            self.winning_trades += 1
        self.total_pnl += pnl
        
        return pnl
    
    def close_all(self, close_price: float) -> float:
        """全平台"""
        total_pnl = 0.0
        for order in list(self.orders):
            pnl = self.partial_close(order, order.volume, close_price)
            total_pnl += pnl
        return total_pnl
    
    def get_grid_status(self) -> dict:
        """获取网格状态"""
        if not self.orders:
            return {'position_count': 0, 'avg_price': 0, 'total_volume': 0}
        
        return {
            'position_count': self.position_count,
            'avg_price': self.avg_entry_price(),
            'total_volume': self.current_volume,
            'orders': [str(o) for o in self.orders],
        }
    
    def max_drawdown(self) -> float:
        """计算当前浮亏比例 (简化为最大值)"""
        if not self.orders:
            return 0.0
        return 0.0  # 由风控模块计算


if __name__ == '__main__':
    gm = GridManager()
    
    # 模拟网格：先在500开多，然后每次跌10点加仓
    prices = [500, 495, 488, 480, 475, 478, 482, 490, 498, 505]
    atr = 8.0
    
    for i, price in enumerate(prices):
        if i == 0:
            order = gm.open_position(PositionSide.LONG, price)
            print(f"首单: {order}")
        elif gm.should_add_grid(Signal.BUY, price, atr, PositionSide.LONG):
            order = gm.open_position(PositionSide.LONG, price)
            print(f"网格加仓 #{i}: {order}")
        
        # 检查部分平仓
        to_close = gm.should_partial_close(price, atr)
        for o, vol in to_close:
            pnl = gm.partial_close(o, vol, price)
            print(f"部分平仓 {o.order_id}: 盈利 {pnl:.2f}")
    
    print(f"\n最终持仓: {gm.get_grid_status()}")
    print(f"总交易: {gm.total_trades}, 总盈亏: {gm.total_pnl:.2f}")
