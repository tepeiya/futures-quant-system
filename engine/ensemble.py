# 多策略组合引擎 v4
# 并行运行多个子策略，动态分配权重
#
# 子策略:
#   S1 - 趋势跟踪网格 (原来v3版本)
#   S2 - 突破追踪 (价格突破关键位追单)
#   S3 - 回调反转 (超买超卖反向开仓)
#   S4 - 波动率突破 (ATR扩张时入场)
#   S5 - 套利通道 (跨期价差回归)
#   S6 - 均线交叉 (快慢线金叉死叉)

import numpy as np
import pandas as pd
from enum import Enum

class Signal(Enum):
    STRONG_SELL = -2
    SELL = -1
    NEUTRAL = 0
    BUY = 1
    STRONG_BUY = 2


class StrategyBase:
    """子策略基类"""
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight  # 权重，可动态调整
        self.trades = 0
        self.wins = 0
        self.pnl = 0.0
    
    def update_performance(self, pnl: float, won: bool):
        """更新策略绩效"""
        self.trades += 1
        if won: self.wins += 1
        self.pnl += pnl
        # 根据胜率动态调权
        if self.trades >= 5:
            win_rate = self.wins / self.trades
            # 胜率低于30%降权，高于60%加权
            if win_rate < 0.3:
                self.weight *= 0.95
            elif win_rate > 0.6:
                self.weight *= 1.05
            self.weight = max(0.1, min(3.0, self.weight))
    
    def signal(self, df: pd.DataFrame, i: int, pos_info: dict) -> Signal:
        """返回信号 (子类实现)"""
        return Signal.NEUTRAL


# ========== 6 个子策略 ==========

class S1_TrendGrid(StrategyBase):
    """策略1: 趋势跟踪网格 (原v3核心)"""
    def __init__(self):
        super().__init__('TrendGrid', weight=1.0)
    
    def signal(self, df, i, pos):
        if i < 30: return Signal.NEUTRAL
        c = float(df['close'].iloc[i])
        ef = float(df['ema_f'].iloc[i])
        es = float(df['ema_s'].iloc[i])
        rsi = float(df['rsi'].iloc[i])
        
        # 大趋势方向
        trend_up = ef > es and c > ef
        trend_dn = ef < es and c < ef
        
        if trend_up and rsi < 55:
            return Signal.BUY
        elif trend_dn and rsi > 45:
            return Signal.SELL
        return Signal.NEUTRAL


class S2_Breakout(StrategyBase):
    """策略2: 突破追踪 (价格突破N日高低点)"""
    def __init__(self):
        super().__init__('Breakout', weight=0.8)
    
    def signal(self, df, i, pos):
        if i < 20: return Signal.NEUTRAL
        c = float(df['close'].iloc[i])
        h = float(df['high'].iloc[i])
        l = float(df['low'].iloc[i])
        
        # 20日高低点
        h20 = max(df['high'].iloc[i-19:i+1])
        l20 = min(df['low'].iloc[i-19:i+1])
        
        # 突破上轨
        if c >= h20 and c > float(df['close'].iloc[i-1]):
            return Signal.STRONG_BUY
        # 突破下轨
        elif c <= l20 and c < float(df['close'].iloc[i-1]):
            return Signal.STRONG_SELL
        
        return Signal.NEUTRAL


class S3_Reversal(StrategyBase):
    """策略3: 回调反转 (超买超卖反向)"""
    def __init__(self):
        super().__init__('Reversal', weight=0.6)
    
    def signal(self, df, i, pos):
        if i < 14: return Signal.NEUTRAL
        rsi = float(df['rsi'].iloc[i])
        c = float(df['close'].iloc[i])
        
        # RSI极端区域反向
        if rsi < 25:
            # 超卖反弹
            return Signal.BUY
        elif rsi > 75:
            # 超买回落
            return Signal.SELL
        
        return Signal.NEUTRAL


class S4_VolBreakout(StrategyBase):
    """策略4: 波动率突破 (ATR扩张)"""
    def __init__(self):
        super().__init__('VolBreakout', weight=0.7)
    
    def signal(self, df, i, pos):
        if i < 30: return Signal.NEUTRAL
        atr = float(df['atr'].iloc[i])
        c = float(df['close'].iloc[i])
        atr_ma = float(pd.Series(df['atr']).iloc[max(0,i-19):i+1].mean())
        
        # ATR比均值大1.5倍 → 波动爆发
        if atr > atr_ma * 1.5:
            # 往哪个方向突破？
            direction = c - float(df['close'].iloc[i-5])
            if direction > atr:
                return Signal.STRONG_BUY
            elif direction < -atr:
                return Signal.STRONG_SELL
        
        return Signal.NEUTRAL


class S5_MeanReversion(StrategyBase):
    """策略5: 均值回归 (价格远离均线回归)"""
    def __init__(self):
        super().__init__('MeanRev', weight=0.5)
    
    def signal(self, df, i, pos):
        if i < 20: return Signal.NEUTRAL
        c = float(df['close'].iloc[i])
        ema = float(df['ema_s'].iloc[i])
        atr = float(df['atr'].iloc[i])
        
        # 价格离均线超过2倍ATR → 回归
        dist = (c - ema) / max(atr, 0.1)
        
        if dist > 2.5:
            return Signal.SELL  # 太远了，等回归
        elif dist < -2.5:
            return Signal.BUY
        
        return Signal.NEUTRAL


class S6_EMACross(StrategyBase):
    """策略6: 均线交叉 (快慢线金叉死叉)"""
    def __init__(self):
        super().__init__('EMACross', weight=1.2)
    
    def signal(self, df, i, pos):
        if i < 2: return Signal.NEUTRAL
        ef = float(df['ema_f'].iloc[i])
        es = float(df['ema_s'].iloc[i])
        ef_p = float(df['ema_f'].iloc[i-1])
        es_p = float(df['ema_s'].iloc[i-1])
        
        # 金叉
        if ef > es and ef_p <= es_p:
            return Signal.STRONG_BUY
        # 死叉
        elif ef < es and ef_p >= es_p:
            return Signal.STRONG_SELL
        
        return Signal.NEUTRAL


class EnsembleEngine:
    """多策略组合引擎"""
    
    def __init__(self):
        self.strategies = [
            S1_TrendGrid(),
            S2_Breakout(),
            S3_Reversal(),
            S4_VolBreakout(),
            S5_MeanReversion(),
            S6_EMACross(),
        ]
    
    def get_signals(self, df: pd.DataFrame, i: int, positions: list) -> dict:
        """获取所有策略的信号和加权总分"""
        pos_info = {
            'n_pos': len(positions),
            'long_pos': sum(1 for p in positions if p['s'] == 'L'),
            'short_pos': sum(1 for p in positions if p['s'] == 'S'),
        }
        
        results = {}
        total_score = 0.0
        total_weight = 0.0
        
        for s in self.strategies:
            sig = s.signal(df, i, pos_info)
            score = sig.value * s.weight
            results[s.name] = {'signal': sig, 'score': score, 'weight': s.weight}
            total_score += score
            total_weight += s.weight
        
        return {
            'signals': results,
            'total_score': total_score,
            'avg_score': total_score / max(total_weight, 0.1),
            'n_buy': sum(1 for r in results.values() if r['signal'].value > 0),
            'n_sell': sum(1 for r in results.values() if r['signal'].value < 0),
        }
    
    def update_strategy_performance(self, name: str, pnl: float, won: bool):
        """更新子策略绩效"""
        for s in self.strategies:
            if s.name == name:
                s.update_performance(pnl, won)
                break
    
    def print_weights(self):
        """打印各策略当前权重"""
        total = sum(s.weight for s in self.strategies)
        print(f"\n  策略权重分布:")
        for s in sorted(self.strategies, key=lambda x: -x.weight):
            pct = s.weight / total * 100
            wr = (s.wins / s.trades * 100) if s.trades > 0 else 0
            print(f"    {s.name:12s} w={s.weight:.2f} ({pct:4.0f}%)  "
                  f"胜率{wr:4.0f}%  PnL:{s.pnl:+.0f}  ({s.trades}次)")
