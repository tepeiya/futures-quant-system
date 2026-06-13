# 多策略信号引擎
# 包含 RSI, ATR, 动量, ICT Order Block, 支撑阻力等

import numpy as np
import pandas as pd
from enum import Enum

class PositionSide(Enum):
    LONG = 1
    SHORT = -1

class Signal(Enum):
    """信号方向"""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


class StrategyEngine:
    """多策略信号引擎 - 类似量子女王的6策略共振"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # 默认参数
        self.params = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'momentum_period': 10,
            'ob_lookback': 20,         # Order Block 回溯K线数
            'ob_body_ratio': 0.6,      # Order Block 实体占比阈值
            'sr_lookback': 30,         # 支撑阻力回溯期
            'sr_sensitivity': 0.02,    # 支撑阻力灵敏度
            'trend_ema_fast': 9,
            'trend_ema_slow': 21,
            'confluence_min': 3,       # 最少需要几个策略共振
        }
        self.params.update(self.config.get('strategy_params', {}))
    
    def compute_all(self, df: pd.DataFrame):
        """计算所有策略信号
        
        Returns:
            df: 附加指标列
            signals: 各策略信号字典 {strategy_name: Signal}
        """
        df = df.copy()
        
        # 1. RSI 信号
        df = self._add_rsi(df)
        rsi_signal = self._rsi_signal(df)
        
        # 2. ATR 波动率信号
        df = self._add_atr(df)
        atr_signal = self._atr_signal(df)
        
        # 3. 动量信号
        df = self._add_momentum(df)
        momentum_signal = self._momentum_signal(df)
        
        # 4. 趋势信号 (EMA)
        df = self._add_ema(df)
        trend_signal = self._trend_signal(df)
        
        # 5. 支撑阻力
        sr = self._support_resistance(df)
        
        # 6. ICT Order Block (简化版)
        df = self._add_order_block(df)
        ob_signal = self._order_block_signal(df)
        
        # 组合信号
        signals = {
            'rsi': rsi_signal,
            'atr': atr_signal,
            'momentum': momentum_signal,
            'trend': trend_signal,
            'support_resistance': sr,
            'order_block': ob_signal,
        }
        
        return df, signals
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 RSI"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(span=self.params['rsi_period'], adjust=False).mean()
        avg_loss = loss.ewm(span=self.params['rsi_period'], adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        return df
    
    def _rsi_signal(self, df: pd.DataFrame) -> Signal:
        """RSI 信号判断"""
        if df.empty:
            return Signal.NEUTRAL
        
        current = df['rsi'].iloc[-1]
        prev = df['rsi'].iloc[-2] if len(df) > 1 else current
        
        ob = self.params['rsi_overbought']
        os = self.params['rsi_oversold']
        
        if current < os and prev >= os:
            return Signal.STRONG_BUY   # 超卖反弹
        elif current > ob and prev <= ob:
            return Signal.STRONG_SELL  # 超买回落
        elif current < os:
            return Signal.BUY
        elif current > ob:
            return Signal.SELL
        else:
            return Signal.NEUTRAL
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 ATR (平均真实波幅)"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.params['atr_period']).mean()
        return df
    
    def _atr_signal(self, df: pd.DataFrame) -> Signal:
        """ATR 波动率信号 (高波动+趋势 = 顺势入场机会)"""
        if len(df) < self.params['atr_period'] * 2:
            return Signal.NEUTRAL
        
        atr_current = df['atr'].iloc[-1]
        atr_avg = df['atr'].iloc[-self.params['atr_period']*2:-self.params['atr_period']].mean()
        
        # 波动率扩张 => 可能有趋势
        if atr_current > atr_avg * self.params['atr_multiplier']:
            # 结合方向判断
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-self.params['atr_period']]) / df['close'].iloc[-self.params['atr_period']]
            if price_change > 0.02:
                return Signal.BUY
            elif price_change < -0.02:
                return Signal.SELL
        return Signal.NEUTRAL
    
    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量"""
        df['momentum'] = df['close'] - df['close'].shift(self.params['momentum_period'])
        df['momentum_pct'] = df['momentum'] / df['close'].shift(self.params['momentum_period'])
        return df
    
    def _momentum_signal(self, df: pd.DataFrame) -> Signal:
        """动量信号"""
        if len(df) < self.params['momentum_period'] + 1:
            return Signal.NEUTRAL
        
        mom = df['momentum_pct'].iloc[-1]
        mom_prev = df['momentum_pct'].iloc[-2]
        
        # 动量转正/转负
        if mom > 0.01 and mom_prev <= 0.005:
            return Signal.BUY
        elif mom < -0.01 and mom_prev >= -0.005:
            return Signal.SELL
        elif mom > 0.05:
            return Signal.BUY
        elif mom < -0.05:
            return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 EMA 均线"""
        df['ema_fast'] = df['close'].ewm(span=self.params['trend_ema_fast'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.params['trend_ema_slow'], adjust=False).mean()
        return df
    
    def _trend_signal(self, df: pd.DataFrame) -> Signal:
        """均线趋势信号"""
        if len(df) < 2:
            return Signal.NEUTRAL
        
        fast = df['ema_fast'].iloc[-1]
        slow = df['ema_slow'].iloc[-1]
        fast_prev = df['ema_fast'].iloc[-2]
        slow_prev = df['ema_slow'].iloc[-2]
        
        # 金叉
        if fast > slow and fast_prev <= slow_prev:
            return Signal.STRONG_BUY
        # 死叉
        elif fast < slow and fast_prev >= slow_prev:
            return Signal.STRONG_SELL
        # 多头排列
        elif fast > slow:
            return Signal.BUY
        elif fast < slow:
            return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _support_resistance(self, df: pd.DataFrame) -> Signal:
        """支撑阻力信号 - 检测价格接近关键位"""
        if len(df) < self.params['sr_lookback']:
            return Signal.NEUTRAL
        
        recent = df.tail(self.params['sr_lookback'])
        current_price = df['close'].iloc[-1]
        
        # 找近期高点和低点作为阻力/支撑
        resistance = recent['high'].rolling(5).max().max()
        support = recent['low'].rolling(5).min().min()
        
        sensitivity = self.params['sr_sensitivity']
        
        # 接近支撑
        if abs(current_price - support) / current_price < sensitivity:
            return Signal.BUY
        # 接近阻力
        elif abs(current_price - resistance) / current_price < sensitivity:
            return Signal.SELL
        
        return Signal.NEUTRAL
    
    def _add_order_block(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测 ICT Order Block (简化版)
        
        规则: 在关键位置找到大实体K线，其"缺口"区域为 Order Block
        """
        df['body'] = abs(df['close'] - df['open'])
        df['body_ratio'] = df['body'] / (df['high'] - df['low'] + 0.001)
        
        df['range'] = df['high'] - df['low']
        df['avg_range'] = df['range'].rolling(10).mean()
        
        # 大实体K线标记
        df['is_big_candle'] = (
            (df['body'] > df['avg_range'] * self.params['ob_body_ratio']) &
            (df['body_ratio'] > 0.5)
        )
        
        # Order Block 区域 (大K线的价区)
        df['ob_high'] = np.where(df['is_big_candle'], df['high'], np.nan)
        df['ob_low'] = np.where(df['is_big_candle'], df['low'], np.nan)
        
        # 向前填充，保持OB区域活跃
        df['ob_high'].ffill(inplace=True)
        df['ob_low'].ffill(inplace=True)
        
        # 价格是否在OB区域内
        lookback = self.params['ob_lookback']
        df['in_ob_zone'] = (
            (df['close'] >= df['ob_low']) & 
            (df['close'] <= df['ob_high']) &
            (df.index % 5 == 0)  # 简化: 不是每个K线都触发
        )
        
        return df
    
    def _order_block_signal(self, df: pd.DataFrame) -> Signal:
        """OB 信号"""
        if len(df) < 5:
            return Signal.NEUTRAL
        
        # 最近5根K线内是否有OB触发
        recent = df.tail(5)
        if recent['in_ob_zone'].any():
            # 方向: OB区域的方向取决于对应大实体K线的颜色
            # 简化处理: 结合当前趋势
            if df['close'].iloc[-1] > df['close'].iloc[-5]:
                return Signal.BUY
            else:
                return Signal.SELL
        
        return Signal.NEUTRAL
    
    def get_weighted_signal(self, signals: dict) -> Signal:
        """根据各策略信号加权计算最终信号
        
        类似量子女王的"多重信号共振"
        """
        weights = {
            'rsi': 0.15,
            'atr': 0.10,
            'momentum': 0.15,
            'trend': 0.25,
            'support_resistance': 0.15,
            'order_block': 0.20,
        }
        weights.update(self.config.get('signal_weights', {}))
        
        score = 0.0
        active_strategies = 0
        
        for name, sig in signals.items():
            w = weights.get(name, 0.1)
            score += sig.value * w
            if sig.value != 0:
                active_strategies += 1
        
        # 需要最少N个策略激活才出手
        min_confluence = self.params['confluence_min']
        if active_strategies < min_confluence:
            return Signal.NEUTRAL
        
        if score >= 0.5:
            return Signal.STRONG_BUY if score >= 1.0 else Signal.BUY
        elif score <= -0.5:
            return Signal.STRONG_SELL if score <= -1.0 else Signal.SELL
        
        return Signal.NEUTRAL


if __name__ == '__main__':
    # 测试
    from data.loader import FuturesDataLoader
    
    loader = FuturesDataLoader()
    engine = StrategyEngine()
    
    df = loader.get_daily_data('AU', months=6)
    df, signals = engine.compute_all(df)
    
    print("=== 策略信号 ===")
    for name, sig in signals.items():
        print(f"  {name:20s}: {sig.name}")
    
    final = engine.get_weighted_signal(signals)
    print(f"\n  {'最终信号':20s}: {final.name}")
    
    print(f"\n最后5行数据:")
    print(df[['date', 'close', 'rsi', 'atr', 'ema_fast', 'ema_slow']].tail())
