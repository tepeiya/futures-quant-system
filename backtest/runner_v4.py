# 回测引擎 v4 - 多策略组合
# 6个子策略并行 + 动态权重分配 + 手续费

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from engine.ensemble import EnsembleEngine, Signal

# 缓存
_DATA_CACHE = {}


class BacktestRunnerV4:
    def __init__(self):
        self.equity_curve = []
        self.trades = []
        self.flat_pnl = []
        self.ensemble = EnsembleEngine()
    
    def run(self, symbol='AU', months=12, balance=100000, volume=0.5,
            sl_mult=4.0, commission=0.0003):
        """回测主函数
        Args:
            commission: 手续费率 (万三)
        """
        print(f"\n{'='*60}")
        print(f"  🚀 量子女王 v4 - 多策略组合 (6策略并行)")
        print(f"{'='*60}")
        print(f"  品种: {symbol} | 资金: ¥{balance:,.0f} | 首单: {volume}手")
        print(f"  止损: {sl_mult}xATR | 手续费: {commission*10000:.1f}/万")
        print(f"{'='*60}")
        
        # 加载数据
        key = f'{symbol}_{months}'
        if key not in _DATA_CACHE:
            from data.loader import FuturesDataLoader
            loader = FuturesDataLoader()
            _DATA_CACHE[key] = loader.get_daily_data(symbol, months=months+3)
        df = _DATA_CACHE[key]
        
        if df is None or df.empty:
            print("❌ 没有数据"); return None
        
        df = df.tail(months * 21 + 60).reset_index(drop=True)
        self._calc_indicators(df)
        
        print(f"  数据: {len(df)} 日K | {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        print(f"{'='*60}")
        
        # ---- 回放 ----
        positions = []  # [{id, s(L/S), e(entry), v(vol), sl, strategy(来源策略名)}]
        nid = 0
        bal = float(balance)
        peak_bal = bal
        total_commission = 0.0
        
        for i in range(60, len(df)):
            r = df.iloc[i]
            dt = r['date']
            c = float(r['close']); h = float(r['high']); l = float(r['low'])
            atr_val = float(r['atr'])
            rsi_val = float(r['rsi'])
            
            # ---- 多策略信号 ----
            signal_result = self.ensemble.get_signals(df, i, positions)
            total_score = signal_result['total_score']
            n_buy = signal_result['n_buy']
            n_sell = signal_result['n_sell']
            
            # ---- 止损 ----
            for p in list(positions):
                if p['s'] == 'L' and p['sl'] and l <= p['sl']:
                    pnl = (p['sl'] - p['e']) * p['v']
                    fee = abs(pnl) * commission * 0.1
                    bal += pnl - fee
                    total_commission += fee
                    self.flat_pnl.append({'dt': dt, 't': 'SL', 's': 'L', 'pnl': pnl-fee})
                    self.trades.append(f"[{dt.date()}] 🛑 SL L@{p['sl']:.0f} {pnl:+.0f}(费{fee:.1f})")
                    self.ensemble.update_strategy_performance(p['strategy'], pnl-fee, pnl > 0)
                    positions.remove(p)
                elif p['s'] == 'S' and p['sl'] and h >= p['sl']:
                    pnl = (p['e'] - p['sl']) * p['v']
                    fee = abs(pnl) * commission * 0.1
                    bal += pnl - fee
                    total_commission += fee
                    self.flat_pnl.append({'dt': dt, 't': 'SL', 's': 'S', 'pnl': pnl-fee})
                    self.trades.append(f"[{dt.date()}] 🛑 SL S@{p['sl']:.0f} {pnl:+.0f}(费{fee:.1f})")
                    self.ensemble.update_strategy_performance(p['strategy'], pnl-fee, pnl > 0)
                    positions.remove(p)
            
            # ---- 开仓 ----
            # 信号阈值放宽: 总得分 > 0.8 且 至少1个策略看多
            can_open_short = total_score < -0.8 and n_sell >= 1
            can_open_long = total_score > 0.8 and n_buy >= 1
            
            if can_open_long:
                same_side = [p for p in positions if p['s'] == 'L']
                if not same_side:
                    sl = c - atr_val * sl_mult
                    nid += 1
                    positions.append({'id': nid, 's': 'L', 'e': c, 'v': volume, 'sl': sl,
                                      'strategy': 'ensemble'})
                    # 记录哪些策略触发了
                    active = [k for k, v in signal_result['signals'].items() if v['signal'].value > 0]
                    self.trades.append(f"[{dt.date()}] 🟢 L#{nid} @{c:.0f} | {'+'.join(active)}")
                elif len(same_side) < 2 and (same_side[-1]['e'] - c) >= atr_val * 1.5:
                    sl = c - atr_val * sl_mult
                    nid += 1
                    positions.append({'id': nid, 's': 'L', 'e': c, 'v': volume, 'sl': sl,
                                      'strategy': 'ensemble'})
                    self.trades.append(f"[{dt.date()}] ➕ +L#{nid} @{c:.0f} GRID")
            
            elif can_open_short:
                same_side = [p for p in positions if p['s'] == 'S']
                if not same_side:
                    sl = c + atr_val * sl_mult
                    nid += 1
                    positions.append({'id': nid, 's': 'S', 'e': c, 'v': volume, 'sl': sl,
                                      'strategy': 'ensemble'})
                    active = [k for k, v in signal_result['signals'].items() if v['signal'].value < 0]
                    self.trades.append(f"[{dt.date()}] 🔴 S#{nid} @{c:.0f} | {'+'.join(active)}")
                elif len(same_side) < 2 and (c - same_side[-1]['e']) >= atr_val * 1.5:
                    sl = c + atr_val * sl_mult
                    nid += 1
                    positions.append({'id': nid, 's': 'S', 'e': c, 'v': volume, 'sl': sl,
                                      'strategy': 'ensemble'})
                    self.trades.append(f"[{dt.date()}] ➕ +S#{nid} @{c:.0f} GRID")
            
            # 趋势反转 → 不做平仓，持有等止损（避免反转成本）
            # 只做新方向的开仓信号
            
            # ---- 止盈 ----
            for p in list(positions):
                pf = (c - p['e']) / p['e'] if p['s'] == 'L' else (p['e'] - c) / p['e']
                if pf >= 0.02:
                    if 'tp' not in p: p['tp'] = 0
                    lvl = int(pf / 0.02)
                    if lvl > p['tp']:
                        cv = p['v'] * 0.5
                        pnl = (c - p['e']) * cv if p['s'] == 'L' else (p['e'] - c) * cv
                        fee = abs(pnl) * commission
                        p['v'] -= cv; p['tp'] = lvl
                        bal += pnl - fee
                        total_commission += fee
                        self.flat_pnl.append({'dt': dt, 't': 'TP', 's': p['s'], 'pnl': pnl-fee})
                        self.trades.append(f"[{dt.date()}] ✅ TP{lvl} #{p['id']} @{c:.0f} +{pnl:.0f}")
                        if p['v'] <= 0: positions.remove(p)
            
            # ---- 移动止损 ----
            for p in positions:
                if p['s'] == 'L':
                    pf = (c - p['e']) / p['e']
                    if pf >= 0.015 and p.get('sl_orig', p['sl']) == p['sl']:
                        p['sl_orig'] = p['sl']
                        p['sl'] = p['e'] + (c - p['e']) * 0.3
                else:
                    pf = (p['e'] - c) / p['e']
                    if pf >= 0.015 and p.get('sl_orig', p['sl']) == p['sl']:
                        p['sl_orig'] = p['sl']
                        p['sl'] = p['e'] - (p['e'] - c) * 0.3
            
            # ---- 权益 ----
            ur = sum((c - p['e']) * p['v'] if p['s'] == 'L' else (p['e'] - c) * p['v'] for p in positions)
            eq = bal + ur
            if eq > peak_bal: peak_bal = eq
            self.equity_curve.append({
                'date': dt, 'bal': bal, 'ur': ur, 'eq': eq, 'n': len(positions),
                'dd': (peak_bal - eq) / peak_bal * 100 if peak_bal > 0 else 0,
                'score': total_score,
                'n_buy': n_buy, 'n_sell': n_sell,
            })
        
        # 收盘平
        fc = float(df['close'].iloc[-1])
        for p in list(positions):
            pnl = (fc - p['e']) * p['v'] if p['s'] == 'L' else (p['e'] - fc) * p['v']
            fee = abs(pnl) * commission
            bal += pnl - fee
            total_commission += fee
            positions.remove(p)
        
        self._report(symbol, balance, bal, total_commission)
        self.ensemble.print_weights()
        print(f"{'='*60}\n")
        
        return {'bal': bal}
    
    def _calc_indicators(self, df):
        """计算指标"""
        c = df['close']
        # EMA
        df['ema_f'] = c.ewm(span=9, adjust=False).mean()
        df['ema_s'] = c.ewm(span=21, adjust=False).mean()
        df['ema55'] = c.ewm(span=55, adjust=False).mean()
        df['ema120'] = c.ewm(span=120, adjust=False).mean()
        # RSI
        delta = c.diff()
        g = delta.where(delta > 0, 0.0); l = (-delta).where(delta < 0, 0.0)
        ag = g.ewm(span=14, adjust=False).mean(); al = l.ewm(span=14, adjust=False).mean()
        df['rsi'] = 100 - (100 / (1 + ag / al.replace(0, np.nan))); df['rsi'] = df['rsi'].fillna(50)
        # ATR
        hl = df['high'] - df['low']
        hc = abs(df['high'] - c.shift(1))
        lc = abs(df['low'] - c.shift(1))
        df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean().bfill()
    
    def _report(self, symbol, init, final, commission):
        ret = (final - init) / init * 100
        eqdf = pd.DataFrame(self.equity_curve)
        max_dd = eqdf['dd'].max() if not eqdf.empty else 0
        
        if len(eqdf) > 20:
            rtns = eqdf['eq'].pct_change().dropna()
            sharpe = rtns.mean() / rtns.std() * np.sqrt(252) if rtns.std() > 0 else 0
        else:
            sharpe = 0
        
        wins = sum(1 for t in self.flat_pnl if t['pnl'] > 0)
        loss = sum(1 for t in self.flat_pnl if t['pnl'] < 0)
        tt = wins + loss
        wr = wins / tt * 100 if tt > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"  📊 回测报告 - {symbol} (v4多策略)")
        print(f"{'='*60}")
        print(f"  初始资金    ¥{init:>10,.2f}")
        print(f"  最终资金    ¥{final:>10,.2f}")
        print(f"  总盈亏      ¥{final-init:>+10,.2f}")
        print(f"  收益率      {ret:>+9.2f}%")
        print(f"  手续费      ¥{commission:>8.2f}")
        print(f"{'─'*60}")
        print(f"  交易 {tt}次 | 胜率 {wr:.1f}% ({wins}赢/{loss}亏)")
        if len(eqdf) > 0:
            print(f"  夏普比率    {sharpe:.2f} | 最大回撤 {max_dd:.2f}%")
        print(f"{'─'*60}")
        
        # 策略信号统计
        signals_log = eqdf[['n_buy', 'n_sell']].mean() if not eqdf.empty else None
        if signals_log is not None:
            print(f" 日均信号: 买入 {signals_log['n_buy']:.1f}个 卖出 {signals_log['n_sell']:.1f}个")
        
        if self.trades:
            print(f"\n  交易日志 (最后15笔):")
            for t in self.trades[-15:]:
                print(f"    {t}")
        print(f"\n{'='*60}\n")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='AU')
    p.add_argument('--months', type=int, default=12)
    p.add_argument('--balance', type=float, default=100000)
    p.add_argument('--volume', type=float, default=0.5)
    p.add_argument('--sl', type=float, default=4.0)
    p.add_argument('--commission', type=float, default=0.0003)
    args = p.parse_args()
    
    r = BacktestRunnerV4()
    r.run(args.symbol, args.months, args.balance, args.volume, args.sl, args.commission)

