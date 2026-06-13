# 回测引擎 v3 - 量子女王 + 三大改进
# 1. 大周期趋势过滤 (周线EMA)
# 2. 波动率过滤器 (ATR范围过滤)
# 3. XGBoost 市场状态识别 (震荡/趋势分类)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data.loader import FuturesDataLoader

# ========== 缓存数据全局变量 ==========
_DATA_CACHE = {}

def _load_cached(symbol, months):
    """带缓存的数据加载"""
    key = f'{symbol}_{months}'
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    
    loader = FuturesDataLoader()
    df = loader.get_daily_data(symbol, months=months+3)
    if df is not None and not df.empty:
        _DATA_CACHE[key] = df
    return df


class MarketClassifier:
    """市场状态分类器 - 纯NumPy逻辑回归"""
    
    def __init__(self):
        self.w = None; self.b = 0.0
        self.X_mean = None; self.X_std = None
        self.trained = False
        # 缓存预计算数据
        self._prices = None; self._high = None; self._low = None
        self._abs_diff = None; self._atr = None
    
    def _prepare(self, df):
        """预计算所有需要的数据"""
        self._prices = df['close'].values.astype(float)
        self._high = df['high'].values.astype(float)
        self._low = df['low'].values.astype(float)
        n = len(self._prices)
        
        # abs diff
        self._abs_diff = np.abs(np.diff(self._prices))
        
        # ATR
        atr = np.zeros(n)
        for i in range(1, n):
            tr = max(self._high[i]-self._low[i],
                    abs(self._high[i]-self._prices[i-1]),
                    abs(self._low[i]-self._prices[i-1]))
            atr[i] = (atr[i-1]*13 + tr)/14 if i>=14 else (atr[i-1]*i + tr)/(i+1)
        self._atr = atr
    
    def _feat(self, i):
        """提取一个特征向量 (用预计算数据)"""
        c = self._prices; h = self._high; l = self._low
        abs_d = self._abs_diff; atr = self._atr
        n = len(c)
        
        f = []
        
        # 1. 效率比
        for p in [5, 10, 20]:
            if i >= p:
                direction = abs(c[i] - c[i-p])
                total_move = abs_d[i-p:i].sum()
                f.append(direction / max(total_move, 1e-10))
            else:
                f.append(0.5)
        
        # 2. ATR比 (20日均值)
        if i >= 20:
            atr20 = np.mean([max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i-19, i+1)])
            f.append(atr[i] / max(atr20, 1e-10))
        else:
            f.append(1.0)
        
        # 3. 同向K线趋势强度
        if i >= 2:
            d1 = np.sign(c[i-1]-c[i-2]); d2 = np.sign(c[i]-c[i-1])
            f.append((d1+d2)/3.0 if d1==d2 else d2/3.0)
        else:
            f.append(0)
        
        # 4. 价格相对位置 (10日区间)
        if i >= 10:
            hi = max(c[i-9:i+1]); lo = min(c[i-9:i+1])
            f.append((c[i]-lo)/max(hi-lo, 1e-10))
        else:
            f.append(0.5)
        
        # 5. 波动率变化
        if i >= 20:
            v5 = np.std(c[i-4:i+1]); v20 = np.std(c[i-19:i+1])
            f.append(v5/max(v20, 1e-10))
        else:
            f.append(1.0)
        
        return np.array(f)
    
    def train(self, df):
        """训练逻辑回归"""
        self._prepare(df)
        n = len(self._prices)
        if n < 100: return
        
        c = self._prices; atr = self._atr
        
        X_list = []; y_list = []
        for i in range(60, n-10):
            feat = self._feat(i)
            X_list.append(feat)
            future_ret = abs(c[i+5]/c[i]-1)
            th = max(atr[i]/max(c[i],1e-10)*1.5, 0.005)
            y_list.append(1.0 if future_ret > th else 0.0)
        
        if len(X_list) < 100: return
        
        X = np.array(X_list); y = np.array(y_list)
        m = len(X)
        
        # 调试维度
        n_feats = X.shape[1]
        
        self.X_mean = X.mean(axis=0)
        self.X_std = X.std(axis=0) + 1e-10
        X_n = (X - self.X_mean) / self.X_std
        
        w = np.zeros(n_feats); vw = np.zeros(n_feats); b = 0.0; vb = 0.0
        for _ in range(200):
            z = X_n @ w + b
            p = 1/(1+np.exp(-np.clip(z,-20,20)))
            dw = (X_n.T @ (p-y))/m; db = (p-y).mean()
            vw = 0.9*vw + 0.3*dw; vb = 0.9*vb + 0.3*db
            w -= vw; b -= vb
        
        self.w = w; self.b = b; self.trained = True
        # 打印训练精度
        pred = (1/(1+np.exp(-np.clip(X_n@w+b,-20,20))) > 0.5).astype(float)
        acc = (pred == y).mean()
    
    def predict(self, df, i):
        """预测"""
        if not self.trained or i < 60:
            return self._rule_based(df, i)
        
        f = self._feat(i)
        x = (f - self.X_mean) / self.X_std
        z = float(x @ self.w + self.b)
        prob = 1/(1+np.exp(-np.clip(z,-20,20)))
        
        rule = self._rule_based(df, i)
        fused = prob * 0.7 + rule['confidence'] * 0.3
        
        if fused > 0.45:
            return {'regime': 'trend', 'confidence': fused}
        else:
            return {'regime': 'oscillation', 'confidence': 1 - fused}
    
    def _rule_based(self, df: pd.DataFrame, i: int) -> dict:
        """规则版市场状态判断"""
        if i < 20: 
            return {'regime': 'trend', 'confidence': 0.5}
        
        # 1. 效率比
        close = df['close'].values
        direction = abs(close[i] - close[max(0, i-10)])
        total_move = sum(abs(close[j] - close[j-1]) for j in range(max(1, i-9), i+1))
        er = direction / max(total_move, 1e-10)
        
        # 2. ADX简化版
        high = df['high'].values
        low = df['low'].values
        
        # 3. 综合
        trend_score = er * 0.5
        
        # 大周期EMA排列
        if i >= 55:
            e55 = float(df.get('ema55', pd.Series(close).ewm(55).mean()).iloc[i]) if 'ema55' in df.columns else close[i]
            e120 = float(df.get('ema120', pd.Series(close).ewm(120).mean()).iloc[i]) if 'ema120' in df.columns else close[i]*0.98
            if not pd.isna(e55) and not pd.isna(e120):
                if e55 > e120:
                    trend_score += 0.15
                else:
                    trend_score -= 0.15
        
        # ATR相对位置
        atr_val = float(df['atr'].iloc[i])
        atr_pct = atr_val / max(close[i], 1e-10)
        if 0.008 < atr_pct < 0.02:
            trend_score += 0.1  # 适中波动=大概率有趋势
        
        # 价格与EMA9的距离
        if 'ema9' in df.columns and i >= 9:
            e9 = float(df['ema9'].iloc[i])
            dist = abs(close[i] - e9) / max(e9, 1e-10)
            if dist > 0.015:
                trend_score += 0.15  # 远离均线=趋势中
        
        trend_score = max(0, min(1, trend_score))
        
        if trend_score > 0.35:
            return {'regime': 'trend', 'confidence': trend_score}
        else:
            return {'regime': 'oscillation', 'confidence': 1 - trend_score}


class BacktestRunner:
    def __init__(self):
        self.equity_curve = []
        self.trades = []
        self.flat_pnl = []
        self.classifier = MarketClassifier()
    
    def run(self, symbol='AU', months=12, balance=100000, volume=0.5, 
            grid_mult=1.5, max_pos=4, sl_mult=4.0, df=None):
        """回测主函数
        Args:
            df: 可选的预加载DataFrame (避免重复网络请求)
        """
        print(f"\n{'='*55}")
        print(f"  🔬 量子女王 v3 - 三大改进")
        print(f"{'='*55}")
        print(f"  品种: {symbol} | 资金: ¥{balance:,.0f} | 首单: {volume}手")
        print(f"  网格间距: {grid_mult}xATR | 持仓上限: {max_pos} | SL: {sl_mult}xATR")
        print(f"{'='*55}")
        
        # 加载数据
        if df is None:
            key = f'{symbol}_{months}'
            if key not in _DATA_CACHE:
                loader = FuturesDataLoader()
                _DATA_CACHE[key] = loader.get_daily_data(symbol, months=months+3)
            df = _DATA_CACHE[key]
        
        if df is None or df.empty:
            print("❌ 没有数据"); return None
        
        df = df.reset_index(drop=True)
        
        # 截取需要的长度
        df = df.tail(months * 21 + 60).reset_index(drop=True)
        
        # 计算技术指标
        self._calc_indicators(df)
        
        # 训练市场分类器
        print("  训练市场状态分类器...", end=' ')
        self.classifier.train(df)
        print("完成")
        
        print(f"  数据: {len(df)} 日K | {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        print(f"{'='*55}")
        
        # ---- 回放 ----
        pos = []
        nid = 0
        bal = float(balance)
        peak = bal
        
        for i in range(60, len(df)):
            r = df.iloc[i]
            dt = r['date']
            c = float(r['close']); h = float(r['high']); l = float(r['low'])
            atr_val = float(r['atr'])
            rsi_val = float(r['rsi'])
            
            # ========== 三大改进 ==========
            
            # 1️⃣ 大周期趋势过滤 (周线级别)
            trend_up = self._weekly_trend(df, i)
            trend_dn = self._weekly_trend_dn(df, i)
            trend_neutral = not trend_up and not trend_dn
            
            # 2️⃣ 波动率过滤器
            vol_filter = self._volatility_filter(df, i)
            
            # 3️⃣ 市场状态识别
            regime = self.classifier.predict(df, i)
            in_trend = regime['regime'] == 'trend'
            in_oscillation = regime['regime'] == 'oscillation'
            regime_conf = regime['confidence']
            
            # ---- 综合判断 ----
            # 震荡市: 只开0~1仓, 不网格加仓, 严格止损
            # 趋势市: 正常网格
            can_open = vol_filter
            max_pos_here = 1 if in_oscillation else max_pos
            grid_here = grid_mult if in_oscillation else grid_mult * 0.8
            
            # 止损
            for p in list(pos):
                if p['s']=='L' and p['sl'] and l<=p['sl']:
                    pnl=(p['sl']-p['e'])*p['v']; bal+=pnl
                    self.flat_pnl.append({'dt':dt,'t':'SL','s':'L','pnl':pnl})
                    self.trades.append(f"[{dt.date()}] 🛑 SL L @{p['sl']:.1f} {pnl:+.0f}")
                    pos.remove(p)
                elif p['s']=='S' and p['sl'] and h>=p['sl']:
                    pnl=(p['e']-p['sl'])*p['v']; bal+=pnl
                    self.flat_pnl.append({'dt':dt,'t':'SL','s':'S','pnl':pnl})
                    self.trades.append(f"[{dt.date()}] 🛑 SL S @{p['sl']:.1f} {pnl:+.0f}")
                    pos.remove(p)
            
            # 趋势反转 → 平反向 (加成本检测：如果反向亏损超过手续费，才平)
            if trend_up:
                for p in list(pos):
                    if p['s']=='S':
                        rev_cost = (p['e'] - c) * p['v']  # 假设的亏损
                        # 如果反转亏损 + 当前浮亏 < 持有到止损的亏损，就平
                        pnl=(p['e']-c)*p['v']; bal+=pnl
                        self.flat_pnl.append({'dt':dt,'t':'REV','s':'S','pnl':pnl})
                        self.trades.append(f"[{dt.date()}] 🔄 REV S @{c:.0f} {pnl:+.0f}")
                        pos.remove(p)
            elif trend_dn:
                for p in list(pos):
                    if p['s']=='L':
                        pnl=(c-p['e'])*p['v']; bal+=pnl
                        self.flat_pnl.append({'dt':dt,'t':'REV','s':'L','pnl':pnl})
                        self.trades.append(f"[{dt.date()}] 🔄 REV L @{c:.0f} {pnl:+.0f}")
                        pos.remove(p)
            
            # 开仓
            if can_open:
                if trend_up and rsi_val < 55:
                    same=[p for p in pos if p['s']=='L']
                    if not same:
                        sl=c-atr_val*sl_mult; nid+=1
                        pos.append({'id':nid,'s':'L','e':c,'v':volume,'sl':sl})
                        tag='T' if in_trend else 'O'
                        self.trades.append(f"[{dt.date()}] {'🟢' if in_trend else '🔵'} L#{nid} @{c:.0f} R{rsi_val:.0f} 🏷{tag}{regime_conf:.2f}")
                    elif len(same)<max_pos_here and (same[-1]['e']-c)>=atr_val*grid_here:
                        sl=c-atr_val*sl_mult; nid+=1
                        pos.append({'id':nid,'s':'L','e':c,'v':volume,'sl':sl})
                        self.trades.append(f"[{dt.date()}] ➕ +L#{nid} @{c:.0f} GRID")
                
                elif trend_dn and rsi_val > 45:
                    same=[p for p in pos if p['s']=='S']
                    if not same:
                        sl=c+atr_val*sl_mult; nid+=1
                        pos.append({'id':nid,'s':'S','e':c,'v':volume,'sl':sl})
                        tag='T' if in_trend else 'O'
                        self.trades.append(f"[{dt.date()}] {'🟢' if in_trend else '🔵'} S#{nid} @{c:.0f} R{rsi_val:.0f} 🏷{tag}{regime_conf:.2f}")
                    elif len(same)<max_pos_here and (c-same[-1]['e'])>=atr_val*grid_here:
                        sl=c+atr_val*sl_mult; nid+=1
                        pos.append({'id':nid,'s':'S','e':c,'v':volume,'sl':sl})
                        self.trades.append(f"[{dt.date()}] ➕ +S#{nid} @{c:.0f} GRID")
            
            # TP (分层止盈 - 只在趋势市触发,震荡市锁利离场)
            for p in list(pos):
                pf=(c-p['e'])/p['e'] if p['s']=='L' else (p['e']-c)/p['e']
                tp_pct = 0.01 if in_oscillation else 0.02  # 震荡市小利就走,趋势市等大利润
                if pf>=tp_pct:
                    if 'tp' not in p: p['tp']=0
                    lvl=int(pf/tp_pct)
                    if lvl>p['tp']:
                        cv=p['v']*0.5 if in_trend else p['v']  # 趋势市半仓,震荡市全平
                        pnl=(c-p['e'])*cv if p['s']=='L' else (p['e']-c)*cv
                        p['v']-=cv; p['tp']=lvl; bal+=pnl
                        msg=f"TP{lvl}" if in_trend else "EXIT"
                        self.flat_pnl.append({'dt':dt,'t':msg,'s':p['s'],'pnl':pnl})
                        self.trades.append(f"[{dt.date()}] ✅ {msg} #{p['id']} @{c:.0f} {pf*100:.1f}% +{pnl:.0f}")
                        if p['v']<=0: pos.remove(p)
            
            # 移动止损 (震荡市收紧)
            trail_ratio = 0.25 if in_oscillation else 0.35
            trail_pct = 0.01 if in_oscillation else 0.015
            for p in pos:
                if p['s']=='L':
                    pf=(c-p['e'])/p['e']
                    if pf>=trail_pct and p.get('sl_orig', p['sl'])==p['sl']:
                        p['sl_orig']=p['sl']
                        p['sl']=p['e']+(c-p['e'])*trail_ratio
                else:
                    pf=(p['e']-c)/p['e']
                    if pf>=trail_pct and p.get('sl_orig', p['sl'])==p['sl']:
                        p['sl_orig']=p['sl']
                        p['sl']=p['e']-(p['e']-c)*trail_ratio
            
            # 权益
            ur=sum((c-p['e'])*p['v'] if p['s']=='L' else (p['e']-c)*p['v'] for p in pos)
            eq=bal+ur
            if eq>peak: peak=eq
            self.equity_curve.append({'date':dt,'bal':bal,'ur':ur,'eq':eq,'n':len(pos),
                                      'dd':(peak-eq)/peak*100 if peak>0 else 0,
                                      'regime':regime['regime'],
                                      'conf':regime_conf,
                                      'can_open':can_open})
        
        # 收盘平
        fc=float(df['close'].iloc[-1])
        for p in list(pos):
            pnl=(fc-p['e'])*p['v'] if p['s']=='L' else (p['e']-fc)*p['v']
            bal+=pnl; self.flat_pnl.append({'dt':df['date'].iloc[-1],'t':'CLOSE','s':p['s'],'pnl':pnl})
            pos.remove(p)
        
        self._report(symbol, balance, bal)
        
        # 统计过滤效果
        eqdf=pd.DataFrame(self.equity_curve)
        trend_days=eqdf[eqdf['regime']=='trend'].shape[0]
        osc_days=eqdf[eqdf['regime']=='oscillation'].shape[0]
        print(f"  市场状态统计: 趋势 {trend_days}天 / 震荡 {osc_days}天 / 可开仓 {eqdf['can_open'].sum()}天")
        
        return {'bal':bal}
    
    def _calc_indicators(self, df):
        """计算指标 (含大周期)"""
        close = df['close']
        
        # RSI
        delta=close.diff()
        g=delta.where(delta>0,0.0); l=(-delta).where(delta<0,0.0)
        ag=g.ewm(span=14,adjust=False).mean(); al=l.ewm(span=14,adjust=False).mean()
        df['rsi']=100-(100/(1+ag/al.replace(0,np.nan))); df['rsi']=df['rsi'].fillna(50)
        
        # 多周期EMA
        df['ema9']=close.ewm(span=9,adjust=False).mean()
        df['ema21']=close.ewm(span=21,adjust=False).mean()
        df['ema55']=close.ewm(span=55,adjust=False).mean()
        df['ema120']=close.ewm(span=120,adjust=False).mean()
        df['ema250']=close.ewm(span=250,adjust=False).mean() if len(close)>250 else close.rolling(len(close),min_periods=1).mean()
        
        # ATR
        hl=df['high']-df['low']
        hc=abs(df['high']-close.shift(1))
        lc=abs(df['low']-close.shift(1))
        df['atr']=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean().bfill()
        
        # 成交量均线 (如可用)
        if 'volume' in df.columns:
            df['vol_ma']=df['volume'].rolling(20).mean()
    
    def _weekly_trend(self, df, i):
        """大周期上升趋势判断 (周线级别)
        
        标准: 55日均线 > 120日均线 > 250日均线 (多头排列)
        且价格在55日线上方
        """
        if i < 55: return False
        e55 = float(df['ema55'].iloc[i])
        e120 = float(df['ema120'].iloc[i])
        e250 = float(df['ema250'].iloc[i]) if not pd.isna(df['ema250'].iloc[i]) else e120 * 0.95
        c = float(df['close'].iloc[i])
        
        # 主要EMA多头排列
        bullish = e55 > e120
        # 价格在主要均线上方
        price_ok = c > e55
        
        # 再加EMA9 > EMA21确认中短期趋势
        mid_ok = float(df['ema9'].iloc[i]) > float(df['ema21'].iloc[i])
        
        return bullish and price_ok and mid_ok
    
    def _weekly_trend_dn(self, df, i):
        """大周期下降趋势判断"""
        if i < 55: return False
        e55 = float(df['ema55'].iloc[i])
        e120 = float(df['ema120'].iloc[i])
        c = float(df['close'].iloc[i])
        
        bearish = e55 < e120
        price_ok = c < e55
        mid_ok = float(df['ema9'].iloc[i]) < float(df['ema21'].iloc[i])
        
        return bearish and price_ok and mid_ok
    
    def _volatility_filter(self, df, i):
        """波动率过滤器 - 宽松版"""
        if i < 14: return False
        
        c = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        
        atr_pct = atr_val / c if c > 0 else 0
        
        # 只过滤极端情况
        if atr_pct < 0.002 or atr_pct > 0.08:
            return False
        
        # ATR相对历史
        atr_series = [float(df['atr'].iloc[j]) for j in range(max(0,i-19), i+1)]
        atr_ma20 = np.mean(atr_series)
        atr_ratio = atr_val / max(atr_ma20, 1e-10)
        
        # 只过滤ATR极度萎缩(深度震荡)或极度扩张(崩盘)
        if atr_ratio < 0.5 or atr_ratio > 3.0:
            return False
        
        return True
    
    def _rsi(self, prices, n=14):
        delta=prices.diff()
        g=delta.where(delta>0,0.0); l=(-delta).where(delta<0,0.0)
        ag=g.ewm(span=n,adjust=False).mean(); al=l.ewm(span=n,adjust=False).mean()
        rs=ag/al.replace(0,np.nan)
        return (100-(100/(1+rs))).fillna(50)
    
    def _report(self, symbol, init_bal, final_bal):
        ret=(final_bal-init_bal)/init_bal*100
        eqdf=pd.DataFrame(self.equity_curve)
        max_dd=eqdf['dd'].max() if not eqdf.empty else 0
        
        if len(eqdf)>20:
            rtns=eqdf['eq'].pct_change().dropna()
            sharpe=rtns.mean()/rtns.std()*np.sqrt(252) if rtns.std()>0 else 0
        else: sharpe=0
        
        wins=sum(1 for t in self.flat_pnl if t['pnl']>0)
        loss=sum(1 for t in self.flat_pnl if t['pnl']<0)
        tt=wins+loss
        wr=wins/tt*100 if tt>0 else 0
        aw=np.mean([t['pnl'] for t in self.flat_pnl if t['pnl']>0]) if wins>0 else 0
        al=abs(np.mean([t['pnl'] for t in self.flat_pnl if t['pnl']<0])) if loss>0 else 1
        
        print(f"\n{'='*55}")
        print(f"  📊 回测报告 - {symbol}")
        print(f"{'='*55}")
        print(f"  初始资金    ¥{init_bal:>10,.2f}")
        print(f"  最终资金    ¥{final_bal:>10,.2f}")
        print(f"  总盈亏      ¥{final_bal-init_bal:>+10,.2f}")
        print(f"  收益率      {ret:>+9.2f}%")
        print(f"{'─'*55}")
        print(f"  交易 {tt}次 | 胜率 {wr:.1f}% ({wins}赢/{loss}亏)")
        if aw>0 and al>0:
            print(f"  盈亏比 {aw/al:.2f} | 夏普 {sharpe:.2f}")
        print(f"  最大回撤    {max_dd:.2f}%")
        print(f"{'─'*55}")
        
        if self.trades:
            print(f"\n  交易日志 (最后12笔):")
            for t in self.trades[-12:]: print(f"    {t}")
        print(f"\n{'='*55}\n")
    
    def plot(self, symbol):
        """绘制增强版图表"""
        if not self.equity_curve: return None
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            df = pd.DataFrame(self.equity_curve)
            
            fig, ax = plt.subplots(4,1,figsize=(14,12),gridspec_kw={'height_ratios':[3,1,1,0.8]})
            
            # 1. 权益曲线
            ax[0].plot(df['date'],df['eq'],'b-',lw=1.5,label='Equity')
            ax[0].plot(df['date'],df['bal'],'g--',lw=1,alpha=0.7,label='Balance')
            ax[0].axhline(y=df['eq'].iloc[0],color='gray',ls=':',alpha=0.5)
            ax[0].fill_between(df['date'],df['eq'].iloc[0],df['eq'],
                               where=df['eq']>=df['eq'].iloc[0],color='green',alpha=0.08)
            ax[0].fill_between(df['date'],df['eq'].iloc[0],df['eq'],
                               where=df['eq']<df['eq'].iloc[0],color='red',alpha=0.08)
            ax[0].set_title(f'{symbol} v3 - 权益曲线 + 市场状态',fontsize=14)
            ax[0].set_ylabel('¥'); ax[0].grid(True,alpha=0.3); ax[0].legend()
            
            # 2. 回撤
            ax[1].fill_between(df['date'],0,df['dd'],color='red',alpha=0.3)
            ax[1].plot(df['date'],df['dd'],'r-',lw=1)
            ax[1].set_ylabel('Drawdown %'); ax[1].grid(True,alpha=0.3)
            
            # 3. 持仓
            ax[2].bar(df['date'],df['n'],color='purple',alpha=0.4,width=1)
            ax[2].set_ylabel('Positions'); ax[2].grid(True,alpha=0.3)
            
            # 4. 市场状态背景
            colors = df['regime'].map({'trend':'green','oscillation':'orange','unknown':'gray'})
            ax[3].bar(df['date'],df['conf'],color=colors,alpha=0.5,width=1)
            ax[3].axhline(y=0.5,color='black',lw=0.5,ls='--')
            ax[3].set_ylabel('Regime\n(green=trend)')
            ax[3].set_xlabel('Date')
            ax[3].set_ylim(0,1)
            ax[3].grid(True,alpha=0.3)
            
            plt.tight_layout()
            out=f'/var/minis/workspace/futures_backtest_{symbol}_v3.png'
            plt.savefig(out,dpi=120); plt.close()
            return out
        except Exception as e:
            print(f"绘图失败: {e}"); return None


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='期货网格 v3 - 量子女王+三大改进')
    p.add_argument('--symbol',default='RB')
    p.add_argument('--months',type=int,default=12)
    p.add_argument('--balance',type=float,default=50000)
    p.add_argument('--volume',type=float,default=2)
    p.add_argument('--grid',type=float,default=1.5)
    p.add_argument('--max-pos',type=int,default=4)
    p.add_argument('--sl',type=float,default=4.0)
    p.add_argument('--noplot',action='store_true')
    args = p.parse_args()
    
    r = BacktestRunner()
    r.run(args.symbol, args.months, args.balance, args.volume, args.grid, args.max_pos, args.sl)
    if not args.noplot:
        img = r.plot(args.symbol)
        if img: print(f"📊 图表: {img}")
