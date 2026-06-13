# 跨品种套利引擎 v3 - 多品种组合
#
# 品种对:
#   RB-HC  螺纹钢-热卷 (钢材)
#   Y-P    豆油-棕榈油 (油脂)
#   CU-ZN  铜-锌 (有色金属)
#   I-RB   铁矿石-螺纹钢 (产业链)
#   AU-AG  黄金-白银 (贵金属)
#   SC-MA  原油-甲醇 (化工)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ========== 品种对配置 ==========
PAIRS = {
    'RB-HC': {'name': '螺纹钢-热卷', 'a': 'RB', 'b': 'HC', 'ratio': 1.0, 'type': '钢材'},
    'Y-P':   {'name': '豆油-棕榈油', 'a': 'Y',  'b': 'P', 'ratio': 1.0, 'type': '油脂'},
    'CU-ZN': {'name': '铜-锌',       'a': 'CU', 'b': 'ZN','ratio': 1.0, 'type': '有色'},
    'I-RB':  {'name': '铁矿石-螺纹钢','a': 'I',  'b': 'RB','ratio': 1.0, 'type': '黑色'},
    'AU-AG': {'name': '黄金-白银',    'a': 'AU', 'b': 'AG','ratio': 80,  'type': '贵金属'},
    'SC-MA': {'name': '原油-甲醇',    'a': 'SC', 'b': 'MA','ratio': 1.0, 'type': '化工'},
}


def load_pair_data(sym_a, sym_b):
    """加载品种对数据并合并"""
    try:
        da = pd.read_csv(f'/tmp/{sym_a.lower()}_data.csv')
        db = pd.read_csv(f'/tmp/{sym_b.lower()}_data.csv')
    except:
        return None
    
    da = da.rename(columns={'日期':'date','收盘价':'close_a'})
    db = db.rename(columns={'日期':'date','收盘价':'close_b'})
    da['date'] = pd.to_datetime(da['date'])
    db['date'] = pd.to_datetime(db['date'])
    
    df = pd.merge(da[['date','close_a']], db[['date','close_b']], on='date')
    return df.sort_values('date').reset_index(drop=True)


def analyze_pair(pair_key):
    """分析一个品种对的价差特性和相关性"""
    cfg = PAIRS.get(pair_key)
    if not cfg:
        return None
    
    df = load_pair_data(cfg['a'], cfg['b'])
    if df is None or len(df) < 100:
        return None
    
    pa = df['close_a'].values * cfg['ratio']
    pb = df['close_b'].values
    spread = pa - pb
    
    # 相关性
    corr = np.corrcoef(pa[-500:], pb[-500:])[0,1]
    
    # 价差统计
    spr_mean = spread[-500:].mean()
    spr_std = spread[-500:].std()
    spr_current = spread[-1]
    z_current = (spr_current - spr_mean) / spr_std if spr_std > 0 else 0
    
    # 回归特性（检验价差是否均值回归）
    spr_50d = pd.Series(spread).rolling(50).mean().values[-250:]
    spr_50d_diff = np.diff(spr_50d)
    mean_reversion = np.mean(spr_50d_diff * np.sign(np.random.randn(len(spr_50d_diff))))  # 简化版
    
    return {
        'pair': pair_key,
        'name': cfg['name'],
        'type': cfg['type'],
        'a': cfg['a'], 'b': cfg['b'],
        'corr': round(corr, 3),
        'spr_mean': round(spr_mean, 1),
        'spr_std': round(spr_std, 1),
        'spr_current': round(spr_current, 1),
        'z_current': round(z_current, 2),
        'n_days': len(df),
    }


def run_pair(pair_key, balance=100000, vol=10,
             z_entry=1.8, z_exit=0.2, lookback=50, sl_z=3.0,
             vol_per_pnl=300, max_vol=20,
             plot=False):
    """运行单个品种对的套利回测"""
    cfg = PAIRS.get(pair_key)
    if not cfg:
        print(f"❌ 未知品种对: {pair_key}"); return None
    
    sym_a, sym_b, name = cfg['a'], cfg['b'], cfg['name']
    ratio = cfg['ratio']
    
    df = load_pair_data(sym_a, sym_b)
    if df is None:
        print(f"❌ 无法加载 {sym_a}-{sym_b} 数据"); return None
    
    print(f"\n{'='*60}")
    print(f"  🔗 {name} ({sym_a}:{sym_b} = {ratio}:1)")
    print(f"{'='*60}")
    print(f"  资金: ¥{balance:,.0f} | 基础手数: {vol} | 最大: {max_vol}")
    print(f"  开仓: {z_entry}σ | 平仓: {z_exit}σ | 止损: {sl_z}σ")
    print(f"  窗口: {lookback}天")
    print(f"  数据: {len(df)}条 | {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    
    pa = df['close_a'].values * ratio
    pb = df['close_b'].values
    spr_arr = pa - pb
    
    bal = float(balance); peak = bal; cum_pnl = 0.0
    pos = None; trades = []; equity = []
    wins = 0; losses = 0
    
    for i in range(lookback, len(df)):
        dt = df['date'].iloc[i]
        spr = spr_arr[i]
        
        hist = spr_arr[i-lookback:i]
        spr_ma = hist.mean(); spr_std = hist.std()
        if spr_std < 1: spr_std = 1
        z = (spr - spr_ma) / spr_std
        
        # 动态手数
        bonus = int(abs(cum_pnl) / max(vol_per_pnl, 1))
        cv = max(vol, min(vol + bonus, max_vol))
        
        if pos is None:
            if z > z_entry:
                pos = {'side':'SHORT','es':spr,'ez':z,'v':cv,'dt':dt}
                trades.append(f"[{dt.date()}] 🟢 SHORT z={z:.2f} spr={spr:.0f} v={cv}")
            elif z < -z_entry:
                pos = {'side':'LONG','es':spr,'ez':z,'v':cv,'dt':dt}
                trades.append(f"[{dt.date()}] 🟢 LONG  z={z:.2f} spr={spr:.0f} v={cv}")
        else:
            if pos['side'] == 'SHORT':
                pnl_chg = (pos['es'] - spr) * pos['v']
            else:
                pnl_chg = (spr - pos['es']) * pos['v']
            
            # 反向信号
            if pos['side'] == 'SHORT' and z < -z_entry:
                bal += pnl_chg; cum_pnl += pnl_chg
                if pnl_chg > 0: wins += 1; trades.append(f"[{dt.date()}] ✅ 平SHORT→LONG +{pnl_chg:.0f}")
                else: losses += 1; trades.append(f"[{dt.date()}] ⚡ 反手SHORT→LONG {pnl_chg:.0f}")
                pos = {'side':'LONG','es':spr,'ez':z,'v':cv,'dt':dt}
                continue
            elif pos['side'] == 'LONG' and z > z_entry:
                bal += pnl_chg; cum_pnl += pnl_chg
                if pnl_chg > 0: wins += 1; trades.append(f"[{dt.date()}] ✅ 平LONG→SHORT +{pnl_chg:.0f}")
                else: losses += 1; trades.append(f"[{dt.date()}] ⚡ 反手LONG→SHORT {pnl_chg:.0f}")
                pos = {'side':'SHORT','es':spr,'ez':z,'v':cv,'dt':dt}
                continue
            
            # 止损
            if abs(z) >= sl_z:
                bal += pnl_chg; cum_pnl += pnl_chg
                if pnl_chg > 0: wins += 1; trades.append(f"[{dt.date()}] 🛑 止损 {pos['side']} +{pnl_chg:.0f}")
                else: losses += 1; trades.append(f"[{dt.date()}] 🛑 止损 {pos['side']} {pnl_chg:.0f}")
                pos = None; continue
            
            # 回归平仓
            if abs(z) <= z_exit:
                bal += pnl_chg; cum_pnl += pnl_chg
                if pnl_chg > 0: wins += 1; trades.append(f"[{dt.date()}] ✅ 回归 +{pnl_chg:.0f}")
                else: losses += 1; trades.append(f"[{dt.date()}] ⚠️ 离场 {pnl_chg:.0f}")
                pos = None; continue
        
        if pos:
            ur = (pos['es']-spr)*pos['v'] if pos['side']=='SHORT' else (spr-pos['es'])*pos['v']
        else: ur = 0
        eq = bal + ur
        if eq > peak: peak = eq
        equity.append({'date':dt,'eq':eq,'bal':bal,'ur':ur,'z':z,'in_pos':1 if pos else 0,'dd':(peak-eq)/peak*100 if peak>0 else 0})
    
    if pos:
        pnl = (pos['es']-spr)*pos['v'] if pos['side']=='SHORT' else (spr-pos['es'])*pos['v']
        bal += pnl; cum_pnl += pnl
        trades.append(f"[{dt.date()}] 🔚 收盘 {pos['side']} {pnl:.0f}")
    
    # 报告
    eqdf = pd.DataFrame(equity)
    ret = (bal-balance)/balance*100
    md = eqdf['dd'].max() if not eqdf.empty else 0
    if len(eqdf)>20:
        r = eqdf['eq'].pct_change().dropna()
        sp = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    else: sp = 0
    wr = wins/(wins+losses)*100 if wins+losses>0 else 0
    
    print(f"\n{'='*60}")
    print(f"  📊 套利报告 - {name}")
    print(f"{'='*60}")
    print(f"  初始资金    ¥{balance:>10,.2f}")
    print(f"  最终资金    ¥{bal:>10,.2f}")
    print(f"  总盈亏      ¥{bal-balance:>+10,.2f}")
    print(f"  收益率      {ret:>+9.2f}%")
    print(f"{'─'*60}")
    print(f"  交易 {wins+losses}次 | 胜率 {wr:.1f}% ({wins}赢/{losses}亏)")
    print(f"  夏普 {sp:.2f} | 最大回撤 {md:.2f}%")
    print(f"  动态手数: {vol}→{max_vol}")
    if trades:
        print(f"\n  交易日志 (最后12笔):")
        for t in trades[-12:]: print(f"    {t}")
    print(f"\n{'='*60}\n")
    
    return {'bal':bal,'pnl':bal-balance,'sharpe':sp,'dd':md}


def analyze_all():
    """分析所有品种对"""
    print(f"\n{'='*60}")
    print(f"  📊 品种对分析")
    print(f"{'='*60}")
    print(f"  {'对':<10} {'类型':<8} {'相关性':>6} {'价差均值':>10} {'标准差':>8} {'当前Z':>6} {'天数':>5}")
    print(f"  {'-'*55}")
    
    results = []
    for pk in sorted(PAIRS.keys()):
        r = analyze_pair(pk)
        if r:
            results.append(r)
            print(f"  {pk:<10} {r['type']:<8} {r['corr']:>6.3f} {r['spr_mean']:>10.1f} {r['spr_std']:>8.1f} {r['z_current']:>6.2f} {r['n_days']:>5d}")
    
    print(f"\n  可交易品种对: {len(results)}")
    
    # 推荐排序
    print(f"\n  推荐排序 (按相关性+数据量):")
    ranked = sorted(results, key=lambda x: -abs(x['corr']) * x['n_days'])
    for r in ranked:
        print(f"    {r['pair']:<10} {r['name']:<16} corr={r['corr']:>6.3f} z={r['z_current']:>5.2f}")
    
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pair', default='all', help='品种对: RB-HC, Y-P, CU-ZN, I-RB, AU-AG, SC-MA, all')
    p.add_argument('--balance', type=float, default=100000)
    p.add_argument('--vol', type=float, default=10)
    p.add_argument('--z-entry', type=float, default=1.8)
    p.add_argument('--z-exit', type=float, default=0.2)
    p.add_argument('--lookback', type=int, default=50)
    p.add_argument('--sl-z', type=float, default=3.0)
    p.add_argument('--analyze', action='store_true', help='分析所有品种对')
    args = p.parse_args()
    
    if args.analyze:
        analyze_all()
    elif args.pair == 'all':
        print("\n===== 多品种套利回测 =====")
        for pk in ['RB-HC','Y-P','CU-ZN','I-RB','AU-AG','SC-MA']:
            run_pair(pk, balance=args.balance, vol=args.vol,
                    z_entry=args.z_entry, z_exit=args.z_exit,
                    lookback=args.lookback, sl_z=args.sl_z)
    else:
        run_pair(args.pair, balance=args.balance, vol=args.vol,
                z_entry=args.z_entry, z_exit=args.z_exit,
                lookback=args.lookback, sl_z=args.sl_z)
