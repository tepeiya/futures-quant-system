# 参数优化器 - 自动扫描最佳参数
# 对每个品种对遍历 z_entry/lookback/sl_z 组合

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime

from backtest.arbitrage import load_pair_data, PAIRS


def optimize_pair(pair_key, balance=100000, vol=10, commission=0.0003):
    """对一个品种对做参数扫描"""
    cfg = PAIRS.get(pair_key)
    if not cfg:
        print(f"❌ 未知: {pair_key}"); return None
    
    sym_a, sym_b, name, ratio = cfg['a'], cfg['b'], cfg['name'], cfg['ratio']
    
    df = load_pair_data(sym_a, sym_b)
    if df is None or len(df) < 200:
        print(f"❌ 数据不足: {pair_key}"); return None
    
    pa = df['close_a'].values * ratio
    pb = df['close_b'].values
    spr_arr = pa - pb
    
    # 参数范围
    param_grid = {
        'z_entry': [1.5, 1.8, 2.0, 2.2],
        'lookback': [40, 50, 60, 80],
        'sl_z': [2.5, 3.0, 3.5, 4.0],
    }
    
    results = []
    total = len(param_grid['z_entry']) * len(param_grid['lookback']) * len(param_grid['sl_z'])
    n = 0
    
    for ze, lb, sl in product(param_grid['z_entry'], param_grid['lookback'], param_grid['sl_z']):
        n += 1
        
        # 运行回测
        bal_init = float(balance)
        bal = bal_init; peak = bal
        pos = None; cum_pnl = 0.0; wins = 0; losses = 0
        max_dd = 0.0
        
        for i in range(lb, len(df)):
            spr = spr_arr[i]
            hist = spr_arr[i-lb:i]
            spr_ma = hist.mean(); spr_std = hist.std()
            if spr_std < 1: spr_std = 1
            z = (spr - spr_ma) / spr_std
            
            cv = max(vol, min(vol, 20))
            
            if pos is None:
                if z > ze:
                    pos = {'side':'SHORT','es':spr,'v':cv}
                elif z < -ze:
                    pos = {'side':'LONG','es':spr,'v':cv}
            else:
                if pos['side'] == 'SHORT':
                    pnl_chg = (pos['es'] - spr) * pos['v']
                else:
                    pnl_chg = (spr - pos['es']) * pos['v']
                
                # 反向
                if (pos['side']=='SHORT' and z<-ze) or (pos['side']=='LONG' and z>ze):
                    bal += pnl_chg; cum_pnl += pnl_chg
                    if pnl_chg > 0: wins += 1
                    else: losses += 1
                    pos = {'side':'LONG' if pos['side']=='SHORT' else 'SHORT','es':spr,'v':cv}
                    continue
                
                # 止损
                if abs(z) >= sl:
                    bal += pnl_chg; cum_pnl += pnl_chg
                    if pnl_chg > 0: wins += 1
                    else: losses += 1
                    pos = None; continue
                
                # 回归
                if abs(z) <= 0.2:
                    bal += pnl_chg; cum_pnl += pnl_chg
                    if pnl_chg > 0: wins += 1
                    else: losses += 1
                    pos = None; continue
            
            if pos:
                ur = (pos['es']-spr)*pos['v'] if pos['side']=='SHORT' else (spr-pos['es'])*pos['v']
            else: ur = 0
            eq = bal + ur
            if eq > peak: peak = eq
            dd = (peak-eq)/peak*100 if peak>0 else 0
            if dd > max_dd: max_dd = dd
        
        if pos:
            pnl = (pos['es']-spr)*pos['v'] if pos['side']=='SHORT' else (spr-pos['es'])*pos['v']
            bal += pnl; cum_pnl += pnl
        
        ret = (bal-bal_init)/bal_init*100
        sharpe_approx = ret / max(max_dd + 1, 0.1)  # 简化夏普
        score = ret * 0.6 - max_dd * 0.4  # 综合评分: 收益*0.6 - 回撤*0.4
        
        results.append({
            'z_entry': ze, 'lookback': lb, 'sl_z': sl,
            'return_pct': round(ret, 2),
            'max_dd': round(max_dd, 2),
            'trades': wins+losses,
            'win_rate': round(wins/(wins+losses)*100, 1) if wins+losses>0 else 0,
            'score': round(score, 2),
        })
    
    # 排序
    results.sort(key=lambda x: -x['score'])
    
    # 输出
    print(f"\n{'='*55}")
    print(f"  📊 参数优化 - {name} ({pair_key})")
    print(f"  共扫描 {len(results)} 组参数")
    print(f"{'='*55}")
    print(f"  {'z':>4} {'lb':>3} {'sl':>3} {'收益%':>7} {'回撤%':>7} {'交易':>5} {'胜率':>5} {'评分':>6}")
    print(f"  {'-'*42}")
    
    for r in results[:10]:
        print(f"  {r['z_entry']:>4.1f} {r['lookback']:>3d} {r['sl_z']:>3.1f} "
              f"{r['return_pct']:>7.2f} {r['max_dd']:>7.2f} "
              f"{r['trades']:>5d} {r['win_rate']:>5.1f} {r['score']:>6.2f}")
    
    best = results[0]
    print(f"\n  🏆 最佳: z_entry={best['z_entry']} lookback={best['lookback']} "
          f"sl_z={best['sl_z']} → 收益{best['return_pct']:+.2f}% 回撤{best['max_dd']:.2f}%")
    print(f"{'='*55}\n")
    
    return {'pair': pair_key, 'name': name, 'results': results, 'best': best}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='参数优化')
    p.add_argument('--pair', default='RB-HC', help='品种对: RB-HC, SC-MA, Y-P, all')
    args = p.parse_args()
    
    if args.pair == 'all':
        pairs = ['RB-HC', 'SC-MA', 'Y-P']
        best_params = {}
        for pk in pairs:
            r = optimize_pair(pk)
            if r:
                best_params[pk] = r['best']
        
        print(f"\n{'='*55}")
        print(f"  🏆 各品种最佳参数汇总")
        print(f"{'='*55}")
        print(f"  {'品种':<10} {'z_entry':>7} {'lookback':>8} {'sl_z':>4} {'收益%':>7} {'回撤%':>7}")
        print(f"  {'-'*48}")
        for pk, bp in best_params.items():
            print(f"  {PAIRS[pk]['name']:<10} {bp['z_entry']:>7.1f} {bp['lookback']:>8d} "
                  f"{bp['sl_z']:>4.1f} {bp['return_pct']:>7.2f} {bp['max_dd']:>7.2f}")
        print()
    else:
        optimize_pair(args.pair)
