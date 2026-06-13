#!/usr/bin/env python3
"""
实时行情监控 - 检查各套利品种对当前价差状态
每天运行一次，看有没有开仓/平仓信号

用法:
  python3 monitor.py                  # 检查所有品种对
  python3 monitor.py --pair RB-HC     # 指定品种
  python3 monitor.py --notify         # 有信号时发送通知
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from urllib.request import urlopen
import warnings
warnings.filterwarnings('ignore')


# ========== 品种对配置 ==========
# 格式: {key: {name, sym_a, sym_b, ratio, ze, lb, sl, z_exit}}
PAIR_CONFIG = {
    'RB-HC': {'name': '螺纹钢-热卷', 'a': 'RB', 'b': 'HC', 'ratio': 1.0, 'ze': 1.5, 'lb': 40, 'sl': 3.0, 'zx': 0.2},
    'SC-MA': {'name': '原油-甲醇',   'a': 'SC', 'b': 'MA', 'ratio': 1.0, 'ze': 1.5, 'lb': 50, 'sl': 2.5, 'zx': 0.2},
    'Y-P':   {'name': '豆油-棕榈油', 'a': 'Y',  'b': 'P',  'ratio': 1.0, 'ze': 1.5, 'lb': 50, 'sl': 3.0, 'zx': 0.2},
}

DATA_DIR = '/tmp'


def fetch_live_data(symbol):
    """从新浪API获取最新数据"""
    url = f'https://stock.finance.sina.com.cn/futures/api/jsonp.php//InnerFuturesNewService.getDailyKLine?symbol={symbol}0'
    try:
        resp = urlopen(url, timeout=8)
        raw = resp.read().decode('utf-8')
        data = json.loads(raw.split('(', 1)[1].rsplit(')', 1)[0])
        
        # 保存到本地
        fn = f'{DATA_DIR}/{symbol.lower()}_data.csv'
        with open(fn, 'w') as f:
            f.write('日期,开盘价,最高价,最低价,收盘价,成交量,持仓量,动态结算价\n')
            for row in data:
                f.write(f"{row['d']},{row['o']},{row['h']},{row['l']},{row['c']},{row['v']},{row['p']},{row['s']}\n")
        
        closes = [float(r['c']) for r in data]
        return closes
    except Exception as e:
        print(f"    ❌ 获取 {symbol} 失败: {e}")
        return None


def load_history(symbol):
    """从本地缓存加载历史数据"""
    fn = f'{DATA_DIR}/{symbol.lower()}_data.csv'
    try:
        df = pd.read_csv(fn)
        return df['收盘价'].values.tolist()
    except:
        return None


def check_pair(key, cfg, use_live=True):
    """检查一个品种对的当前状态"""
    name = cfg['name']
    sym_a = cfg['a']
    sym_b = cfg['b']
    ratio = cfg['ratio']
    ze = cfg['ze']
    lb = cfg['lb']
    sl = cfg['sl']
    zx = cfg['zx']
    
    # 获取数据
    if use_live:
        print(f"  📡 获取 {sym_a} 数据...")
        close_a = fetch_live_data(sym_a)
        print(f"  📡 获取 {sym_b} 数据...")
        close_b = fetch_live_data(sym_b)
    else:
        close_a = load_history(sym_a)
        close_b = load_history(sym_b)
    
    if not close_a or not close_b:
        return {'error': '数据获取失败'}
    
    # 对齐
    n = min(len(close_a), len(close_b))
    pa = np.array(close_a[-n:]) * ratio
    pb = np.array(close_b[-n:])
    spr = pa - pb
    
    # 当前价差
    current_spr = spr[-1]
    
    # Z-Score (用最近 lb 天)
    if len(spr) < lb + 1:
        return {'error': '数据不足'}
    
    hist = spr[-(lb+1):-1]  # 只用历史数据
    spr_mean = np.mean(hist)
    spr_std = np.std(hist)
    z = (current_spr - spr_mean) / max(spr_std, 1)
    
    # 最近N天价差走势
    n_days = min(30, len(spr))
    recent_spr = spr[-n_days:]
    
    result = {
        'pair': key,
        'name': name,
        'sym_a': sym_a,
        'sym_b': sym_b,
        'price_a': round(close_a[-1], 2),
        'price_b': round(close_b[-1], 2),
        'current_spr': round(current_spr, 2),
        'spr_mean': round(spr_mean, 2),
        'spr_std': round(spr_std, 2),
        'z_score': round(z, 2),
        'lookback': lb,
        'data_days': len(spr),
        'spr_min': round(min(recent_spr), 2),
        'spr_max': round(max(recent_spr), 2),
        'spr_trend': '上涨' if len(recent_spr) > 5 and recent_spr[-1] > recent_spr[-5] else '下跌',
    }
    
    # 信号判断
    signals = []
    
    if z > ze:
        result['signal'] = '做空价差 🟢'
        trigger = f'价差过高达{z:.2f}σ，卖{sym_a}买{sym_b}'
        signals.append({
            'direction': 'SHORT',
            'reason': trigger,
            'z_score': round(z, 2),
            'entry_spr': round(current_spr, 2),
        })
    elif z < -ze:
        result['signal'] = '做多价差 🟢'
        trigger = f'价差过低达{abs(z):.2f}σ，买{sym_a}卖{sym_b}'
        signals.append({
            'direction': 'LONG',
            'reason': trigger,
            'z_score': round(z, 2),
            'entry_spr': round(current_spr, 2),
        })
    elif abs(z) <= zx:
        result['signal'] = '已回归，应平仓 ✅'
    elif abs(z) >= sl:
        result['signal'] = f'超止损! {abs(z):.2f}σ ⚠️'
    else:
        result['signal'] = '无信号（持仓观望）⏳'
    
    result['signals'] = signals
    return result


def print_result(result):
    """打印监控结果"""
    if 'error' in result:
        print(f"  ❌ {result['error']}\n")
        return
    
    pair = result['pair']
    name = result['name']
    signal = result['signal']
    
    print(f"\n{'='*55}")
    print(f"  {name} ({pair})")
    print(f"{'='*55}")
    print(f"  {result['sym_a']}: ¥{result['price_a']:<10.2f}  {result['sym_b']}: ¥{result['price_b']:.2f}")
    print(f"  价差: {result['current_spr']:>+8.2f}  |  均值: {result['spr_mean']:>+8.2f}  |  Z值: {result['z_score']:>+5.2f}σ")
    print(f"  30日价差: {result['spr_min']} ~ {result['spr_max']}  |  趋势: {result['spr_trend']}")
    print(f"  {'─'*55}")
    
    # 信号状态用颜色
    if '🟢' in signal:
        status = '🔔 开仓信号!'
    elif '✅' in signal:
        status = '📗 平仓信号'
    elif '⚠️' in signal:
        status = '🔴 止损警告!'
    else:
        status = '⏳ 观望'
    
    print(f"  状态: {status}")
    print(f"  建议: {signal}")
    
    if result['signals']:
        for sig in result['signals']:
            print(f"    入场: {sig['direction']} @价差{sig['entry_spr']} (z={sig['z_score']})")
            print(f"    原因: {sig['reason']}")
    
    print()


def monitor_all(use_live=True, notify=False):
    """监控所有品种对"""
    print(f"\n{'='*55}")
    print(f"  📊 期货套利监控器")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")
    
    has_signals = []
    
    for key in ['RB-HC', 'SC-MA', 'Y-P']:
        cfg = PAIR_CONFIG[key]
        result = check_pair(key, cfg, use_live=use_live)
        print_result(result)
        
        if not ('error' in result) and result['signals']:
            has_signals.append(result)
    
    # 通知
    if notify and has_signals:
        msg_lines = ['🔔 套利开仓信号!']
        for r in has_signals:
            for s in r['signals']:
                msg_lines.append(f"  {r['name']}: {s['direction']} @{s['entry_spr']} (z={s['z_score']})")
        msg = '\n'.join(msg_lines)
        
        # try iOS通知
        import subprocess
        try:
            subprocess.run(['apple-open', 'minis://settings/notifications'], 
                         capture_output=True, timeout=3)
        except:
            pass
        
        print("\n" + "="*55)
        print("  🔔 有开仓信号!")
        for r in has_signals:
            print(f"    {r['name']}: {r['signal']}")
        print("="*55)
    
    return has_signals


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='期货套利实时监控')
    p.add_argument('--pair', default=None, help='指定品种对: RB-HC, SC-MA, Y-P')
    p.add_argument('--notify', action='store_true', help='有信号时推送通知')
    p.add_argument('--local', action='store_true', help='用本地缓存数据(不联网)')
    args = p.parse_args()
    
    use_live = not args.local
    
    if args.pair:
        if args.pair not in PAIR_CONFIG:
            print(f"未知品种对: {args.pair}")
            sys.exit(1)
        cfg = PAIR_CONFIG[args.pair]
        result = check_pair(args.pair, cfg, use_live=use_live)
        print_result(result)
    else:
        monitor_all(use_live=use_live, notify=args.notify)
