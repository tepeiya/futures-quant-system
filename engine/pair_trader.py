"""
配对交易引擎 - 跨品种套利实盘信号生成与交易执行

基于Z-Score均值回归策略，监控价差偏离，生成开仓/平仓/止损指令。
支持手动确认模式和全自动模式。

使用方式:
  python3 -m engine.pair_trader --pair RB-HC --mode signal   查看信号
  python3 -m engine.pair_trader --pair RB-HC --mode trade    执行交易
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from datetime import datetime
from urllib.request import urlopen
import warnings
warnings.filterwarnings('ignore')


# ========== 品种对配置 ==========
PAIR_CONFIG = {
    'RB-HC': {
        'name': '螺纹钢-热卷',
        'a': 'RB', 'b': 'HC',
        'ratio': 1.0,
        'contract_mult_a': 10,    # 螺纹钢10吨/手
        'contract_mult_b': 10,    # 热卷10吨/手
        'margin_pct': 0.10,       # 保证金比例10%
        'commission': 0.0003,     # 手续费万三
        'ze': 1.5, 'lb': 40, 'sl': 3.0, 'zx': 0.2,
    },
    'SC-MA': {
        'name': '原油-甲醇',
        'a': 'SC', 'b': 'MA',
        'ratio': 1.0,
        'contract_mult_a': 1000,  # 原油1000桶/手
        'contract_mult_b': 10,    # 甲醇10吨/手
        'margin_pct': 0.12,
        'commission': 0.0003,
        'ze': 1.5, 'lb': 50, 'sl': 2.5, 'zx': 0.2,
    },
    'Y-P': {
        'name': '豆油-棕榈油',
        'a': 'Y', 'b': 'P',
        'ratio': 1.0,
        'contract_mult_a': 10,
        'contract_mult_b': 10,
        'margin_pct': 0.10,
        'commission': 0.0003,
        'ze': 1.5, 'lb': 50, 'sl': 3.0, 'zx': 0.2,
    },
}

# 信号文件目录（持久化持仓状态）
STATE_DIR = '/var/minis/shared/futures-grid/state'
os.makedirs(STATE_DIR, exist_ok=True)


def fetch_data(symbol):
    """从新浪获取数据并返回收盘价列表"""
    url = f'https://stock.finance.sina.com.cn/futures/api/jsonp.php//InnerFuturesNewService.getDailyKLine?symbol={symbol}0'
    resp = urlopen(url, timeout=8)
    raw = resp.read().decode('utf-8')
    data = json.loads(raw.split('(', 1)[1].rsplit(')', 1)[0])
    return [float(r['c']) for r in data]


def calc_spread_zscore(pa, pb, lb, ratio=1.0):
    """计算价差和Z-Score"""
    n = min(len(pa), len(pb))
    spr = np.array(pa[-n:]) * ratio - np.array(pb[-n:])
    
    if len(spr) < lb + 1:
        return None, None, None
    
    hist = spr[-(lb+1):-1]
    mean = np.mean(hist)
    std = np.std(hist)
    z = (spr[-1] - mean) / max(std, 1)
    
    return spr[-1], z, {'spr': spr.tolist(), 'mean': mean, 'std': std, 'z': z}


def load_position(pair_key):
    """加载当前持仓状态"""
    path = f'{STATE_DIR}/{pair_key}.json'
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_position(pair_key, position):
    """保存持仓状态"""
    path = f'{STATE_DIR}/{pair_key}.json'
    with open(path, 'w') as f:
        json.dump(position, f, indent=2)
    print(f"  💾 持仓已保存: {path}")


def clear_position(pair_key):
    """清除持仓"""
    path = f'{STATE_DIR}/{pair_key}.json'
    if os.path.exists(path):
        os.remove(path)
        print(f"  🗑️ 持仓已清除")


def analyze(pair_key, cfg):
    """分析品种对状态"""
    pa = fetch_data(cfg['a'])
    pb = fetch_data(cfg['b'])
    
    if not pa or not pb:
        return None
    
    spr, z, detail = calc_spread_zscore(pa, pb, cfg['lb'], cfg['ratio'])
    
    if spr is None:
        return None
    
    n = min(len(pa), len(pb))
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'pair': pair_key,
        'name': cfg['name'],
        'a': cfg['a'],
        'b': cfg['b'],
        'price_a': pa[-1],
        'price_b': pb[-1],
        'spread': round(spr, 2),
        'z_score': round(z, 2),
        'lookback': cfg['lb'],
        'data_days': n,
    }
    
    # 加载持仓
    position = load_position(pair_key)
    result['position'] = position
    
    # 信号逻辑
    has_position = position is not None
    entry_z = cfg['ze']
    exit_z = cfg['zx']
    stop_z = cfg['sl']
    
    signals = []
    
    if has_position:
        pos_side = position['direction']
        
        # 止损检查
        if abs(z) >= stop_z:
            signals.append({
                'type': 'STOP_LOSS',
                'action': '平仓止损',
                'direction': pos_side,
                'reason': f'Z值已达{abs(z):.2f}σ，超过止损阈值{stop_z}σ',
                'z_score': round(z, 2),
                'pnl_estimate': round(position.get('pnl', 0), 2),
            })
        
        # 回归平仓
        elif abs(z) <= exit_z:
            signals.append({
                'type': 'EXIT',
                'action': '平仓离场',
                'direction': pos_side,
                'reason': f'Z值回归至{abs(z):.2f}σ，低于平仓阈值{exit_z}σ',
                'z_score': round(z, 2),
                'pnl_estimate': round(position.get('pnl', 0), 2),
            })
        
        # 反向信号
        elif (pos_side == 'SHORT' and z < -entry_z) or (pos_side == 'LONG' and z > entry_z):
            new_side = 'LONG' if pos_side == 'SHORT' else 'SHORT'
            signals.append({
                'type': 'REVERSE',
                'action': f'反向开仓({new_side})',
                'direction': new_side,
                'reason': f'出现反向信号，平{pos_side}开{new_side}',
                'z_score': round(z, 2),
                'entry_spread': round(spr, 2),
            })
        
        else:
            signals.append({
                'type': 'HOLD',
                'action': '持有',
                'direction': pos_side,
                'reason': f'当前Z值{abs(z):.2f}σ，在持有区间内',
                'z_score': round(z, 2),
            })
    else:
        # 无持仓，检查开仓信号
        if z > entry_z:
            signals.append({
                'type': 'ENTER',
                'action': f'做空价差',
                'direction': 'SHORT',
                'reason': f'价差过高，Z={z:.2f}σ > {entry_z}σ',
                'entry_spread': round(spr, 2),
                'z_score': round(z, 2),
            })
        elif z < -entry_z:
            signals.append({
                'type': 'ENTER',
                'action': f'做多价差',
                'direction': 'LONG',
                'reason': f'价差过低，Z={abs(z):.2f}σ > {entry_z}σ',
                'entry_spread': round(spr, 2),
                'z_score': round(z, 2),
            })
        else:
            signals.append({
                'type': 'WAIT',
                'action': '等待',
                'reason': f'当前Z值{abs(z):.2f}σ，未达开仓阈值{entry_z}σ',
                'z_score': round(z, 2),
            })
    
    result['signals'] = signals
    return result


def execute_trade(pair_key, cfg, dry_run=True):
    """执行交易（dry_run=True为模拟模式）"""
    result = analyze(pair_key, cfg)
    if not result:
        print(f"  ❌ 分析失败"); return
    
    has_position = result['position'] is not None
    signal = result['signals'][0] if result['signals'] else None
    
    if not signal:
        print(f"  ❌ 无信号"); return
    
    print(f"\n  📊 交易决策 - {result['name']}")
    print(f"  {'-'*35}")
    print(f"  当前价差: {result['spread']:.2f}  |  Z值: {result['z_score']:+.2f}σ")
    print(f"  {result['a']}: ¥{result['price_a']:.2f}  |  {result['b']}: ¥{result['price_b']:.2f}")
    print(f"  持仓: {'有' if has_position else '无'}")
    print(f"  信号: {signal['type']} - {signal['action']}")
    print(f"  原因: {signal['reason']}")
    
    if signal['type'] in ('ENTER', 'REVERSE'):
        # 计算开仓手数和保证金
        vol = 5  # 基础手数
        margin_a = result['price_a'] * cfg['contract_mult_a'] * vol * cfg['margin_pct']
        margin_b = result['price_b'] * cfg['contract_mult_b'] * vol * cfg['margin_pct']
        total_margin = margin_a + margin_b
        
        new_position = {
            'direction': signal['direction'],
            'entry_spread': signal.get('entry_spread', result['spread']),
            'entry_z': signal['z_score'],
            'volume': vol,
            'entry_time': result['timestamp'],
            'entry_price_a': result['price_a'],
            'entry_price_b': result['price_b'],
            'margin_used': round(total_margin, 2),
        }
        
        if dry_run:
            print(f"  💡 模拟交易:")
            print(f"     操作: {'买' if signal['direction']=='LONG' else '卖'}{result['sym_a']} "
                  f"/ {'卖' if signal['direction']=='LONG' else '买'}{result['sym_b']}")
            print(f"     手数: {vol}手")
            print(f"     占用保证金: ¥{total_margin:,.2f}")
        else:
            save_position(pair_key, new_position)
            print(f"  ✅ 实盘开仓完成")
    
    elif signal['type'] in ('EXIT', 'STOP_LOSS'):
        position = result['position']
        if position:
            # 计算盈亏
            spr_diff = result['spread'] - position['entry_spread']
            if position['direction'] == 'SHORT':
                spr_diff = -spr_diff
            pnl = spr_diff * position['volume'] * cfg['contract_mult_a']
            
            if dry_run:
                print(f"  💡 模拟平仓:")
                print(f"     入场价差: {position['entry_spread']:.2f} | 当前价差: {result['spread']:.2f}")
                print(f"     预估盈亏: ¥{pnl:+,.2f}")
            else:
                clear_position(pair_key)
                print(f"  ✅ 实盘平仓完成, 盈亏: ¥{pnl:+,.2f}")
    
    elif signal['type'] == 'HOLD':
        position = result['position']
        if position:
            spr_diff = result['spread'] - position['entry_spread']
            if position['direction'] == 'SHORT':
                spr_diff = -spr_diff
            pnl = spr_diff * position['volume'] * cfg['contract_mult_a']
            print(f"  📈 当前浮动盈亏: ¥{pnl:+,.2f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='配对交易引擎')
    p.add_argument('--pair', default='RB-HC', help='品种对')
    p.add_argument('--mode', default='signal', choices=['signal', 'trade', 'settle'])
    p.add_argument('--execute', action='store_true', help='实盘执行（默认模拟）')
    args = p.parse_args()
    
    cfg = PAIR_CONFIG.get(args.pair)
    if not cfg:
        print(f"未知品种对: {args.pair}"); sys.exit(1)
    
    if args.mode == 'signal':
        result = analyze(args.pair, cfg)
        if result:
            print(f"\n{'='*45}")
            print(f"  {result['name']} ({args.pair})")
            print(f"  时间: {result['timestamp'][:19]}")
            print(f"{'='*45}")
            print(f"  {result['a']}: ¥{result['price_a']:.2f}")
            print(f"  {result['b']}: ¥{result['price_b']:.2f}")
            print(f"  价差: {result['spread']:>+8.2f}  |  Z值: {result['z_score']:>+5.2f}σ")
            print(f"  {'─'*45}")
            for s in result['signals']:
                tag = {'ENTER':'🟢','EXIT':'✅','STOP_LOSS':'🛑','REVERSE':'🔄','HOLD':'⏳','WAIT':'⏳'}.get(s['type'],'📌')
                print(f"  {tag} {s['action']}")
                print(f"     {s['reason']}")
                if 'entry_spread' in s:
                    print(f"     入场价差: {s['entry_spread']}")
                if 'pnl_estimate' in s:
                    print(f"     预估盈亏: ¥{s['pnl_estimate']:+,.2f}")
            print()
    
    elif args.mode == 'trade':
        dry_run = not args.execute
        execute_trade(args.pair, cfg, dry_run=dry_run)
    
    elif args.mode == 'settle':
        position = load_position(args.pair)
        if position:
            print(f"\n  {cfg['name']} 当前持仓:")
            for k,v in position.items():
                print(f"    {k}: {v}")
            print()
        else:
            print(f"\n  {cfg['name']} 无持仓\n")
