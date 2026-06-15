#!/usr/bin/env python3
"""
期货量化系统 v4
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest.arbitrage import run_pair, analyze_all, PAIRS
from monitor import monitor_all


def arb(args):
    if args.pair == 'all':
        print(f"\n{'='*60}")
        print(f"  多品种套利组合")
        print(f"{'='*60}")
        for pk in ['RB-HC', 'SC-MA', 'Y-P']:
            run_pair(pk, balance=args.balance, vol=args.vol,
                    z_entry=args.z_entry, z_exit=args.z_exit,
                    lookback=args.lookback, sl_z=args.sl_z)
    else:
        run_pair(args.pair, balance=args.balance, vol=args.vol,
                z_entry=args.z_entry, z_exit=args.z_exit,
                lookback=args.lookback, sl_z=args.sl_z)


def lst(args):
    analyze_all()


def main():
    p = argparse.ArgumentParser(description='期货量化系统')
    sub = p.add_subparsers(dest='cmd')
    
    ap = sub.add_parser('arb', help='跨品种套利')
    ap.add_argument('--pair', default='RB-HC', help='品种对: RB-HC, SC-MA, Y-P, CU-ZN, I-RB, AU-AG, all')
    ap.add_argument('--balance', type=float, default=100000)
    ap.add_argument('--vol', type=float, default=10)
    ap.add_argument('--z-entry', type=float, default=1.8)
    ap.add_argument('--z-exit', type=float, default=0.2)
    ap.add_argument('--lookback', type=int, default=50)
    ap.add_argument('--sl-z', type=float, default=3.0)
    
    sub.add_parser('list', help='分析所有品种对')
    mp = sub.add_parser('monitor', help='实时行情监控')
    mp.add_argument('--pair', default=None, help='品种对: RB-HC, SC-MA, Y-P')
    mp.add_argument('--local', action='store_true', help='用本地缓存(不联网)')
    mp.add_argument('--notify', action='store_true', help='有信号时通知')
    
    tp = sub.add_parser('trade', help='配对交易引擎')
    tp.add_argument('--pair', default='RB-HC')
    tp.add_argument('--mode', default='signal', choices=['signal','trade','settle'])
    tp.add_argument('--execute', action='store_true', help='实盘执行')
    
    args = p.parse_args()
    if args.cmd == 'arb': arb(args)
    elif args.cmd == 'list': lst(args)
    elif args.cmd == 'monitor':
        if args.pair:
            from monitor import check_pair, PAIR_CONFIG, print_result
            cfg = PAIR_CONFIG[args.pair]
            r = check_pair(args.pair, cfg, use_live=not args.local)
            print_result(r)
        else:
            monitor_all(use_live=not args.local, notify=args.notify)
    
    elif args.cmd == 'trade':
        from engine.pair_trader import PAIR_CONFIG, analyze, execute_trade, load_position
        cfg = PAIR_CONFIG[args.pair]
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
                    print(f"  {tag} {s['action']}: {s['reason']}")
                print()
        elif args.mode == 'trade':
            execute_trade(args.pair, cfg, dry_run=not args.execute)
        elif args.mode == 'settle':
            pos = load_position(args.pair)
            if pos: print(f"\n  {cfg['name']} 持仓:\n    " + "\n    ".join(f"{k}: {v}" for k,v in pos.items()) + "\n")
            else: print(f"\n  {cfg['name']} 无持仓\n")
    else: p.print_help()


if __name__ == '__main__':
    main()
