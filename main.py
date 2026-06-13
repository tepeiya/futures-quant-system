#!/usr/bin/env python3
"""
期货量化系统 v4
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest.arbitrage import run_pair, analyze_all, PAIRS


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
    
    args = p.parse_args()
    if args.cmd == 'arb': arb(args)
    elif args.cmd == 'list': lst(args)
    else: p.print_help()


if __name__ == '__main__':
    main()
