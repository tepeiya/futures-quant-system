# 数据加载模块 - 使用 akshare 获取期货数据

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FuturesDataLoader:
    """期货数据加载器"""
    
    def __init__(self):
        self._cache = {}
    
    def list_futures(self):
        """列出国内期货品种"""
        try:
            import akshare as ak
            df = ak.futures_contract_info()
            # 提取主要品种代码
            main_contracts = df[df['symbol'].str.len() <= 6]
            return main_contracts[['symbol', 'name']].drop_duplicates().to_dict('records')
        except Exception as e:
            print(f"获取期货列表失败: {e}")
            # 返回常见主力品种
            return [
                {'symbol': 'AU', 'name': '沪黄金'},
                {'symbol': 'AG', 'name': '沪白银'},
                {'symbol': 'RB', 'name': '螺纹钢'},
                {'symbol': 'CU', 'name': '沪铜'},
                {'symbol': 'AL', 'name': '沪铝'},
                {'symbol': 'ZN', 'name': '沪锌'},
                {'symbol': 'NI', 'name': '沪镍'},
                {'symbol': 'SN', 'name': '沪锡'},
                {'symbol': 'PB', 'name': '沪铅'},
                {'symbol': 'I',  'name': '铁矿石'},
                {'symbol': 'HC', 'name': '热卷'},
                {'symbol': 'FG', 'name': '玻璃'},
                {'symbol': 'MA', 'name': '甲醇'},
                {'symbol': 'TA', 'name': 'PTA'},
                {'symbol': 'SC', 'name': '原油'},
                {'symbol': 'FU', 'name': '燃料油'},
                {'symbol': 'BU', 'name': '沥青'},
                {'symbol': 'RU', 'name': '橡胶'},
                {'symbol': 'P',  'name': '棕榈油'},
                {'symbol': 'Y',  'name': '豆油'},
                {'symbol': 'M',  'name': '豆粕'},
                {'symbol': 'CF', 'name': '棉花'},
                {'symbol': 'SR', 'name': '白糖'},
                {'symbol': 'JM', 'name': '焦煤'},
                {'symbol': 'J',  'name': '焦炭'},
            ]
    
    def get_realtime_quote(self, symbol: str):
        """获取实时行情"""
        try:
            import akshare as ak
            df = ak.futures_spot_price(symbol=symbol)
            return df
        except:
            return None
    
    def get_daily_data(self, symbol: str, start_date: str = None, end_date: str = None,
                       months: int = 12):
        """获取历史日线数据
        
        Args:
            symbol: 品种代码, 如 'AU', 'RB'
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            months: 回溯月数 (start_date未提供时使用)
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume, open_interest
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start = datetime.now() - timedelta(days=months*30)
            start_date = start.strftime('%Y%m%d')
        
        cache_key = f"{symbol}_{start_date}_{end_date}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            import akshare as ak
            
            # 尝试获取日线数据 - 新浪主力连续
            df = ak.futures_main_sina(symbol=f'{symbol}0')
            if df is not None and not df.empty:
                # 处理日期
                df = df.rename(columns={'日期': 'date', '开盘价': 'open', '收盘价': 'close', 
                                        '最高价': 'high', '最低价': 'low', '成交量': 'volume',
                                        '持仓量': 'open_interest'})
                df['date'] = pd.to_datetime(df['date'])
                
                # 标准化列名
                col_map = {
                    '开盘': 'open', '开盘价': 'open', '开盘_o': 'open',
                    '收盘': 'close', '收盘价': 'close', '收盘_c': 'close',
                    '最高': 'high', '最高价': 'high', '最高_h': 'high',
                    '最低': 'low', '最低价': 'low', '最低_l': 'low',
                    '成交量': 'volume',
                    '持仓量': 'open_interest', 'open_interest_o': 'open_interest', 
                    'open_interest_oi': 'open_interest', 'oi': 'open_interest',
                    '日期': 'date',
                }
                df.rename(columns=col_map, inplace=True)
                
                # 只保留需要的列
                needed = ['date', 'open', 'high', 'low', 'close', 'volume']
                if 'open_interest' in df.columns:
                    needed.append('open_interest')
                
                available = [c for c in needed if c in df.columns]
                df = df[available]
                
                df = df.sort_values('date').reset_index(drop=True)
                df = df[(df['date'] >= pd.Timestamp(start_date)) & 
                        (df['date'] <= pd.Timestamp(end_date))]
                
                self._cache[cache_key] = df
                return df
        
        except Exception as e:
            print(f"拉取 {symbol} 数据失败: {e}")
        
        # fallback: 生成模拟数据用于测试
        print(f"⚠️ 使用模拟数据用于 {symbol}")
        return self._generate_sample_data(symbol, start_date, end_date)
    
    def _generate_sample_data(self, symbol: str, start_date: str, end_date: str):
        """生成模拟期货数据（用于架构测试）"""
        np.random.seed(hash(symbol) % 2**31)
        
        # 不同品种的基准价格
        base_prices = {
            'AU': 580, 'AG': 7500, 'RB': 3600, 'CU': 75000, 
            'I': 800, 'SC': 550, 'MA': 2400, 'P': 7800,
        }
        base = base_prices.get(symbol, 5000)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(dates)
        
        # 随机游走 + 趋势
        returns = np.random.randn(n) * 0.01 + 0.0002
        price = base * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'open': price * (1 + np.random.randn(n) * 0.003),
            'high': price * (1 + abs(np.random.randn(n)) * 0.006),
            'low': price * (1 - abs(np.random.randn(n)) * 0.006),
            'close': price,
            'volume': np.random.randint(10000, 500000, n),
            'open_interest': np.random.randint(50000, 500000, n),
        })
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].round(2)
        
        return df
    
    def get_tick_size(self, symbol: str) -> float:
        """获取最小变动价位"""
        ticks = {
            'AU': 0.02, 'AG': 1, 'RB': 1, 'CU': 10, 'I': 0.5,
            'SC': 0.1, 'MA': 1, 'P': 2, 'TA': 2, 'BU': 2,
            'RU': 5, 'M': 1, 'Y': 2, 'CF': 5, 'SR': 1,
        }
        return ticks.get(symbol, 1)
    
    def get_contract_multiplier(self, symbol: str) -> int:
        """获取合约乘数"""
        multipliers = {
            'AU': 1000, 'AG': 15, 'RB': 10, 'CU': 5, 'I': 100,
            'SC': 1000, 'MA': 10, 'P': 10, 'TA': 5, 'BU': 10,
            'RU': 10, 'M': 10, 'Y': 10, 'CF': 5, 'SR': 10,
        }
        return multipliers.get(symbol, 10)


if __name__ == '__main__':
    loader = FuturesDataLoader()
    # 测试
    print("当前可用品种:")
    for s in loader.list_futures()[:10]:
        print(f"  {s['symbol']}: {s['name']}")
    
    print("\n获取沪黄金日线数据:")
    df = loader.get_daily_data('AU', months=3)
    print(df.tail())
