import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env
_env = ROOT / ".env"
if _env.exists():
    with open(_env) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# Load config.yaml
_cfg = {"source": "mock", "start_date": "2017-01-01", "end_date": "2026-03-31"}
_cfg_path = ROOT / "config.yaml"
if _cfg_path.exists():
    try:
        import yaml
        with open(_cfg_path) as _f:
            _file_cfg = yaml.safe_load(_f) or {}
        _cfg.update({k: v for k, v in _file_cfg.items() if v is not None})
    except ImportError:
        pass

from src.data_loader import load_data
from src.features import calculate_returns, rolling_momentum, volatility_signal

data = load_data(source=_cfg["source"], start_date=_cfg["start_date"], end_date=_cfg["end_date"])
prices = data['prices']
macro = data['macro']
returns = calculate_returns(prices)
momentum = rolling_momentum(prices)
vol = volatility_signal(returns)

latest_fundamentals = data['fundamentals'].sort_values(['ticker', 'date']).groupby('ticker').tail(1).set_index('ticker')
fundamentals_df = latest_fundamentals.reset_index()
universe_df = data['universe'][['ticker', 'liquidity_score', 'market_cap_mxn', 'usd_exposure']]

daily_macro = (
    macro.set_index('date')
         .reindex(prices.index, method='ffill')
         .rename_axis('date')
         .reset_index()
)
print('daily_macro cols', daily_macro.columns.tolist())
print(daily_macro.head(2))

price_stack = prices.stack().rename('price').reset_index().rename(columns={'level_0': 'date', 'level_1': 'ticker'})
print('price_stack cols', price_stack.columns.tolist())
print(price_stack.head(2))

momentum_stack = momentum.stack().rename('momentum').reset_index().rename(columns={'level_0': 'date', 'level_1': 'ticker'})
vol_stack = vol.stack().rename('volatility').reset_index().rename(columns={'level_0': 'date', 'level_1': 'ticker'})
merged = price_stack.merge(momentum_stack, on=['date', 'ticker'], how='left').merge(vol_stack, on=['date', 'ticker'], how='left').merge(universe_df, on='ticker', how='left').merge(fundamentals_df, on='ticker', how='left')
print('merged cols before macro', merged.columns.tolist())
print('merged head', merged.head(2))
print('date dtype', merged['date'].dtype)
print('daily_macro date dtype', daily_macro['date'].dtype)
print('daily_macro sample', daily_macro['date'].head(2))
