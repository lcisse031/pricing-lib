# options-pricing-lib

A Python library I built to price structured products and derivatives on CAC 40 stocks, from simple vanilla options to autocall products like Phoenix and Athena.

It handles pricing (Black-Scholes + Monte Carlo), Greeks, stress tests, and a full backtest engine that breaks down P&L day by day into delta, gamma, vega, theta, and unexplained.

Market data (spot, rates, dividends, implied vol surface) is pulled automatically from `yfinance`, no manual inputs needed.

---

## What it can price

Vanilla options, warrants, reverse convertibles, bonus certificates, twin-win, airbag, sprint, outperformance, discount, capital-protected notes, Phoenix and Athena autocalls.

Two pricing modes everywhere: **analytical** (Black-Scholes closed-form) or **Monte Carlo**.

---

## Getting started

```bash
git clone https://github.com/lcisse031/pricing-lib.git
cd pricing-lib
pip install -e .
```

---

## Pricing a product

```python
from pricing_lib.api import Call, Phoenix, Greeks

# A 3-month ATM call on Société Générale
result = Call('AL', 'GLE', spot=22.50, strike=22.50, start_date='02/01/2025', maturity='3M')
# PricingResult(price=1.84, delta=0.52, gamma=0.08, vega=3.21, theta=-4.15)

# A 5-year Phoenix on Total, 20k paths
result = Phoenix(
    'MC', 'TTE', n_paths=20_000, start_date='02/01/2025', spot=60.0,
    barrier_coupon=80, barrier_recall=100, capital_barrier=70,
    freq_months=3, maturity='60M', kg='no'
)

# Greeks + scenario grid
greeks = Greeks('AL', 'GLE', spot=22.50, strike=22.50, start_date='02/01/2025', maturity='3M')
```

---

## Running a backtest

```python
from pricing_lib.backtest_api import Backtest, PositionSpec

spec = PositionSpec(
    product_id   = 'CALL_GLE_ATM',
    ticker       = 'GLE',
    product_type = 'call',
    spot         = 22.50,
    start_date   = '02/01/2023',
    maturity     = '3M',
    nominal      = 100_000,   # how much you invested in the product
    mode         = 'AL',
    params       = {'strike': 22.50},
)

results = Backtest(spec, freq='daily')
```

The output gives you, for each day:

- `mtm_total` : real EUR value of the position
- `pnl` : daily P&L vs. previous day
- `pnl_delta`, `pnl_gamma`, `pnl_vega`, `pnl_theta` : what drove the move
- `pnl_unexplained` : what the Greeks didn't capture

The quantity convention is `nominal / mtm0`: if you invest 100k in a product worth 95 on day 0, you hold 1052.6 units. Day-0 MTM equals exactly 100k by construction.

For a portfolio backtest, just pass a list of specs:

```python
results = Backtest([spec_call, spec_phoenix], freq='weekly')
```

---

## Structure

```
pricing_lib/
├── api.py              # entry point: Call, Put, Phoenix, Greeks, Stress...
├── backtest_api.py     # entry point: Backtest, PositionSpec
├── backtest/           # engine, P&L attribution, market loading
├── pricers/            # analytical, MC, autocalls, Greeks
├── market_data/        # snapshot, vol surface, rates, dividends
├── models/             # Heston
└── risk/               # sensitivities, scenario tools
```

---

## License

MIT
