# options-pricing-lib

A Python library I built to price structured products and derivatives on CAC 40 stocks, from simple vanilla options to autocall products like Phoenix and Athena.

It handles pricing (Black-Scholes, Heston Monte Carlo), Greeks, stress tests, and a full backtest engine that breaks down P&L day by day into delta, gamma, vega, theta and unexplained.

Market data is pulled automatically:

| Data | Source | Fallback if unreachable |
|---|---|---|
| Implied vol surface | Euronext option chains (Brent inversion, PCHIP, SSVI arbitrage-free fit) | last EOD cache in `market_data/.vol_cache/`, then flat 25 % |
| Rates | €STR swap curve (bluegamma.io), bootstrapped | flat 2 % curve (warning printed) |
| Dividends | zonebourse.com (needs `playwright`) | q = 0 (warning printed) |
| Historical spots (backtest only) | `yfinance` | — |

Market data is always **today's**: pricing with a past `start_date` uses today's vol, rates and dividends (a warning is printed).

---

## What it can price

Vanilla options, warrants, reverse convertibles, bonus / reverse bonus, discount, twin-win, airbag, sprint, outperformance, capital-guaranteed notes, Phoenix and Athena autocalls.

Two modes everywhere: **`'AL'`** (closed form / replication) or **`'MC'`** (Monte Carlo: GBM for vanillas, Heston QE calibrated on the surface for exotics).

---

## Getting started

```bash
git clone https://github.com/lcisse031/pricing-lib.git
cd pricing-lib
pip install -e ".[dev]"            # + ".[scraping]" for dividends, ".[plot]" for charts
pytest                             # 75 offline tests, no network needed
```

---

## Pricing a product

Arguments are **positional**: `mode, ticker, [n_paths,] spot, ..., start_date, maturity`.

```python
from pricing_lib.pricer_api import Call, RC, Phoenix, Stress

# 3-month ATM call on Société Générale
res = Call('AL', 'GLE', 22.50, 22.50, '02/01/2026', '3M')
res.price, res.greeks            # greeks: delta, gamma, vega (per vol pt), theta (per day), rho (per bp)

# Same call in Monte Carlo (n_paths comes right after the ticker)
res = Call('MC', 'GLE', 50_000, 22.50, 22.50, '02/01/2026', '3M', seed=1)

# Reverse convertible, quarterly coupons: AL returns the fair coupon (price = 100 %)
res = RC('AL', 'GLE', 22.50, '02/01/2026', '12M', freq='Q')
res.fair_coupon

# 5-year Phoenix on Total, 20k paths — barriers as a FRACTION of spot (≤ 3) or absolute levels
res = Phoenix('AL', 'TTE', 20_000, '02/01/2026', 60.0,
              barrier_coupon=0.80, barrier_recall=1.00, capital_barrier=0.70,
              freq_months=3, maturity='60M', kg='no', compute_greeks=True)
res.fair_coupon                  # coupon PER OBSERVATION PERIOD (quarterly here); repr shows the p.a. equivalent

# Stress test — terms (coupon, participation) are frozen on the base market
Stress('AL', 'GLE', 22.50, '02/01/2026', '12M', product='rc', freq='Q')
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
- `pnl_delta`, `pnl_gamma`, `pnl_vega`, `pnl_vomma`, `pnl_theta` : what drove the move
- `pnl_unexplained` : what the Greeks didn't capture

How the engine treats products:

- **Terms are frozen at inception**, like on a term sheet: RC coupon and cap, certificate participation, autocall coupon (computed fair if not given). They are not re-struck every day.
- **Autocalls** keep their original observation schedule. Past fixings are replayed on historical spots: coupons received and early redemption are added to the position value (total return, no reinvestment).
- The quantity convention is `nominal / mtm0`: if you invest 100k in a product worth 95 on day 0, you hold 1052.6 units. Day-0 MTM equals exactly 100k by construction.
- Historical vol is a flat 21-day realized vol; rates and dividends are today's (one fetch).

For a portfolio backtest, just pass a list of specs:

```python
results = Backtest([spec_call, spec_phoenix], freq='weekly')
```

---

## Structure

```
pricing_lib/
├── pricer_api.py       # entry point: Call, Put, RC, Phoenix, Athena, certificates, Stress, ScenarioGrid
├── backtest_api.py     # entry point: Backtest, PositionSpec
├── backtest/           # engine, P&L attribution, market loading
├── pricers/            # analytical (core/pdts.py), MC, autocalls
├── market_data/        # snapshot, vol surface, SSVI, rates, dividends
├── models/             # Heston (CF pricing, calibration, QE simulation)
└── risk/               # Greeks, stress tests, scenario grids
tests/                  # offline test suite (synthetic markets)
```

---

## Known limitations

- Heston calibration (`differential_evolution`) takes a few seconds per ticker and date; MC backtests are slow.
- The Euronext symbol map (`market_data/vol_surface.py`) has a duplicate `BN4` code for BNP and Danone; check it against Euronext before using those two tickers.
- Autocall mid-life pricing assumes the product is still alive; the backtest engine handles past redemptions.

---

## License

MIT
