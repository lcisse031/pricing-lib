"""
pricing_lib/backtest_api.py
-----------------------------------------------------------------------------
Interface publique du moteur de backtest.

Usage -- Position unique
------------------------
    from pricing_lib.backtest_api import Backtest, PositionSpec

    call = PositionSpec(
        product_id   = 'CALL_GLE_100',
        ticker       = 'GLE',
        product_type = 'call',
        spot         = 22.50,
        start_date   = '02/01/2023',
        maturity     = '3M',
        nominal      = 100_000,
        mode         = 'AL',
        params       = {'strike': 22.50},
    )

    results = Backtest(call, freq='daily')

    results.positions.head()
    # spot | T_residual | realized_vol | mtm | quantity | nominal | mtm_total
    # delta | gamma | vega | theta
    # pnl | pnl_delta | pnl_gamma | pnl_vega | pnl_theta | pnl_unexplained

    results.portfolio.head()
    # portfolio_value | delta | gamma | vega | theta
    # pnl | pnl_delta | pnl_gamma | pnl_vega | pnl_theta | pnl_unexplained

Usage -- Portefeuille
---------------------
    results = Backtest(
        [call, phoenix],
        start_date  = '02/01/2023',
        end_date    = '31/12/2024',
        freq        = 'weekly',
    )

Parametres
----------
products       : PositionSpec | list[PositionSpec]
start_date     : 'DD/MM/YYYY' | None  (deduit de la plus ancienne position)
end_date       : 'DD/MM/YYYY' | None  (deduit de la maturite la plus longue)
freq           : 'daily' | 'weekly' | 'monthly'           (defaut 'daily')
aggregation    : 'quantity'      greques = sum(greek * qty)  (defaut)
               | 'custom_weights' greques ponderees par weights
weights        : dict {product_id: float}  si aggregation='custom_weights'
compute_greeks : bool                                        (defaut True)
verbose        : bool                                        (defaut True)

Convention quantity (Option B)
------------------------------
Si PositionSpec.quantity est None (defaut), l'engine calcule :
    quantity = nominal / mtm0
ou mtm0 est le prix unitaire du produit a start_date.
=> Au jour 0 : mtm_total = mtm0 * quantity = nominal  (coherent)
=> P&L = mtm_total(t) - mtm_total(t-1)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pricing_lib.backtest.specs import PositionSpec
from pricing_lib.backtest.results import BacktestResults
from pricing_lib.backtest.engine import BacktestEngine

__all__ = ["Backtest", "PositionSpec", "BacktestResults"]


def Backtest(
    products:       Union[PositionSpec, List[PositionSpec]],
    start_date:     Optional[str]            = None,
    end_date:       Optional[str]            = None,
    freq:           str                      = "daily",
    aggregation:    str                      = "quantity",
    weights:        Optional[Dict[str, float]] = None,
    compute_greeks: bool                     = True,
    verbose:        bool                     = True,
) -> BacktestResults:
    """Lance un backtest sur un produit ou un portefeuille."""
    if isinstance(products, PositionSpec):
        products = [products]

    ids = [p.product_id for p in products]
    if len(ids) != len(set(ids)):
        dups = list({pid for pid in ids if ids.count(pid) > 1})
        raise ValueError(f"product_id dupliques : {dups}")

    engine = BacktestEngine(
        positions      = products,
        start_date     = start_date,
        end_date       = end_date,
        freq           = freq,
        aggregation    = aggregation,
        weights        = weights,
        compute_greeks = compute_greeks,
        verbose        = verbose,
    )
    return engine.run()

if __name__ == "__main__":
    '''call = PositionSpec(
        product_id   = 'CALL_GLE_100',
        ticker       = 'GLE',
        product_type = 'call',
        spot         = 20.68,
        start_date   = '02/01/2023',
        maturity     = '3M',
        nominal      = 100_000,
        mode         = 'AL',
        params       = {'strike': 20.685},
    )
    print(Backtest(call, freq='daily').positions.head(100))'''


    # ── Produit 1 : Call vanilla GLE ─────────────────────────────────────────────
    call = PositionSpec(
        product_id   = 'CALL_GLE',
        ticker       = 'GLE',
        product_type = 'call',
        spot         = 20.68,
        start_date   = '02/01/2023',
        maturity     = '6M',
        nominal      = 100_000,
        mode         = 'AL',
        params       = {'strike': 20.68},
    )

    # ── Produit 2 : Put GLE (couverture) ─────────────────────────────────────────
    put = PositionSpec(
        product_id   = 'PUT_GLE',
        ticker       = 'GLE',
        product_type = 'put',
        spot         = 20.68,
        start_date   = '02/01/2023',
        maturity     = '6M',
        nominal      = 50_000,
        mode         = 'AL',
        params       = {'strike': 18.50},   # put OTM à ~90%
    )

    # ── Produit 3 : Phoenix autocall MC ──────────────────────────────────────────
    phoenix = PositionSpec(
        product_id   = 'PHOENIX_GLE',
        ticker       = 'GLE',
        product_type = 'phoenix',
        spot         = 20.68,
        start_date   = '02/01/2023',
        maturity     = '12M',
        nominal      = 200_000,
        mode         = 'MC',
        n_paths      = 5_000,
        seed         = 42,
        params       = {
            'barrier_coupon'  : 0.70,   # coupon servi si spot > 70% S0
            'barrier_recall'  : 1.00,   # rappel anticipé si spot > 100% S0
            'capital_barrier' : 0.60,   # protection capital si spot > 60% S0
            'freq_months'     : 3,      # observation trimestrielle
            'coupon'          : 0.08,   # 8% annuel
            'kg'              :'no',
            'autocall_start'  : 1,
        },
    )

    # ── Backtest portefeuille ─────────────────────────────────────────────────────
    results = Backtest(
        [call, put, phoenix],
        start_date = '02/01/2023',
        end_date   = '30/06/2023',
        freq       = 'daily',
    )

    print(results.portfolio.head(100))

    print(results.positions.head(100))