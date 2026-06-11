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

print(Backtest(call, freq='daily').positions.head())    