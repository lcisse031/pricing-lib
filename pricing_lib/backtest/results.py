"""
pricing_lib/backtest/results.py
-----------------------------------------------------------------------------
Conteneur de resultats du backtest.

positions : DataFrame MultiIndex (date, product_id)
    spot          : cours historique du sous-jacent
    T_residual    : maturite residuelle en annees
    realized_vol  : volatilite realisee rolling (decimal)
    mtm           : prix unitaire en euros (sortie du pricer)
    quantity      : nb d'unites achetees = nominal / mtm0
    nominal       : notionnel en euros
    mtm_total     : valeur reelle de la position = mtm * quantity
    delta         : d(price)/d(S)
    gamma         : d2(price)/d(S2)
    vega          : d(price)/d(sigma)  par unite de sigma decimal
    theta         : d(price)/d(T) annualise
    pnl           : mtm_total(t) - mtm_total(t-1)
    pnl_delta     : delta(t-1) * dS * quantity
    pnl_gamma     : 0.5 * gamma(t-1) * dS**2 * quantity
    pnl_vega      : vega(t-1) * dvol * quantity
    pnl_theta     : theta(t-1) * dT * quantity
    pnl_unexplained : residuel = pnl - sum(composants)

    Grecques NaN en mode MC. Premiere date de chaque position = NaN (pas de J-1).

portfolio : DataFrame indexe par date
    portfolio_value : sum(mtm_total)
    delta...theta   : sum(greek * quantity)
    pnl_*           : sum(pnl_*) sur toutes les positions actives
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResults:
    positions: pd.DataFrame   # MultiIndex (date, product_id)
    portfolio: pd.DataFrame   # index = date

    def __repr__(self) -> str:
        n_dates = self.portfolio.shape[0]
        n_pos   = self.positions.index.get_level_values("product_id").nunique()
        pv_min  = self.portfolio["portfolio_value"].min()
        pv_max  = self.portfolio["portfolio_value"].max()
        return (
            f"BacktestResults("
            f"{n_pos} position(s), "
            f"{n_dates} dates, "
            f"portfolio_value [{pv_min:.2f} ; {pv_max:.2f}])"
        )
