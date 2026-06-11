"""
pricing_lib/risk/sensitivities.py
─────────────────────────────────────────────────────────────────────────────
Analyse de scénarios et de sensibilités.

ScenarioAnalyzer — grille de prix (spot × vol)
StressTest       — scénarios extrêmes prédéfinis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Dict

import numpy as np

from ..market_data.market_snapshot import MarketSnapshot
from .greeks import _BumpedMarket


# ─── ScenarioAnalyzer ────────────────────────────────────────────────────────

class ScenarioAnalyzer:
    """
    Grille de prix en fonction de (spot, vol).

    Paramètres
    ----------
    pricer  : objet avec .price(product, market) → obj.price
    product : produit
    market  : MarketSnapshot de référence
    model   : modèle optionnel

    Exemple
    -------
    >>> grid = ScenarioAnalyzer(pricer, product, market)
    >>> df = grid.spot_vol_grid(
    ...     spot_shocks=[-0.20, -0.10, 0, +0.10, +0.20],
    ...     vol_shocks=[-0.05, 0, +0.05],
    ... )
    >>> print(df)
    """

    def __init__(self, pricer, product, market: MarketSnapshot, model=None) -> None:
        self._pricer  = pricer
        self._product = product
        self._market  = market
        self._model   = model

    def spot_vol_grid(
        self,
        spot_shocks: List[float],   # ex: [-0.20, -0.10, 0, 0.10, 0.20]
        vol_shocks:  List[float],   # ex: [-0.05, 0, 0.05]
    ) -> "pd.DataFrame":
        """
        Matrice de prix pour chaque combinaison (spot_shock, vol_shock).

        spot_shock : variation relative du spot (ex: -0.20 = -20%)
        vol_shock  : variation absolue de la vol (ex: +0.05 = +5 points de vol)

        Retourne un DataFrame indexé sur vol_shock, colonnes = spot_shock.
        """
        import pandas as pd

        S0 = self._market.spot
        rows: Dict[str, Dict[str, float]] = {}

        for dv in vol_shocks:
            row: Dict[str, float] = {}
            for ds in spot_shocks:
                m = _BumpedMarket(self._market, spot=S0 * (1 + ds), sigma_bump=dv)
                price = self._price(m)
                row[f"{ds:+.0%}"] = round(price, 4)
            rows[f"vol {dv:+.0%}"] = row

        df = pd.DataFrame(rows).T
        df.index.name   = "Vol shock"
        df.columns.name = "Spot shock"
        return df

    def _price(self, market) -> float:
        if self._model is not None:
            return self._pricer.price(self._product, market, self._model).price
        return self._pricer.price(self._product, market).price


# ─── StressTest ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StressScenario:
    name:      str
    spot_pct:  float   # choc relatif sur le spot  (ex: -0.30)
    vol_abs:   float   # choc absolu sur la vol     (ex: +0.15)
    rate_abs:  float   # choc absolu sur le taux    (ex: +0.01)


# Scénarios classiques desk quant
STANDARD_STRESSES: List[StressScenario] = [
    StressScenario("Base",              0.00,  0.00,  0.00),
    StressScenario("Crash -20%",       -0.20, +0.15, -0.01),
    StressScenario("Crash -30%",       -0.30, +0.25, -0.02),
    StressScenario("Crash -40%",       -0.40, +0.35, -0.03),
    StressScenario("Rally +20%",       +0.20, -0.05,  0.00),
    StressScenario("Vol spike +10pt",   0.00, +0.10,  0.00),
    StressScenario("Vol crush -10pt",   0.00, -0.10,  0.00),
    StressScenario("Rate +100bp",       0.00,  0.00, +0.01),
    StressScenario("Rate -100bp",       0.00,  0.00, -0.01),
    StressScenario("2008 crisis",      -0.40, +0.40, -0.02),
    StressScenario("COVID shock",      -0.35, +0.50, -0.03),
]


class StressTest:
    """
    Stress test d'un produit sur des scénarios prédéfinis ou personnalisés.

    Exemple
    -------
    >>> st = StressTest(pricer, product, market)
    >>> df = st.run()
    >>> print(df)
    """

    def __init__(
        self,
        pricer,
        product,
        market:    MarketSnapshot,
        model      = None,
        scenarios: Optional[List[StressScenario]] = None,
    ) -> None:
        self._pricer    = pricer
        self._product   = product
        self._market    = market
        self._model     = model
        self._scenarios = scenarios or STANDARD_STRESSES

    def run(self) -> "pd.DataFrame":
        """
        Exécute tous les scénarios et retourne un DataFrame.

        Colonnes : scenario | price | pnl | pnl_pct
        """
        import pandas as pd
        import math

        base_price = self._price(self._market)
        rows = []

        for sc in self._scenarios:
            bumped = _BumpedMarket(
                self._market,
                spot      = self._market.spot * (1 + sc.spot_pct),
                sigma_bump= sc.vol_abs,
                rate_bump = sc.rate_abs,
            )
            price = self._price(bumped)
            pnl   = price - base_price
            rows.append({
                "Scenario":  sc.name,
                "Spot choc": f"{sc.spot_pct:+.0%}",
                "Vol choc":  f"{sc.vol_abs:+.0%}",
                "Rate choc": f"{sc.rate_abs*100:+.0f}bp",
                "Price":     round(price, 4),
                "PnL":       round(pnl, 4),
                "PnL %":     f"{pnl/abs(base_price)*100:+.2f}%" if base_price != 0 else "N/A",
            })

        df = pd.DataFrame(rows).set_index("Scenario")
        return df

    def _price(self, market) -> float:
        if self._model is not None:
            return self._pricer.price(self._product, market, self._model).price
        return self._pricer.price(self._product, market).price
