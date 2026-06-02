from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from .rates import OISCurve
from .dividends import DividendCurve
from .vol_surface import VolSurface


@dataclass
class MarketSnapshot:
    """Données de marché agrégées pour un sous-jacent."""

    ticker:         str
    valuation_date: date
    spot:           float
    ois_curve:      OISCurve
    div_curve:      DividendCurve
    vol_surface:    VolSurface

    @property
    def r(self) -> float:
        """Taux sans risque continu 1 an (OIS)."""
        return self.ois_curve.zero_rate_tau(1.0)

    @property
    def q(self) -> float:
        """Taux de dividende continu."""
        return self.div_curve.q

    def sigma(self, T: float, K: float) -> float:
        """Vol implicite σ(T, K) depuis la surface."""
        return self.vol_surface.vol(T, K)

    def atm_vol(self, T: float) -> float:
        """Vol ATM pour la maturité T."""
        return self.vol_surface.atm_vol(T)

    def df(self, tau: float) -> float:
        """Discount factor pour τ années."""
        return self.ois_curve.df_tau(tau)

    def __repr__(self) -> str:
        return (
            f"MarketSnapshot(ticker={self.ticker!r}, "
            f"date={self.valuation_date}, spot={self.spot}, "
            f"r={self.r:.4%}, q={self.q:.4%})"
        )


@dataclass
class MultiAssetSnapshot:
    """Données de marché pour un panier multi-actifs (worst-of, basket options)."""

    import numpy as np

    assets:         list
    corr_matrix:    "np.ndarray"
    valuation_date: date
    ois_curve:      OISCurve

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    @property
    def spots(self) -> list:
        return [a.spot for a in self.assets]

    @property
    def sigmas_atm(self) -> list:
        """Vols ATM 1 an pour chaque actif."""
        return [a.atm_vol(1.0) for a in self.assets]

    @property
    def qs(self) -> list:
        return [a.q for a in self.assets]

    def df(self, tau: float) -> float:
        return self.ois_curve.df_tau(tau)

    def __repr__(self) -> str:
        tickers = [a.ticker for a in self.assets]
        return (
            f"MultiAssetSnapshot(assets={tickers}, "
            f"date={self.valuation_date})"
        )
