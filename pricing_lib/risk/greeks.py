"""
pricing_lib/risk/greeks.py
─────────────────────────────────────────────────────────────────────────────
Greeks par différences finies centrées.

FiniteDiffGreeks encapsule n'importe quel pricer callable et calcule
Delta, Gamma, Vega, Theta, Rho par perturbation des paramètres.

Compatible avec AnalyticalPricer et MonteCarloPricer (les deux retournent
un objet avec un attribut .price).

Usage
-----
>>> from pricing_lib.pricers import AnalyticalPricer
>>> from pricing_lib.risk.greeks import FiniteDiffGreeks
>>>
>>> pricer  = AnalyticalPricer()
>>> greeks  = FiniteDiffGreeks(pricer, product, market)
>>> print(greeks.delta())
>>> print(greeks.all())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..market_data.market_snapshot import MarketSnapshot


# ─── résultat ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GreeksResult:
    delta: float
    gamma: float
    vega:  float
    theta: float
    rho:   float

    def __str__(self) -> str:
        return (
            f"Delta : {self.delta:>10.6f}\n"
            f"Gamma : {self.gamma:>10.6f}\n"
            f"Vega  : {self.vega:>10.4f}  (pour +1% vol)\n"
            f"Theta : {self.theta:>10.4f}  (par jour calendaire)\n"
            f"Rho   : {self.rho:>10.4f}  (pour +1bp taux)"
        )


# ─── FiniteDiffGreeks ────────────────────────────────────────────────────────

class FiniteDiffGreeks:
    """
    Greeks par différences finies centrées sur le MarketSnapshot.

    Paramètres
    ----------
    pricer  : objet avec méthode .price(product, market[, model]) -> obj.price
    product : produit à pricer
    market  : MarketSnapshot de référence
    model   : modèle optionnel (passé au pricer si fourni)

    Bumps par défaut
    ----------------
    dS    : 0.1% de S     (Delta, Gamma)
    dsig  : 1bp de vol    (Vega)
    dT    : 1 jour        (Theta)
    dr    : 1bp de taux   (Rho)
    """

    def __init__(
        self,
        pricer,
        product,
        market:  MarketSnapshot,
        model    = None,
        dS_frac: float = 0.001,
        dsig:    float = 0.0001,
        dT:      float = 1/365,
        dr:      float = 0.0001,
    ) -> None:
        self._pricer  = pricer
        self._product = product
        self._market  = market
        self._model   = model
        self._dS_frac = dS_frac
        self._dsig    = dsig
        self._dT      = dT
        self._dr      = dr

    # ── interface publique ────────────────────────────────────────────────────

    def delta(self) -> float:
        dS  = self._market.spot * self._dS_frac
        p_u = self._price(_bump_spot(self._market, +dS))
        p_d = self._price(_bump_spot(self._market, -dS))
        return (p_u - p_d) / (2 * dS)

    def gamma(self) -> float:
        dS  = self._market.spot * self._dS_frac
        p_u = self._price(_bump_spot(self._market, +dS))
        p_0 = self._price(self._market)
        p_d = self._price(_bump_spot(self._market, -dS))
        return (p_u - 2 * p_0 + p_d) / dS**2

    def vega(self, per_percent: bool = True) -> float:
        p_u = self._price(_bump_vol(self._market, +self._dsig))
        p_d = self._price(_bump_vol(self._market, -self._dsig))
        raw = (p_u - p_d) / (2 * self._dsig)
        return raw * 0.01 if per_percent else raw

    def theta(self) -> float:
        prod = self._product
        dT   = self._dT
        try:
            if hasattr(prod, "maturity"):
                prod_bumped = _bump_product_maturity(prod, -dT)
            else:
                return float("nan")
        except Exception:
            return float("nan")
        p_0 = self._price(self._market)
        p_d = self._price_product(prod_bumped, self._market)
        return (p_d - p_0) / dT

    def rho(self, per_bp: bool = True) -> float:
        p_u = self._price(_bump_rate(self._market, +self._dr))
        p_d = self._price(_bump_rate(self._market, -self._dr))
        raw = (p_u - p_d) / (2 * self._dr)
        return raw * 0.0001 if per_bp else raw

    def all(self) -> GreeksResult:
        return GreeksResult(
            delta = self.delta(),
            gamma = self.gamma(),
            vega  = self.vega(),
            theta = self.theta(),
            rho   = self.rho(),
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _price(self, market: MarketSnapshot) -> float:
        return self._price_product(self._product, market)

    def _price_product(self, product, market: MarketSnapshot) -> float:
        if self._model is not None:
            result = self._pricer.price(product, market, self._model)
        else:
            result = self._pricer.price(product, market)
        return result.price


# ─── bump helpers ─────────────────────────────────────────────────────────────

class _BumpedVolSurface:
    """
    Wrapper sur une VolSurface qui décale toutes les vols d'un montant fixe.

    Override sigma() ET vol() car AnalyticalPricer utilise vol_surface.vol(T, K)
    via _iv(), tandis que les produits exotiques utilisent vol_surface.sigma(T, K)
    via dupire_local_vol().
    """

    def __init__(self, base_surface, sigma_bump: float) -> None:
        self._base = base_surface
        self._bump = sigma_bump

    def sigma(self, T: float, K: float) -> float:
        return max(1e-4, self._base.sigma(T, K) + self._bump)

    def vol(self, T: float, K: float) -> float:
        return max(1e-4, self._base.vol(T, K) + self._bump)

    def __getattr__(self, name: str):
        return getattr(self._base, name)


class _BumpedMarket:
    """Wrapper léger autour de MarketSnapshot avec paramètres modifiés."""

    def __init__(self, base: MarketSnapshot, **overrides) -> None:
        self._base = base
        self._ov   = overrides

    def __getattr__(self, name: str):
        if name in self._ov:
            return self._ov[name]
        return getattr(self._base, name)

    @property
    def vol_surface(self):
        if "sigma_bump" in self._ov:
            return _BumpedVolSurface(self._base.vol_surface, self._ov["sigma_bump"])
        return self._base.vol_surface

    def sigma(self, T: float, K: float) -> float:
        return self.vol_surface.sigma(T, K)

    def df(self, tau: float) -> float:
        if "rate_bump" in self._ov:
            r_new = self._base.r + self._ov["rate_bump"]
            return math.exp(-r_new * tau)
        return self._base.df(tau)

    @property
    def r(self) -> float:
        if "rate_bump" in self._ov:
            return self._base.r + self._ov["rate_bump"]
        return self._base.r

    @property
    def q(self) -> float:
        return self._base.q

    @property
    def spot(self) -> float:
        return self._ov.get("spot", self._base.spot)


def _bump_spot(market: MarketSnapshot, dS: float) -> _BumpedMarket:
    return _BumpedMarket(market, spot=market.spot + dS)


def _bump_vol(market: MarketSnapshot, dsig: float) -> _BumpedMarket:
    return _BumpedMarket(market, sigma_bump=dsig)


def _bump_rate(market: MarketSnapshot, dr: float) -> _BumpedMarket:
    return _BumpedMarket(market, rate_bump=dr)


def _bump_product_maturity(product, dT: float):
    """Retourne une copie du produit avec maturity décalée de dT."""
    import dataclasses
    if dataclasses.is_dataclass(product):
        if hasattr(product, "obs_times"):
            new_obs = [t + dT for t in product.obs_times]
            return dataclasses.replace(product, obs_times=new_obs)
        if hasattr(product, "maturity"):
            return dataclasses.replace(product, maturity=product.maturity + dT)
    return product


# ─── Adaptateur générique ─────────────────────────────────────────────────────

class _ClosurePricer:
    """
    Adapte n'importe quelle fonction f(market) -> float a l'interface pricer
    attendue par FiniteDiffGreeks, ScenarioAnalyzer et StressTest.
    """

    def __init__(self, fn) -> None:
        self._fn = fn

    def price(self, product, market) -> object:
        class _R:
            pass
        r = _R()
        r.price = self._fn(market)
        return r
