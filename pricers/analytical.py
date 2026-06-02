from __future__ import annotations

import math
import sys
import os
from datetime import date, timedelta
from typing import List, Optional
from functools import partial

import numpy as np
import pandas as pd

# Import pdts depuis Produit_pricing
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from Produit_pricing.pdts import (
    # Cat 1
    warrant, straddle, strangle,
    turbo_certificate, turbo_strangle,
    discount_certificate, discount_plus_certificate,
    discount_call, discount_put,
    reverse_discount_certificate, reverse_discount_plus_certificate,
    bonus_certificate, capped_bonus_certificate,
    reverse_bonus_certificate, capped_reverse_bonus_certificate,
    leveraged_bonus_certificate,
    # Cat 2
    implied_coupon_reverse_convertible, implied_coupon_rc_plus_pro,
    implied_redemption_easy_express, implied_bonus_return_certificate,
    _rc, _rc_plus_pro,
    # Cat 3
    implied_participation_outperformance, implied_participation_capped_outperformance,
    implied_participation_outperformance_plus, implied_participation_sprint,
    implied_participation_twin_win, implied_participation_reverse_outperformance,
    implied_participation_garantie, implied_participation_airbag,
    implied_participation_airbag_plus,
    _outperformance, _capped_outperformance, _outperformance_plus,
    _sprint, _twin_win, _reverse_outperformance, _garantie, _airbag, _airbag_plus,
    _gbs,
)

from pricing_lib.market_data.market_snapshot import MarketSnapshot
from pricing_lib.market_data.vol_surface import dupire_local_vol
from pricing_lib.pricers.base import PricingResult
from pricing_lib.pricers.greeks import compute_greeks


def _freq_map(freq: str) -> int:
    """'M'->12, 'Q'->4, 'S'->2, 'A'->1 paiements par an."""
    return {"M": 12, "Q": 4, "S": 2, "A": 1}[freq.upper()]


def _coupon_dates(start: date, T_years: float, freq: str) -> List[date]:
    """Génère les dates de paiement de coupon."""
    import calendar
    n_per_year  = _freq_map(freq)
    months_step = 12 // n_per_year
    n_periods   = max(1, round(T_years * n_per_year))
    dates   = []
    current = start
    for _ in range(n_periods):
        month = current.month + months_step
        year  = current.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day   = min(current.day, calendar.monthrange(year, month)[1])
        current = current.replace(year=year, month=month, day=day)
        dates.append(current)
    return dates


def _bond_value(
    nominal: float,
    coupon: float,
    freq: str,
    T_years: float,
    start: date,
    snapshot: MarketSnapshot,
) -> float:
    """Valeur actualisée de l'obligation (coupons périodiques + remboursement)."""
    n_per_year = _freq_map(freq)
    coupon_period = coupon / n_per_year
    pay_dates = _coupon_dates(start, T_years, freq)

    pv = 0.0
    for d in pay_dates:
        tau = (d - snapshot.valuation_date).days / 365.25
        if tau > 0:
            pv += nominal * coupon_period * snapshot.ois_curve.df_tau(tau)

    # Remboursement du nominal à maturité
    pv += nominal * snapshot.ois_curve.df_tau(T_years)
    return pv


def _cashflow_df(
    nominal: float,
    coupon: float,
    freq: str,
    T_years: float,
    start: date,
    snapshot: MarketSnapshot,
) -> pd.DataFrame:
    """DataFrame des flux pour RC avec fréquence."""
    n_per_year = _freq_map(freq)
    coupon_period = coupon / n_per_year
    pay_dates = _coupon_dates(start, T_years, freq)

    rows = []
    for d in pay_dates:
        tau = (d - snapshot.valuation_date).days / 365.25
        if tau <= 0:
            continue
        df_v = snapshot.ois_curve.df_tau(tau)
        rows.append({
            "Date":        str(d),
            "Coupon_rate": round(coupon_period * 100, 4),
            "DF":          round(df_v, 6),
            "PV_coupon":   round(nominal * coupon_period * df_v, 4),
        })
    # Ajout du nominal final
    tau_T = T_years
    df_T  = snapshot.ois_curve.df_tau(tau_T)
    rows[-1]["PV_coupon"] = round(
        rows[-1]["PV_coupon"] + nominal * df_T, 4
    )
    return pd.DataFrame(rows)


def _rc_price_with_freq(
    S: float, cap: float, T: float,
    r: float, q: float, sigma: float,
    nominal: float, coupon: float, freq: str,
    start: date, snapshot: MarketSnapshot,
) -> float:
    """RC avec fréquence : bond actualisé - short put × ratio."""
    bond  = _bond_value(nominal, coupon, freq, T, start, snapshot)
    ratio = nominal / cap
    sp    = _gbs("p", S, cap, T, r, r - q, sigma)
    return (bond - sp * ratio) / nominal * 100.0


class AnalyticalPricer:
    """Pricer analytique unifié."""

    def __init__(self, snapshot: MarketSnapshot):
        self.snap = snapshot

    @property
    def S(self) -> float:
        return self.snap.spot

    @property
    def r(self) -> float:
        return self.snap.r

    @property
    def q(self) -> float:
        return self.snap.q

    def _iv(self, T: float, K: float) -> float:
        """Vol IMPLICITE de marché σ_IV(T, K) — interpolation directe de la surface.
        À utiliser pour les vanilles (call, put, warrant).
        BS(σ_IV) = prix de marché par définition."""
        return self.snap.vol_surface.vol(T, K)

    def _atm_iv(self, T: float) -> float:
        """Vol implicite ATM (forward)."""
        F = self.S * math.exp((self.r - self.q) * T)
        return self.snap.vol_surface.vol(T, F)

    def _sigma(self, T: float, K: float) -> float:
        """Vol LOCALE σ_loc(T, K) via Dupire — pour exotiques / barrières."""
        return dupire_local_vol(self.snap.vol_surface, T, K)

    def _atm_sigma(self, T: float) -> float:
        """Vol locale ATM via Dupire."""
        F = self.S * math.exp((self.r - self.q) * T)
        return dupire_local_vol(self.snap.vol_surface, T, F)

    def _result(
        self, price: float, price_fn, S, r, q, sigma, T,
        product: str, model: str = "analytical",
        K: float = None, vol_type: str = "σ_IV(T,K)", **kw
    ) -> PricingResult:
        greeks = compute_greeks(price_fn, price, S, r, q, sigma, T, **kw)

        # Paramètres BS : calculés si K est fourni (vanilles), sinon ATM
        K_ref = K if K is not None else S * math.exp((r - q) * T)
        F     = S * math.exp((r - q) * T)
        if sigma > 0 and T > 0:
            d1 = (math.log(S / K_ref) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            d2 = d1 - sigma * math.sqrt(T)
        else:
            d1 = d2 = float("nan")

        bs = {
            "S": S, "K": K_ref, "T": T,
            "r": r, "q": q, "sigma": sigma,
            "F": F, "d1": d1, "d2": d2,
            "vol_type": vol_type,
        }

        return PricingResult(
            price=round(price, 6),
            greeks=greeks,
            ticker=self.snap.ticker,
            product=product,
            model=model,
            bs_params=bs,
        )

    # ── Vanilles ─────────────────────────────────────────────────────────────

    def call(self, K: float, T: float, ratio: float = 1.0) -> PricingResult:
        sigma = self._iv(T, K)
        S, r, q = self.S, self.r, self.q
        price = warrant("c", S, K, T, r, q, sigma, ratio)
        return self._result(price, lambda S, r, q, sigma, T: warrant("c", S, K, T, r, q, sigma, ratio),
                            S, r, q, sigma, T, product="Call", K=K, vol_type="σ_IV(T,K)")

    def put(self, K: float, T: float, ratio: float = 1.0) -> PricingResult:
        sigma = self._iv(T, K)
        S, r, q = self.S, self.r, self.q
        price = warrant("p", S, K, T, r, q, sigma, ratio)
        return self._result(price, lambda S, r, q, sigma, T: warrant("p", S, K, T, r, q, sigma, ratio),
                            S, r, q, sigma, T, product="Put", K=K, vol_type="σ_IV(T,K)")

    def warrant_product(self, flag: str, K: float, T: float, ratio: float = 1.0) -> PricingResult:
        sigma = self._iv(T, K)
        S, r, q = self.S, self.r, self.q
        price = warrant(flag, S, K, T, r, q, sigma, ratio)
        return self._result(price, lambda S, r, q, sigma, T: warrant(flag, S, K, T, r, q, sigma, ratio),
                            S, r, q, sigma, T, product=f"Warrant {flag.upper()}", K=K, vol_type="σ_IV(T,K)")

    # ── Catégorie 1 : structures pures ───────────────────────────────────────

    def bonus_certificate(self, X: float, B: float, T: float,
                          ratio: float = 1.0, barrier_hit: bool = False) -> PricingResult:
        sigma = self._atm_sigma(T)
        S, r, q = self.S, self.r, self.q
        price = bonus_certificate(S, X, B, T, r, q, sigma, ratio, barrier_hit)
        return self._result(
            price,
            lambda S, r, q, sigma, T: bonus_certificate(S, X, B, T, r, q, sigma, ratio, barrier_hit),
            S, r, q, sigma, T, product="Bonus Certificate",
        )

    def capped_bonus_certificate(self, X: float, B: float, cap: float, T: float,
                                  ratio: float = 1.0, barrier_hit: bool = False) -> PricingResult:
        sigma = self._atm_sigma(T)
        S, r, q = self.S, self.r, self.q
        price = capped_bonus_certificate(S, X, B, cap, T, r, q, sigma, ratio, barrier_hit)
        return self._result(
            price,
            lambda S, r, q, sigma, T: capped_bonus_certificate(S, X, B, cap, T, r, q, sigma, ratio, barrier_hit),
            S, r, q, sigma, T, product="Capped Bonus Certificate",
        )

    def reverse_bonus_certificate(self, X: float, B: float, T: float,
                                   ratio: float = 1.0, barrier_hit: bool = False) -> PricingResult:
        sigma = self._atm_sigma(T)
        S, r, q = self.S, self.r, self.q
        S0 = S
        price = reverse_bonus_certificate(S, S0, X, B, T, r, q, sigma, ratio, barrier_hit)
        return self._result(
            price,
            lambda S, r, q, sigma, T: reverse_bonus_certificate(S, S0, X, B, T, r, q, sigma, ratio, barrier_hit),
            S, r, q, sigma, T, product="Reverse Bonus Certificate",
        )

    def discount_certificate(self, X: float, T: float, ratio: float = 1.0) -> PricingResult:
        sigma = self._sigma(T, X)
        S, r, q = self.S, self.r, self.q
        price = discount_certificate(S, X, T, r, q, sigma, ratio)
        return self._result(
            price,
            lambda S, r, q, sigma, T: discount_certificate(S, X, T, r, q, sigma, ratio),
            S, r, q, sigma, T, product="Discount Certificate",
        )

    def turbo_certificate(self, flag: str, X: float, B: float, T: float,
                           ratio: float = 1.0) -> PricingResult:
        sigma = self._atm_sigma(T)
        S, r, q = self.S, self.r, self.q
        price = turbo_certificate(flag, S, X, B, T, r, q, sigma, ratio)
        return self._result(
            price,
            lambda S, r, q, sigma, T: turbo_certificate(flag, S, X, B, T, r, q, sigma, ratio),
            S, r, q, sigma, T, product=f"Turbo {'Call' if flag=='c' else 'Put'}",
        )

    def airbag_certificate(self, X: float, B: float, T: float,
                            ratio: float = 1.0) -> PricingResult:
        sigma = self._atm_sigma(T)
        S, r, q = self.S, self.r, self.q
        part = implied_participation_airbag(S, S, X, B, T, r, q, sigma)
        price = _airbag(S, X, B, T, r, q, sigma, part)
        res = self._result(
            price,
            lambda S, r, q, sigma, T: _airbag(S, X, B, T, r, q, sigma, part),
            S, r, q, sigma, T, product="Airbag Certificate",
        )
        res.fair_participation = round(part, 6)
        return res

    def garantie_certificate(self, X: float, T: float,
                              nominal: float = 100.0, ratio: float = 1.0) -> PricingResult:
        sigma = self._sigma(T, X)
        S, r, q = self.S, self.r, self.q
        part = implied_participation_garantie(S, S, X, T, r, q, sigma, nominal)
        price = _garantie(S, X, T, r, q, sigma, part, nominal)
        res = self._result(
            price,
            lambda S, r, q, sigma, T: _garantie(S, X, T, r, q, sigma, part, nominal),
            S, r, q, sigma, T, product="Garantie Certificate",
        )
        res.fair_participation = round(part, 6)
        return res

    # ── Catégorie 2 : coupon produits ─────────────────────────────────────────

    def reverse_convertible(
        self,
        cap: float,
        T: float,
        freq: str = "A",
        nominal: float = 100.0,
        ratio: float = 1.0,
        start: Optional[date] = None,
    ) -> PricingResult:
        """RC avec fréquence de coupon. Retourne le coupon fair annualisé."""
        if start is None:
            start = self.snap.valuation_date
        sigma = self._sigma(T, cap)
        S, r, q = self.S, self.r, self.q
        snap = self.snap

        # Coupon fair : prix = 100% du nominal
        from scipy.optimize import brentq
        def price_at_coupon(c):
            return _rc_price_with_freq(S, cap, T, r, q, sigma, nominal, c, freq, start, snap) - 100.0

        coupon = brentq(price_at_coupon, 0.0, 2.0, xtol=1e-8)
        price  = _rc_price_with_freq(S, cap, T, r, q, sigma, nominal, coupon, freq, start, snap)

        # Greeks sur le prix
        def _pfn(S, r, q, sigma, T):
            return _rc_price_with_freq(S, cap, T, r, q, sigma, nominal, coupon, freq, start, snap)

        greeks = compute_greeks(_pfn, price, S, r, q, sigma, T)
        cf_df  = _cashflow_df(nominal, coupon, freq, T, start, snap)

        return PricingResult(
            price=round(price, 6),
            greeks=greeks,
            ticker=self.snap.ticker,
            product=f"Reverse Convertible ({freq})",
            model="analytical",
            fair_coupon=round(coupon, 6),
            cashflows=cf_df,
        )

    def reverse_convertible_plus_pro(
        self,
        cap: float,
        B: float,
        T: float,
        freq: str = "A",
        nominal: float = 100.0,
        barrier_hit: bool = False,
        start: Optional[date] = None,
    ) -> PricingResult:
        """RC Plus Pro (knock-in put) avec fréquence de coupon."""
        if start is None:
            start = self.snap.valuation_date
        sigma = self._sigma(T, cap)
        S, r, q = self.S, self.r, self.q
        snap = self.snap

        from scipy.optimize import brentq

        def price_at_coupon(c):
            bond  = _bond_value(nominal, c, freq, T, start, snap)
            ratio = nominal / cap
            from Produit_pricing.pdts import _gbs, _barrier
            b = r - q
            sp  = _gbs("p", S, cap, T, r, b, sigma)
            dop = 0.0 if barrier_hit else _barrier("pdo", S, cap, B, 0.0, T, r, b, sigma)
            return (bond - sp * ratio + dop * ratio) / nominal * 100.0 - 100.0

        coupon = brentq(price_at_coupon, 0.0, 2.0, xtol=1e-8)

        def _pfn(S, r, q, sigma, T):
            from Produit_pricing.pdts import _gbs, _barrier
            bond  = _bond_value(nominal, coupon, freq, T, start, snap)
            ratio = nominal / cap
            b = r - q
            sp  = _gbs("p", S, cap, T, r, b, sigma)
            dop = 0.0 if barrier_hit else _barrier("pdo", S, cap, B, 0.0, T, r, b, sigma)
            return (bond - sp * ratio + dop * ratio) / nominal * 100.0

        price  = _pfn(S, r, q, sigma, T)
        greeks = compute_greeks(_pfn, price, S, r, q, sigma, T)
        cf_df  = _cashflow_df(nominal, coupon, freq, T, start, snap)

        return PricingResult(
            price=round(price, 6),
            greeks=greeks,
            ticker=self.snap.ticker,
            product=f"RC Plus Pro ({freq})",
            model="analytical",
            fair_coupon=round(coupon, 6),
            cashflows=cf_df,
        )

    # ── Catégorie 3 : participation produits ──────────────────────────────────

    def outperformance_certificate(self, X: float, T: float, ratio: float = 1.0) -> PricingResult:
        sigma = self._sigma(T, X)
        S, r, q = self.S, self.r, self.q
        part = implied_participation_outperformance(S, S, X, T, r, q, sigma)
        price = _outperformance(S, X, T, r, q, sigma, part)
        res = self._result(
            price,
            lambda S, r, q, sigma, T: _outperformance(S, X, T, r, q, sigma, part),
            S, r, q, sigma, T, product="Outperformance Certificate",
        )
        res.fair_participation = round(part, 6)
        return res

    def sprint_certificate(self, X: float, cap: float, T: float, ratio: float = 1.0) -> PricingResult:
        sigma = self._atm_sigma(T)
        S, r, q = self.S, self.r, self.q
        part = implied_participation_sprint(S, S, X, cap, T, r, q, sigma)
        price = _sprint(S, X, cap, T, r, q, sigma, part)
        res = self._result(
            price,
            lambda S, r, q, sigma, T: _sprint(S, X, cap, T, r, q, sigma, part),
            S, r, q, sigma, T, product="Sprint Certificate",
        )
        res.fair_participation = round(part, 6)
        return res

    def twin_win_certificate(self, X: float, B: float, T: float, ratio: float = 1.0) -> PricingResult:
        sigma = self._atm_sigma(T)
        S, r, q = self.S, self.r, self.q
        part = implied_participation_twin_win(S, S, X, B, T, r, q, sigma, bracket=(0., 5.))
        price = _twin_win(S, X, B, T, r, q, sigma, part)
        res = self._result(
            price,
            lambda S, r, q, sigma, T: _twin_win(S, X, B, T, r, q, sigma, part),
            S, r, q, sigma, T, product="Twin Win Certificate",
        )
        res.fair_participation = round(part, 6)
        return res
