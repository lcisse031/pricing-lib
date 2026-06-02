from __future__ import annotations

import math
import os
import sys
from datetime import date
from typing import Dict, Optional

import numpy as np
import pandas as pd

from pricing_lib.market_data.market_snapshot import MarketSnapshot
from pricing_lib.market_data.vol_surface import dupire_local_vol
from pricing_lib.models.heston import (
    HestonParams, DEFAULT_HESTON,
    heston_simulate, heston_simulate_paths,
    calibrate_heston,
)
from pricing_lib.models.heston2 import heston_monte_carlo
from pricing_lib.pricers.base import PricingResult
from pricing_lib.pricers.greeks import compute_greeks_mc


_GREEK_SEED: int = 42   # toutes les closures pfn utilisent ce seed → aléas identiques


_HESTON_CACHE: Dict[str, HestonParams] = {}


def get_heston_params(
    snapshot: MarketSnapshot,
    force_recalibrate: bool = False,
) -> HestonParams:
    """
    Retourne les paramètres Heston calibrés pour le ticker du snapshot.
    Utilise le cache si disponible, recalibre sinon.
    """
    key = snapshot.ticker
    if key in _HESTON_CACHE and not force_recalibrate:
        return _HESTON_CACHE[key]

    try:
        params = calibrate_heston(
            snapshot.vol_surface,
            S=snapshot.spot,
            r=snapshot.r,
            q=snapshot.q,
        )
    except Exception as e:
        print(f"[MC] Calibration Heston échouée pour {key} : {e}")
        print(f"[MC] Utilisation des paramètres par défaut.")
        params = DEFAULT_HESTON

    _HESTON_CACHE[key] = params
    return params


class MCPricer:
    """Pricer Monte Carlo Heston."""

    def __init__(
        self,
        snapshot: MarketSnapshot,
        n_paths: int,
        params: Optional[HestonParams] = None,
        antithetic: bool = True,
        seed: Optional[int] = None,
    ):
        self.snap      = snapshot
        self.n_paths   = n_paths
        self.antithetic = antithetic
        self.seed      = seed
        self.params    = params or get_heston_params(snapshot)

    @property
    def S(self) -> float:
        return self.snap.spot

    @property
    def r(self) -> float:
        return self.snap.r

    @property
    def q(self) -> float:
        return self.snap.q

    def _simulate(self, T: float, S_override: Optional[float] = None,
                  r_override: Optional[float] = None,
                  params_override: Optional[HestonParams] = None) -> np.ndarray:
        """Simule N prix terminaux."""
        return heston_simulate(
            S=S_override or self.S,
            r=r_override or self.r,
            q=self.q,
            T=T,
            params=params_override or self.params,
            n_paths=self.n_paths,
            antithetic=self.antithetic,
            seed=self.seed,
        )

    def _simulate_paths(self, T: float, obs_times: np.ndarray,
                        S_override: Optional[float] = None,
                        r_override: Optional[float] = None,
                        params_override: Optional[HestonParams] = None):
        """Simule les chemins complets aux dates d'observation."""
        return heston_simulate_paths(
            S=S_override or self.S,
            r=r_override or self.r,
            q=self.q,
            T=T,
            params=params_override or self.params,
            n_paths=self.n_paths,
            observation_times=obs_times,
            antithetic=self.antithetic,
            seed=self.seed,
        )

    def _price_fn_terminal(self, payoff_fn, T: float):
        """Retourne une fonction price_fn(S, r, q, T, params) pour les Greeks MC."""
        def fn(S, r, q, T, params=None):
            ST = heston_simulate(
                S=S, r=r, q=q, T=T,
                params=params or self.params,
                n_paths=self.n_paths,
                antithetic=self.antithetic,
                seed=self.seed,
            )
            df = self.snap.ois_curve.df_tau(T)
            return float(np.mean(payoff_fn(ST)) * df)
        return fn

    def _greeks(self, price0: float, price_fn_mc, S, r, q, T) -> dict:
        return compute_greeks_mc(
            price_fn_mc, price0, S, r, q, T, self.params
        )

    # ── Vanilles ─────────────────────────────────────────────────────────────

    def call(self, K: float, T: float, ratio: float = 1.0) -> PricingResult:
        S, r, q = self.S, self.r, self.q
        df = self.snap.ois_curve.df_tau(T)

        ST    = self._simulate(T)
        price = float(np.mean(np.maximum(ST - K, 0.0)) * df * ratio)

        def pfn(S, r, q, T, params=None):
            _ST = heston_simulate(S=S, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            return float(np.mean(np.maximum(_ST - K, 0.0)) * self.snap.ois_curve.df_tau(T) * ratio)

        greeks = self._greeks(price, pfn, S, r, q, T)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Call", model="heston_mc")

    def put(self, K: float, T: float, ratio: float = 1.0) -> PricingResult:
        S, r, q = self.S, self.r, self.q
        df = self.snap.ois_curve.df_tau(T)

        ST    = self._simulate(T)
        price = float(np.mean(np.maximum(K - ST, 0.0)) * df * ratio)

        def pfn(S, r, q, T, params=None):
            _ST = heston_simulate(S=S, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            return float(np.mean(np.maximum(K - _ST, 0.0)) * self.snap.ois_curve.df_tau(T) * ratio)

        greeks = self._greeks(price, pfn, S, r, q, T)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Put", model="heston_mc")

    # ── Barrières (chemin complet requis) ─────────────────────────────────────

    def bonus_certificate(
        self, X: float, B: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Bonus Certificate : ZSC + Down-and-Out Put(X, B).
        La barrière est surveillée en continu sur les trajectoires simulées.
        """
        S, r, q = self.S, self.r, self.q
        n_steps  = max(int(252 * T), 50)
        obs_times = np.linspace(T / n_steps, T, n_steps)

        _, paths = self._simulate_paths(T, obs_times)
        # paths shape : (n_paths, n_steps)

        barrier_hit = np.any(paths <= B, axis=1)          # (n_paths,)
        ST          = paths[:, -1]                          # prix final

        # Payoff : min(S_T, X) si barrière touchée, sinon max(S_T, X)
        payoff = np.where(
            barrier_hit,
            ST,                          # barrière touchée → participation directe
            np.maximum(ST, X),           # non touchée → bonus garanti
        ) * ratio

        df    = self.snap.ois_curve.df_tau(T)
        price = float(np.mean(payoff) * df)

        # Greeks simplifiés (terminal — approximation)
        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths // 4,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            # Approximation sans barrière pour les Greeks
            _payoff = np.maximum(_ST, X) * ratio
            return float(np.mean(_payoff) * self.snap.ois_curve.df_tau(T))

        greeks = compute_greeks_mc(pfn, price, S, r, q, T, self.params)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Bonus Certificate", model="heston_mc")

    def reverse_bonus_certificate(
        self, X: float, B: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Reverse Bonus Certificate.
        Payoff baissier : max(2S0 - S_T, 0) si barrière non touchée → bonus garanti.
        """
        S0, r, q = self.S, self.r, self.q
        n_steps  = max(int(252 * T), 50)
        obs_times = np.linspace(T / n_steps, T, n_steps)

        _, paths = self._simulate_paths(T, obs_times)

        barrier_hit = np.any(paths >= B, axis=1)
        ST          = paths[:, -1]

        payoff_no_hit = np.maximum(2 * S0 - ST, X - ST)  # bonus inverse
        payoff_hit    = np.maximum(2 * S0 - ST, 0.0)

        payoff = np.where(barrier_hit, payoff_hit, payoff_no_hit) * ratio

        df    = self.snap.ois_curve.df_tau(T)
        price = float(np.mean(payoff) * df)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths // 4,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            return float(np.mean(np.maximum(2 * S0 - _ST, X - _ST)) * ratio
                         * self.snap.ois_curve.df_tau(T))

        greeks = compute_greeks_mc(pfn, price, S0, r, q, T, self.params)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Reverse Bonus Certificate",
                             model="heston_mc")

    def discount_certificate(self, X: float, T: float, ratio: float = 1.0) -> PricingResult:
        """Discount Certificate : min(S_T, X)."""
        S, r, q = self.S, self.r, self.q
        ST    = self._simulate(T)
        df    = self.snap.ois_curve.df_tau(T)
        price = float(np.mean(np.minimum(ST, X)) * df * ratio)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            return float(np.mean(np.minimum(_ST, X)) * self.snap.ois_curve.df_tau(T) * ratio)

        greeks = compute_greeks_mc(pfn, price, S, r, q, T, self.params)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Discount Certificate",
                             model="heston_mc")

    # ── Reverse Convertible MC ────────────────────────────────────────────────

    def reverse_convertible(
        self,
        cap: float,
        T: float,
        coupon: float,
        freq: str = "A",
        nominal: float = 100.0,
        ratio: float = 1.0,
        start: Optional[date] = None,
    ) -> PricingResult:
        """
        Reverse Convertible Monte Carlo avec fréquence.

        Coupons : déterministes, actualisés par OISCurve.
        Remboursement final : min(S_T, cap) × nominal/cap.
        Coupon fourni en input — pour le fair coupon, utiliser le mode AL.
        """
        from pricing_lib.pricers.analytical import _bond_value, _cashflow_df

        if start is None:
            start = self.snap.valuation_date

        S, r, q = self.S, self.r, self.q
        df_T    = self.snap.ois_curve.df_tau(T)

        # Coupons déterministes
        bond_pv = _bond_value(nominal, coupon, freq, T, start, self.snap)

        # Remboursement final par MC
        ST         = self._simulate(T)
        ratio_conv = nominal / cap
        payoff_T   = np.where(ST >= cap, nominal, ST * ratio_conv)
        redemption = float(np.mean(payoff_T) * df_T * ratio)

        price = (bond_pv + redemption - nominal) / nominal * 100.0 + 100.0

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _redemption = float(np.mean(np.where(_ST >= cap, nominal, _ST * ratio_conv))
                                * self.snap.ois_curve.df_tau(T) * ratio)
            _bond = _bond_value(nominal, coupon, freq, T, start, self.snap)
            return (_bond + _redemption - nominal) / nominal * 100.0 + 100.0

        greeks = compute_greeks_mc(pfn, price, S, r, q, T, self.params)
        cf_df  = _cashflow_df(nominal, coupon, freq, T, start, self.snap)

        return PricingResult(
            price=round(price, 6),
            greeks=greeks,
            ticker=self.snap.ticker,
            product=f"Reverse Convertible MC ({freq})",
            model="heston_mc",
            fair_coupon=round(coupon, 6),
            cashflows=cf_df,
        )

    # ── Catégorie 3 : produits à participation ────────────────────────────────

    def _sigma(self, T: float, K: float) -> float:
        """Volatilité locale Dupire σ_loc(T, K) — pour calibration participation."""
        return dupire_local_vol(self.snap.vol_surface, T, K)

    def _atm_sigma(self, T: float) -> float:
        """Volatilité locale ATM."""
        F = self.S * math.exp((self.r - self.q) * T)
        return self._sigma(T, F)

    def _get_participation(self, fn_name: str, *args) -> float:
        """
        Calibre le taux de participation via Produit_pricing.pdts.
        Retourne 1.0 en cas d'échec.
        """
        try:
            _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            import importlib
            pdts = importlib.import_module("Produit_pricing.pdts")
            fn = getattr(pdts, fn_name)
            return float(fn(*args))
        except Exception as e:
            print(f"[MC] Calibration participation ({fn_name}) échouée : {e} — part=1.0")
            return 1.0

    def outperformance_certificate(
        self, X: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Outperformance Certificate MC.
        Payoff = S_T + part × max(S_T − X, 0)
        Le taux de participation `part` est calibré analytiquement (même surface).
        """
        S, r, q = self.S, self.r, self.q
        df = self.snap.ois_curve.df_tau(T)

        sigma = self._sigma(T, X)
        part  = self._get_participation(
            "implied_participation_outperformance", S, S, X, T, r, q, sigma
        )

        ST    = self._simulate(T)
        payoff = (ST + part * np.maximum(ST - X, 0.0)) * ratio
        price  = float(np.mean(payoff) * df)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _p = (_ST + part * np.maximum(_ST - X, 0.0)) * ratio
            return float(np.mean(_p) * self.snap.ois_curve.df_tau(T))

        greeks = compute_greeks_mc(pfn, price, S, r, q, T, self.params)
        res = PricingResult(price=round(price, 6), greeks=greeks,
                            ticker=self.snap.ticker,
                            product="Outperformance Certificate", model="heston_mc")
        res.fair_participation = round(part, 6)
        return res

    def sprint_certificate(
        self, X: float, cap: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Sprint Certificate MC.
        Payoff = S_T + part × (max(S_T − X, 0) − max(S_T − cap, 0))
        Équivalent à un outperformance capé.
        """
        S, r, q = self.S, self.r, self.q
        df = self.snap.ois_curve.df_tau(T)

        sigma = self._atm_sigma(T)
        part  = self._get_participation(
            "implied_participation_sprint", S, S, X, cap, T, r, q, sigma
        )

        ST    = self._simulate(T)
        bull_spread = np.maximum(ST - X, 0.0) - np.maximum(ST - cap, 0.0)
        payoff = (ST + part * bull_spread) * ratio
        price  = float(np.mean(payoff) * df)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _bs = np.maximum(_ST - X, 0.0) - np.maximum(_ST - cap, 0.0)
            return float(np.mean((_ST + part * _bs) * ratio)
                         * self.snap.ois_curve.df_tau(T))

        greeks = compute_greeks_mc(pfn, price, S, r, q, T, self.params)
        res = PricingResult(price=round(price, 6), greeks=greeks,
                            ticker=self.snap.ticker,
                            product="Sprint Certificate", model="heston_mc")
        res.fair_participation = round(part, 6)
        return res

    def twin_win_certificate(
        self, X: float, B: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Twin Win Certificate MC (barrière continue basse B).
        Payoff si barrière non touchée : X + part × |S_T − X|
        Payoff si barrière touchée     : S_T
        """
        S, r, q = self.S, self.r, self.q
        n_steps   = max(int(252 * T), 50)
        obs_times = np.linspace(T / n_steps, T, n_steps)

        sigma = self._atm_sigma(T)
        part  = self._get_participation(
            "implied_participation_twin_win", S, S, X, B, T, r, q, sigma
        )

        _, paths   = self._simulate_paths(T, obs_times)
        barrier_hit = np.any(paths <= B, axis=1)
        ST          = paths[:, -1]

        payoff = np.where(
            barrier_hit,
            ST,
            X + part * np.abs(ST - X),
        ) * ratio

        df    = self.snap.ois_curve.df_tau(T)
        price = float(np.mean(payoff) * df)

        # Greeks approx. — simulation terminale sans barrière
        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths // 4,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _p = (X + part * np.abs(_ST - X)) * ratio
            return float(np.mean(_p) * self.snap.ois_curve.df_tau(T))

        greeks = compute_greeks_mc(pfn, price, S, r, q, T, self.params)
        res = PricingResult(price=round(price, 6), greeks=greeks,
                            ticker=self.snap.ticker,
                            product="Twin Win Certificate", model="heston_mc")
        res.fair_participation = round(part, 6)
        return res

    def airbag_certificate(
        self, X: float, B: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Airbag Certificate MC (barrière européenne à maturité).
        Payoff :
          S_T ≥ X          → S_T + (part−1) × (S_T − X)   [upside levier]
          B ≤ S_T < X      → X                              [capital protégé]
          S_T < B          → S_T                            [sous la barrière]
        """
        S, r, q = self.S, self.r, self.q
        df = self.snap.ois_curve.df_tau(T)

        sigma = self._atm_sigma(T)
        part  = self._get_participation(
            "implied_participation_airbag", S, S, X, B, T, r, q, sigma
        )

        ST = self._simulate(T)
        payoff = (
            ST
            + (part - 1.0) * np.maximum(ST - X, 0.0)          # levier upside
            + np.maximum(X - ST, 0.0) * (ST >= B).astype(float)  # protection [B, X)
        ) * ratio
        price = float(np.mean(payoff) * df)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _p = (
                _ST
                + (part - 1.0) * np.maximum(_ST - X, 0.0)
                + np.maximum(X - _ST, 0.0) * (_ST >= B).astype(float)
            ) * ratio
            return float(np.mean(_p) * self.snap.ois_curve.df_tau(T))

        greeks = compute_greeks_mc(pfn, price, S, r, q, T, self.params)
        res = PricingResult(price=round(price, 6), greeks=greeks,
                            ticker=self.snap.ticker,
                            product="Airbag Certificate", model="heston_mc")
        res.fair_participation = round(part, 6)
        return res

    def garantie_certificate(
        self, X: float, T: float, nominal: float = 100.0, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Garantie Certificate MC.
        Payoff = nominal + part × max(S_T − X, 0)
        Le taux de participation `part` est calibré pour que prix ≈ nominal.
        """
        S, r, q = self.S, self.r, self.q
        df = self.snap.ois_curve.df_tau(T)

        sigma = self._sigma(T, X)
        part  = self._get_participation(
            "implied_participation_garantie", S, S, X, T, r, q, sigma, nominal
        )

        ST    = self._simulate(T)
        payoff = (nominal + part * np.maximum(ST - X, 0.0)) * ratio
        price  = float(np.mean(payoff) * df)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _p = (nominal + part * np.maximum(_ST - X, 0.0)) * ratio
            return float(np.mean(_p) * self.snap.ois_curve.df_tau(T))

        greeks = compute_greeks_mc(pfn, price, S, r, q, T, self.params)
        res = PricingResult(price=round(price, 6), greeks=greeks,
                            ticker=self.snap.ticker,
                            product="Garantie Certificate", model="heston_mc")
        res.fair_participation = round(part, 6)
        return res
