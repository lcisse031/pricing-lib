"""
pricing_lib/pricers/mc.py
─────────────────────────────────────────────────────────────────────────────
Pricer Monte Carlo — vanilles (GBM) et produits structurés (Heston).

Vanilles (Call, Put, Warrant Call, Warrant Put) :
    Modèle GBM standard : S_T = S · exp((r-q-σ²/2)T + σ√T·Z)
    Vol : σ_IV(T, K) lue sur la surface implicite.
    Greeks : différences finies centrées via compute_greeks_fd_vol.

Autres produits (barrières, participation, RC) :
    Modèle Heston calibré sur la surface.
    Greeks : différences finies centrées via compute_greeks_fd_heston.

Cache de calibration : {ticker: HestonParams} — 1 calibration par session par ticker.
"""


from __future__ import annotations

import math
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
from pricing_lib.pricers.base import PricingResult
from pricing_lib.risk.greeks import (
    compute_greeks_fd_vol,      # vanilles GBM (bump σ, centré)
    compute_greeks_fd_heston,   # exotiques Heston (bump v0+θ, centré)
)


# ─── Seed fixe pour les Greeks (différences finies reproductibles) ───────────
_GREEK_SEED: int = 42   # toutes les closures pfn utilisent ce seed → aléas identiques


# ─── Simulation GBM standard ──────────────────────────────────────────────────

def _gbm_simulate(
    S: float, r: float, q: float, sigma: float, T: float,
    n_paths: int, antithetic: bool = True, seed: Optional[int] = None,
    qmc: bool = False,
) -> np.ndarray:
    """
    S_T = S * exp((r - q - σ²/2)*T + σ*√T*Z).

    qmc=True : suite de Sobol scramblée + inverse-CDF (quasi-Monte-Carlo).
    Super-convergence : ~30× moins d'erreur qu'un tirage pseudo-aléatoire à
    nombre de trajectoires égal (estimateur non biaisé grâce au scrambling).
    """
    if qmc:
        from scipy.stats import qmc as _qmc, norm as _norm
        sob = _qmc.Sobol(d=1, scramble=True, seed=seed)
        m   = int(np.ceil(np.log2(max(n_paths, 2))))
        u   = sob.random_base2(m=m).ravel()[:n_paths]
        Z   = _norm.ppf(np.clip(u, 1e-12, 1.0 - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        n   = n_paths // 2 if antithetic else n_paths
        Z   = rng.standard_normal(n)
        if antithetic:
            Z = np.concatenate([Z, -Z])
    return S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)


# ─── Cache calibration ────────────────────────────────────────────────────────

_HESTON_CACHE: Dict[str, HestonParams] = {}


def get_heston_params(
    snapshot: MarketSnapshot,
    force_recalibrate: bool = False,
) -> HestonParams:
    """
    Retourne les paramètres Heston calibrés pour le ticker du snapshot.
    Utilise le cache si disponible, recalibre sinon.

    Clé de cache = (ticker, date de valorisation) : on recalibre par date
    (la surface change chaque jour) au lieu de geler les params au 1er appel.
    """
    key = (snapshot.ticker, snapshot.valuation_date)
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


# ─── Pricer MC ────────────────────────────────────────────────────────────────

class MCPricer:
    """
    Pricer Monte Carlo.

    Vanilles (Call, Put, Warrant Call, Warrant Put) : GBM standard, σ_IV(T,K).
    Autres produits : Heston calibré.
    """

    def __init__(
        self,
        snapshot: MarketSnapshot,
        n_paths: int,
        params: Optional[HestonParams] = None,
        antithetic: bool = True,
        seed: Optional[int] = None,
        qmc: bool = False,
    ):
        self.snap      = snapshot
        self.n_paths   = n_paths
        self.antithetic = antithetic
        self.qmc       = qmc   # Sobol/QMC pour les vanilles GBM (super-convergence)
        self.seed      = seed
        # Calibration Heston PARESSEUSE : déclenchée seulement au 1er accès à
        # `self.params`, c.-à-d. uniquement pour les produits exotiques.
        # Les vanilles (GBM, σ_IV) ne touchent jamais params → pas de calibration.
        self._params   = params

    @property
    def params(self) -> HestonParams:
        """Paramètres Heston — calibrés à la demande (cache via get_heston_params)."""
        if self._params is None:
            self._params = get_heston_params(self.snap)
        return self._params

    @property
    def S(self) -> float:
        return self.snap.spot

    @property
    def r(self) -> float:
        return self.snap.r

    @property
    def q(self) -> float:
        return self.snap.q

    def _rT(self, T: float) -> float:
        """
        Taux forward OIS continu jusqu'à T : r_T = −ln(df(T))/T.

        Garantit la cohérence drift/actualisation : avec ce taux,
        df = exp(−r_T·T) = df_OIS(T) exactement. Un bump de r_T décale donc
        drift ET actualisation (rho « complet », pas seulement le drift).
        """
        T = max(float(T), 1e-6)
        ois = self.snap.ois_curve
        return -math.log(max(ois.df_tau(T), 1e-12)) / T

    def _iv(self, T: float, K: float) -> float:
        """Vol implicite de marché σ_IV(T, K)."""
        return self.snap.vol_surface.vol(T, K)

    def _simulate(self, T: float, S_override: Optional[float] = None,
                  r_override: Optional[float] = None,
                  params_override: Optional[HestonParams] = None) -> np.ndarray:
        """Simule N prix terminaux (Heston)."""
        return heston_simulate(
            S=S_override or self.S,
            r=r_override if r_override is not None else self._rT(T),
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
        """Simule les chemins complets aux dates d'observation (Heston)."""
        return heston_simulate_paths(
            S=S_override or self.S,
            r=r_override if r_override is not None else self._rT(T),
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
            df = math.exp(-r * T)
            return float(np.mean(payoff_fn(ST)) * df)
        return fn

    def _greeks(self, price0: float, price_fn_mc, S, r, q, T) -> dict:
        return compute_greeks_fd_heston(
            price_fn_mc, price0, S, r, q, T, self.params
        )

    @staticmethod
    def _mc_stat(disc_payoff: np.ndarray,
                 control: Optional[np.ndarray] = None,
                 control_mean: Optional[float] = None) -> tuple:
        """
        (prix, erreur standard) à partir du payoff ACTUALISÉ par trajectoire.

        Si `control` (array, ex. S_T·df) et `control_mean` (espérance connue,
        ex. S·e^(−qT)) sont fournis, applique une VARIABLE DE CONTRÔLE à β
        optimal :
            X_cv = X − β·(C − E[C]),   β = Cov(X, C)/Var(C)
        Estimateur non biaisé (E[C−E[C]]=0) et de variance réduite.
        """
        n = len(disc_payoff)
        if n == 0:
            return 0.0, 0.0
        x = disc_payoff
        if control is not None and control_mean is not None:
            var_c = float(np.var(control))
            if var_c > 1e-12:
                beta = float(np.cov(x, control, bias=True)[0, 1] / var_c)
                x = x - beta * (control - control_mean)
        price  = float(np.mean(x))
        stderr = float(np.std(x) / math.sqrt(n))
        return price, stderr

    # ── Vanilles GBM ─────────────────────────────────────────────────────────

    def call(self, K: float, T: float, ratio: float = 1.0) -> PricingResult:
        S, r, q  = self.S, self._rT(T), self.q
        sigma    = self._iv(T, K)
        df       = math.exp(-r * T)

        ST    = _gbm_simulate(S, r, q, sigma, T, self.n_paths, self.antithetic, self.seed, qmc=self.qmc)
        pay   = np.maximum(ST - K, 0.0) * df * ratio
        price, stderr = self._mc_stat(pay, control=ST * df, control_mean=S * math.exp(-q * T))

        def pfn(S, r, q, sigma, T):
            _ST = _gbm_simulate(S, r, q, sigma, T, self.n_paths, self.antithetic, _GREEK_SEED, qmc=self.qmc)
            return float(np.mean(np.maximum(_ST - K, 0.0)) * math.exp(-r * T) * ratio)

        greeks = compute_greeks_fd_vol(pfn, price, S, r, q, sigma, T)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Call", model="gbm_mc",
                             mc_stderr=round(stderr, 6))

    def put(self, K: float, T: float, ratio: float = 1.0) -> PricingResult:
        S, r, q  = self.S, self._rT(T), self.q
        sigma    = self._iv(T, K)
        df       = math.exp(-r * T)

        ST    = _gbm_simulate(S, r, q, sigma, T, self.n_paths, self.antithetic, self.seed, qmc=self.qmc)
        pay   = np.maximum(K - ST, 0.0) * df * ratio
        price, stderr = self._mc_stat(pay, control=ST * df, control_mean=S * math.exp(-q * T))

        def pfn(S, r, q, sigma, T):
            _ST = _gbm_simulate(S, r, q, sigma, T, self.n_paths, self.antithetic, _GREEK_SEED, qmc=self.qmc)
            return float(np.mean(np.maximum(K - _ST, 0.0)) * math.exp(-r * T) * ratio)

        greeks = compute_greeks_fd_vol(pfn, price, S, r, q, sigma, T)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Put", model="gbm_mc",
                             mc_stderr=round(stderr, 6))

    def warrant_call(self, K: float, T: float, ratio: float = 1.0) -> PricingResult:
        S, r, q  = self.S, self._rT(T), self.q
        sigma    = self._iv(T, K)
        df       = math.exp(-r * T)

        ST    = _gbm_simulate(S, r, q, sigma, T, self.n_paths, self.antithetic, self.seed, qmc=self.qmc)
        pay   = np.maximum(ST - K, 0.0) * df * ratio
        price, stderr = self._mc_stat(pay, control=ST * df, control_mean=S * math.exp(-q * T))

        def pfn(S, r, q, sigma, T):
            _ST = _gbm_simulate(S, r, q, sigma, T, self.n_paths, self.antithetic, _GREEK_SEED, qmc=self.qmc)
            return float(np.mean(np.maximum(_ST - K, 0.0)) * math.exp(-r * T) * ratio)

        greeks = compute_greeks_fd_vol(pfn, price, S, r, q, sigma, T)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Warrant Call", model="gbm_mc",
                             mc_stderr=round(stderr, 6))

    def warrant_put(self, K: float, T: float, ratio: float = 1.0) -> PricingResult:
        S, r, q  = self.S, self._rT(T), self.q
        sigma    = self._iv(T, K)
        df       = math.exp(-r * T)

        ST    = _gbm_simulate(S, r, q, sigma, T, self.n_paths, self.antithetic, self.seed, qmc=self.qmc)
        pay   = np.maximum(K - ST, 0.0) * df * ratio
        price, stderr = self._mc_stat(pay, control=ST * df, control_mean=S * math.exp(-q * T))

        def pfn(S, r, q, sigma, T):
            _ST = _gbm_simulate(S, r, q, sigma, T, self.n_paths, self.antithetic, _GREEK_SEED, qmc=self.qmc)
            return float(np.mean(np.maximum(K - _ST, 0.0)) * math.exp(-r * T) * ratio)

        greeks = compute_greeks_fd_vol(pfn, price, S, r, q, sigma, T)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Warrant Put", model="gbm_mc",
                             mc_stderr=round(stderr, 6))

    # ── Barrières (chemin complet requis) ─────────────────────────────────────

    def bonus_certificate(
        self, X: float, B: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Bonus Certificate : ZSC + Down-and-Out Put(X, B).
        La barrière est surveillée en continu sur les trajectoires simulées.
        """
        S, r, q = self.S, self._rT(T), self.q
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

        df    = math.exp(-r * T)
        price, stderr = self._mc_stat(payoff * df, control=ST * df,
                                      control_mean=S * math.exp(-q * T))

        def pfn(S_arg, r, q, T, params=None):
            # Barrière CONSERVÉE dans le bump (plus d'approximation sans-barrière) :
            # on simule les chemins et on applique le monitoring continu.
            _ns  = max(int(252 * T), 50)
            _obs = np.linspace(T / _ns, T, _ns)
            _, _paths = heston_simulate_paths(
                S=S_arg, r=r, q=q, T=T, params=params or self.params,
                n_paths=self.n_paths // 2, observation_times=_obs,
                antithetic=self.antithetic, seed=_GREEK_SEED)
            _hit = np.any(_paths <= B, axis=1)
            _STe = _paths[:, -1]
            _payoff = np.where(_hit, _STe, np.maximum(_STe, X)) * ratio
            return float(np.mean(_payoff) * math.exp(-r * T))

        greeks = compute_greeks_fd_heston(pfn, price, S, r, q, T, self.params)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Bonus Certificate", model="heston_mc",
                             mc_stderr=round(stderr, 6))

    def reverse_bonus_certificate(
        self, X: float, B: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Reverse Bonus Certificate.
        Payoff baissier : max(2S0 - S_T, 0) si barrière non touchée → bonus garanti.
        """
        S0, r, q = self.S, self._rT(T), self.q
        n_steps  = max(int(252 * T), 50)
        obs_times = np.linspace(T / n_steps, T, n_steps)

        _, paths = self._simulate_paths(T, obs_times)

        barrier_hit = np.any(paths >= B, axis=1)
        ST          = paths[:, -1]

        payoff_no_hit = np.maximum(2 * S0 - ST, X - ST)  # bonus inverse
        payoff_hit    = np.maximum(2 * S0 - ST, 0.0)

        payoff = np.where(barrier_hit, payoff_hit, payoff_no_hit) * ratio

        df    = math.exp(-r * T)
        price, stderr = self._mc_stat(payoff * df, control=ST * df,
                                      control_mean=S0 * math.exp(-q * T))

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths // 4,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            return float(np.mean(np.maximum(2 * S0 - _ST, X - _ST)) * ratio
                         * math.exp(-r * T))

        greeks = compute_greeks_fd_heston(pfn, price, S0, r, q, T, self.params)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Reverse Bonus Certificate",
                             model="heston_mc", mc_stderr=round(stderr, 6))

    def discount_certificate(self, X: float, T: float, ratio: float = 1.0) -> PricingResult:
        """Discount Certificate : min(S_T, X)."""
        S, r, q = self.S, self._rT(T), self.q
        ST    = self._simulate(T)
        df    = math.exp(-r * T)
        price, stderr = self._mc_stat(np.minimum(ST, X) * df * ratio,
                                      control=ST * df, control_mean=S * math.exp(-q * T))

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            return float(np.mean(np.minimum(_ST, X)) * math.exp(-r * T) * ratio)

        greeks = compute_greeks_fd_heston(pfn, price, S, r, q, T, self.params)
        return PricingResult(price=round(price, 6), greeks=greeks,
                             ticker=self.snap.ticker, product="Discount Certificate",
                             model="heston_mc", mc_stderr=round(stderr, 6))

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
        from pricing_lib.pricers.analytical import _cashflow_df, _coupon_dates, _freq_map

        if start is None:
            start = self.snap.valuation_date

        S, r, q = self.S, self._rT(T), self.q
        df_T    = math.exp(-r * T)

        # Échéancier de coupons (jambe obligataire) — actualisé au taux pour que
        # le bump de taux le ré-actualise AUSSI (rho complet, pas seulement la
        # jambe de remboursement). Le nominal est remboursé via la jambe MC,
        # donc PAS compté ici (sinon double comptage).
        _pay  = _coupon_dates(start, T, freq)
        _taus = np.array([(d - self.snap.valuation_date).days / 365.25 for d in _pay])
        _taus = _taus[_taus > 0]
        _c_per = coupon * nominal / _freq_map(freq)

        def _coupons_pv(rate: float) -> float:
            return float(np.sum(_c_per * np.exp(-rate * _taus)))

        coupons_pv = _coupons_pv(r)

        # Remboursement final par MC : min(S_T, cap)·nominal/cap (variable de contrôle)
        ST         = self._simulate(T)
        ratio_conv = nominal / cap
        payoff_T   = np.where(ST >= cap, nominal, ST * ratio_conv)
        redemption, red_se = self._mc_stat(payoff_T * df_T * ratio,
                                           control=ST * df_T, control_mean=S * math.exp(-q * T))

        price  = (coupons_pv + redemption) / nominal * 100.0
        stderr = red_se / nominal * 100.0   # erreur standard MC du prix (% du nominal)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _redemption = float(np.mean(np.where(_ST >= cap, nominal, _ST * ratio_conv))
                                * math.exp(-r * T) * ratio)
            _coupons = _coupons_pv(r)   # coupons actualisés au taux BUMPÉ → rho complet
            return (_coupons + _redemption) / nominal * 100.0

        greeks = compute_greeks_fd_heston(pfn, price, S, r, q, T, self.params)
        cf_df  = _cashflow_df(nominal, coupon, freq, T, start, self.snap)

        return PricingResult(
            price=round(price, 6),
            greeks=greeks,
            ticker=self.snap.ticker,
            product=f"Reverse Convertible MC ({freq})",
            model="heston_mc",
            fair_coupon=round(coupon, 6),
            cashflows=cf_df,
            mc_stderr=round(stderr, 6),
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
        [Obsolète — conservé pour compatibilité]
        Calibrait le taux de participation via pdts (BS + vol locale).
        Remplacé par _solve_participation (auto-cohérent sur les paths Heston).
        """
        try:
            from pricing_lib.pricers.core import pdts
            fn = getattr(pdts, fn_name)
            return float(fn(*args))
        except Exception as e:
            print(f"[MC] Calibration participation ({fn_name}) échouée : {e} — part=1.0")
            return 1.0

    @staticmethod
    def _solve_participation(base: np.ndarray, option: np.ndarray,
                             par: float, df: float) -> tuple:
        """
        Participation fair AUTO-COHÉRENTE, résolue sur les MÊMES trajectoires
        Heston que celles du pricing (zéro dépendance à Black-Scholes ou à la
        vol locale).

        Le payoff de ces produits est LINÉAIRE en `part` :
            payoff = base(S_T) + part · option(S_T)
        On résout `part` tel que le prix actualisé vaille le pair :
            E[payoff]·df = par
            ⇒  part = (par − E[base]·df) / (E[option]·df)

        Le prix qui en découle vaut alors `par` par construction, et `part`
        est le vrai niveau fair sous le modèle de pricing (Heston), pas sous
        un modèle auxiliaire. base/option sont les payoffs NON scalés (ratio
        appliqué ensuite).

        Retourne (part, part_stderr) — l'erreur standard MC de la participation,
        obtenue par delta method (ratio de deux moyennes MC corrélées) :
            part = (par − μ_b)/μ_o,  ∂part/∂μ_b = −1/μ_o,  ∂part/∂μ_o = −(par−μ_b)/μ_o²
        """
        n = len(base)
        b = np.asarray(base,   dtype=float) * df   # base actualisée par trajectoire
        o = np.asarray(option, dtype=float) * df   # option actualisée par trajectoire
        mu_b = float(np.mean(b))
        mu_o = float(np.mean(o))
        if abs(mu_o) < 1e-10 or n < 2:
            return 0.0, 0.0
        part = (par - mu_b) / mu_o

        var_b  = float(np.var(b))  / n
        var_o  = float(np.var(o))  / n
        cov_bo = float(np.cov(b, o, bias=True)[0, 1]) / n
        g_b = -1.0 / mu_o
        g_o = -(par - mu_b) / (mu_o * mu_o)
        var_part = g_b * g_b * var_b + g_o * g_o * var_o + 2.0 * g_b * g_o * cov_bo
        return part, math.sqrt(max(var_part, 0.0))

    def outperformance_certificate(
        self, X: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Outperformance Certificate MC.
        Payoff = S_T + (part − 1) × max(S_T − X, 0)   [convention AL : participation TOTALE]
        Participation fair résolue AUTO-COHÉRENTE sur les trajectoires Heston
        (prix = pair par construction).
        """
        S, r, q = self.S, self._rT(T), self.q
        df = math.exp(-r * T)

        ST     = self._simulate(T)
        option = np.maximum(ST - X, 0.0)
        base   = ST - option                 # part = participation TOTALE (= 1 + incrémentale)
        part, part_se = self._solve_participation(base, option, S, df)

        payoff = (base + part * option) * ratio
        price  = float(np.mean(payoff) * df)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _opt = np.maximum(_ST - X, 0.0)
            _p = (_ST - _opt + part * _opt) * ratio       # = S_T + (part−1)·call
            return float(np.mean(_p) * math.exp(-r * T))

        greeks = compute_greeks_fd_heston(pfn, price, S, r, q, T, self.params)
        res = PricingResult(price=round(price, 6), greeks=greeks,
                            ticker=self.snap.ticker,
                            product="Outperformance Certificate", model="heston_mc")
        res.fair_participation = round(part, 6)
        res.mc_stderr = round(part_se, 6)   # incertitude MC sur la participation (prix = pair exact)
        return res

    def sprint_certificate(
        self, X: float, cap: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Sprint Certificate MC (convention AL canonique).
        Payoff = S_T + (part − 1) × (max(S_T − X, 0) − 2·max(S_T − cap, 0))
        Participation fair auto-cohérente sur les trajectoires Heston.
        """
        S, r, q = self.S, self._rT(T), self.q
        df = math.exp(-r * T)

        ST     = self._simulate(T)
        option = np.maximum(ST - X, 0.0) - 2.0 * np.maximum(ST - cap, 0.0)
        base   = ST - option
        part, part_se = self._solve_participation(base, option, S, df)

        payoff = (base + part * option) * ratio
        price  = float(np.mean(payoff) * df)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _opt = np.maximum(_ST - X, 0.0) - 2.0 * np.maximum(_ST - cap, 0.0)
            return float(np.mean((_ST - _opt + part * _opt) * ratio)
                         * math.exp(-r * T))

        greeks = compute_greeks_fd_heston(pfn, price, S, r, q, T, self.params)
        res = PricingResult(price=round(price, 6), greeks=greeks,
                            ticker=self.snap.ticker,
                            product="Sprint Certificate", model="heston_mc")
        res.fair_participation = round(part, 6)
        res.mc_stderr = round(part_se, 6)   # incertitude MC sur la participation (prix = pair exact)
        return res

    def twin_win_certificate(
        self, X: float, B: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Twin Win Certificate MC (convention AL : ZSC + (part−1)·call(X) + 2·DOP(X,B)).
        DOP = put down-and-out (paie max(X−S_T,0) si la barrière basse B n'a
        jamais été touchée) → transforme la baisse en gain tant que B tient.
        Participation fair auto-cohérente sur les trajectoires Heston.
        """
        S, r, q = self.S, self._rT(T), self.q
        n_steps   = max(int(252 * T), 50)
        obs_times = np.linspace(T / n_steps, T, n_steps)

        df = math.exp(-r * T)

        _, paths   = self._simulate_paths(T, obs_times)
        barrier_hit = np.any(paths <= B, axis=1)
        ST          = paths[:, -1]

        option = np.maximum(ST - X, 0.0)
        dop    = np.where(barrier_hit, 0.0, np.maximum(X - ST, 0.0))   # down-and-out put
        base   = ST - option + 2.0 * dop
        part, part_se = self._solve_participation(base, option, S, df)

        payoff = (base + part * option) * ratio
        price  = float(np.mean(payoff) * df)

        def pfn(S_arg, r, q, T, params=None):
            # Barrière basse CONSERVÉE dans le bump (monitoring continu sur chemins).
            _ns  = max(int(252 * T), 50)
            _obs = np.linspace(T / _ns, T, _ns)
            _, _paths = heston_simulate_paths(
                S=S_arg, r=r, q=q, T=T, params=params or self.params,
                n_paths=self.n_paths // 2, observation_times=_obs,
                antithetic=self.antithetic, seed=_GREEK_SEED)
            _hit = np.any(_paths <= B, axis=1)
            _STe = _paths[:, -1]
            _opt = np.maximum(_STe - X, 0.0)
            _dop = np.where(_hit, 0.0, np.maximum(X - _STe, 0.0))
            _p = (_STe - _opt + part * _opt + 2.0 * _dop) * ratio
            return float(np.mean(_p) * math.exp(-r * T))

        greeks = compute_greeks_fd_heston(pfn, price, S, r, q, T, self.params)
        res = PricingResult(price=round(price, 6), greeks=greeks,
                            ticker=self.snap.ticker,
                            product="Twin Win Certificate", model="heston_mc")
        res.fair_participation = round(part, 6)
        res.mc_stderr = round(part_se, 6)   # incertitude MC sur la participation (prix = pair exact)
        return res

    def airbag_certificate(
        self, X: float, B: float, T: float, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Airbag Certificate MC (convention AL : cash(X) + part·call(X) − (X/B)·put(B)).
        Capital X protégé ; au-dessus de X, participation ; sous la barrière B,
        perte amplifiée par le ratio X/B (le « dégonflage » de l'airbag).
        """
        S, r, q = self.S, self._rT(T), self.q
        df = math.exp(-r * T)

        ST = self._simulate(T)
        option = np.maximum(ST - X, 0.0)
        base   = X - (X / B) * np.maximum(B - ST, 0.0)   # cash X − (X/B)·put(B)
        part, part_se = self._solve_participation(base, option, S, df)

        payoff = (base + part * option) * ratio
        price  = float(np.mean(payoff) * df)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _p = (X - (X / B) * np.maximum(B - _ST, 0.0)
                  + part * np.maximum(_ST - X, 0.0)) * ratio
            return float(np.mean(_p) * math.exp(-r * T))

        greeks = compute_greeks_fd_heston(pfn, price, S, r, q, T, self.params)
        res = PricingResult(price=round(price, 6), greeks=greeks,
                            ticker=self.snap.ticker,
                            product="Airbag Certificate", model="heston_mc")
        res.fair_participation = round(part, 6)
        res.mc_stderr = round(part_se, 6)   # incertitude MC sur la participation (prix = pair exact)
        return res

    def garantie_certificate(
        self, X: float, T: float, nominal: float = 100.0, ratio: float = 1.0,
    ) -> PricingResult:
        """
        Garantie Certificate MC.
        Payoff = nominal + part × max(S_T − X, 0)
        """
        S, r, q = self.S, self._rT(T), self.q
        df = math.exp(-r * T)

        ST     = self._simulate(T)
        base   = np.full_like(ST, float(nominal))   # capital garanti
        option = np.maximum(ST - X, 0.0)
        part, part_se = self._solve_participation(base, option, float(nominal), df)

        payoff = (base + part * option) * ratio
        price  = float(np.mean(payoff) * df)

        def pfn(S_arg, r, q, T, params=None):
            _ST = heston_simulate(S=S_arg, r=r, q=q, T=T,
                                  params=params or self.params,
                                  n_paths=self.n_paths,
                                  antithetic=self.antithetic, seed=_GREEK_SEED)
            _p = (nominal + part * np.maximum(_ST - X, 0.0)) * ratio
            return float(np.mean(_p) * math.exp(-r * T))

        greeks = compute_greeks_fd_heston(pfn, price, S, r, q, T, self.params)
        res = PricingResult(price=round(price, 6), greeks=greeks,
                            ticker=self.snap.ticker,
                            product="Garantie Certificate", model="heston_mc")
        res.fair_participation = round(part, 6)
        res.mc_stderr = round(part_se, 6)   # incertitude MC sur la participation (prix = pair exact)
        return res

       