"""
pricing_lib/risk/greeks.py
─────────────────────────────────────────────────────────────────────────────
Moteur de Greeks unifié — spec « Pricing & Greeks ».

Trois voies, une seule convention de sortie :

  1. compute_greeks_bs        — AL vanille (Call/Put/Warrant).
                                100 % formules fermées Black-Scholes.
                                Aucun bump, aucun bloc p_*.

  2. compute_greeks_fd_vol    — différences finies CENTRÉES, vol = paramètre σ.
                                Utilisé par :
                                  • AL exotiques (repricing déterministe)
                                  • MC vanilles GBM (repricing simulé, seed fixe)
                                price_fn(S, r, q, sigma, T) -> float

  3. compute_greeks_fd_heston — différences finies CENTRÉES, vol = paramètres Heston.
                                Utilisé par les exotiques MC (Heston).
                                Vega = bump simultané v0 ET θ (choc de niveau de variance).
                                price_fn(S, r, q, T, params) -> float

Convention de différences finies (identique AL et MC, cf. décision projet) :

    delta = (p_S_up   − p_S_down)  / (2·dS)
    gamma = (p_S_up   − 2·p_0 + p_S_down) / dS²
    vega  = (p_vol_up − p_vol_down)/ (2·dσ)          [voir note vega ci-dessous]
    rho   = (p_r_up   − p_r_down)  / (2·dr)  · 1e-4   [pour +1 bp]
    theta = (p_T_down − p_T_up)    / (2·dT_j)         [par jour calendaire]

Le bloc de prix intermédiaires renvoyé (p_0, p_S_up, p_S_down, p_vol_up,
p_vol_down, p_r_up, p_r_down, p_T_up, p_T_down) rend chaque Greek recomposable
à la main → sortie auto-vérifiable.

Note vega (unité homogène : +1 point de vol partout) :
    • Vanilles (fd_vol)  : bump de σ de ±1 pt, vega × 0.01.
    • Heston  (fd_heston): shift parallèle de ±1 pt de vol sur √v0 ET √θ
                           (reconvertis en variance), vega × 0.01.
      → Même unité que le vega vanille : sensibilité à un déplacement de +1
        point du niveau de vol Heston (vol initiale et vol long terme).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable, Dict, Optional

from scipy.stats import norm

_N = norm.cdf      # fonction de répartition  N(·)
_n = norm.pdf      # densité                  N'(·) = φ(·)

_GREEK_KEYS = ("delta", "gamma", "vega", "theta", "rho")
_PRICE_KEYS = (
    "p_0",
    "p_S_up", "p_S_down",
    "p_vol_up", "p_vol_down",
    "p_r_up", "p_r_down",
    "p_T_up", "p_T_down",
)


# ─── 1. AL vanille — Greeks fermés Black-Scholes ─────────────────────────────

def compute_greeks_bs(
    flag: str,
    S: float, K: float, T: float,
    r: float, q: float, sigma: float,
    ratio: float = 1.0,
) -> Dict[str, float]:
    """
    Greeks fermés d'une option vanille européenne (Garman-Kohlhagen, b = r − q).

    flag  : 'c' (call) | 'p' (put)
    ratio : quotité / rapport de souscription, appliqué à tous les Greeks.

    Conventions :
        delta : ∂V/∂S
        gamma : ∂²V/∂S²
        vega  : ∂V/∂σ   pour +1 point de vol     (× 0.01)
        theta : ∂V/∂t   par jour calendaire        (÷ 365)
        rho   : ∂V/∂r   pour +1 bp de taux         (× 0.0001)
    """
    flag = flag.lower()
    if flag not in ("c", "p"):
        raise ValueError(f"flag attendu 'c' ou 'p', reçu {flag!r}")

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    sqrtT = math.sqrt(T)
    sqT   = sigma * sqrtT
    d1    = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sqT
    d2    = d1 - sqT

    eqT = math.exp(-q * T)
    erT = math.exp(-r * T)
    phi = _n(d1)

    gamma = eqT * phi / (S * sqT)
    vega  = S * eqT * phi * sqrtT

    if flag == "c":
        delta = eqT * _N(d1)
        theta = (-S * eqT * phi * sigma / (2.0 * sqrtT)
                 - r * K * erT * _N(d2)
                 + q * S * eqT * _N(d1))
        rho = K * T * erT * _N(d2)
    else:
        delta = -eqT * _N(-d1)
        theta = (-S * eqT * phi * sigma / (2.0 * sqrtT)
                 + r * K * erT * _N(-d2)
                 - q * S * eqT * _N(-d1))
        rho = -K * T * erT * _N(-d2)

    return {
        "delta": round(ratio * delta,         6),
        "gamma": round(ratio * gamma,         6),
        "vega":  round(ratio * vega * 0.01,   6),
        "theta": round(ratio * theta / 365.0, 6),
        "rho":   round(ratio * rho * 0.0001,  6),
    }


# ─── Assemblage du bloc commun (prix intermédiaires + Greeks) ────────────────

def _assemble(
    p_0: float,
    p_S_up: float, p_S_down: float, dS: float,
    p_vol_up: float, p_vol_down: float, dvol_eff: float, vega_scale: float,
    p_r_up: float, p_r_down: float, dr_eff: float,
    p_T_up: float, p_T_down: float, dT_days_eff: float,
) -> Dict[str, float]:
    """
    Recompose le bloc unifié (prix p_* + 5 Greeks centrés) à partir des
    repricings. `vega_scale` = 0.01 (par point de vol) pour le bump σ vanille,
    1.0 pour le bump de variance Heston.
    """
    delta = (p_S_up - p_S_down) / (2.0 * dS)
    gamma = (p_S_up - 2.0 * p_0 + p_S_down) / (dS * dS)
    vega  = (p_vol_up - p_vol_down) / dvol_eff * vega_scale
    rho   = (p_r_up - p_r_down) / dr_eff * 0.0001
    theta = (p_T_down - p_T_up) / dT_days_eff   # par jour calendaire

    return {
        "p_0":        round(p_0,        6),
        "p_S_up":     round(p_S_up,     6),
        "p_S_down":   round(p_S_down,   6),
        "p_vol_up":   round(p_vol_up,   6),
        "p_vol_down": round(p_vol_down, 6),
        "p_r_up":     round(p_r_up,     6),
        "p_r_down":   round(p_r_down,   6),
        "p_T_up":     round(p_T_up,     6),
        "p_T_down":   round(p_T_down,   6),
        "delta":      round(delta,      6),
        "gamma":      round(gamma,      6),
        "vega":       round(vega,       6),
        "theta":      round(theta,      6),
        "rho":        round(rho,        6),
    }


# ─── 2. FD centré, vol = σ (AL exotiques + MC vanilles GBM) ──────────────────

def compute_greeks_fd_vol(
    price_fn: Callable[[float, float, float, float, float], float],
    price0: float,
    S: float, r: float, q: float, sigma: float, T: float,
    *,
    dS_frac: float = 0.01,
    dvol: float = 0.01,
    dr: float = 1e-4,
    dT_days: float = 1.0,
) -> Dict[str, float]:
    """
    Greeks par différences finies CENTRÉES, le paramètre de vol étant σ.

    price_fn(S, r, q, sigma, T) -> float
    price0 : conservé pour compatibilité ; p_0 est repricé avec price_fn pour
             que base et chocs partagent le même tirage en MC.
    Le `ratio`/quotité doit déjà être incorporé dans price_fn.

    Mode d'emploi :
      • AL exotique : price_fn = repricing analytique déterministe.
      • MC vanille  : price_fn = repricing GBM avec le MÊME seed que p_0.
    """
    dS   = S * dS_frac
    dT   = dT_days / 365.0
    sv_dn = max(sigma - dvol, 1e-6)

    p_0        = price_fn(S, r, q, sigma, T)
    p_S_up     = price_fn(S + dS, r,      q, sigma,      T)
    p_S_down   = price_fn(S - dS, r,      q, sigma,      T)
    p_vol_up   = price_fn(S,      r,      q, sigma + dvol, T)
    p_vol_down = price_fn(S,      r,      q, sv_dn,        T)
    p_r_up     = price_fn(S,      r + dr, q, sigma,      T)
    p_r_down   = price_fn(S,      r - dr, q, sigma,      T)
    T_dn       = max(T - dT, 1e-6)
    p_T_up     = price_fn(S,      r,      q, sigma,      T + dT)
    p_T_down   = price_fn(S,      r,      q, sigma,      T_dn)

    dvol_eff    = (sigma + dvol) - sv_dn        # = 2·dvol sauf clipping
    dT_days_eff = ((T + dT) - T_dn) * 365.0     # = 2·dT_days sauf clipping

    return _assemble(
        p_0,
        p_S_up, p_S_down, dS,
        p_vol_up, p_vol_down, dvol_eff, 0.01,    # vega par +1 pt de vol
        p_r_up, p_r_down, 2.0 * dr,
        p_T_up, p_T_down, dT_days_eff,
    )


# ─── 3. FD centré, vol = paramètres Heston (exotiques MC) ────────────────────

def compute_greeks_fd_heston(
    price_fn: Callable[..., float],
    price0: float,
    S: float, r: float, q: float, T: float,
    params,
    *,
    dS_frac: float = 0.01,
    dv: float = 0.01,
    dr: float = 1e-4,
    dT_days: float = 1.0,
) -> Dict[str, float]:
    """
    Greeks par différences finies CENTRÉES sous Heston.

    price_fn(S, r, q, T, params) -> float
    price0 : conservé pour compatibilité (p_0 est repricé).

    Vega : choc de NIVEAU DE VOL — shift parallèle de ±dv POINTS de vol
           appliqué à √v0 ET √θ, puis reconverti en variance (v = σ²).
           Rapporté par +1 point de vol (× 0.01), MÊME unité que le vega vanille.

    price_fn doit utiliser le même seed pour tous les repricings.
    """
    dS = S * dS_frac
    dT = dT_days / 365.0

    p_0      = price_fn(S, r, q, T, params)
    p_S_up   = price_fn(S + dS, r,      q, T,      params)
    p_S_down = price_fn(S - dS, r,      q, T,      params)

    # Shift parallèle en VOL : ±dv points sur √v0 et √θ, reconvertis en variance.
    vol_v0 = math.sqrt(max(params.v0,    0.0))
    vol_th = math.sqrt(max(params.theta, 0.0))
    v0_up, v0_dn = (vol_v0 + dv) ** 2, max(vol_v0 - dv, 1e-4) ** 2
    th_up, th_dn = (vol_th + dv) ** 2, max(vol_th - dv, 1e-4) ** 2
    params_vup = replace(params, v0=v0_up, theta=th_up)
    params_vdn = replace(params, v0=v0_dn, theta=th_dn)
    p_vol_up   = price_fn(S,      r,      q, T,      params_vup)
    p_vol_down = price_fn(S,      r,      q, T,      params_vdn)

    p_r_up   = price_fn(S,      r + dr, q, T,      params)
    p_r_down = price_fn(S,      r - dr, q, T,      params)

    T_dn     = max(T - dT, 1e-6)
    p_T_up   = price_fn(S,      r,      q, T + dT, params)
    p_T_down = price_fn(S,      r,      q, T_dn,   params)
    dT_days_eff = ((T + dT) - T_dn) * 365.0

    return _assemble(
        p_0,
        p_S_up, p_S_down, dS,
        p_vol_up, p_vol_down, 2.0 * dv, 0.01,    # vega Heston par +1 point de vol
        p_r_up, p_r_down, 2.0 * dr,
        p_T_up, p_T_down, dT_days_eff,
    )


# Alias historique
compute_greeks_gbm_vanilla = compute_greeks_fd_vol


# ─── Marché bumpé (pour sensitivities : Stress / ScenarioGrid) ───────────────

class _BumpedOIS:
    """Courbe OIS décalée additivement de `rate_bump` (taux continu)."""

    def __init__(self, base, rate_bump: float = 0.0) -> None:
        self._base = base
        self._b = rate_bump

    def df_tau(self, tau: float) -> float:
        return self._base.df_tau(tau) * math.exp(-self._b * tau)

    def zero_rate_tau(self, tau: float) -> float:
        return self._base.zero_rate_tau(tau) + self._b

    def __getattr__(self, name):
        return getattr(self._base, name)


class _BumpedMarket:
    """
    Vue « choquée » d'un MarketSnapshot, sans modifier l'original.

    spot       : nouveau spot absolu (None = inchangé)
    sigma_bump : choc ADDITIF sur toute la surface de vol (+0.05 = +5 pts)
    rate_bump  : choc ADDITIF sur le taux continu (+0.01 = +100 bp)

    Expose la même interface que MarketSnapshot pour les pricers
    (spot, r, q, vol_surface, ois_curve, div_curve, valuation_date, ticker).
    """

    def __init__(self, base, spot: Optional[float] = None,
                 sigma_bump: float = 0.0, rate_bump: float = 0.0) -> None:
        self._base          = base
        self.vol_shift      = float(sigma_bump or 0.0)
        self.ticker         = base.ticker
        self.valuation_date = base.valuation_date
        self.div_curve      = base.div_curve
        self.spot           = base.spot if spot is None else float(spot)
        self.ois_curve      = _BumpedOIS(base.ois_curve, rate_bump) if rate_bump else base.ois_curve

        if sigma_bump:
            from pricing_lib.market_data.vol_surface import VolSurface
            bs = base.vol_surface
            self.vol_surface = VolSurface(
                bs.valuation_date, bs.spot, bs.r, bs.q,
                bs.tenors, bs.strikes, bs.vols + sigma_bump,
            )
        else:
            self.vol_surface = base.vol_surface

    @property
    def r(self) -> float:
        return self.ois_curve.zero_rate_tau(1.0)

    @property
    def q(self) -> float:
        return self.div_curve.q

    def sigma(self, T: float, K: float) -> float:
        return self.vol_surface.vol(T, K)

    def atm_vol(self, T: float) -> float:
        return self.vol_surface.atm_vol(T)

    def df(self, tau: float) -> float:
        return self.ois_curve.df_tau(tau)
