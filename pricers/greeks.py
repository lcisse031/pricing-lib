from __future__ import annotations

from typing import Callable, Dict, Optional


def compute_greeks(
    price_fn:  Callable[..., float],
    price0:    float,
    S:         float,
    r:         float,
    q:         float,
    sigma:     float,
    T:         float,
    dS_frac:   float = 0.01,
    dv:        float = 0.01,
    dr:        float = 0.0001,
    dT_days:   float = 1.0,
    **kwargs,
) -> Dict[str, float]:
    """Greeks numériques pour n'importe quelle fonction de pricing."""
    dS = S * dS_frac
    dT = dT_days / 365.0

    p_up   = price_fn(S + dS, r, q, sigma, T, **kwargs)
    p_down = price_fn(S - dS, r, q, sigma, T, **kwargs)
    p_vega = price_fn(S, r, q, sigma + dv, T, **kwargs)
    p_rho  = price_fn(S, r + dr, q, sigma, T, **kwargs)
    p_theta = price_fn(S, r, q, sigma, max(T - dT, 1e-6), **kwargs)

    delta = (p_up - p_down) / (2.0 * dS)
    gamma = (p_up - 2.0 * price0 + p_down) / dS ** 2
    vega  = (p_vega - price0) / dv
    rho   = (p_rho - price0) / dr
    theta = (p_theta - price0) / (-dT_days)

    return {
        "delta": round(float(delta), 6),
        "gamma": round(float(gamma), 6),
        "vega":  round(float(vega),  6),
        "theta": round(float(theta), 6),
        "rho":   round(float(rho),   6),
    }


def compute_greeks_mc(
    price_fn_mc: Callable[..., float],
    price0:      float,
    S:           float,
    r:           float,
    q:           float,
    T:           float,
    params,                   # HestonParams
    dS_frac:     float = 0.01,
    dv:          float = 0.01,
    dr:          float = 0.0001,
    dT_days:     float = 1.0,
    **kwargs,
) -> Dict[str, float]:
    """Greeks MC Heston par différences finies. Signature : price_fn_mc(S, r, q, T, params) -> float"""
    dS   = S * dS_frac
    dT   = dT_days / 365.0

    from pricing_lib.models.heston import HestonParams
    params_vega = HestonParams(
        kappa=params.kappa, theta=params.theta,
        sigma_v=params.sigma_v, rho=params.rho,
        v0=max(params.v0 + dv, 1e-4),
    )

    p_up   = price_fn_mc(S + dS, r, q, T, params, **kwargs)
    p_down = price_fn_mc(S - dS, r, q, T, params, **kwargs)
    p0_g   = price_fn_mc(S, r, q, T, params, **kwargs)
    p_vega = price_fn_mc(S, r, q, T, params_vega, **kwargs)
    p_rho  = price_fn_mc(S, r + dr, q, T, params, **kwargs)
    T_sh   = max(T - dT, 1e-6)
    p_th   = price_fn_mc(S, r, q, T_sh, params, **kwargs)

    delta = (p_up - p_down) / (2.0 * dS)
    gamma = (p_up - 2.0 * p0_g + p_down) / dS ** 2
    vega  = (p_vega - p0_g) / dv
    rho_g = (p_rho - p0_g) / dr
    theta = (p_th - p0_g) / (-dT_days)

    return {
        "delta": round(float(delta), 6),
        "gamma": round(float(gamma), 6),
        "vega":  round(float(vega),  6),
        "theta": round(float(theta), 6),
        "rho":   round(float(rho_g), 6),
    }
