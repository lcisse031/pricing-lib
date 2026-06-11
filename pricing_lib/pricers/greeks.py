"""
pricing_lib/pricers/greeks.py
─────────────────────────────────────────────────────────────────────────────
Calcul des Greeks par différences finies centrées.

Compatible avec toute fonction de pricing de signature :
    price_fn(S, r, q, sigma, T, **kwargs) -> float

Pas de différentiation :
    Delta / Gamma : dS = 1% du spot
    Vega          : +1 vol point (0.01)
    Rho           : +1 bp (0.0001)
    Theta         : -1 jour calendaire
"""

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
    """
    Greeks numériques pour n'importe quelle fonction de pricing.

    Paramètres
    ----------
    price_fn  : callable(S, r, q, sigma, T, **kwargs) -> float
    price0    : prix de base déjà calculé (évite un appel supplémentaire)
    dS_frac   : pas relatif pour delta/gamma (défaut 1%)
    dv        : choc vol absolu pour vega (défaut +1 vol point)
    dr        : choc taux pour rho (défaut +1 bp)
    dT_days   : choc temporel pour theta en jours calendaires (défaut 1 jour)
    **kwargs  : paramètres supplémentaires passés à price_fn

    Retourne
    --------
    dict avec clés : delta, gamma, vega, theta, rho
    Theta est exprimé en valeur par jour calendaire.
    """
    dS = S * dS_frac
    dT = dT_days / 365.0

    # ── Spot ─────────────────────────────────────────────────────────────────
    p_up   = price_fn(S + dS, r, q, sigma, T, **kwargs)
    p_down = price_fn(S - dS, r, q, sigma, T, **kwargs)

    delta = (p_up - p_down) / (2.0 * dS)
    gamma = (p_up - 2.0 * price0 + p_down) / (dS ** 2)

    # ── Vol (vega) — différence centrée ──────────────────────────────────────
    # Bug #9 fix : centré (O(dv²)) au lieu de forward (O(dv))
    sigma_vup   = sigma + dv
    sigma_vdown = max(1e-6, sigma - dv)   # évite sigma ≤ 0 → overflow dans _barrier
    p_vega_up   = price_fn(S, r, q, sigma_vup,   T, **kwargs)
    p_vega_down = price_fn(S, r, q, sigma_vdown, T, **kwargs)
    p_vega = p_vega_up   # conservé pour compatibilité dict de retour
    vega   = (p_vega_up - p_vega_down) / (sigma_vup - sigma_vdown)

    # ── Taux (rho) ────────────────────────────────────────────────────────────
    p_rho = price_fn(S, r + dr, q, sigma, T, **kwargs)
    rho   = (p_rho - price0) / dr

    # ── Temps (theta) ─────────────────────────────────────────────────────────
    T_shifted = max(T - dT, 1e-6)
    p_theta   = price_fn(S, r, q, sigma, T_shifted, **kwargs)
    theta     = (p_theta - price0) / (-dT_days)   # variation par jour (négatif = perte de valeur)

    return {
                "p_up": p_up, 
        "p_down": p_down,  
        "p_vega": p_vega,
        "p_rho": p_rho,
        "p_th": p_theta,

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
    """
    Greeks numériques pour le pricer MC Heston.

    price_fn_mc signature : price_fn_mc(S, r, q, T, params, **kwargs) -> float

    Note : les Greeks MC sont bruités — augmenter n_paths pour les réduire.
    On utilise un pas dS plus grand (2%) pour atténuer le bruit.
    """
    dS   = S * dS_frac
    dT   = dT_days / 365.0

    # Bug #3 fix : choquer v0 ET theta pour capturer le vega court ET long terme
    # Choquer v0 seul ne donne que la sensibilité à la variance instantanée.
    # theta doit être choqué symétriquement pour représenter un déplacement
    # uniforme de la surface de vol implicite (+1 vol point).
    from pricing_lib.models.heston import HestonParams
    params_vega = HestonParams(
        kappa=params.kappa,
        theta=max(params.theta + dv, 1e-4),
        sigma_v=params.sigma_v, rho=params.rho,
        v0=max(params.v0 + dv, 1e-4),
    )

    p_up   = price_fn_mc(S + dS, r, q, T, params, **kwargs)
    p_down = price_fn_mc(S - dS, r, q, T, params, **kwargs)
    # p0_g : base recalculée avec la même graine que les bumps (seed=_GREEK_SEED dans les pfn)
    # → gamma / vega / rho / theta ne sont plus bruités par des aléas différents
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
