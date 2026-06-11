"""
fCertificates.py
================
Python reimplementation of the R fCertificates package.

Products are split into three categories:

  CATEGORY 1 — PURE STRUCTURE
    price = f(S, X, ..., r, r_d, sigma)
    No coupon, no participation to derive.

  CATEGORY 2 — COUPON PRODUCTS
    Coupon / redemption / bonus is NEVER an explicit input.
    Public API → implied_coupon_xxx(target_price, S, ...) returns the fair coupon.

  CATEGORY 3 — PARTICIPATION PRODUCTS
    Participation rate is NEVER an explicit input.
    Public API → implied_participation_xxx(target_price, S, ...) returns the fair participation.

Pricing model: Generalized Black-Scholes (GBS) + Standard Barrier Options
               (Reiner & Rubinstein 1991 / Haug 2007 Table 4-1).
Greeks: numerical finite differences.
Solvers: Brent's method (scipy.optimize.brentq).
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Sequence, Union


# =============================================================================
#  BUILDING BLOCKS  (private)
# =============================================================================

def _gbs(flag: str, S: float, X: float, T: float,
         r: float, b: float, sigma: float) -> float:
    """
    Generalized Black-Scholes price.
    flag  : 'c' call | 'p' put
    b     : cost of carry  (b = r − r_d for equity with continuous dividend yield)
    """
    if X <= 1e-10:                                    # zero-strike call limit
        return S * np.exp((b - r) * T) if flag == "c" else 0.0
    if T <= 0:
        return max((1. if flag == "c" else -1.) * (S - X), 0.)
    sqT = sigma * np.sqrt(T)
    d1  = (np.log(S / X) + (b + .5 * sigma**2) * T) / sqT
    d2  = d1 - sqT
    eta = 1. if flag == "c" else -1.
    return max(eta * (S * np.exp((b - r) * T) * norm.cdf(eta * d1)
                      - X * np.exp(-r * T) * norm.cdf(eta * d2)), 0.)


def _barrier(flag: str, S: float, X: float, H: float, K: float,
             T: float, r: float, b: float, sigma: float) -> float:
    """
    Standard barrier option — Haug (2007) Table 4-1.
    flag : 'cdo','cdi','cuo','cui','pdo','pdi','puo','pui'
    K    : cash rebate at expiry if knocked out (0 for most structured products)
    """
    if T <= 0:
        eta       = 1. if flag[0] == "c" else -1.
        intrinsic = max(eta * (S - X), 0.)
        bt        = flag[1:]
        if   bt == "do": return intrinsic if S > H else K
        elif bt == "uo": return intrinsic if S < H else K
        elif bt == "di": return intrinsic if S <= H else 0.
        elif bt == "ui": return intrinsic if S >= H else 0.
        return intrinsic

    sigma = max(sigma, 1e-6)   # évite mu → ±∞ et HS**(2*(mu+1)) → overflow
    sqT = sigma * np.sqrt(T)
    mu  = (b - .5 * sigma**2) / sigma**2
    lam = np.sqrt(max(mu**2 + 2. * r / sigma**2, 0.))
    HS  = H / S

    x1 = np.log(S / X)          / sqT + (1. + mu) * sqT
    x2 = np.log(S / H)          / sqT + (1. + mu) * sqT
    y1 = np.log(H**2 / (S * X)) / sqT + (1. + mu) * sqT
    y2 = np.log(H / S)          / sqT + (1. + mu) * sqT
    z  = np.log(H / S)          / sqT + lam        * sqT

    eta = 1. if flag[0] == "c" else -1.
    phi = 1. if flag[1] == "d" else -1.
    N   = norm.cdf

    A  = eta * (S * np.exp((b-r)*T) * N(eta*x1)
              - X * np.exp(-r*T)    * N(eta*(x1-sqT)))
    B  = eta * (S * np.exp((b-r)*T) * N(eta*x2)
              - X * np.exp(-r*T)    * N(eta*(x2-sqT)))
    C  = eta * (S * np.exp((b-r)*T) * HS**(2*(mu+1)) * N(phi*eta*y1)
              - X * np.exp(-r*T)    * HS**(2*mu)       * N(phi*eta*(y1-sqT)))
    D  = eta * (S * np.exp((b-r)*T) * HS**(2*(mu+1)) * N(phi*eta*y2)
              - X * np.exp(-r*T)    * HS**(2*mu)       * N(phi*eta*(y2-sqT)))
    E  = K * np.exp(-r*T) * (N(eta*(x2-sqT)) - HS**(2*mu) * N(eta*(y2-sqT)))
    F_ = K * (HS**(mu+lam) * N(phi*z) + HS**(mu-lam) * N(phi*(z - 2*lam*sqT)))

    # --- 8 cases (Haug 2007 Table 4-1) ---
    if   flag == "cdi": p = (C + E)                if X >= H else (A - B + D + E)
    elif flag == "cdo": p = (A - C + F_)            if X >= H else (B - D + F_)
    elif flag == "cui": p = (A + E)                if X >= H else (B - C + D + E)
    elif flag == "cuo": p = (F_)                   if X >= H else (A - B + C - D + F_)
    elif flag == "pdi": p = (B - C + D + E)        if X >= H else (A + E)
    elif flag == "pdo": p = (A - B + C - D + F_)   if X >= H else (F_)
    elif flag == "pui": p = (A - B + D + E)        if X >= H else (C + E)
    elif flag == "puo": p = (B - D + F_)           if X >= H else (A - C + F_)
    else: raise ValueError(f"Unknown barrier flag: {flag!r}")

    return max(p, 0.)


def _con(flag: str, S: float, X: float, K: float,
         T: float, r: float, b: float, sigma: float) -> float:
    """Cash-or-Nothing option: pays K if S_T > X (call) or S_T < X (put)."""
    if T <= 0:
        return K if ((flag == "c" and S > X) or (flag == "p" and S < X)) else 0.
    sqT = sigma * np.sqrt(T)
    d2  = (np.log(S / X) + (b - .5 * sigma**2) * T) / sqT
    eta = 1. if flag == "c" else -1.
    return K * np.exp(-r * T) * norm.cdf(eta * d2)


def _cndo(S: float, H: float, K: float,
          T: float, r: float, b: float, sigma: float) -> float:
    """
    Cash-or-Nothing Down-and-Out: pays K at maturity if S never touches H.
    Used for periodic bonus payments in ReturnCertificate.
    """
    if T <= 0:
        return K if S > H else 0.
    sqT = sigma * np.sqrt(T)
    mu  = (b - .5 * sigma**2) / sigma**2
    x2  = np.log(S / H) / sqT + (1. + mu) * sqT
    y2  = np.log(H / S) / sqT + (1. + mu) * sqT
    return max(K * np.exp(-r * T) * (norm.cdf(x2 - sqT)
               - (H/S)**(2*mu) * norm.cdf(y2 - sqT)), 0.)


# =============================================================================
#  SOLVER UTILITY  (private)
# =============================================================================

def _brent(f, lo: float, hi: float, xtol: float = 1e-8) -> float:
    """Brent's method with automatic bracket widening."""
    fa = f(lo)
    for scale in [1, 2, 5, 10, 50, 200]:
        new_hi = lo + (hi - lo) * scale
        fb = f(new_hi)
        if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
            return brentq(f, lo, new_hi, xtol=xtol)
    raise ValueError(
        f"Could not bracket the root on [{lo}, {lo + (hi-lo)*200:.4g}]. "
        "Check target price and market parameters."
    )


# =============================================================================
#  CATEGORY 1 — PURE STRUCTURE PRODUCTS
#  API: price = f(S, ..., r, r_d, sigma)
# =============================================================================

def warrant(flag: str, S, X, T, r, r_d, sigma, ratio=1.) -> float:
    """Vanilla call/put warrant."""
    return _gbs(flag, S, X, T, r, r - r_d, sigma) * ratio


def straddle(S, X, T, r, r_d, sigma, ratio=1.) -> float:
    """Long straddle (call + put at the same strike)."""
    b = r - r_d
    return (_gbs("c", S, X, T, r, b, sigma) + _gbs("p", S, X, T, r, b, sigma)) * ratio


def strangle(S, X1, X2, T, r, r_d, sigma, ratio=1.) -> float:
    """Long strangle: put @ X1 + call @ X2  (X1 < X2)."""
    b = r - r_d
    return (_gbs("p", S, X1, T, r, b, sigma) + _gbs("c", S, X2, T, r, b, sigma)) * ratio


def turbo_certificate(flag: str, S, X, B, T, r, r_d, sigma, ratio=1.) -> float:
    """
    Turbo / Knock-Out Certificate.
    flag='c' → long leverage  (Down-and-Out Call, B below S)
    flag='p' → short leverage (Up-and-Out Put,   B above S)
    """
    bflag = "cdo" if flag == "c" else "puo"
    return _barrier(bflag, S, X, B, 0., T, r, r - r_d, sigma) * ratio


def turbo_strangle(S, X_call, B_call, X_put, B_put, T, r, r_d, sigma, ratio=1.) -> float:
    """Down-and-Out Call + Up-and-Out Put."""
    b = r - r_d
    c = _barrier("cdo", S, X_call, B_call, 0., T, r, b, sigma)
    p = _barrier("puo", S, X_put,  B_put,  0., T, r, b, sigma)
    return (c + p) * ratio


def discount_certificate(S, X, T, r, r_d, sigma, ratio=1.) -> float:
    """
    Discount Certificate = ZSC − Short Call(X).
    Payoff at maturity: min(S_T, X).
    """
    b = r - r_d
    return max(_gbs("c", S, 1e-12, T, r, b, sigma)
               - _gbs("c", S, X,   T, r, b, sigma), 0.) * ratio


def discount_plus_certificate(S, X, B, T, r, r_d, sigma,
                               ratio=1., barrier_active=True, barrier_hit=False) -> float:
    """Discount Plus = ZSC − Short Call(X) + Down-and-Out Put(X, B)."""
    b   = r - r_d
    zsc = _gbs("c", S, 1e-12, T, r, b, sigma)
    sc  = _gbs("c", S, X,     T, r, b, sigma)
    dop = _barrier("pdo", S, X, B, 0., T, r, b, sigma) if (barrier_active and not barrier_hit) else 0.
    return (zsc - sc + dop) * ratio


def discount_call(S, X, cap, T, r, r_d, sigma, ratio=1.) -> float:
    """Bull call spread: Long Call(X) − Short Call(cap)."""
    b = r - r_d
    return max(_gbs("c", S, X,   T, r, b, sigma)
               - _gbs("c", S, cap, T, r, b, sigma), 0.) * ratio


def discount_put(S, X, cap, T, r, r_d, sigma, ratio=1.) -> float:
    """Bear put spread: Long Put(cap) − Short Put(X)."""
    b = r - r_d
    return max(_gbs("p", S, cap, T, r, b, sigma)
               - _gbs("p", S, X,   T, r, b, sigma), 0.) * ratio


def reverse_discount_certificate(S, S0, X, T, r, r_d, sigma, ratio=1.) -> float:
    """Reverse Discount = Long Put(2·S0) − Short Put(X)."""
    b = r - r_d
    return max(_gbs("p", S, 2*S0, T, r, b, sigma)
               - _gbs("p", S, X,   T, r, b, sigma), 0.) * ratio


def reverse_discount_plus_certificate(S, S0, X, B, T, r, r_d, sigma,
                                       ratio=1., barrier_active=True) -> float:
    """Reverse Discount Plus = Long Put(2S0) − Short Put(X) + Up-and-Out Call(X, B)."""
    b   = r - r_d
    p1  = _gbs("p", S, 2*S0, T, r, b, sigma)
    p2  = _gbs("p", S, X,    T, r, b, sigma)
    uoc = _barrier("cuo", S, X, B, 0., T, r, b, sigma) if barrier_active else 0.
    return (p1 - p2 + uoc) * ratio


def bonus_certificate(S, X, B, T, r, r_d, sigma, ratio=1., barrier_hit=False) -> float:
    """
    Bonus Certificate = ZSC + Down-and-Out Put(X, B).
    X = bonus level, B = knock-out barrier (B < S < X typically).
    """
    b   = r - r_d
    zsc = _gbs("c", S, 1e-12, T, r, b, sigma)
    dop = 0. if barrier_hit else _barrier("pdo", S, X, B, 0., T, r, b, sigma)
    return (zsc + dop) * ratio


def capped_bonus_certificate(S, X, B, cap, T, r, r_d, sigma,
                              ratio=1., barrier_hit=False) -> float:
    """Capped Bonus = ZSC + Down-and-Out Put(X, B) − Short Call(cap)."""
    b   = r - r_d
    zsc = _gbs("c", S, 1e-12, T, r, b, sigma)
    dop = 0. if barrier_hit else _barrier("pdo", S, X, B, 0., T, r, b, sigma)
    sc  = _gbs("c", S, cap, T, r, b, sigma)
    return (zsc + dop - sc) * ratio


def reverse_bonus_certificate(S, S0, X, B, T, r, r_d, sigma,
                               ratio=1., barrier_hit=False) -> float:
    """Reverse Bonus = Long Put(2·S0) + Up-and-Out Call(X, B)."""
    b   = r - r_d
    p1  = _gbs("p", S, 2*S0, T, r, b, sigma)
    uoc = 0. if barrier_hit else _barrier("cuo", S, X, B, 0., T, r, b, sigma)
    return (p1 + uoc) * ratio


def capped_reverse_bonus_certificate(S, S0, X, B, cap, T, r, r_d, sigma,
                                      ratio=1., barrier_hit=False) -> float:
    """Capped Reverse Bonus = Long Put(2S0) + Up-and-Out Call(X,B) − Short Put(cap)."""
    b   = r - r_d
    p1  = _gbs("p", S, 2*S0, T, r, b, sigma)
    uoc = 0. if barrier_hit else _barrier("cuo", S, X, B, 0., T, r, b, sigma)
    sp  = _gbs("p", S, cap, T, r, b, sigma)
    return (p1 + uoc - sp) * ratio


def leveraged_bonus_certificate(S, X, B, B2, T, r, r_d, sigma,
                                 ratio=1., barrier_hit=False) -> float:
    """Leveraged Bonus = Down-and-Out Call(B2, B2) + Down-and-Out Put(X, B)."""
    b   = r - r_d
    doc = _barrier("cdo", S, B2, B2, 0., T, r, b, sigma)
    dop = 0. if barrier_hit else _barrier("pdo", S, X, B, 0., T, r, b, sigma)
    return (doc + dop) * ratio


# =============================================================================
#  CATEGORY 2 — COUPON PRODUCTS
#  Coupon / redemption / bonus is NEVER an explicit input.
#  Public API: implied_coupon_xxx(target_price, ...) → fair coupon
# =============================================================================

# ── internal pricers (coupon as explicit arg) ─────────────────────────────────

def _rc(S, cap, T, r, r_d, sigma, nominal, coupon) -> float:
    """Reverse Convertible — price as % of nominal."""
    bond  = nominal * (1. + coupon * T) / (1. + r) ** T
    sp    = _gbs("p", S, cap, T, r, r - r_d, sigma)
    return (bond - sp * (nominal / cap)) / nominal * 100.


def _rc_plus_pro(S, cap, B, T, r, r_d, sigma, nominal, coupon,
                 barrier_hit=False) -> float:
    """Reverse Convertible Plus Pro — price as % of nominal."""
    b     = r - r_d
    ratio = nominal / cap
    bond  = nominal * (1. + coupon * T) / (1. + r) ** T
    sp    = _gbs("p", S, cap, T, r, b, sigma)
    dop   = 0. if barrier_hit else _barrier("pdo", S, cap, B, 0., T, r, b, sigma)
    return (bond - sp * ratio + dop * ratio) / nominal * 100.


def _easy_express(S, B, T, r, r_d, sigma, S0) -> float:
    """Easy Express Certificate — price given redemption level S0."""
    b     = r - r_d
    p_con = _con("p", S, B, S0 - B, T, r, b, sigma)
    sp    = _gbs("p", S, B,   T, r, b, sigma)
    return S0 * np.exp(-r * T) - p_con - sp


def _return_cert(S, B, cap, obs_times, T, r, r_d, sigma,
                 bonus, barrier_hit=False) -> float:
    """Return Certificate — price given (uniform) bonus per observation."""
    b    = r - r_d
    zsc  = _gbs("c", S, 1e-12, T, r, b, sigma)
    sc   = _gbs("c", S, cap,   T, r, b, sigma)
    if barrier_hit:
        return zsc - sc
    times   = list(obs_times) if hasattr(obs_times, "__iter__") else [T]
    bonuses = ([bonus] * len(times) if not hasattr(bonus, "__iter__") else list(bonus))
    bpv = sum(_cndo(S, B, k_i, t_i, r, b, sigma) for t_i, k_i in zip(times, bonuses))
    return zsc - sc + bpv


# ── public API ────────────────────────────────────────────────────────────────

def implied_coupon_reverse_convertible(
    target_price_pct: float,
    S: float, cap: float, T: float, r: float, r_d: float, sigma: float,
    nominal: float = 100.,
    bracket: tuple = (0., 2.)
) -> float:
    """
    Implied annual coupon rate for a Reverse Convertible.

    Parameters
    ----------
    target_price_pct : fair value as % of nominal (e.g. 100.0 at issuance)
    bracket          : search interval for coupon (default: 0 % – 200 % p.a.)

    Returns
    -------
    coupon : float  e.g. 0.08 = 8 % p.a.
    """
    f = lambda c: _rc(S, cap, T, r, r_d, sigma, nominal, c) - target_price_pct
    return _brent(f, *bracket)


def implied_coupon_rc_plus_pro(
    target_price_pct: float,
    S: float, cap: float, B: float, T: float, r: float, r_d: float, sigma: float,
    nominal: float = 100., barrier_hit: bool = False,
    bracket: tuple = (0., 2.)
) -> float:
    """
    Implied annual coupon for a Reverse Convertible Plus Pro.
    """
    f = lambda c: _rc_plus_pro(S, cap, B, T, r, r_d, sigma, nominal, c, barrier_hit) - target_price_pct
    return _brent(f, *bracket)


def implied_redemption_easy_express(
    target_price: float,
    S: float, B: float, T: float, r: float, r_d: float, sigma: float,
) -> float:
    """
    Implied redemption level S0 for an Easy Express Certificate.

    Returns
    -------
    S0 : float — fair maximum payout at maturity
    """
    f = lambda s0: _easy_express(S, B, T, r, r_d, sigma, s0) - target_price
    return _brent(f, S * 0.5, S * 3.)


def implied_bonus_return_certificate(
    target_price: float,
    S: float, B: float, cap: float,
    obs_times: Union[float, Sequence[float]],
    T: float, r: float, r_d: float, sigma: float,
    barrier_hit: bool = False,
) -> float:
    """
    Implied (uniform) bonus payment per observation date for a Return Certificate.

    Returns
    -------
    bonus : float — fair bonus per observation period
    """
    f = lambda bon: _return_cert(S, B, cap, obs_times, T, r, r_d, sigma, bon, barrier_hit) - target_price
    return _brent(f, 0., S * 0.5)


# =============================================================================
#  CATEGORY 3 — PARTICIPATION PRODUCTS
#  Participation rate is NEVER an explicit input.
#  Public API: implied_participation_xxx(target_price, ...) → fair participation
# =============================================================================

# ── internal pricers (participation as explicit arg) ──────────────────────────

def _outperformance(S, X, T, r, r_d, sigma, part) -> float:
    b = r - r_d
    return _gbs("c", S, 1e-12, T, r, b, sigma) + _gbs("c", S, X, T, r, b, sigma) * (part - 1.)


def _capped_outperformance(S, X, cap, T, r, r_d, sigma, part) -> float:
    b   = r - r_d
    zsc = _gbs("c", S, 1e-12, T, r, b, sigma)
    c   = _gbs("c", S, X,   T, r, b, sigma) * (part - 1.)
    sc  = _gbs("c", S, cap, T, r, b, sigma) * (part - 1.)
    return zsc + c - 2. * sc


def _outperformance_plus(S, X, B, T, r, r_d, sigma, part, barrier_hit=False) -> float:
    b   = r - r_d
    zsc = _gbs("c", S, 1e-12, T, r, b, sigma)
    c   = _gbs("c", S, X,     T, r, b, sigma) * (part - 1.)
    dop = 0. if barrier_hit else _barrier("pdo", S, X, B, 0., T, r, b, sigma)
    return zsc + c + dop


def _sprint(S, X, cap, T, r, r_d, sigma, part) -> float:
    b   = r - r_d
    zsc = _gbs("c", S, 1e-12, T, r, b, sigma)
    lc  = _gbs("c", S, X,   T, r, b, sigma) * (part - 1.)
    sc  = _gbs("c", S, cap, T, r, b, sigma) * (part - 1.)
    return zsc + lc - 2. * sc


def _twin_win(S, X, B, T, r, r_d, sigma, part) -> float:
    b   = r - r_d
    zsc = _gbs("c", S, 1e-12, T, r, b, sigma)
    c   = _gbs("c", S, X,     T, r, b, sigma) * (part - 1.)
    dop = _barrier("pdo", S, X, B, 0., T, r, b, sigma)
    return zsc + c + 2. * dop


def _reverse_outperformance(S, S0, X, T, r, r_d, sigma, part) -> float:
    b  = r - r_d
    p1 = _gbs("p", S, 2*S0, T, r, b, sigma)
    p2 = _gbs("p", S, X,    T, r, b, sigma) * (part - 1.)
    return p1 + p2


def _garantie(S, X, T, r, r_d, sigma, part, nominal=100.) -> float:
    b    = r - r_d
    bond = nominal * np.exp(-r * T)
    c    = _gbs("c", S, X, T, r, b, sigma) * part
    return bond + c


def _airbag(S, X, B, T, r, r_d, sigma, part) -> float:
    b    = r - r_d
    cash = X * np.exp(-r * T)
    lc   = _gbs("c", S, X, T, r, b, sigma) * part
    sp   = _gbs("p", S, B, T, r, b, sigma) * (X / B)
    return cash + lc - sp


def _airbag_plus(S, X, B, B2, T, r, r_d, sigma, part, barrier_hit=False) -> float:
    b    = r - r_d
    cash = X * np.exp(-r * T)
    lc   = _gbs("c", S, X, T, r, b, sigma) * part
    sp   = _gbs("p", S, B, T, r, b, sigma) * (X / B)
    dop  = 0. if barrier_hit else _barrier("pdo", S, X, B2, 0., T, r, b, sigma)
    return cash + lc - sp + dop


# ── public API ────────────────────────────────────────────────────────────────

def implied_participation_outperformance(
    target_price: float,
    S: float, X: float, T: float, r: float, r_d: float, sigma: float,
    bracket: tuple = (1., 10.)
) -> float:
    """Implied participation for an Outperformance Certificate."""
    f = lambda p: _outperformance(S, X, T, r, r_d, sigma, p) - target_price
    return _brent(f, *bracket)


def implied_participation_capped_outperformance(
    target_price: float,
    S: float, X: float, cap: float, T: float, r: float, r_d: float, sigma: float,
    bracket: tuple = (1., 10.)
) -> float:
    """Implied participation for a Capped Outperformance Certificate."""
    f = lambda p: _capped_outperformance(S, X, cap, T, r, r_d, sigma, p) - target_price
    return _brent(f, *bracket)


def implied_participation_outperformance_plus(
    target_price: float,
    S: float, X: float, B: float, T: float, r: float, r_d: float, sigma: float,
    barrier_hit: bool = False, bracket: tuple = (1., 10.)
) -> float:
    """Implied participation for an Outperformance Plus Certificate."""
    f = lambda p: _outperformance_plus(S, X, B, T, r, r_d, sigma, p, barrier_hit) - target_price
    return _brent(f, *bracket)


def implied_participation_sprint(
    target_price: float,
    S: float, X: float, cap: float, T: float, r: float, r_d: float, sigma: float,
    bracket: tuple = (1., 10.)
) -> float:
    """Implied participation for a Sprint (Double Chance) Certificate."""
    f = lambda p: _sprint(S, X, cap, T, r, r_d, sigma, p) - target_price
    return _brent(f, *bracket)


def implied_participation_twin_win(
    target_price: float,
    S: float, X: float, B: float, T: float, r: float, r_d: float, sigma: float,
    bracket: tuple = (1., 10.)
) -> float:
    """Implied participation for a Twin-Win Certificate."""
    f = lambda p: _twin_win(S, X, B, T, r, r_d, sigma, p) - target_price
    return _brent(f, *bracket)


def implied_participation_reverse_outperformance(
    target_price: float,
    S: float, S0: float, X: float, T: float, r: float, r_d: float, sigma: float,
    bracket: tuple = (1., 10.)
) -> float:
    """Implied participation for a Reverse Outperformance Certificate."""
    f = lambda p: _reverse_outperformance(S, S0, X, T, r, r_d, sigma, p) - target_price
    return _brent(f, *bracket)


def implied_participation_garantie(
    target_price: float,
    S: float, X: float, T: float, r: float, r_d: float, sigma: float,
    nominal: float = 100., bracket: tuple = (0., 10.)
) -> float:
    """Implied participation for a Guarantee Certificate."""
    f = lambda p: _garantie(S, X, T, r, r_d, sigma, p, nominal) - target_price
    return _brent(f, *bracket)


def implied_participation_airbag(
    target_price: float,
    S: float, X: float, B: float, T: float, r: float, r_d: float, sigma: float,
    bracket: tuple = (0., 10.)
) -> float:
    """Implied participation for an Airbag Certificate."""
    f = lambda p: _airbag(S, X, B, T, r, r_d, sigma, p) - target_price
    return _brent(f, *bracket)


def implied_participation_airbag_plus(
    target_price: float,
    S: float, X: float, B: float, B2: float,
    T: float, r: float, r_d: float, sigma: float,
    barrier_hit: bool = False, bracket: tuple = (0., 10.)
) -> float:
    """Implied participation for an Airbag Plus Certificate."""
    f = lambda p: _airbag_plus(S, X, B, B2, T, r, r_d, sigma, p, barrier_hit) - target_price
    return _brent(f, *bracket)


# =============================================================================
#  GREEKS  — numerical finite differences (works on any pricing function)
# =============================================================================

def greeks(price_fn, S, sigma, r, T,
           dS=0.01, dv=1e-3, dr=1e-3, dT=1/365, **kw) -> dict:
    """
    Numerical Greeks for any certificate pricing function.

    Parameters
    ----------
    price_fn : callable with signature price_fn(S, sigma, r, T, **kw)
    dS, dv, dr, dT : finite-difference step sizes

    Returns
    -------
    dict with keys: delta, gamma, vega, rho, theta
    """
    p0 = price_fn(S=S,     sigma=sigma,    r=r,    T=T,    **kw)
    pu = price_fn(S=S+dS,  sigma=sigma,    r=r,    T=T,    **kw)
    pd = price_fn(S=S-dS,  sigma=sigma,    r=r,    T=T,    **kw)
    pv = price_fn(S=S,     sigma=sigma+dv, r=r,    T=T,    **kw)
    pr = price_fn(S=S,     sigma=sigma,    r=r+dr, T=T,    **kw)
    pt = price_fn(S=S,     sigma=sigma,    r=r,    T=T+dT, **kw)
    return {
        "delta": (pu - pd) / (2 * dS),
        "gamma": (pu - 2*p0 + pd) / dS**2,
        "vega" : (pv - p0) / dv,
        "rho"  : (pr - p0) / dr,
        "theta": (pt - p0) / dT,
    }


# =============================================================================
#  IMPLIED VOLATILITY  — works on any pricing function
# =============================================================================

def implied_volatility(market_price: float, price_fn,
                        bracket=(0.001, 5.), **kw) -> float:
    """
    Implied volatility for any certificate pricing function.

    Parameters
    ----------
    market_price : observed market price
    price_fn     : callable  price_fn(sigma=..., **kw)
    bracket      : search interval for sigma (default: 0.1 % – 500 %)
    """
    f = lambda sig: price_fn(sigma=sig, **kw) - market_price
    return _brent(f, *bracket)


# =============================================================================
#  QUICK USAGE EXAMPLES
# =============================================================================
if __name__ == "__main__":
    # ── Market parameters ──────────────────────────────────────────────
    S, r, r_d, sigma, T = 100., 0.04, 0.01, 0.20, 1.0

    # ── Category 1: pure pricing ───────────────────────────────────────
    bc  = bonus_certificate(S=S, X=110., B=80., T=T, r=r, r_d=r_d, sigma=sigma)
    dc  = discount_certificate(S=S, X=110., T=T, r=r, r_d=r_d, sigma=sigma)
    print(f"Bonus Certificate       : {bc:.4f}")
    print(f"Discount Certificate    : {dc:.4f}")

    # ── Category 2: implied coupon ─────────────────────────────────────
    coupon = implied_coupon_reverse_convertible(
        target_price_pct=100., S=S, cap=100., T=T,
        r=r, r_d=r_d, sigma=sigma, nominal=1000.
    )
    print(f"RC implied coupon       : {coupon*100:.2f}% p.a.")

    redemption = implied_redemption_easy_express(
        target_price=S, S=S, B=85., T=T, r=r, r_d=r_d, sigma=sigma
    )
    print(f"Easy Express implied S0 : {redemption:.4f}")

    # ── Category 3: implied participation ─────────────────────────────
    part = implied_participation_outperformance(
        target_price=S, S=S, X=S, T=T, r=r, r_d=r_d, sigma=sigma
    )
    print(f"Outperformance implied participation: {part*100:.1f}%")

    # ── Greeks on a bonus certificate ─────────────────────────────────
    from functools import partial
    fn = partial(bonus_certificate, X=110., B=80., r_d=r_d)
    g  = greeks(fn, S=S, sigma=sigma, r=r, T=T)
    print(f"Bonus cert Greeks: {g}")