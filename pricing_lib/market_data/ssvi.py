"""
pricing_lib/market_data/ssvi.py
─────────────────────────────────────────────────────────────────────────────
Surface de volatilité SSVI sans-arbitrage  (Gatheral-Jacquier, 2014).

Pourquoi
────────
La surface brute interpolée depuis Euronext est bruitée et peut violer la
non-arbitrage (calendar / butterfly). En aval, Dupire produit alors des
variances locales négatives (d'où les replis ATM) et la calibration Heston
part en coin. SSVI fournit une surface :
  • lisse et paramétrique (3 paramètres de smile + structure par terme ATM),
  • garantie sans-arbitrage si les conditions de Gatheral-Jacquier sont tenues,
  • dont la vol locale de Dupire est ANALYTIQUE (dérivées en strike fermées),
    donc stable.

Paramétrisation (power-law SSVI)
────────────────────────────────
Variance totale  w(k) = σ_imp(k)²·T  en log-moneyness k = ln(K/F) :

    w(k ; θ) = (θ/2) · [ 1 + ρ·φ·k + √( (φ·k + ρ)² + (1 − ρ²) ) ]

avec θ = θ_T la variance totale ATM (structure par terme, pinnée au marché),
et  φ = φ(θ) = η · θ^(−γ)   (power-law).

Paramètres de smile : ρ (corrélation/skew), η (niveau de courbure), γ (decay).

Non-arbitrage (conditions suffisantes Gatheral-Jacquier)
───────────────────────────────────────────────────────
  • Calendar : θ_T croissante en T.
  • Butterfly: θφ(1+|ρ|) ≤ 4  et  θφ²(1+|ρ|) ≤ 4   pour tout θ.

Interface
─────────
calibrate_ssvi(...)  → (SSVIParams, theta_nodes)
SSVISurface          → vol(T,K), atm_vol(T), local_vol(T,K) [Dupire analytique],
                       + attributs compatibles VolSurface (tenors/strikes/vols).
ssvi_from_grid(surf) → construit une SSVISurface depuis une VolSurface brute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# ─── Noyau SSVI ───────────────────────────────────────────────────────────────

# Borne butterfly de Gatheral-Jacquier (4) avec une petite marge
_BF_TARGET = 3.92


def _phi(theta: float, eta: float, gamma: float) -> float:
    """φ(θ) = η·θ^(−γ)  (power-law)."""
    return eta / (max(theta, 1e-8) ** gamma)


def ssvi_total_variance(k, theta: float, rho: float, eta: float, gamma: float):
    """Variance totale w(k ; θ). k scalaire ou np.ndarray."""
    k   = np.asarray(k, dtype=float)
    phi = _phi(theta, eta, gamma)
    u   = phi * k + rho
    return 0.5 * theta * (1.0 + rho * phi * k + np.sqrt(u * u + (1.0 - rho * rho)))


def _w_and_k_derivs(k: float, theta: float, rho: float, eta: float, gamma: float):
    """
    Retourne (w, ∂w/∂k, ∂²w/∂k²) ANALYTIQUES à θ fixe.

    Avec u = φk+ρ, D = √(u² + 1−ρ²) :
        w     = (θ/2)(1 + ρφk + D)
        w_k   = (θφ/2)(ρ + u/D)
        w_kk  = (θφ²(1−ρ²)) / (2 D³)
    """
    phi = _phi(theta, eta, gamma)
    u   = phi * k + rho
    D   = math.sqrt(u * u + (1.0 - rho * rho))
    w   = 0.5 * theta * (1.0 + rho * phi * k + D)
    w_k = 0.5 * theta * phi * (rho + u / D)
    w_kk = 0.5 * theta * phi * phi * (1.0 - rho * rho) / (D ** 3)
    return w, w_k, w_kk


def butterfly_free(theta: float, rho: float, eta: float, gamma: float) -> bool:
    """Conditions suffisantes de non-arbitrage papillon (Gatheral-Jacquier)."""
    phi = _phi(theta, eta, gamma)
    tp  = theta * phi
    return (tp * (1.0 + abs(rho)) <= 4.0) and (tp * phi * (1.0 + abs(rho)) <= 4.0)


# ─── Paramètres calibrés ──────────────────────────────────────────────────────

@dataclass
class SSVIParams:
    rho:   float
    eta:   float
    gamma: float

    def __repr__(self) -> str:
        return (f"SSVIParams(ρ={self.rho:+.3f}, η={self.eta:.3f}, "
                f"γ={self.gamma:.3f})")


# ─── Extraction de la structure par terme ATM ────────────────────────────────

def _atm_total_variance(tenors, strikes, vols, spot, r, q):
    """
    θ_i = σ_ATM(t_i)²·t_i pour chaque tenor, σ_ATM interpolée au forward.
    Force la croissance de θ (non-arbitrage calendaire).
    """
    tenors  = np.asarray(tenors, float)
    strikes = np.asarray(strikes, float)
    vols    = np.asarray(vols, float)

    theta = np.empty(len(tenors))
    for i, T in enumerate(tenors):
        F   = spot * math.exp((r - q) * T)
        row = vols[i]
        valid = np.isfinite(row) & (row > 0)
        if valid.sum() >= 2:
            sig_atm = float(np.interp(F, strikes[valid], row[valid]))
        elif valid.sum() == 1:
            sig_atm = float(row[valid][0])
        else:
            sig_atm = 0.20
        theta[i] = max(sig_atm, 1e-3) ** 2 * T

    # Monotonie croissante (calendar no-arb) — clamp cumulatif
    for i in range(1, len(theta)):
        theta[i] = max(theta[i], theta[i - 1] + 1e-8)
    return theta


# ─── Calibration ──────────────────────────────────────────────────────────────

def calibrate_ssvi(
    tenors, strikes, vols, spot: float, r: float, q: float,
    seed: int = 0,
) -> Tuple[SSVIParams, np.ndarray]:
    """
    Calibre (ρ, η, γ) en variance totale, θ_T pinnée à l'ATM de marché.

    • Cible : variances totales de marché w_ij = σ_ij²·t_i sur la grille.
    • Pondération gaussienne en log-moneyness (ATM prioritaire).
    • Pénalité de non-arbitrage papillon.
    Retourne (SSVIParams, theta_nodes aux tenors).
    """
    tenors  = np.asarray(tenors, float)
    strikes = np.asarray(strikes, float)
    vols    = np.asarray(vols, float)

    theta = _atm_total_variance(tenors, strikes, vols, spot, r, q)

    # Points de marché (k, w) valides
    K_pts, W_pts, TH_pts, WT_pts = [], [], [], []
    for i, T in enumerate(tenors):
        F = spot * math.exp((r - q) * T)
        sd = math.sqrt(max(theta[i], 1e-8))        # σ_ATM √T
        for j, K in enumerate(strikes):
            sig = vols[i, j]
            if not np.isfinite(sig) or sig <= 0:
                continue
            k = math.log(K / F)
            if abs(k) > 3.0 * sd:      # ailes cotées au tick, pas d'information
                continue
            K_pts.append(k)
            W_pts.append(sig * sig * T)
            TH_pts.append(theta[i])
            WT_pts.append(math.exp(-2.0 * k * k))   # poids gaussien ATM
    K_pts  = np.array(K_pts);  W_pts = np.array(W_pts)
    TH_pts = np.array(TH_pts); WT_pts = np.array(WT_pts)

    if len(K_pts) < 5:
        # Pas assez de données → smile plat raisonnable
        return SSVIParams(rho=-0.5, eta=1.0, gamma=0.4), theta

    def objective(x):
        rho, eta, gamma = x
        phi = eta / np.power(np.maximum(TH_pts, 1e-8), gamma)
        u   = phi * K_pts + rho
        w   = 0.5 * TH_pts * (1.0 + rho * phi * K_pts
                              + np.sqrt(u * u + (1.0 - rho * rho)))
        resid = WT_pts * (w - W_pts) ** 2

        # Pénalité butterfly sur les θ nodaux
        pen = 0.0
        for th in theta:
            ph = eta / (max(th, 1e-8) ** gamma)
            tp = th * ph
            pen += max(0.0, tp * (1 + abs(rho)) - _BF_TARGET) ** 2
            pen += max(0.0, tp * ph * (1 + abs(rho)) - _BF_TARGET) ** 2
        return float(np.sum(resid)) + 1.0e3 * pen

    bounds = [(-0.99, 0.99), (0.01, 10.0), (0.01, 0.50)]
    best, best_f = None, np.inf
    rng = np.random.default_rng(seed)
    starts = [(-0.6, 1.0, 0.35), (-0.3, 0.5, 0.45), (-0.8, 2.0, 0.25)]
    starts += [(rng.uniform(-0.9, 0.0), rng.uniform(0.1, 3.0),
                rng.uniform(0.05, 0.49)) for _ in range(4)]
    for x0 in starts:
        try:
            res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
            if res.fun < best_f:
                best_f, best = res.fun, res.x
        except Exception:
            continue

    if best is None:
        return SSVIParams(rho=-0.5, eta=1.0, gamma=0.4), theta

    rho_b, eta_b, gamma_b = float(best[0]), float(best[1]), float(best[2])

    # Si les conditions butterfly sont violées (c1 = θφ(1+|ρ|), c2 = θφ²(1+|ρ|),
    # c1 ∝ η et c2 ∝ η²), on réduit η juste assez pour repasser sous la borne.
    c1 = max(th * (eta_b / max(th, 1e-8) ** gamma_b) * (1 + abs(rho_b))
             for th in theta)
    c2 = max(th * (eta_b / max(th, 1e-8) ** gamma_b) ** 2 * (1 + abs(rho_b))
             for th in theta)
    if c1 > _BF_TARGET or c2 > _BF_TARGET:
        s = min(_BF_TARGET / c1 if c1 > 0 else 1.0,
                math.sqrt(_BF_TARGET / c2) if c2 > 0 else 1.0)
        eta_b *= s
        print(f"[SSVI] Projection butterfly — η réduit ×{s:.3f} → {eta_b:.3f} "
              f"(surface garantie sans-arbitrage)")

    params_out = SSVIParams(rho=rho_b, eta=eta_b, gamma=gamma_b)
    rmse = math.sqrt(best_f / max(len(K_pts), 1))
    print(f"[SSVI] Calibration — {params_out}  (résidu w≈{rmse:.2e}, {len(K_pts)} points)")
    if rmse > 5e-3:
        print(f"[SSVI] ⚠ Résidu élevé — surface de marché bruitée ou incohérente, "
              f"vérifier les cotations.")
    return params_out, theta


# ─── Surface SSVI ─────────────────────────────────────────────────────────────

class SSVISurface:
    """
    Surface de vol SSVI sans-arbitrage — drop-in pour VolSurface.

    Expose vol(T,K), atm_vol(T) et la vol locale Dupire ANALYTIQUE local_vol(T,K).
    Fournit aussi tenors/strikes/vols (grille ré-évaluée) pour rester compatible
    avec les consommateurs existants (Dupire FD, calibration Heston).
    """

    def __init__(self, valuation_date: date, spot: float, r: float, q: float,
                 tenors: np.ndarray, theta_nodes: np.ndarray, params: SSVIParams,
                 strikes: Optional[np.ndarray] = None):
        self.valuation_date = valuation_date
        self.spot = spot
        self.r = r
        self.q = q
        self.tenors  = np.asarray(tenors, float)
        self._theta_nodes = np.asarray(theta_nodes, float)
        self.params  = params

        # Grille de strikes compatible (par défaut ±30 % autour du spot)
        if strikes is None:
            strikes = spot * np.linspace(0.6, 1.4, 21)
        self.strikes = np.asarray(strikes, float)

        # Grille de vols ré-évaluée (pour Dupire FD / calibration Heston)
        self.vols = np.array([[self.vol(T, K) for K in self.strikes]
                              for T in self.tenors])

    # -- structure par terme θ_T (monotone, extrapolation ATM constante) -------
    def _theta(self, T: float) -> float:
        t = self.tenors
        th = self._theta_nodes
        if T <= t[0]:
            return max(th[0] * (T / t[0]), 1e-10)      # σ_ATM constante en deçà
        if T >= t[-1]:
            return th[-1] * (T / t[-1])                # σ_ATM constante au-delà
        return float(np.interp(T, t, th))

    def _forward(self, T: float) -> float:
        return self.spot * math.exp((self.r - self.q) * T)

    def vol(self, T: float, K: float) -> float:
        """Vol implicite SSVI σ(T,K) = √(w/T)."""
        T = max(float(T), 1e-6)
        F = self._forward(T)
        k = math.log(max(K, 1e-8) / F)
        theta = self._theta(T)
        p = self.params
        w = float(ssvi_total_variance(k, theta, p.rho, p.eta, p.gamma))
        return max(math.sqrt(max(w, 1e-12) / T), 1e-4)

    def __call__(self, T: float, K: float) -> float:
        return self.vol(T, K)

    def atm_vol(self, T: float) -> float:
        return self.vol(T, self._forward(T))

    def sigma(self, T: float, K: float) -> float:   # alias (compat _BumpedSnap)
        return self.vol(T, K)

    def local_vol(self, T: float, K: float, dT: float = 1e-3) -> float:
        """
        Vol locale de Dupire ANALYTIQUE (dérivées en strike fermées) :

            σ_loc² = w_T / [ 1 − (k/w)·w_k
                             + ¼(−¼ − 1/w + k²/w²)·w_k²
                             + ½·w_kk ]

        ∂w/∂T par différence finie sur la structure θ_T (lisse), w_k et w_kk
        analytiques → bien plus stable que Dupire tout-FD sur surface brute.
        """
        T = max(float(T), 1e-4)
        F = self._forward(T)
        k = math.log(max(K, 1e-8) / F)
        p = self.params

        theta = self._theta(T)
        w, w_k, w_kk = _w_and_k_derivs(k, theta, p.rho, p.eta, p.gamma)
        if w <= 1e-10:
            return self.atm_vol(T)

        # ∂w/∂T à k fixe (FD centrée sur la structure θ lisse)
        h    = 0.5 * dT
        T_up = T + h
        T_dn = max(T - h, 1e-5)
        w_up = float(ssvi_total_variance(k, self._theta(T_up), p.rho, p.eta, p.gamma))
        w_dn = float(ssvi_total_variance(k, self._theta(T_dn), p.rho, p.eta, p.gamma))
        w_T  = (w_up - w_dn) / (T_up - T_dn)

        g = (1.0
             - (k / w) * w_k
             + 0.25 * (-0.25 - 1.0 / w + k * k / (w * w)) * w_k * w_k
             + 0.5 * w_kk)

        if g <= 1e-10 or w_T <= 0:
            return self.atm_vol(T)
        local_var = w_T / g
        if local_var <= 0:
            return self.atm_vol(T)
        return float(np.clip(math.sqrt(local_var), 0.01, 3.0))

    def no_arbitrage_report(self) -> dict:
        """Diagnostic non-arbitrage sur les nœuds de la structure."""
        p = self.params
        cal_ok = bool(np.all(np.diff(self._theta_nodes) >= -1e-12))
        bf = [butterfly_free(th, p.rho, p.eta, p.gamma) for th in self._theta_nodes]
        return {"calendar_free": cal_ok, "butterfly_free": bool(np.all(bf)),
                "params": p}

    def __repr__(self) -> str:
        rep = self.no_arbitrage_report()
        return (f"SSVISurface({self.params}, tenors={len(self.tenors)}, "
                f"calendar_free={rep['calendar_free']}, "
                f"butterfly_free={rep['butterfly_free']})")


# ─── Constructeur depuis une VolSurface brute ────────────────────────────────

def _observed_mask(vols: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Masque des cellules observées. Les cellules issues d'un prolongement plat
    sont identiques à leur voisine (maturité précédente ou strike adjacent).
    """
    v = np.asarray(vols, float)
    mask = np.isfinite(v) & (v > 0)
    if v.shape[0] > 1:
        mask[1:] &= ~(np.abs(v[1:] - v[:-1]) < tol)
    if v.shape[1] > 1:
        same_left  = np.zeros_like(mask); same_left[:, 1:]   = np.abs(v[:, 1:] - v[:, :-1]) < tol
        same_right = np.zeros_like(mask); same_right[:, :-1] = np.abs(v[:, :-1] - v[:, 1:]) < tol
        mask &= ~(same_left | same_right)

    if mask.sum() < 5:
        mask = np.isfinite(v) & (v > 0)
    return mask


def ssvi_from_grid(surf) -> SSVISurface:
    """
    Calibre une SSVISurface sans-arbitrage à partir d'une VolSurface brute
    (mêmes tenors/strikes/spot/r/q).
    """
    vols = np.asarray(surf.vols, float)
    mask = _observed_mask(vols)
    params, theta = calibrate_ssvi(
        surf.tenors, surf.strikes, np.where(mask, vols, np.nan),
        surf.spot, surf.r, surf.q,
    )
    return SSVISurface(
        valuation_date=surf.valuation_date, spot=surf.spot, r=surf.r, q=surf.q,
        tenors=np.asarray(surf.tenors, float), theta_nodes=theta, params=params,
        strikes=np.asarray(surf.strikes, float),
    )
