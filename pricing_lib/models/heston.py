"""
pricing_lib/models/heston.py
─────────────────────────────────────────────────────────────────────────────
Modèle de Heston (Stochastic Volatility).

Contenu
───────
HestonParams          — dataclass des 5 paramètres du modèle
heston_cf()           — fonction caractéristique (Gatheral / Albrecher 2007)
heston_price()        — prix call/put semi-fermé via intégration trapézoïdale
heston_implied_vol()  — conversion prix Heston → vol implicite BS
_heston_call_vec()    — prix calls vectorisé sur un tableau de strikes (interne)
calibrate_heston()    — calibration sur VolSurface (DE + L-BFGS-B)
heston_simulate()     — simulation MC terminale (payoffs vanilles)
heston_simulate_paths()— simulation MC chemin complet (produits path-dep.)

Optimisations v2
────────────────
• heston_price    : quad → np.trapz sur grille fixe ; heston_cf est vectorisé
                    sur u (numpy), élimine ~200× le surcoût Python/appel.
• _heston_call_vec: évalue simultanément tous les strikes pour un tenor ;
                    utilisé dans calibrate_heston pour supprimer la boucle K.
• calibrate_heston: objectif pré-calcule les prix de marché + normalisation
                    vega → zéro appel brentq dans la boucle interne.
                    maxiter DE 100→50, polishing désactivé en DE.
• heston_simulate : n_steps par défaut 252→100 /an (suffisant en terminal).

Optimisations v3
────────────────
• _cf_static / _cf_finish : la CF Heston est décomposée en deux phases.
  Phase statique  _cf_static(u, params) : calcule d, g, coef — dépend de u
                  et des params, PAS de T. Résultat réutilisable sur tous les
                  tenors pour un même jeu de paramètres.
  Phase dynamique _cf_finish(u, d, g, coef, S, v0, r, q, T, kappa, theta,
                  sigma_v) : ne calcule que les termes en exp(-d·T) et le
                  log restant.
• calibrate_heston : _cf_static est appelé UNE SEULE FOIS par évaluation
                  d'objectif (hors boucle tenor), puis _cf_finish est appelé
                  par tenor (5×). Économie : ~(n_T - 1) × 2 sqrt complexe
                  par éval, soit ~80 % du coût CF éliminé.
• Grille u réduite 200 → 128 points (précision identique pour la calibration).
• Exponentielles scalaires (exp(-q·T), etc.) pré-calculées hors objectif.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm as _norm


# ─── Dataclass paramètres ────────────────────────────────────────────────────

@dataclass
class HestonParams:
    """
    Paramètres du modèle de Heston.

    kappa   : vitesse de retour à la moyenne de la variance
    theta   : variance long terme (vol long terme = sqrt(theta))
    sigma_v : vol de vol
    rho     : corrélation browniens (S, V)  — typiquement négatif pour actions
    v0      : variance initiale (vol initiale = sqrt(v0))

    Condition de Feller (variance strictement positive) : 2·κ·θ > σ_v²
    """
    kappa:   float
    theta:   float
    sigma_v: float
    rho:     float
    v0:      float

    @property
    def vol0(self) -> float:
        """Vol initiale annualisée."""
        return math.sqrt(max(self.v0, 0.0))

    @property
    def vol_inf(self) -> float:
        """Vol long terme annualisée."""
        return math.sqrt(max(self.theta, 0.0))

    @property
    def feller(self) -> float:
        """Ratio de Feller (> 1 garantit V_t > 0 p.s.)."""
        return 2 * self.kappa * self.theta / (self.sigma_v ** 2)

    def to_array(self) -> np.ndarray:
        return np.array([self.kappa, self.theta, self.sigma_v, self.rho, self.v0])

    @classmethod
    def from_array(cls, arr) -> "HestonParams":
        return cls(*arr)

    def __repr__(self) -> str:
        return (
            f"HestonParams(κ={self.kappa:.3f}, θ={self.theta:.4f} "
            f"[σ_∞={self.vol_inf:.2%}], σ_v={self.sigma_v:.3f}, "
            f"ρ={self.rho:.3f}, v₀={self.v0:.4f} [σ₀={self.vol0:.2%}], "
            f"Feller={self.feller:.2f})"
        )


# ─── Fonction caractéristique ─────────────────────────────────────────────────

def heston_cf(
    u,          # float scalaire OU np.ndarray complexe — vectorisé sur u
    S: float,
    v0: float,
    r: float,
    q: float,
    T: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
):
    """
    Fonction caractéristique de Heston  E[e^{i·u·ln(S_T)}]
    sous la mesure risque-neutre.

    Formulation Albrecher et al. (2007) — numériquement stable
    (évite la discontinuité de branche du log complexe).

    u peut être un scalaire ou un np.ndarray — la fonction est entièrement
    vectorisée (toutes les opérations sont numpy element-wise).
    """
    u  = np.asarray(u, dtype=complex)
    x  = np.log(S) + (r - q) * T

    d = np.sqrt(
        (rho * sigma_v * 1j * u - kappa) ** 2
        + sigma_v ** 2 * (1j * u + u ** 2)
    )
    g = (kappa - rho * sigma_v * 1j * u - d) / (
        kappa - rho * sigma_v * 1j * u + d
    )

    eps = 1e-300   # garde-fou : évite log(0)/div 0 sur params dégénérés
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        exp_dT          = np.exp(-d * T)
        one_minus_g_exp = 1.0 - g * exp_dT
        one_minus_g     = 1.0 - g

        # NB : la dérive (r−q) est déjà portée par x = ln S + (r−q)T (terme i·u·x).
        # Ne PAS la rajouter ici sinon double comptage (φ(−i) ≠ E[S_T], prix biaisés).
        C = (kappa * theta / sigma_v ** 2) * (
            (kappa - rho * sigma_v * 1j * u - d) * T
            - 2.0 * np.log((one_minus_g_exp + eps) / (one_minus_g + eps))
        )
        D = ((kappa - rho * sigma_v * 1j * u - d) / sigma_v ** 2) * (
            (1.0 - exp_dT) / (one_minus_g_exp + eps)
        )

        return np.exp(C + D * v0 + 1j * u * x)


# ─── Pricing semi-fermé (scalaire K) ─────────────────────────────────────────

# Grille de quadrature partagée (évite de la recréer à chaque appel)
# 128 points suffisent pour la calibration (vs 200 avant) — gain ~36 %.
_U_GRID: Optional[np.ndarray] = None
_N_QUAD  = 128
_U_UPPER = 200.0


def _get_u_grid() -> np.ndarray:
    global _U_GRID
    if _U_GRID is None:
        _U_GRID = np.linspace(1e-6, _U_UPPER, _N_QUAD)
    return _U_GRID


# ─── Décomposition statique / dynamique de la CF ─────────────────────────────

def _cf_static(
    u: np.ndarray,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
):
    """
    Partie STATIQUE de la fonction caractéristique Heston.

    Calcule d, g et coef qui dépendent uniquement de (u, params) et PAS de T.
    Ces quantités sont invariantes pour un même jeu de params sur tous les tenors.

    Retourne
    --------
    d    : np.ndarray complexe (N,)
    g    : np.ndarray complexe (N,)
    coef : np.ndarray complexe (N,) — kappa − ρ·σ_v·i·u − d
    """
    iu = 1j * u
    d    = np.sqrt((rho * sigma_v * iu - kappa) ** 2 + sigma_v ** 2 * (iu + u ** 2))
    neg  = kappa - rho * sigma_v * iu - d      # = coef
    pos  = kappa - rho * sigma_v * iu + d
    g    = neg / pos
    return d, g, neg   # neg == coef


def _cf_finish(
    u: np.ndarray,
    d: np.ndarray,
    g: np.ndarray,
    coef: np.ndarray,
    S: float,
    v0: float,
    r: float,
    q: float,
    T: float,
    kappa: float,
    theta: float,
    sigma_v: float,
) -> np.ndarray:
    """
    Partie DYNAMIQUE de la fonction caractéristique Heston.

    Reçoit les parties statiques pré-calculées (d, g, coef) et finit le
    calcul pour un tenor T donné.  Coût : 2 exp complexe + 1 log complexe.

    Retourne
    --------
    np.ndarray complexe (N,) — valeurs de la CF
    """
    # Garde-fous numériques : sur des params dégénérés (calibration sur surface
    # pauvre/plate), one_minus_g_exp ou (1−g) peuvent friser 0 → log(0)/div 0.
    # eps complexe + np.errstate évitent un crash (certains environnements
    # remontent les warnings numpy en erreurs).
    eps = 1e-300
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x               = np.log(S) + (r - q) * T
        exp_dT          = np.exp(-d * T)
        one_minus_g_exp = 1.0 - g * exp_dT
        # (r−q) déjà dans x = ln S + (r−q)T — pas de double comptage ici.
        C = (kappa * theta / sigma_v ** 2) * (
            coef * T - 2.0 * np.log((one_minus_g_exp + eps) / (1.0 - g + eps))
        )
        D = (coef / sigma_v ** 2) * ((1.0 - exp_dT) / (one_minus_g_exp + eps))
        return np.exp(C + D * v0 + 1j * u * x)


def heston_price(
    flag: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    # upper_limit et n_limit conservés pour compatibilité ascendante (ignorés)
    upper_limit: float = _U_UPPER,
    n_limit: int = _N_QUAD,
) -> float:
    """
    Prix call ou put Heston via inversion de Gil-Pelaez.

    P_1, P_2 calculées par intégration trapézoïdale vectorisée :
    heston_cf est évalué en une seule passe numpy sur la grille u,
    ce qui élimine le surcoût des ~200 appels scalaires de quad.

    flag : 'c' (call) | 'p' (put)
    """
    kappa, theta, sigma_v, rho, v0 = (
        params.kappa, params.theta, params.sigma_v, params.rho, params.v0
    )
    u     = _get_u_grid()                        # (N,) réel
    log_K = np.log(K)

    # CF vectorisée sur toute la grille u en un seul appel numpy
    phi_u    = heston_cf(u,      S, v0, r, q, T, kappa, theta, sigma_v, rho)  # (N,) complexe
    phi_u_sh = heston_cf(u - 1j, S, v0, r, q, T, kappa, theta, sigma_v, rho)  # (N,) complexe
    phi_mi   = heston_cf(np.complex128(-1j), S, v0, r, q, T, kappa, theta, sigma_v, rho)  # scalaire

    e_term = np.exp(-1j * u * log_K)             # (N,)

    int_P1 = np.real(e_term * phi_u_sh / (1j * u * phi_mi))   # (N,)
    int_P2 = np.real(e_term * phi_u    / (1j * u))             # (N,)

    P1 = float(np.clip(0.5 + (1.0 / np.pi) * np.trapezoid(int_P1, u), 0.0, 1.0))
    P2 = float(np.clip(0.5 + (1.0 / np.pi) * np.trapezoid(int_P2, u), 0.0, 1.0))

    fwd_S = S * math.exp(-q * T)
    fwd_K = K * math.exp(-r * T)

    call = max(fwd_S * P1 - fwd_K * P2, max(fwd_S - fwd_K, 0.0))

    if flag.lower() == "c":
        return float(call)
    put = call - fwd_S + fwd_K
    return float(max(put, max(fwd_K - fwd_S, 0.0)))


# ─── Pricing vectorisé multi-strikes (interne, utilisé par calibrate_heston) ─

def _heston_call_vec(
    S: float,
    K_array: np.ndarray,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
    _precomp=None,   # tuple (d_u, g_u, coef_u, d_w, g_w, coef_w, w) pré-calculé
) -> np.ndarray:
    """
    Prix calls Heston pour un tableau de strikes, un seul tenor.

    Si _precomp est fourni (optimisation calibration), les parties statiques
    de la CF (d, g, coef) ne sont pas recalculées — gain ~2× par appel.

    Retourne
    --------
    np.ndarray shape (n_K,) — prix call pour chaque strike
    """
    kappa, theta, sigma_v, rho, v0 = (
        params.kappa, params.theta, params.sigma_v, params.rho, params.v0
    )
    u     = _get_u_grid()    # (N,)
    log_K = np.log(K_array)  # (n_K,)

    if _precomp is not None:
        d_u, g_u, coef_u, d_w, g_w, coef_w, w, phi_mi = _precomp
        phi_u    = _cf_finish(u, d_u, g_u, coef_u, S, v0, r, q, T, kappa, theta, sigma_v)
        phi_u_sh = _cf_finish(w, d_w, g_w, coef_w, S, v0, r, q, T, kappa, theta, sigma_v)
    else:
        w        = u - 1j
        d_u, g_u, coef_u = _cf_static(u, kappa, theta, sigma_v, rho)
        d_w, g_w, coef_w = _cf_static(w, kappa, theta, sigma_v, rho)
        phi_u    = _cf_finish(u, d_u, g_u, coef_u, S, v0, r, q, T, kappa, theta, sigma_v)
        phi_u_sh = _cf_finish(w, d_w, g_w, coef_w, S, v0, r, q, T, kappa, theta, sigma_v)
        phi_mi   = heston_cf(np.complex128(-1j), S, v0, r, q, T, kappa, theta, sigma_v, rho)

    if _precomp is not None:
        # phi_mi pré-calculé hors boucle mais T-dépendant → recalculé ici (scalaire, rapide)
        phi_mi = heston_cf(np.complex128(-1j), S, v0, r, q, T, kappa, theta, sigma_v, rho)

    # Broadcasting (N,1) × (1,n_K) → (N, n_K)
    e_term = np.exp(-1j * u[:, None] * log_K[None, :])

    int_P1 = np.real(e_term * phi_u_sh[:, None] / (1j * u[:, None] * phi_mi))
    int_P2 = np.real(e_term * phi_u[:, None]    / (1j * u[:, None]))

    P1 = np.clip(0.5 + (1.0 / np.pi) * np.trapezoid(int_P1, u, axis=0), 0.0, 1.0)
    P2 = np.clip(0.5 + (1.0 / np.pi) * np.trapezoid(int_P2, u, axis=0), 0.0, 1.0)

    fwd_S = S * math.exp(-q * T)
    fwd_K = K_array * math.exp(-r * T)
    return np.maximum(fwd_S * P1 - fwd_K * P2, np.maximum(fwd_S - fwd_K, 0.0))


# ─── Vol implicite Heston → BS ────────────────────────────────────────────────

def heston_implied_vol(
    flag: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    params: HestonParams,
) -> Optional[float]:
    """
    Convertit le prix Heston en volatilité implicite Black-Scholes.
    Retourne None si pas de solution (option trop profondément OTM).
    """
    from pricing_lib.market_data.vol_surface import implied_vol

    price = heston_price(flag, S, K, T, r, q, params)
    return implied_vol(price, S, K, T, r, q, flag)


# ─── Détection forward-fill ──────────────────────────────────────────────────

def _mask_forwardfilled(vols: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Retourne un masque booléen shape (n_T, n_K) où True indique qu'une cellule
    est suspecte de forward-fill : valeur identique (à tol près) à celle de la
    maturité immédiatement précédente ET non nulle.

    Logique : si la vol n'a pas bougé entre deux maturités consécutives pour
    un strike donné, c'est quasi-certainement un forward-fill Euronext plutôt
    qu'une vraie donnée de marché.  La première ligne (maturité la plus courte)
    est toujours considérée valide.

    Paramètre
    ---------
    vols : np.ndarray (n_T, n_K) — grille de vols implicites
    """
    mask = np.zeros_like(vols, dtype=bool)
    if vols.shape[0] < 2:
        return mask
    diff    = np.abs(np.diff(vols, axis=0))   # (n_T-1, n_K)
    nonzero = vols[:-1] > 0                   # exclut les cases vraiment vides
    mask[1:] = (diff < tol) & nonzero
    return mask


# ─── Calibration ──────────────────────────────────────────────────────────────

def calibrate_heston(
    vol_surface,
    S: float,
    r: float,
    q: float,
    weights: Optional[np.ndarray] = None,
    seed: int = 42,
    prev_params: Optional[HestonParams] = None,
    reg_weight: float = 0.0,
    rmse_warn: float = 0.02,
) -> HestonParams:
    """
    Calibre les paramètres Heston sur une VolSurface Euronext.

    Minimise la RMSE vol-normalisée entre prix Heston et prix de marché.

    Stratégie :
    • Objectif vectorisé — _heston_call_vec() calcule tous les strikes d'un
      tenor en une seule passe numpy ; zéro appel brentq dans la boucle.
    • Normalisation par vega BSM (pre-calculé hors boucle) — équivalent
      à une RMSE de vol implicite au premier ordre sans brentq.
    • Étape 1 : differential_evolution  (maxiter=50, polish=False)
    • Étape 2 : raffinement L-BFGS-B

    Paramètres
    ----------
    vol_surface : VolSurface (grille tenors × strikes × vols)
    S           : spot
    r, q        : taux et dividende continus
    weights     : pondération optionnelle (ignoré, recalculé en interne)
    seed        : graine pour differential_evolution
    """
    tenors_all   = vol_surface.tenors
    strikes_all  = vol_surface.strikes
    mkt_vols_all = vol_surface.vols

    # ── Masque forward-fill : exclure les vols dupliquées entre maturités ───
    ff_mask      = _mask_forwardfilled(mkt_vols_all)
    n_masked     = int(ff_mask.sum())
    if n_masked:
        print(f"[Heston] {n_masked} cellule(s) forward-fill détectée(s) et exclues de la calibration.")
    mkt_vols_all = np.where(ff_mask, np.nan, mkt_vols_all)

    # ── Sous-échantillonnage ─────────────────────────────────────────────────
    n_T, n_K = len(tenors_all), len(strikes_all)

    # 5 maturités les plus denses en données
    valid_per_row = np.sum(~np.isnan(mkt_vols_all) & (mkt_vols_all > 0), axis=1)
    top5_t = np.argsort(valid_per_row)[::-1][:5]
    t_idx  = np.sort(top5_t)
    tenors = tenors_all[t_idx]

    # 5 strikes par moneyness cible {0.90, 0.95, 1.00, 1.05, 1.10}
    target_m  = [0.90, 0.95, 1.00, 1.05, 1.10]
    K_targets = np.array([m * S for m in target_m])
    strikes   = K_targets

    # Interpolation linéaire de mkt_vols sur K_targets pour chaque tenor
    mkt_vols = np.full((len(t_idx), len(K_targets)), np.nan)
    for i, ti in enumerate(t_idx):
        row   = mkt_vols_all[ti]
        valid = ~np.isnan(row) & (row > 0)
        if valid.sum() < 2:
            continue
        mkt_vols[i] = np.interp(K_targets, strikes_all[valid], row[valid])

    # ── Ancrage ATM : v0 ≈ variance ATM courte, θ ≈ variance ATM longue ────────
    # (approche desk : on pin le NIVEAU de variance aux vols ATM observables ;
    # seule la FORME du smile — σ_v, ρ — est laissée libre au calibrateur).
    _atm_col = 2   # moneyness 1.00 dans target_m
    _atm = np.array([mkt_vols[i, _atm_col] if (np.isfinite(mkt_vols[i, _atm_col])
                     and mkt_vols[i, _atm_col] > 0) else np.nan
                     for i in range(len(tenors))])
    _fin = _atm[np.isfinite(_atm)]
    v_front = float(_fin[0] ** 2)  if len(_fin) else 0.0784
    v_back  = float(_fin[-1] ** 2) if len(_fin) else 0.0784
    v_front = min(max(v_front, 0.0025), 0.39)   # vol ATM bornée ∈ [5 %, ~62 %]
    v_back  = min(max(v_back,  0.0025), 0.39)

    # ── Pondération gaussienne ATM ────────────────────────────────────────────
    moneyness = strikes / S
    w_k = np.exp(-2.0 * np.log(moneyness) ** 2)       # (n_K_sub,)
    w_mat = np.outer(np.ones(len(tenors)), w_k)        # (n_T_sub, n_K_sub)

    # ── Pré-calcul hors boucle : prix et vega de marché ──────────────────────
    # (une seule fois, pas à chaque évaluation de l'objectif)
    sqrt_2pi = math.sqrt(2.0 * math.pi)
    mkt_prices  = np.zeros_like(mkt_vols)
    vega_norm   = np.zeros_like(mkt_vols)
    w_sigma     = np.ones_like(mkt_vols)   # correction 6 : poids niveau de vol

    for i, T in enumerate(tenors):
        if T <= 0:
            continue
        sqrt_T = math.sqrt(T)
        fwd_S  = S * math.exp(-q * T)
        for j, K in enumerate(strikes):
            sv = mkt_vols[i, j]
            if sv <= 0 or np.isnan(sv):
                continue
            fwd_K = K * math.exp(-r * T)
            d1 = (math.log(S / K) + (r - q + 0.5 * sv ** 2) * T) / (sv * sqrt_T)
            d2 = d1 - sv * sqrt_T
            call_mkt = fwd_S * _norm.cdf(d1) - fwd_K * _norm.cdf(d2)
            if K > S * math.exp((r - q) * T):
                mkt_prices[i, j] = call_mkt + fwd_K - fwd_S
            else:
                mkt_prices[i, j] = call_mkt
            vega_norm[i, j]  = max(fwd_S * sqrt_T * math.exp(-0.5 * d1 ** 2) / sqrt_2pi, 1e-4)
            w_sigma[i, j]    = 1.0 / (1.0 + sv)   # correction 6

    # Pré-calcul des exponentielles scalaires par tenor (hors objectif)
    tenor_fwd_S = np.array([S * math.exp(-q * T) for T in tenors])       # (n_T,)
    tenor_fwd_K = np.outer(np.ones(len(tenors)), strikes * 1.0)           # (n_T, n_K) — rempli ci-dessous
    for i, T in enumerate(tenors):
        tenor_fwd_K[i] = strikes * math.exp(-r * T)
    tenor_atm_fwd = np.array([S * math.exp((r - q) * T) for T in tenors])  # (n_T,) — seuil call/put

    u = _get_u_grid()   # (N,) — récupéré ici pour être utilisé dans l'objectif
    w = u - 1j          # (N,) pour P1

    # ── Fonction objectif (vectorisée sur K et T, _cf_static hors boucle tenor) ─
    def objective(x: np.ndarray) -> float:
        try:
            hp = HestonParams.from_array(x)
            kappa, theta, sigma_v, rho, v0 = hp.kappa, hp.theta, hp.sigma_v, hp.rho, hp.v0

            # ── Partie statique : calculée UNE SEULE FOIS pour ce jeu de params ──
            d_u, g_u, coef_u = _cf_static(u, kappa, theta, sigma_v, rho)  # (N,)
            d_w, g_w, coef_w = _cf_static(w, kappa, theta, sigma_v, rho)  # (N,)

            precomp = (d_u, g_u, coef_u, d_w, g_w, coef_w, w, None)

            # ── Pénalité Feller : λ·max(0, σ_v² − 2κθ)² ──────────────────────
            feller_viol = max(0.0, sigma_v ** 2 - 2.0 * kappa * theta)
            penalty = 15.0 * feller_viol ** 2   # λ=15 : moins bloquant que 50

            # Ancrage ATM (fort) : v0 et θ collés aux variances ATM de marché.
            # Empêche le niveau de partir en vrille sur surface pauvre/plate.
            penalty += 50.0 * ((v0 - v_front) ** 2 + (theta - v_back) ** 2)

            # Prior doux sur la FORME du smile (σ_v, ρ) et la vitesse κ : n'agit
            # que si la surface ne les contraint pas (cas plat → κ indéterminé
            # quand v0=θ) → évite les paramètres aux bornes.
            penalty += 0.5 * ((sigma_v - 0.40) ** 2 + (rho + 0.50) ** 2)
            penalty += 0.002 * (kappa - 3.0) ** 2

            # Régularisation : pénalise les sauts vs la calibration précédente
            # (stabilité jour à jour façon desk).
            if prev_params is not None and reg_weight > 0.0:
                penalty += reg_weight * float(np.sum((x - prev_params.to_array()) ** 2))

            total = 0.0
            n_pts = 0
            for i, T in enumerate(tenors):
                if T <= 0:
                    continue

                model_calls = _heston_call_vec(S, strikes, T, r, q, hp, _precomp=precomp)

                # Parité : K > F → put OTM
                is_put       = strikes > tenor_atm_fwd[i]
                model_prices = np.where(
                    is_put,
                    model_calls + tenor_fwd_K[i] - tenor_fwd_S[i],
                    model_calls,
                )

                valid = vega_norm[i] > 1e-5
                if not np.any(valid):
                    continue
                w_T = 1.0 / max(T, 0.05)   # pondération maturité
                residuals = (
                    w_mat[i][valid]
                    * w_sigma[i][valid]                              # correction 6
                    * (model_prices[valid] - mkt_prices[i][valid])
                    / vega_norm[i][valid]
                )
                total += w_T * float(np.sum(residuals ** 2))
                n_pts += int(np.sum(valid))

            return total / max(n_pts, 1) + penalty
        except Exception as e:
            print(f"[Heston] objectif exception: {e}")
            return 1e6

    # ── Bornes : [kappa, theta, sigma_v, rho, v0] ────────────────────────────
    bounds = [
        (0.50, 25.0),    # kappa
        (0.01,  1.0),    # theta
        (0.05,  1.50),   # sigma_v — resserré [0.05, 1.5]
        (-0.95, -0.30),  # rho     — skew marqué forcé
        (0.01,  0.40),   # v0
    ]

    # Initialisation ancrée sur l'ATM : v0=variance ATM courte, θ=variance ATM
    # longue (warm-start si une calibration précédente est fournie).
    x0 = (prev_params.to_array() if prev_params is not None
          else np.array([4.0, v_back, 0.40, -0.50, v_front]))

    print("[Heston] Calibration — recherche globale (differential_evolution)...")
    result_de = differential_evolution(
        objective, bounds,
        seed=seed,
        init='sobol',
        x0=x0,
        maxiter=80,
        tol=1e-5,
        popsize=8,
        mutation=(0.5, 1.5),
        recombination=0.7,
        workers=1,
        polish=False,
    )

    print(f"[Heston] DE terminé — résidu={math.sqrt(result_de.fun)*100:.4f}%")
    print("[Heston] Raffinement local (L-BFGS-B)...")

    result = minimize(
        objective, result_de.x,
        method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": 500},
    )

    params  = HestonParams.from_array(result.x)
    rmse    = math.sqrt(max(result.fun, 0.0))      # résidu vol-normalisé (fraction)
    bnames  = ("kappa", "theta", "sigma_v", "rho", "v0")

    # ── Rapport de calibration (gates desk) ──────────────────────────────────
    at_bound = []
    for v, (lo, hi), nm in zip(result.x, bounds, bnames):
        if abs(v - lo) <= 1e-3 * max(abs(lo), 1.0) or abs(v - hi) <= 1e-3 * max(abs(hi), 1.0):
            at_bound.append(nm)

    print(f"[Heston] Calibration terminée — résidu RMSE={rmse*100:.4f}% vol")
    print(f"[Heston] {params}")
    if params.feller < 1.0:
        print(f"[Heston] ⚠ Feller {params.feller:.2f} < 1 — variance peut toucher 0 "
              f"(QE gère, mais smile court terme à surveiller).")
    if at_bound:
        print(f"[Heston] ⚠ Paramètre(s) collé(s) aux bornes : {', '.join(at_bound)} "
              f"— calibration possiblement mal posée (surface pauvre ?).")
    if rmse > rmse_warn:
        print(f"[Heston] ⚠ Résidu {rmse*100:.2f}% > seuil {rmse_warn*100:.1f}% "
              f"— fit médiocre, prix exotiques à prendre avec prudence.")
    return params


# ─── Schéma Quadratic-Exponential (Andersen 2008) ────────────────────────────

_QE_PSI_C = 1.5   # seuil de bascule quadratique / exponentielle


def _simulate_heston_qe(
    S: float, r: float, q: float, T: float, params: HestonParams,
    n_paths: int, observation_times: np.ndarray,
    n_steps: int, antithetic: bool, seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulation Heston par le schéma QE d'Andersen (2008).

    • Variance : Quadratic-Exponential (exact en loi sur le moment conditionnel),
      sans biais de troncature contrairement à Euler full-truncation.
    • Log-spot : discrétisation centrale (γ₁=γ₂=½) avec correction martingale
      (K0..K4), corrélation ρ injectée via les termes en V (pas en corrélant
      les browniens) — propriété clé du QE.

    Retourne (observation_times, paths[n_paths, n_obs]).
    """
    kappa, theta, sigma_v, rho, v0 = (
        params.kappa, params.theta, params.sigma_v, params.rho, params.v0
    )
    dt = T / n_steps
    E  = math.exp(-kappa * dt)

    # Constantes log-spot (Andersen, central discretization γ1=γ2=0.5)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma_v
    K1 = g1 * dt * (kappa * rho / sigma_v - 0.5) - rho / sigma_v
    K2 = g2 * dt * (kappa * rho / sigma_v - 0.5) + rho / sigma_v
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    rng = np.random.default_rng(seed)
    if antithetic:
        n_half = n_paths // 2
        n_act  = n_half * 2
        ZV = rng.standard_normal((n_steps, n_half))
        ZS = rng.standard_normal((n_steps, n_half))
        UU = rng.random((n_steps, n_half))
        # Antithétique : réflexion des normales, U → 1−U (lois préservées → non biaisé)
        ZV = np.concatenate([ZV, -ZV], axis=1)
        ZS = np.concatenate([ZS, -ZS], axis=1)
        UU = np.concatenate([UU, 1.0 - UU], axis=1)
    else:
        n_act = n_paths
        ZV = rng.standard_normal((n_steps, n_act))
        ZS = rng.standard_normal((n_steps, n_act))
        UU = rng.random((n_steps, n_act))

    log_S = np.full(n_act, math.log(S))
    V     = np.full(n_act, v0)

    step_times = np.linspace(dt, T, n_steps)
    obs_idx = np.array([int(np.argmin(np.abs(step_times - t)))
                        for t in observation_times])
    obs_map = {idx: i for i, idx in enumerate(obs_idx)}
    obs_set = set(obs_idx.tolist())
    paths   = np.zeros((n_act, len(observation_times)))

    eps = 1e-12
    for t in range(n_steps):
        m  = theta + (V - theta) * E
        m  = np.maximum(m, eps)
        s2 = (V * sigma_v ** 2 * E * (1.0 - E) / kappa
              + theta * sigma_v ** 2 * (1.0 - E) ** 2 / (2.0 * kappa))
        psi = s2 / (m * m)

        V_next = np.empty_like(V)

        # Branche quadratique  (psi <= psi_c)
        quad = psi <= _QE_PSI_C
        if np.any(quad):
            inv = 2.0 / psi[quad]
            b2  = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
            a   = m[quad] / (1.0 + b2)
            V_next[quad] = a * (np.sqrt(b2) + ZV[t, quad]) ** 2

        # Branche exponentielle (psi > psi_c)
        expb = ~quad
        if np.any(expb):
            pe   = (psi[expb] - 1.0) / (psi[expb] + 1.0)
            beta = (1.0 - pe) / m[expb]
            u    = UU[t, expb]
            vv   = np.where(u <= pe, 0.0,
                            np.log(np.maximum((1.0 - pe) / np.maximum(1.0 - u, eps), eps)) / beta)
            V_next[expb] = vv

        log_S = (log_S + (r - q) * dt + K0
                 + K1 * V + K2 * V_next
                 + np.sqrt(np.maximum(K3 * V + K4 * V_next, 0.0)) * ZS[t])
        V = V_next

        if t in obs_set:
            paths[:, obs_map[t]] = np.exp(log_S)

    return observation_times, paths


# ─── Simulation MC terminale ──────────────────────────────────────────────────

def heston_simulate(
    S: float,
    r: float,
    q: float,
    T: float,
    params: HestonParams,
    n_paths: int,
    n_steps: Optional[int] = None,
    antithetic: bool = True,
    seed: Optional[int] = None,
    scheme: str = "qe",
) -> np.ndarray:
    """
    Simulation MC Heston — retourne les prix terminaux S_T.

    scheme="qe"    : Quadratic-Exponential d'Andersen (défaut, sans biais).
    scheme="euler" : Euler-Maruyama full-truncation (legacy).

    n_steps par défaut : 100/an — suffisant pour les produits à payoff terminal.
    Pour les produits path-dépendants utilisez heston_simulate_paths.

    Retourne
    --------
    np.ndarray shape (n_paths,)
    """
    if n_steps is None:
        n_steps = max(int(100 * T), 30)   # 100/an suffit pour payoff terminal

    if scheme == "qe":
        _, paths = _simulate_heston_qe(
            S, r, q, T, params, n_paths, np.array([T]), n_steps, antithetic, seed
        )
        return paths[:, -1]

    dt      = T / n_steps
    sqrt_dt = math.sqrt(dt)

    kappa, theta, sigma_v, rho, v0 = (
        params.kappa, params.theta, params.sigma_v, params.rho, params.v0
    )
    sqrt_1_rho2 = math.sqrt(max(1.0 - rho ** 2, 0.0))

    rng = np.random.default_rng(seed)

    if antithetic:
        n_half = n_paths // 2
        Z1 = rng.standard_normal((n_steps, n_half))
        Z2 = rng.standard_normal((n_steps, n_half))
        ZS = rho * Z1 + sqrt_1_rho2 * Z2
        ZV = Z1
        ZS = np.concatenate([ZS, -ZS], axis=1)
        ZV = np.concatenate([ZV, -ZV], axis=1)
        n_actual = n_half * 2
    else:
        n_actual = n_paths
        Z1 = rng.standard_normal((n_steps, n_actual))
        Z2 = rng.standard_normal((n_steps, n_actual))
        ZS = rho * Z1 + sqrt_1_rho2 * Z2
        ZV = Z1

    log_S = np.full(n_actual, math.log(S))
    V     = np.full(n_actual, v0)

    for t in range(n_steps):
        V_pos  = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)
        V      = V + kappa * (theta - V_pos) * dt + sigma_v * sqrt_V * sqrt_dt * ZV[t]
        log_S  = log_S + (r - q - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * ZS[t]

    return np.exp(log_S)


# ─── Simulation MC chemin complet ─────────────────────────────────────────────

def heston_simulate_paths(
    S: float,
    r: float,
    q: float,
    T: float,
    params: HestonParams,
    n_paths: int,
    observation_times: np.ndarray,
    n_steps: Optional[int] = None,
    antithetic: bool = True,
    seed: Optional[int] = None,
    scheme: str = "qe",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulation MC Heston avec enregistrement aux dates d'observation.

    Utilisée pour les produits path-dépendants :
    autocalls, options à barrières, phoenix, athena.

    n_steps conservé à 252/an pour la précision des barrières continues.

    Paramètres
    ----------
    observation_times : np.ndarray (n_obs,) — en années depuis aujourd'hui

    Retourne
    --------
    times  : np.ndarray (n_obs,)
    paths  : np.ndarray (n_paths, n_obs)
    """
    if n_steps is None:
        n_steps = max(int(252 * T), 50)   # 252/an — précision barrière continue

    if scheme == "qe":
        return _simulate_heston_qe(
            S, r, q, T, params, n_paths, observation_times, n_steps, antithetic, seed
        )

    dt       = T / n_steps
    sqrt_dt  = math.sqrt(dt)
    step_times = np.linspace(dt, T, n_steps)

    obs_indices = np.array(
        [int(np.argmin(np.abs(step_times - t))) for t in observation_times]
    )
    obs_set = set(obs_indices.tolist())
    obs_map = {idx: i for i, idx in enumerate(obs_indices)}

    kappa, theta, sigma_v, rho, v0 = (
        params.kappa, params.theta, params.sigma_v, params.rho, params.v0
    )
    sqrt_1_rho2 = math.sqrt(max(1.0 - rho ** 2, 0.0))

    rng = np.random.default_rng(seed)

    if antithetic:
        n_half = n_paths // 2
        Z1 = rng.standard_normal((n_steps, n_half))
        Z2 = rng.standard_normal((n_steps, n_half))
        ZS = rho * Z1 + sqrt_1_rho2 * Z2
        ZV = Z1
        ZS = np.concatenate([ZS, -ZS], axis=1)
        ZV = np.concatenate([ZV, -ZV], axis=1)
        n_actual = n_half * 2
    else:
        n_actual = n_paths
        Z1 = rng.standard_normal((n_steps, n_actual))
        Z2 = rng.standard_normal((n_steps, n_actual))
        ZS = rho * Z1 + sqrt_1_rho2 * Z2
        ZV = Z1

    log_S  = np.full(n_actual, math.log(S))
    V      = np.full(n_actual, v0)
    paths  = np.zeros((n_actual, len(observation_times)))

    for t in range(n_steps):
        V_pos  = np.maximum(V, 0.0)
        sqrt_V = np.sqrt(V_pos)
        V      = V + kappa * (theta - V_pos) * dt + sigma_v * sqrt_V * sqrt_dt * ZV[t]
        log_S  = log_S + (r - q - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * ZS[t]

        if t in obs_set:
            paths[:, obs_map[t]] = np.exp(log_S)

    return observation_times, paths


# ─── Paramètres par défaut (desk) ─────────────────────────────────────────────

DEFAULT_HESTON = HestonParams(
    kappa=2.0,
    theta=0.04,    # vol long terme 20%
    sigma_v=0.30,
    rho=-0.70,
    v0=0.04,       # vol initiale 20%
)

