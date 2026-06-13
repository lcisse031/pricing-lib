"""
pricing_lib/pricers/autocalls.py
─────────────────────────────────────────────────────────────────────────────
Pricing Monte Carlo des autocalls : Phoenix et Athena.

Deux modes disponibles pour chaque produit
───────────────────────────────────────────
  mode="AL"  (Réplication)
      Pas de coupon en entrée.
      Simule les trajectoires Heston → estime les probabilités de recall/coupon
      à chaque date → calibre le coupon fair algébriquement :
          coupon_fair = (1 − Esp_A − Esp_RF) / (Sum_C + Sum_C_final)
      Prix = 100 % par construction.

  mode="MC"  (Monte Carlo pur)
      Coupon en entrée (paramètre `coupon`).
      Simule les trajectoires Heston → actualise les vrais flux path-by-path
      → retourne le prix en % du nominal.
      Le coupon_fair est calculé en bonus (info) avec la même formule.
      Exemple : coupon=0.10 → prix=107 % → coupon trop élevé.
"""

from __future__ import annotations

import math
from dataclasses import replace as _dc_replace
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Optional

import numpy as np
import pandas as pd

from pricing_lib.market_data.market_snapshot import MarketSnapshot
from pricing_lib.models.heston import (
    HestonParams, DEFAULT_HESTON,
    heston_simulate_paths,
)
from pricing_lib.pricers.base import PricingResult
from pricing_lib.pricers.mc import get_heston_params


def _fwd_rate(ois_curve, T: float) -> float:
    """
    Taux forward OIS continu jusqu'à T : r_T = −ln(df(T))/T.

    Utilisé comme drift de simulation des autocalls (cohérent avec les pricers
    mono-maturité). Avec une courbe décalée (_BumpedMarket), r_T se décale aussi
    → le bump de taux touche drift ET actualisation (rho complet).
    """
    T = max(float(T), 1e-6)
    return -math.log(max(ois_curve.df_tau(T), 1e-12)) / T


# ─── Helpers dates ────────────────────────────────────────────────────────────

def _observation_dates(start: date, freq_months: int, maturity_months: int):
    """
    Génère les dates d'observation depuis start avec freq_months de pas
    jusqu'à maturity_months.
    Retourne (liste de dates, array de tau en années).
    """
    dates = []
    current = start
    while True:
        current = current + relativedelta(months=freq_months)
        while current.weekday() >= 5:
            current += relativedelta(days=1)
        dates.append(current)
        if len(dates) >= maturity_months // freq_months:
            break
    taus = np.array([(d - start).days / 365.25 for d in dates])
    return dates, taus


# ─── Phoenix ──────────────────────────────────────────────────────────────────

class PhoenixPricer:
    """
    Pricing Phoenix / Autocall.

    mode="AL"  →  coupon_fair calibré, prix ≈ 100 %
    mode="MC"    →  prix en % pour un coupon donné, coupon_fair en info
    """

    def __init__(
        self,
        snapshot: MarketSnapshot,
        n_paths: int,
        start_date: date,
        S0: float,
        barrier_coupon: float,
        barrier_recall: float,
        capital_barrier: float,
        freq_months: int,
        maturity_months: int,
        kg: str = "no",
        autocall_start: int = 0,
        mode: str = "AL",
        coupon: float = 0.0,
        compute_greeks: bool = False,
        params: Optional[HestonParams] = None,
        antithetic: bool = True,
        seed: Optional[int] = None,
    ):
        self.snap             = snapshot
        self.n_paths          = n_paths
        self.start_date       = start_date
        self.S0               = S0
        self.barrier_coupon   = barrier_coupon
        self.barrier_recall   = barrier_recall
        self.capital_barrier  = capital_barrier
        self.freq_months      = freq_months
        self.maturity_months  = maturity_months
        self.kg               = kg.lower()
        self.autocall_start   = autocall_start
        self.mode             = mode.upper()
        self.coupon           = coupon
        self.compute_greeks   = compute_greeks
        self.antithetic       = antithetic
        self.seed             = seed
        self.params           = params or get_heston_params(snapshot)

        self.obs_dates, self.obs_taus = _observation_dates(
            start_date, freq_months, maturity_months
        )
        self.n_periods = len(self.obs_dates)
        self.T         = self.obs_taus[-1]

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate(self) -> np.ndarray:
        """Simule les trajectoires — retourne paths (n_paths, n_periods)."""
        _, paths = heston_simulate_paths(
            S=self.S0,
            r=_fwd_rate(self.snap.ois_curve, self.T),   # drift = r_T (forward à maturité)
            q=self.snap.q,
            T=self.T,
            params=self.params,
            n_paths=self.n_paths,
            observation_times=self.obs_taus,
            antithetic=self.antithetic,
            seed=self.seed,
        )
        return paths

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def price(self) -> PricingResult:
        paths = self._simulate()
        if self.mode == "MC":
            return self._compute_mc(paths)
        return self._compute_repl(paths)

    # ── Helpers partagés ──────────────────────────────────────────────────────

    def _build_masks(self, paths: np.ndarray):
        """Retourne cumul_r, has_trigger, no_trigger, trigger_r, trigger_c."""
        n_paths, n_periods = paths.shape
        trigger_r = paths > self.barrier_recall
        trigger_c = paths > self.barrier_coupon

        cumul_r = np.zeros((n_paths, n_periods), dtype=bool)
        for i in range(n_periods):
            if i < self.autocall_start:
                cumul_r[:, i] = False
            elif i == self.autocall_start:
                cumul_r[:, i] = trigger_r[:, i]
            else:
                cumul_r[:, i] = cumul_r[:, i - 1] | trigger_r[:, i]

        has_trigger = (
            cumul_r[:, n_periods - 2]
            if self.autocall_start < n_periods - 1
            else np.zeros(n_paths, dtype=bool)
        )
        return cumul_r, has_trigger, ~has_trigger, trigger_r, trigger_c

    def _extract_probs(self, paths, cumul_r, has_trigger, no_trigger, trigger_r, trigger_c):
        """Probabilités recall et coupon depuis les trajectoires simulées."""
        n_paths, n_periods = paths.shape
        p_recall, c_recall = [], []
        for i in range(n_periods - 1):
            if i < self.autocall_start:
                p_recall.append(0.0); c_recall.append(0)
            elif i == self.autocall_start:
                cond = trigger_r[:, i]
                p_recall.append(float(cond.mean())); c_recall.append(int(cond.sum()))
            else:
                alive = ~cumul_r[:, i - 1]
                cond  = trigger_r[:, i] & alive
                p_recall.append(float(cond.sum()) / n_paths); c_recall.append(int(cond.sum()))

        n_no = int(no_trigger.sum())
        p_no = n_no / n_paths

        p_coupon, c_coupon = [], []
        for i in range(n_periods):
            alive = (np.ones(n_paths, dtype=bool) if i <= self.autocall_start
                     else ~cumul_r[:, i - 1])
            cond = trigger_c[:, i] & alive
            p_coupon.append(float(cond.mean())); c_coupon.append(int(cond.sum()))

        return p_recall, c_recall, p_coupon, c_coupon, p_no, n_no

    def _coupon_fair_formula(self, p_recall, p_coupon, p_no, dfs, paths, no_trigger):
        """Formule algébrique du coupon fair."""
        n_periods = self.n_periods
        S0 = self.S0
        ST = paths[:, -1]

        Esp_A = sum(
            p_recall[i] * dfs[i]
            for i in range(n_periods - 1)
            if i >= self.autocall_start
        )
        Sum_C = sum(p_coupon[i] * dfs[i] for i in range(n_periods - 1))
        df_T  = dfs[-1]

        if self.kg == "no":
            ST_no = ST[no_trigger]
            if len(ST_no) > 0:
                mask_loss   = ST_no < self.capital_barrier
                prob_loss   = float(mask_loss.mean()) if mask_loss.sum() > 0 else 0.0
                payoff_loss = float((ST_no[mask_loss] / S0).mean()) if mask_loss.sum() > 0 else 0.0
                prob_h      = float((ST_no >= self.capital_barrier).mean())
            else:
                prob_loss = 0.0; payoff_loss = 0.0; prob_h = 1.0

            Esp_RF = p_no * df_T * (prob_h + payoff_loss * prob_loss)
            mask_cpn_cond = (ST_no > self.barrier_coupon) if len(ST_no) > 0 else np.array([])
            p_cpn_cond    = float(mask_cpn_cond.mean()) if len(mask_cpn_cond) > 0 else 0.0
            Sum_C_final   = p_cpn_cond * p_no * df_T
        else:
            Esp_RF      = p_no * df_T
            ST_no       = ST[no_trigger]
            mask_cpn_cond = (ST_no > self.barrier_coupon) if len(ST_no) > 0 else np.array([])
            p_cpn_cond    = float(mask_cpn_cond.mean()) if len(mask_cpn_cond) > 0 else 0.0
            Sum_C_final   = p_cpn_cond * p_no * df_T

        denom       = Sum_C + Sum_C_final
        coupon_fair = (1.0 - Esp_A - Esp_RF) / denom if denom > 1e-10 else 0.0
        return coupon_fair, Esp_A, Esp_RF, Sum_C, Sum_C_final

    # ── Greeks (delta, gamma, vega) ──────────────────────────────────────────

    def _compute_greeks(self, price_0: float, fair_coupon: float = None) -> dict:
        """
        Sensibilités par différences finies CENTRÉES (re-simulation à n_paths//8,
        même seed), convention unifiée avec le reste de la lib + bloc p_*.
        Delta / Gamma : S ± dS.
        Vega          : shift parallèle ±1 pt de vol sur √v0 ET √θ.
        Rho           : ± dr via courbe OIS décalée (_BumpedMarket).
        Theta         : ± 1 jour calendaire (start_date ∓ 1j).

        Mode AL : le coupon_fair est figé (paramètre fair_coupon) et les
        bumps sont pricés en MC — sinon chaque bump reprice en AL et donne
        toujours ~100 %, ce qui annule delta/gamma.
        """
        from datetime import timedelta
        from pricing_lib.risk.greeks import _assemble, _BumpedMarket

        n_bump = max(self.n_paths // 8, 500)
        dS     = self.S0 * 0.01
        dvol   = 0.01      # 1 point de vol
        dr     = 0.0001

        # Mode AL : on fixe le coupon et on passe en MC pour les bumps
        coupon_b = fair_coupon if (self.mode == "AL" and fair_coupon is not None) \
                   else self.coupon

        def _sub_price(S0_new, params_new=None, snap_new=None, start_new=None) -> float:
            p  = params_new if params_new is not None else self.params
            sn = snap_new   if snap_new   is not None else self.snap
            sd = start_new  if start_new  is not None else self.start_date
            sub = PhoenixPricer(
                sn, n_bump, sd, S0_new,
                self.barrier_coupon, self.barrier_recall, self.capital_barrier,
                self.freq_months, self.maturity_months, self.kg, self.autocall_start,
                mode="MC", coupon=coupon_b, params=p, seed=self.seed,
            )
            return sub._compute_mc(sub._simulate()).price

        def _safe(fn, fallback):
            try:
                return float(fn())
            except Exception:
                return fallback

        # Prix de base MC (AL : price_0 ≈ 100 % → on reprice en MC à coupon figé)
        # p_0 du bloc Greeks recalculé AU MÊME seed/n_bump que les chocs
        # (sinon gamma contaminé par le décalage de trajectoires). Le prix
        # affiché du produit reste price_0 (pleine résolution).
        p0 = _safe(lambda: _sub_price(self.S0), price_0)

        # Spot ± dS (delta, gamma)
        p_S_up   = _safe(lambda: _sub_price(self.S0 + dS), p0)
        p_S_down = _safe(lambda: _sub_price(self.S0 - dS), p0)

        # Vol : shift parallèle ± dvol points sur √v0 et √θ, reconvertis en variance
        vol_v0 = math.sqrt(max(self.params.v0,    0.0))
        vol_th = math.sqrt(max(self.params.theta, 0.0))
        par_up = _dc_replace(self.params, v0=(vol_v0 + dvol) ** 2,
                                          theta=(vol_th + dvol) ** 2)
        par_dn = _dc_replace(self.params, v0=max(vol_v0 - dvol, 1e-4) ** 2,
                                          theta=max(vol_th - dvol, 1e-4) ** 2)
        p_vol_up   = _safe(lambda: _sub_price(self.S0, params_new=par_up), p0)
        p_vol_down = _safe(lambda: _sub_price(self.S0, params_new=par_dn), p0)

        # Taux ± dr (courbe OIS décalée, forme préservée)
        p_r_up   = _safe(lambda: _sub_price(self.S0, snap_new=_BumpedMarket(self.snap, rate_bump=+dr)), p0)
        p_r_down = _safe(lambda: _sub_price(self.S0, snap_new=_BumpedMarket(self.snap, rate_bump=-dr)), p0)

        # Temps ± 1 jour : +1j = maturité plus courte (p_T_down) ; −1j = plus longue (p_T_up)
        p_T_down = _safe(lambda: _sub_price(self.S0, start_new=self.start_date + timedelta(days=1)), p0)
        p_T_up   = _safe(lambda: _sub_price(self.S0, start_new=self.start_date - timedelta(days=1)), p0)

        return _assemble(
            p0,
            p_S_up, p_S_down, dS,
            p_vol_up, p_vol_down, 2.0 * dvol, 0.01,
            p_r_up, p_r_down, 2.0 * dr,
            p_T_up, p_T_down, 2.0,    # ± 1 jour → dénominateur 2 jours
        )

    # ── Mode AL ─────────────────────────────────────────────────────────────

    def _compute_repl(self, paths: np.ndarray) -> PricingResult:
        """
        Réplication : probabilités MC → coupon_fair algébrique.
        Prix ≈ 100 % par construction.
        """
        n_paths, n_periods = paths.shape
        snap = self.snap
        dfs  = np.array([snap.ois_curve.df_tau(tau) for tau in self.obs_taus])

        cumul_r, has_trigger, no_trigger, trigger_r, trigger_c = self._build_masks(paths)
        p_recall, c_recall, p_coupon, c_coupon, p_no, n_no = self._extract_probs(
            paths, cumul_r, has_trigger, no_trigger, trigger_r, trigger_c
        )

        coupon_fair, Esp_A, Esp_RF, Sum_C, Sum_C_final = self._coupon_fair_formula(
            p_recall, p_coupon, p_no, dfs, paths, no_trigger
        )

        price = (Esp_A + coupon_fair * (Sum_C + Sum_C_final) + Esp_RF) * 100.0

        prob_df = pd.DataFrame({
            "Date":        [str(d) for d in self.obs_dates[:-1]] + [str(self.obs_dates[-1])],
            "Scénario":    [f"Autocall T{i+1}" for i in range(n_periods - 1)] + ["No Autocall"],
            "N_recall":    c_recall + [n_no],
            "Prob_recall": p_recall + [p_no],
            "N_coupon":    c_coupon,
            "Prob_coupon": p_coupon,
            "DF":          dfs.round(6),
        })

        cf_df = pd.DataFrame({
            "Date":       [str(d) for d in self.obs_dates],
            "DF":         dfs.round(6),
            "PV_recall":  (np.array(p_recall + [p_no]) * dfs).round(6),
            "PV_coupon":  (np.array(p_coupon) * coupon_fair * dfs).round(6),
        })

        greeks = self._compute_greeks(price, fair_coupon=coupon_fair) if self.compute_greeks \
            else {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

        return PricingResult(
            price=round(float(price), 6),
            greeks=greeks,
            ticker=snap.ticker,
            product=f"Phoenix AL {'KG' if self.kg=='yes' else 'Capital Risque'}",
            model="heston_repl",
            fair_coupon=round(float(coupon_fair), 6),
            cashflows=cf_df,
            probabilities=prob_df,
        )

    # ── Mode MC ───────────────────────────────────────────────────────────────

    def _compute_mc(self, paths: np.ndarray) -> PricingResult:
        """
        MC pur : actualisation des flux path-by-path avec le coupon en entrée.
        Prix en % du nominal.  coupon_fair calculé en info.
        """
        n_paths, n_periods = paths.shape
        S0      = self.S0
        snap    = self.snap
        coupon  = self.coupon
        dfs     = np.array([snap.ois_curve.df_tau(tau) for tau in self.obs_taus])

        cumul_r, has_trigger, no_trigger, trigger_r, trigger_c = self._build_masks(paths)

        # ── Matrice de payoffs (fraction du nominal) ──────────────────────────
        payoff = np.zeros((n_paths, n_periods))

        for i in range(n_periods - 1):
            if i < self.autocall_start:
                mask_ac = np.zeros(n_paths, dtype=bool)
            elif i == self.autocall_start:
                mask_ac = trigger_r[:, i]
            else:
                mask_ac = trigger_r[:, i] & ~cumul_r[:, i - 1]

            # Autocall → remboursement nominal + coupon
            payoff[mask_ac, i] = 1.0 + coupon

            # Coupon seul (vivant, pas autocallé, barrière coupon franchie)
            alive = (np.ones(n_paths, dtype=bool) if i <= self.autocall_start
                     else ~cumul_r[:, i - 1])
            mask_cpn = alive & ~mask_ac & trigger_c[:, i]
            payoff[mask_cpn, i] = coupon

        # Payoff final (maturité)
        ST = paths[:, -1]
        if self.kg == "no":
            payoff[no_trigger & (ST >= self.capital_barrier), -1] = 1.0
            loss = no_trigger & (ST < self.capital_barrier)
            payoff[loss, -1] = ST[loss] / S0
        else:
            payoff[no_trigger, -1] = 1.0

        # Coupon final
        mask_cpn_final = no_trigger & (ST > self.barrier_coupon)
        payoff[mask_cpn_final, -1] += coupon

        # Prix actualisé
        pv    = (payoff * dfs[np.newaxis, :]).sum(axis=1)
        price = float(pv.mean()) * 100.0

        # Coupon fair (info) — même trajectoires
        p_recall, c_recall, p_coupon, c_coupon, p_no, n_no = self._extract_probs(
            paths, cumul_r, has_trigger, no_trigger, trigger_r, trigger_c
        )
        coupon_fair, _, _, _, _ = self._coupon_fair_formula(
            p_recall, p_coupon, p_no, dfs, paths, no_trigger
        )

        prob_df = pd.DataFrame({
            "Date":        [str(d) for d in self.obs_dates[:-1]] + [str(self.obs_dates[-1])],
            "Scénario":    [f"Autocall T{i+1}" for i in range(n_periods - 1)] + ["No Autocall"],
            "N_recall":    c_recall + [n_no],
            "Prob_recall": p_recall + [p_no],
            "N_coupon":    c_coupon,
            "Prob_coupon": p_coupon,
            "DF":          dfs.round(6),
        })

        cf_df = pd.DataFrame({
            "Date":      [str(d) for d in self.obs_dates],
            "DF":        dfs.round(6),
            "PV_coupon": (np.array(p_coupon) * coupon * dfs).round(6),
        })

        greeks = self._compute_greeks(price) if self.compute_greeks \
            else {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

        return PricingResult(
            price=round(float(price), 6),
            greeks=greeks,
            ticker=snap.ticker,
            product=f"Phoenix MC {'KG' if self.kg=='yes' else 'Capital Risque'}",
            model="heston_mc",
            fair_coupon=round(float(coupon_fair), 6),
            cashflows=cf_df,
            probabilities=prob_df,
        )


# ─── Athena ──────────────────────────────────────────────────────────────────

class AthenaPricer:
    """
    Pricing Athena — autocall sans barrière coupon intermédiaire.
    Effet mémoire : à l'autocall T_i reçoit nominal + (i+1) × coupon.
    À maturité (non autocallé, S_T ≥ capital_barrier) : nominal + n_periods × coupon.

    mode="AL"  →  coupon_fair calibré, prix ≈ 100 %
    mode="MC"    →  prix en % pour un coupon donné, coupon_fair en info
    """

    def __init__(
        self,
        snapshot: MarketSnapshot,
        n_paths: int,
        start_date: date,
        S0: float,
        barrier_recall: float,
        capital_barrier: float,
        freq_months: int,
        maturity_months: int,
        kg: str = "no",
        mode: str = "AL",
        coupon: float = 0.0,
        compute_greeks: bool = False,
        params: Optional[HestonParams] = None,
        antithetic: bool = True,
        seed: Optional[int] = None,
    ):
        self.snap            = snapshot
        self.n_paths         = n_paths
        self.start_date      = start_date
        self.S0              = S0
        self.barrier_recall  = barrier_recall
        self.capital_barrier = capital_barrier
        self.freq_months     = freq_months
        self.maturity_months = maturity_months
        self.kg              = kg.lower()
        self.mode            = mode.upper()
        self.coupon          = coupon
        self.compute_greeks  = compute_greeks
        self.antithetic      = antithetic
        self.seed            = seed
        self.params          = params or get_heston_params(snapshot)

        self.obs_dates, self.obs_taus = _observation_dates(
            start_date, freq_months, maturity_months
        )
        self.n_periods = len(self.obs_dates)
        self.T         = self.obs_taus[-1]

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def price(self) -> PricingResult:
        _, paths = heston_simulate_paths(
            S=self.S0, r=_fwd_rate(self.snap.ois_curve, self.T), q=self.snap.q, T=self.T,
            params=self.params, n_paths=self.n_paths,
            observation_times=self.obs_taus,
            antithetic=self.antithetic, seed=self.seed,
        )
        if self.mode == "MC":
            return self._compute_mc(paths)
        return self._compute_repl(paths)

    # ── Helpers partagés ──────────────────────────────────────────────────────

    def _build_masks(self, paths: np.ndarray):
        trigger_r = paths > self.barrier_recall
        # Bug #8 fix : cumul_r[:, i] doit indiquer qu'un autocall a eu lieu
        # STRICTEMENT AVANT la période i, pas à la période i elle-même.
        # np.cumsum > 0 marque la période i comme "déjà autocallée" dès que
        # trigger_r[:, i] est True, ce qui empêche de compter le flux à T_i.
        # Correct : décaler d'une colonne (shifted cumsum).
        cumsum = np.cumsum(trigger_r, axis=1)
        # cumul_r[:, i] = True ssi un trigger a eu lieu en [0, i-1]
        cumul_r = np.zeros_like(trigger_r, dtype=bool)
        cumul_r[:, 1:] = cumsum[:, :-1] > 0
        has_trigger = trigger_r[:, :-1].any(axis=1)
        return trigger_r, cumul_r, has_trigger, ~has_trigger

    def _extract_probs(self, paths, trigger_r, cumul_r, has_trigger, no_trigger):
        n_paths, n_periods = paths.shape
        p_recall, c_recall = [], []
        for i in range(n_periods - 1):
            cond = trigger_r[:, i] if i == 0 else trigger_r[:, i] & ~cumul_r[:, i - 1]
            p_recall.append(float(cond.mean()))
            c_recall.append(int(cond.sum()))
        n_no = int(no_trigger.sum())
        p_no = n_no / n_paths
        return p_recall, c_recall, p_no, n_no

    def _coupon_fair_formula(self, p_recall, p_no, dfs, paths, no_trigger):
        """Formule algébrique du coupon fair Athena (effet mémoire)."""
        n_periods = self.n_periods
        S0 = self.S0
        ST = paths[:, -1]
        ST_no = ST[no_trigger]
        df_T = dfs[-1]

        Sum_A_i = sum(p_recall[i] * dfs[i] * (i + 1) for i in range(n_periods - 1))
        Sum_A   = sum(p_recall[i] * dfs[i]            for i in range(n_periods - 1))

        if self.kg == "no":
            mask_h    = (ST_no >= self.capital_barrier) if len(ST_no) > 0 else np.array([])
            mask_loss = (ST_no <  self.capital_barrier) if len(ST_no) > 0 else np.array([])
            prob_h    = float(mask_h.mean())    if len(mask_h) > 0    else 1.0
            prob_loss = float(mask_loss.mean()) if len(mask_loss) > 0 else 0.0
            payoff_moy = float((ST_no[mask_loss] / S0).mean()) if mask_loss.sum() > 0 else 0.0
            num   = 1.0 - (Sum_A + p_no * df_T * (prob_h + payoff_moy * prob_loss))
            denom = Sum_A_i + n_periods * p_no * df_T * prob_h
        else:
            prob_h = 1.0; prob_loss = 0.0; payoff_moy = 0.0
            num   = 1.0 - (Sum_A + p_no * df_T)
            denom = Sum_A_i + n_periods * p_no * df_T

        coupon_fair = num / denom if denom > 1e-10 else 0.0
        return coupon_fair, Sum_A, Sum_A_i, prob_h, prob_loss, payoff_moy

    # ── Greeks (delta, gamma, vega) ──────────────────────────────────────────

    def _compute_greeks(self, price_0: float, fair_coupon: float = None) -> dict:
        """
        Différences finies CENTRÉES, convention unifiée + bloc p_* (cf.
        PhoenixPricer._compute_greeks). Delta/Gamma sur S, Vega = shift parallèle
        ±1 pt de vol (√v0, √θ), Rho via courbe décalée, Theta ±1 jour.
        Mode AL : coupon_fair figé, bumps pricés en MC.
        """
        from datetime import timedelta
        from pricing_lib.risk.greeks import _assemble, _BumpedMarket

        n_bump = max(self.n_paths // 8, 500)
        dS     = self.S0 * 0.01
        dvol   = 0.01
        dr     = 0.0001

        coupon_b = fair_coupon if (self.mode == "AL" and fair_coupon is not None) \
                   else self.coupon

        def _sub_price(S0_new, params_new=None, snap_new=None, start_new=None) -> float:
            p  = params_new if params_new is not None else self.params
            sn = snap_new   if snap_new   is not None else self.snap
            sd = start_new  if start_new  is not None else self.start_date
            sub = AthenaPricer(
                sn, n_bump, sd, S0_new,
                self.barrier_recall, self.capital_barrier,
                self.freq_months, self.maturity_months, self.kg,
                mode="MC", coupon=coupon_b, params=p, seed=self.seed,
            )
            _, paths_sub = heston_simulate_paths(
                S=S0_new, r=_fwd_rate(sn.ois_curve, sub.T), q=sn.q, T=sub.T,
                params=p, n_paths=n_bump,
                observation_times=sub.obs_taus,
                antithetic=sub.antithetic, seed=sub.seed,
            )
            return sub._compute_mc(paths_sub).price

        def _safe(fn, fallback):
            try:
                return float(fn())
            except Exception:
                return fallback

        # p_0 du bloc Greeks recalculé AU MÊME seed/n_bump que les chocs
        # (sinon gamma contaminé par le décalage de trajectoires). Le prix
        # affiché du produit reste price_0 (pleine résolution).
        p0 = _safe(lambda: _sub_price(self.S0), price_0)

        p_S_up   = _safe(lambda: _sub_price(self.S0 + dS), p0)
        p_S_down = _safe(lambda: _sub_price(self.S0 - dS), p0)

        vol_v0 = math.sqrt(max(self.params.v0,    0.0))
        vol_th = math.sqrt(max(self.params.theta, 0.0))
        par_up = _dc_replace(self.params, v0=(vol_v0 + dvol) ** 2,
                                          theta=(vol_th + dvol) ** 2)
        par_dn = _dc_replace(self.params, v0=max(vol_v0 - dvol, 1e-4) ** 2,
                                          theta=max(vol_th - dvol, 1e-4) ** 2)
        p_vol_up   = _safe(lambda: _sub_price(self.S0, params_new=par_up), p0)
        p_vol_down = _safe(lambda: _sub_price(self.S0, params_new=par_dn), p0)

        p_r_up   = _safe(lambda: _sub_price(self.S0, snap_new=_BumpedMarket(self.snap, rate_bump=+dr)), p0)
        p_r_down = _safe(lambda: _sub_price(self.S0, snap_new=_BumpedMarket(self.snap, rate_bump=-dr)), p0)

        p_T_down = _safe(lambda: _sub_price(self.S0, start_new=self.start_date + timedelta(days=1)), p0)
        p_T_up   = _safe(lambda: _sub_price(self.S0, start_new=self.start_date - timedelta(days=1)), p0)

        return _assemble(
            p0,
            p_S_up, p_S_down, dS,
            p_vol_up, p_vol_down, 2.0 * dvol, 0.01,
            p_r_up, p_r_down, 2.0 * dr,
            p_T_up, p_T_down, 2.0,
        )

    # ── Mode AL ─────────────────────────────────────────────────────────────

    def _compute_repl(self, paths: np.ndarray) -> PricingResult:
        n_paths, n_periods = paths.shape
        snap = self.snap
        dfs  = np.array([snap.ois_curve.df_tau(tau) for tau in self.obs_taus])
        df_T = dfs[-1]
        ST   = paths[:, -1]

        trigger_r, cumul_r, has_trigger, no_trigger = self._build_masks(paths)
        p_recall, c_recall, p_no, n_no = self._extract_probs(
            paths, trigger_r, cumul_r, has_trigger, no_trigger
        )
        coupon_fair, Sum_A, Sum_A_i, prob_h, prob_loss, payoff_moy = self._coupon_fair_formula(
            p_recall, p_no, dfs, paths, no_trigger
        )

        if self.kg == "no":
            price = (Sum_A + coupon_fair * Sum_A_i
                     + p_no * df_T * (prob_h + payoff_moy * prob_loss)
                     + coupon_fair * n_periods * p_no * df_T * prob_h) * 100.0
        else:
            price = (Sum_A + coupon_fair * Sum_A_i
                     + p_no * df_T
                     + coupon_fair * n_periods * p_no * df_T) * 100.0

        prob_df = pd.DataFrame({
            "Date":        [str(d) for d in self.obs_dates[:-1]] + [str(self.obs_dates[-1])],
            "Scénario":    [f"Autocall T{i+1}" for i in range(n_periods - 1)] + ["No Autocall"],
            "N_recall":    c_recall + [n_no],
            "Prob_recall": p_recall + [p_no],
            "DF":          dfs.round(6),
            "PV_recall":   (np.array(p_recall + [p_no]) * dfs).round(6),
        })

        cf_df = pd.DataFrame({
            "Date":      [str(d) for d in self.obs_dates],
            "DF":        dfs.round(6),
            "Mémoire":   list(range(1, n_periods + 1)),
            "PV_coupon": (np.array(p_recall + [p_no]) * dfs
                          * np.arange(1, n_periods + 1) * coupon_fair).round(6),
        })

        greeks = self._compute_greeks(price, fair_coupon=coupon_fair) if self.compute_greeks \
            else {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

        return PricingResult(
            price=round(float(price), 6),
            greeks=greeks,
            ticker=snap.ticker,
            product=f"Athena AL {'KG' if self.kg=='yes' else 'Capital Risque'}",
            model="heston_repl",
            fair_coupon=round(float(coupon_fair), 6),
            cashflows=cf_df,
            probabilities=prob_df,
        )

    # ── Mode MC ───────────────────────────────────────────────────────────────

    def _compute_mc(self, paths: np.ndarray) -> PricingResult:
        """
        MC pur : flux path-by-path avec effet mémoire.
        Prix en % du nominal. coupon_fair en info.
        """
        n_paths, n_periods = paths.shape
        S0     = self.S0
        snap   = self.snap
        coupon = self.coupon
        dfs    = np.array([snap.ois_curve.df_tau(tau) for tau in self.obs_taus])

        trigger_r, cumul_r, has_trigger, no_trigger = self._build_masks(paths)

        # ── Matrice de payoffs ────────────────────────────────────────────────
        payoff = np.zeros((n_paths, n_periods))

        for i in range(n_periods - 1):
            if i == 0:
                mask_ac = trigger_r[:, i]
            else:
                mask_ac = trigger_r[:, i] & ~cumul_r[:, i - 1]
            # Autocall T_i : nominal + (i+1) × coupon  (effet mémoire)
            payoff[mask_ac, i] = 1.0 + (i + 1) * coupon

        # Payoff final
        ST = paths[:, -1]
        if self.kg == "no":
            mask_h    = no_trigger & (ST >= self.capital_barrier)
            mask_loss = no_trigger & (ST <  self.capital_barrier)
            payoff[mask_h,    -1] = 1.0 + n_periods * coupon
            payoff[mask_loss, -1] = ST[mask_loss] / S0
        else:
            payoff[no_trigger, -1] = 1.0 + n_periods * coupon

        # Prix actualisé
        pv    = (payoff * dfs[np.newaxis, :]).sum(axis=1)
        price = float(pv.mean()) * 100.0

        # Coupon fair (info)
        p_recall, c_recall, p_no, n_no = self._extract_probs(
            paths, trigger_r, cumul_r, has_trigger, no_trigger
        )
        coupon_fair, _, _, _, _, _ = self._coupon_fair_formula(
            p_recall, p_no, dfs, paths, no_trigger
        )

        prob_df = pd.DataFrame({
            "Date":        [str(d) for d in self.obs_dates[:-1]] + [str(self.obs_dates[-1])],
            "Scénario":    [f"Autocall T{i+1}" for i in range(n_periods - 1)] + ["No Autocall"],
            "N_recall":    c_recall + [n_no],
            "Prob_recall": p_recall + [p_no],
            "DF":          dfs.round(6),
        })

        cf_df = pd.DataFrame({
            "Date":      [str(d) for d in self.obs_dates],
            "DF":        dfs.round(6),
            "Mémoire":   list(range(1, n_periods + 1)),
            "PV_coupon": (np.array(p_recall + [p_no]) * dfs
                          * np.arange(1, n_periods + 1) * coupon).round(6),
        })

        greeks = self._compute_greeks(price) if self.compute_greeks \
            else {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

        return PricingResult(
            price=round(float(price), 6),
            greeks=greeks,
            ticker=snap.ticker,
            product=f"Athena MC {'KG' if self.kg=='yes' else 'Capital Risque'}",
            model="heston_mc",
            fair_coupon=round(float(coupon_fair), 6),
            cashflows=cf_df,
            probabilities=prob_df,
        )
