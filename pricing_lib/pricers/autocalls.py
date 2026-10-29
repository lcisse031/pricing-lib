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
      Le coupon_fair est aussi renvoyé à titre indicatif.

Les coupons sont exprimés par période d'observation.
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
    """Taux zéro continu OIS jusqu'à T, utilisé comme drift de simulation."""
    T = max(float(T), 1e-6)
    return -math.log(max(ois_curve.df_tau(T), 1e-12)) / T


# ─── Barrières : fraction de S0 ou niveau absolu ─────────────────────────────

_FRACTION_MAX = 3.0


def barrier_level(x: float, S0: float) -> float:
    """
    x ≤ 3 : fraction de S0 (0.70 = 70 %), sinon niveau absolu.
    """
    if x is None:
        return x
    x = float(x)
    return x * float(S0) if x <= _FRACTION_MAX else x


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


def _future_schedule(dates, valuation_date: date):
    """
    Dates d'observation postérieures à valuation_date.
    Retourne (dates, taus, k0), k0 = nombre de dates déjà passées.
    Suppose le produit encore vivant : les fixings passés sont gérés par
    l'appelant (backtest/engine.py).
    """
    fut = [d for d in dates if d > valuation_date]
    if not fut:
        raise ValueError(f"Produit échu au {valuation_date} : plus aucune date d'observation.")
    k0   = len(dates) - len(fut)
    taus = np.array([(d - valuation_date).days / 365.25 for d in fut])
    return fut, taus, k0


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
        valuation_date: Optional[date] = None,
        spot: Optional[float] = None,
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
        self._autocall_start_global = autocall_start
        self.autocall_start   = autocall_start
        self.mode             = mode.upper()
        self.coupon           = coupon
        self.compute_greeks   = compute_greeks
        self.antithetic       = antithetic
        self.seed             = seed
        self.params           = params or get_heston_params(snapshot)

        # S0 : niveau de référence (perte = S_T/S0). spot : point de départ des
        # simulations, différent de S0 en cours de vie.
        self.spot           = float(spot) if spot is not None else float(S0)
        self.valuation_date = (valuation_date
                               or getattr(snapshot, "valuation_date", None)
                               or start_date)
        if self.valuation_date < start_date:
            self.valuation_date = start_date
        all_dates, _ = _observation_dates(start_date, freq_months, maturity_months)
        self.obs_dates, self.obs_taus, self.k0 = _future_schedule(all_dates, self.valuation_date)
        self.n_total   = len(all_dates)
        self.n_periods = len(self.obs_dates)
        self.T         = self.obs_taus[-1]
        self.autocall_start = max(self._autocall_start_global - self.k0, 0)

    # ── Simulation ────────────────────────────────────────────────────────────

    def _simulate(self) -> np.ndarray:
        """Simule les trajectoires — retourne paths (n_paths, n_periods)."""
        _, paths = heston_simulate_paths(
            S=self.spot,
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
        Différences finies centrées, re-simulation à n_paths//8 avec le même seed.
        Delta/gamma sur le spot, vega = ±1 pt sur √v0 et √θ, rho via la courbe
        décalée, theta = date de valo ± 1 jour.
        En mode AL, le coupon fair est figé et les bumps sont pricés en MC.
        """
        from datetime import timedelta
        from pricing_lib.risk.greeks import _assemble, _BumpedMarket

        n_bump = max(self.n_paths // 8, 500)
        dS     = self.spot * 0.01
        dvol   = 0.01      # 1 point de vol
        dr     = 0.0001

        coupon_b = fair_coupon if (self.mode == "AL" and fair_coupon is not None) \
                   else self.coupon

        def _sub_price(S_new, params_new=None, snap_new=None, val_new=None) -> float:
            p  = params_new if params_new is not None else self.params
            sn = snap_new   if snap_new   is not None else self.snap
            vd = val_new    if val_new    is not None else self.valuation_date
            sub = PhoenixPricer(
                sn, n_bump, self.start_date, self.S0,
                self.barrier_coupon, self.barrier_recall, self.capital_barrier,
                self.freq_months, self.maturity_months, self.kg,
                self._autocall_start_global,
                mode="MC", coupon=coupon_b, params=p, seed=self.seed,
                valuation_date=vd, spot=S_new,
            )
            return sub._compute_mc(sub._simulate()).price

        def _safe(fn, fallback):
            try:
                return float(fn())
            except Exception:
                return fallback

        # p0 au même seed et nombre de chemins que les chocs
        p0 = _safe(lambda: _sub_price(self.spot), price_0)

        # Spot ± dS (delta, gamma)
        p_S_up   = _safe(lambda: _sub_price(self.spot + dS), p0)
        p_S_down = _safe(lambda: _sub_price(self.spot - dS), p0)

        # Vol : ± dvol sur √v0 et √θ
        vol_v0 = math.sqrt(max(self.params.v0,    0.0))
        vol_th = math.sqrt(max(self.params.theta, 0.0))
        par_up = _dc_replace(self.params, v0=(vol_v0 + dvol) ** 2,
                                          theta=(vol_th + dvol) ** 2)
        par_dn = _dc_replace(self.params, v0=max(vol_v0 - dvol, 1e-4) ** 2,
                                          theta=max(vol_th - dvol, 1e-4) ** 2)
        p_vol_up   = _safe(lambda: _sub_price(self.spot, params_new=par_up), p0)
        p_vol_down = _safe(lambda: _sub_price(self.spot, params_new=par_dn), p0)

        # Taux ± dr
        p_r_up   = _safe(lambda: _sub_price(self.spot, snap_new=_BumpedMarket(self.snap, rate_bump=+dr)), p0)
        p_r_down = _safe(lambda: _sub_price(self.spot, snap_new=_BumpedMarket(self.snap, rate_bump=-dr)), p0)

        # Theta : date de valorisation ± 1 jour, échéancier inchangé
        p_T_down = _safe(lambda: _sub_price(self.spot, val_new=self.valuation_date + timedelta(days=1)), p0)
        p_T_up   = _safe(lambda: _sub_price(self.spot, val_new=self.valuation_date - timedelta(days=1)), p0)

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
            coupon_periods_per_year=12 // self.freq_months,
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
            coupon_periods_per_year=12 // self.freq_months,
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
        valuation_date: Optional[date] = None,
        spot: Optional[float] = None,
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

        # S0 : niveau de référence (perte = S_T/S0). spot : point de départ des
        # simulations, différent de S0 en cours de vie.
        self.spot           = float(spot) if spot is not None else float(S0)
        self.valuation_date = (valuation_date
                               or getattr(snapshot, "valuation_date", None)
                               or start_date)
        if self.valuation_date < start_date:
            self.valuation_date = start_date
        all_dates, _ = _observation_dates(start_date, freq_months, maturity_months)
        self.obs_dates, self.obs_taus, self.k0 = _future_schedule(all_dates, self.valuation_date)
        self.n_total   = len(all_dates)
        self.n_periods = len(self.obs_dates)
        self.T         = self.obs_taus[-1]

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def price(self) -> PricingResult:
        _, paths = heston_simulate_paths(
            S=self.spot, r=_fwd_rate(self.snap.ois_curve, self.T), q=self.snap.q, T=self.T,
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
        # cumul_r[:, i] : rappel déclenché sur une date strictement antérieure à i
        cumsum = np.cumsum(trigger_r, axis=1)
        cumul_r = np.zeros_like(trigger_r, dtype=bool)
        cumul_r[:, 1:] = cumsum[:, :-1] > 0
        has_trigger = trigger_r[:, :-1].any(axis=1)
        return trigger_r, cumul_r, has_trigger, ~has_trigger

    def _extract_probs(self, paths, trigger_r, cumul_r, has_trigger, no_trigger):
        n_paths, n_periods = paths.shape
        p_recall, c_recall = [], []
        for i in range(n_periods - 1):
            cond = trigger_r[:, i] & ~cumul_r[:, i]
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

        Sum_A_i = sum(p_recall[i] * dfs[i] * (self.k0 + i + 1) for i in range(n_periods - 1))
        n_mem   = self.n_total
        Sum_A   = sum(p_recall[i] * dfs[i]            for i in range(n_periods - 1))

        if self.kg == "no":
            mask_h    = (ST_no >= self.capital_barrier) if len(ST_no) > 0 else np.array([])
            mask_loss = (ST_no <  self.capital_barrier) if len(ST_no) > 0 else np.array([])
            prob_h    = float(mask_h.mean())    if len(mask_h) > 0    else 1.0
            prob_loss = float(mask_loss.mean()) if len(mask_loss) > 0 else 0.0
            payoff_moy = float((ST_no[mask_loss] / S0).mean()) if mask_loss.sum() > 0 else 0.0
            num   = 1.0 - (Sum_A + p_no * df_T * (prob_h + payoff_moy * prob_loss))
            denom = Sum_A_i + n_mem * p_no * df_T * prob_h
        else:
            prob_h = 1.0; prob_loss = 0.0; payoff_moy = 0.0
            num   = 1.0 - (Sum_A + p_no * df_T)
            denom = Sum_A_i + n_mem * p_no * df_T

        coupon_fair = num / denom if denom > 1e-10 else 0.0
        return coupon_fair, Sum_A, Sum_A_i, prob_h, prob_loss, payoff_moy

    # ── Greeks (delta, gamma, vega) ──────────────────────────────────────────

    def _compute_greeks(self, price_0: float, fair_coupon: float = None) -> dict:
        """
        Même méthode que PhoenixPricer._compute_greeks.
        """
        from datetime import timedelta
        from pricing_lib.risk.greeks import _assemble, _BumpedMarket

        n_bump = max(self.n_paths // 8, 500)
        dS     = self.spot * 0.01
        dvol   = 0.01
        dr     = 0.0001

        coupon_b = fair_coupon if (self.mode == "AL" and fair_coupon is not None) \
                   else self.coupon

        def _sub_price(S_new, params_new=None, snap_new=None, val_new=None) -> float:
            p  = params_new if params_new is not None else self.params
            sn = snap_new   if snap_new   is not None else self.snap
            vd = val_new    if val_new    is not None else self.valuation_date
            sub = AthenaPricer(
                sn, n_bump, self.start_date, self.S0,
                self.barrier_recall, self.capital_barrier,
                self.freq_months, self.maturity_months, self.kg,
                mode="MC", coupon=coupon_b, params=p, seed=self.seed,
                valuation_date=vd, spot=S_new,
            )
            _, paths_sub = heston_simulate_paths(
                S=S_new, r=_fwd_rate(sn.ois_curve, sub.T), q=sn.q, T=sub.T,
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

        # p0 au même seed et nombre de chemins que les chocs
        p0 = _safe(lambda: _sub_price(self.spot), price_0)

        p_S_up   = _safe(lambda: _sub_price(self.spot + dS), p0)
        p_S_down = _safe(lambda: _sub_price(self.spot - dS), p0)

        vol_v0 = math.sqrt(max(self.params.v0,    0.0))
        vol_th = math.sqrt(max(self.params.theta, 0.0))
        par_up = _dc_replace(self.params, v0=(vol_v0 + dvol) ** 2,
                                          theta=(vol_th + dvol) ** 2)
        par_dn = _dc_replace(self.params, v0=max(vol_v0 - dvol, 1e-4) ** 2,
                                          theta=max(vol_th - dvol, 1e-4) ** 2)
        p_vol_up   = _safe(lambda: _sub_price(self.spot, params_new=par_up), p0)
        p_vol_down = _safe(lambda: _sub_price(self.spot, params_new=par_dn), p0)

        p_r_up   = _safe(lambda: _sub_price(self.spot, snap_new=_BumpedMarket(self.snap, rate_bump=+dr)), p0)
        p_r_down = _safe(lambda: _sub_price(self.spot, snap_new=_BumpedMarket(self.snap, rate_bump=-dr)), p0)

        # Theta : date de valorisation ± 1 jour, échéancier inchangé
        p_T_down = _safe(lambda: _sub_price(self.spot, val_new=self.valuation_date + timedelta(days=1)), p0)
        p_T_up   = _safe(lambda: _sub_price(self.spot, val_new=self.valuation_date - timedelta(days=1)), p0)

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
                     + coupon_fair * self.n_total * p_no * df_T * prob_h) * 100.0
        else:
            price = (Sum_A + coupon_fair * Sum_A_i
                     + p_no * df_T
                     + coupon_fair * self.n_total * p_no * df_T) * 100.0

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
            "Mémoire":   list(range(self.k0 + 1, self.k0 + n_periods + 1)),
            "PV_coupon": (np.array(p_recall + [p_no]) * dfs
                          * np.arange(self.k0 + 1, self.k0 + n_periods + 1) * coupon_fair).round(6),
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
            coupon_periods_per_year=12 // self.freq_months,
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
            mask_ac = trigger_r[:, i] & ~cumul_r[:, i]
            # nominal + coupons mémorisés
            payoff[mask_ac, i] = 1.0 + (self.k0 + i + 1) * coupon

        # Payoff final
        ST = paths[:, -1]
        if self.kg == "no":
            mask_h    = no_trigger & (ST >= self.capital_barrier)
            mask_loss = no_trigger & (ST <  self.capital_barrier)
            payoff[mask_h,    -1] = 1.0 + self.n_total * coupon
            payoff[mask_loss, -1] = ST[mask_loss] / S0
        else:
            payoff[no_trigger, -1] = 1.0 + self.n_total * coupon

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
            "Mémoire":   list(range(self.k0 + 1, self.k0 + n_periods + 1)),
            "PV_coupon": (np.array(p_recall + [p_no]) * dfs
                          * np.arange(self.k0 + 1, self.k0 + n_periods + 1) * coupon).round(6),
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
            coupon_periods_per_year=12 // self.freq_months,
            cashflows=cf_df,
            probabilities=prob_df,
        )
