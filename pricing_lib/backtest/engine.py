"""
pricing_lib/backtest/engine.py
-----------------------------------------------------------------------------
Moteur de backtest.

Pour chaque date du calendrier et chaque position active :
1. Recupere le spot historique (pre-charge)
2. Construit le MarketSnapshot avec vol realisee (pre-calculee)
3. Calcule le MtM via le pricer existant
4. Calcule Delta, Gamma, Vega, Theta par differences finies (mode AL)
5. Calcule la decomposition P&L : delta, gamma, vega, theta, inexplique

Quantite (Option B) :
    quantity  = nominal / mtm0
    mtm0      = prix unitaire du produit a start_date
    mtm_total = mtm * quantity  (valeur reelle en euros)

    Au jour 0 : mtm_total = mtm0 * (nominal / mtm0) = nominal
    P&L       = mtm_total(t) - mtm_total(t-1)

Aucun appel HTTP en boucle -- tout est pre-charge avant la boucle principale.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from pricing_lib.backtest.specs import PositionSpec
from pricing_lib.backtest.results import BacktestResults
from pricing_lib.backtest.market_loader import (
    fetch_historical_spots,
    compute_realized_vol,
    fetch_live_market_data,
    build_snapshot,
    _DEFAULT_VOL,
)
from pricing_lib.market_data.market_snapshot import MarketSnapshot
from pricing_lib.market_data.vol_surface import VolSurface
from pricing_lib.pricers.analytical import AnalyticalPricer
from pricing_lib.pricers.mc import MCPricer
from pricing_lib.pricers.autocalls import PhoenixPricer, AthenaPricer


# ---------------------------------------------------------------------------
# Helpers de parsing
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    from datetime import datetime
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Format de date non reconnu : {s!r}")


def _parse_tenor_months(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("M"):
        return int(s[:-1])
    if s.endswith("Y"):
        return int(s[:-1]) * 12
    raise ValueError(f"Format de maturite non reconnu : {s!r}")


def _maturity_date(start: date, maturity_str: str) -> date:
    import calendar as cal
    months = _parse_tenor_months(maturity_str)
    m = start.month + months
    y = start.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    try:
        return date(y, m, start.day)
    except ValueError:
        return date(y, m, cal.monthrange(y, m)[1])


def _business_dates(start: date, end: date, freq: str) -> List[date]:
    """Genere les dates de calcul (jours ouvres uniquement)."""
    import calendar as cal
    dates = []
    cur   = start
    while cur <= end:
        if cur.weekday() < 5:
            dates.append(cur)
        if freq == "daily":
            cur += timedelta(days=1)
        elif freq == "weekly":
            cur += timedelta(weeks=1)
        elif freq == "monthly":
            m = cur.month + 1
            y = cur.year + (m - 1) // 12
            m = (m - 1) % 12 + 1
            try:
                cur = date(y, m, cur.day)
            except ValueError:
                cur = date(y, m, cal.monthrange(y, m)[1])
        else:
            raise ValueError(f"freq='{freq}' invalide. Utiliser daily/weekly/monthly.")
    return dates


def _nearest_spot(spot_map: dict, d: date, max_lookback: int = 5) -> Optional[float]:
    """Retourne le spot a la date d, ou au jour ouvre precedent (jusqu'a max_lookback)."""
    s = spot_map.get(d)
    if s is not None:
        return s
    for back in range(1, max_lookback + 1):
        s = spot_map.get(d - timedelta(days=back))
        if s is not None:
            return s
    return None


# ---------------------------------------------------------------------------
# Bump helpers pour les differences finies
# ---------------------------------------------------------------------------

class _BumpedVolSurface:
    """VolSurface avec toutes les vols decalees d'un montant fixe."""

    def __init__(self, base: VolSurface, bump: float) -> None:
        self._base = base
        self._bump = bump

    def vol(self, T: float, K: float) -> float:
        return max(1e-4, self._base.vol(T, K) + self._bump)

    def sigma(self, T: float, K: float) -> float:
        return max(1e-4, self._base.sigma(T, K) + self._bump)

    def atm_vol(self, T: float) -> float:
        return max(1e-4, self._base.atm_vol(T) + self._bump)

    def __getattr__(self, name: str):
        return getattr(self._base, name)


class _BumpedSnap:
    """
    Proxy leger sur MarketSnapshot avec spot ou vol modifie.
    Implemente explicitement tous les attributs accedes par les pricers.
    """

    def __init__(
        self,
        base:     MarketSnapshot,
        spot:     Optional[float] = None,
        vol_bump: Optional[float] = None,
    ) -> None:
        self._base  = base
        self._spot  = spot if spot is not None else base.spot
        self._vbump = vol_bump

    @property
    def ticker(self) -> str:
        return self._base.ticker

    @property
    def valuation_date(self) -> date:
        return self._base.valuation_date

    @property
    def spot(self) -> float:
        return self._spot

    @property
    def r(self) -> float:
        return self._base.r

    @property
    def q(self) -> float:
        return self._base.q

    @property
    def ois_curve(self):
        return self._base.ois_curve

    @property
    def div_curve(self):
        return self._base.div_curve

    @property
    def vol_surface(self):
        if self._vbump is not None:
            return _BumpedVolSurface(self._base.vol_surface, self._vbump)
        return self._base.vol_surface

    def sigma(self, T: float, K: float) -> float:
        return self.vol_surface.sigma(T, K)

    def atm_vol(self, T: float) -> float:
        return self.vol_surface.atm_vol(T)

    def df(self, tau: float) -> float:
        return self._base.df(tau)

    def __repr__(self) -> str:
        return f"_BumpedSnap(ticker={self.ticker!r}, spot={self._spot})"


# ---------------------------------------------------------------------------
# Pricing d'une position
# ---------------------------------------------------------------------------

def _price_position(spec: PositionSpec, snap: MarketSnapshot, T: float) -> float:
    """
    Price une position. Retourne le prix unitaire (MtM par unite).
    T : maturite residuelle en annees.
    """
    if T <= 0:
        return 0.0

    p  = spec.params
    pt = spec.product_type

    if spec.mode == "AL":
        pr = AnalyticalPricer(snap)
        if pt in ("call",):
            return pr.call(p["strike"], T).price
        elif pt in ("put",):
            return pr.put(p["strike"], T).price
        elif pt in ("warrant",):
            return pr.warrant_product(p.get("flag", "c"), p["strike"], T,
                                      p.get("ratio", 1.0)).price
        elif pt in ("rc", "reverse_convertible"):
            return pr.reverse_convertible(
                cap=p.get("cap", snap.spot), T=T,
                freq=p.get("freq", "A"),
                nominal=p.get("nominal", 100.0),
                ratio=p.get("ratio", 1.0),
                start=snap.valuation_date,
            ).price
        elif pt in ("bonus", "bonus_certificate"):
            return pr.bonus_certificate(p["X"], p["B"], T).price
        elif pt in ("reverse_bonus", "reverse_bonus_certificate"):
            return pr.reverse_bonus_certificate(p["X"], p["B"], T).price
        elif pt in ("twin_win", "twin_win_certificate"):
            return pr.twin_win_certificate(p["X"], p["B"], T).price
        elif pt in ("airbag", "airbag_certificate"):
            return pr.airbag_certificate(p["X"], p["B"], T).price
        elif pt in ("discount", "discount_certificate"):
            cap_val = p.get("X", p.get("cap", snap.spot))
            return pr.discount_certificate(cap_val, T).price
        elif pt in ("sprint", "sprint_certificate"):
            return pr.sprint_certificate(p["X"], p["cap_level"], T).price
        elif pt in ("outperformance", "outperformance_certificate"):
            return pr.outperformance_certificate(p["X"], T).price
        elif pt in ("garantie", "garantie_certificate"):
            return pr.garantie_certificate(p["X"], T).price
        else:
            raise ValueError(f"product_type '{pt}' non supporte en mode AL.")

    else:  # MC
        n_paths = spec.n_paths
        seed    = spec.seed

        if pt in ("phoenix",):
            Bc = p["barrier_coupon"]  * snap.spot if p["barrier_coupon"]  < 10 else p["barrier_coupon"]
            Br = p["barrier_recall"]  * snap.spot if p["barrier_recall"]  < 10 else p["barrier_recall"]
            Bk = p["capital_barrier"] * snap.spot if p["capital_barrier"] < 10 else p["capital_barrier"]
            mat_months = max(1, round(T * 12))
            return PhoenixPricer(
                snapshot=snap, n_paths=n_paths, start_date=snap.valuation_date,
                S0=snap.spot, barrier_coupon=Bc, barrier_recall=Br, capital_barrier=Bk,
                freq_months=int(p.get("freq_months", 3)),
                maturity_months=mat_months,
                kg=p.get("kg", "no"),
                autocall_start=int(p.get("autocall_start", 0)),
                mode="MC", coupon=float(p.get("coupon", 0.0)),
                compute_greeks=False, seed=seed,
            ).price().price

        elif pt in ("athena",):
            Br = p["barrier_recall"]  * snap.spot if p["barrier_recall"]  < 10 else p["barrier_recall"]
            Bk = p["capital_barrier"] * snap.spot if p["capital_barrier"] < 10 else p["capital_barrier"]
            mat_months = max(1, round(T * 12))
            return AthenaPricer(
                snapshot=snap, n_paths=n_paths, start_date=snap.valuation_date,
                S0=snap.spot, barrier_recall=Br, capital_barrier=Bk,
                freq_months=int(p.get("freq_months", 3)),
                maturity_months=mat_months,
                kg=p.get("kg", "no"),
                mode="MC", coupon=float(p.get("coupon", 0.0)),
                compute_greeks=False, seed=seed,
            ).price().price

        else:
            mc = MCPricer(snap, n_paths, seed=seed)
            if pt in ("call",):
                return mc.call(p["strike"], T).price
            elif pt in ("put",):
                return mc.put(p["strike"], T).price
            elif pt in ("rc", "reverse_convertible"):
                return mc.reverse_convertible(
                    cap=p.get("cap", snap.spot), T=T,
                    freq=p.get("freq", "A"),
                    nominal=p.get("nominal", 100.0),
                    ratio=p.get("ratio", 1.0),
                    start=snap.valuation_date,
                ).price
            elif pt in ("bonus", "bonus_certificate"):
                return mc.bonus_certificate(p["X"], p["B"], T).price
            elif pt in ("reverse_bonus", "reverse_bonus_certificate"):
                return mc.reverse_bonus_certificate(p["X"], p["B"], T).price
            elif pt in ("twin_win", "twin_win_certificate"):
                return mc.twin_win_certificate(p["X"], p["B"], T).price
            elif pt in ("airbag", "airbag_certificate"):
                return mc.airbag_certificate(p["X"], p["B"], T).price
            elif pt in ("discount", "discount_certificate"):
                cap_val = p.get("X", p.get("cap", snap.spot))
                return mc.discount_certificate(cap_val, T).price
            elif pt in ("sprint", "sprint_certificate"):
                return mc.sprint_certificate(p["X"], p["cap_level"], T).price
            elif pt in ("outperformance", "outperformance_certificate"):
                return mc.outperformance_certificate(p["X"], T).price
            elif pt in ("garantie", "garantie_certificate"):
                return mc.garantie_certificate(p["X"], T).price
            else:
                raise ValueError(f"product_type '{pt}' non supporte en mode MC.")


# ---------------------------------------------------------------------------
# Greeks par differences finies
# ---------------------------------------------------------------------------

def _compute_greeks(
    spec:    PositionSpec,
    snap:    MarketSnapshot,
    T:       float,
    p0:      float,
    dS_frac: float = 0.001,
    dsig:    float = 0.01,
) -> dict:
    """
    Delta, Gamma, Vega, Theta par differences finies centrees.

    dS_frac : bump spot relatif (0.1% par defaut)
    dsig    : bump vol absolu   (1 vol point = 0.01 par defaut)

    Sorties
    -------
    delta  : d(price)/d(S)                    (par euro de spot)
    gamma  : d2(price)/d(S2)
    vega   : d(price)/d(sigma)                (par unite de sigma decimal)
             PnL_vega  = vega  * dvol         avec dvol en decimal
    theta  : d(price)/d(T) annualise
             PnL_theta = theta * dT           avec dT = T_prev - T_curr (annees)
    """
    S  = snap.spot
    dS = S * dS_frac

    # Delta & Gamma -- convention sticky-strike (bump spot, vol fixe)
    snap_su = _BumpedSnap(snap, spot=S + dS)
    snap_sd = _BumpedSnap(snap, spot=S - dS)
    p_su    = _price_position(spec, snap_su, T)
    p_sd    = _price_position(spec, snap_sd, T)
    delta   = (p_su - p_sd) / (2 * dS)
    gamma   = (p_su - 2 * p0 + p_sd) / (dS ** 2)

    # Vega -- bump vol symetrique, spot fixe
    snap_vu = _BumpedSnap(snap, vol_bump=+dsig)
    snap_vd = _BumpedSnap(snap, vol_bump=-dsig)
    p_vu    = _price_position(spec, snap_vu, T)
    p_vd    = _price_position(spec, snap_vd, T)
    vega    = (p_vu - p_vd) / (2 * dsig)  # par unite de sigma decimal

    # Theta -- decalage d'1 jour calendaire vers la maturite
    dT    = 1.0 / 365.0
    T_sh  = max(T - dT, 1e-6)
    p_th  = _price_position(spec, snap, T_sh)
    theta = (p_th - p0) / dT  # annualise

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega":  float(vega),
        "theta": float(theta),
    }


# ---------------------------------------------------------------------------
# Prix initial pour le calcul de quantity (Option B)
# ---------------------------------------------------------------------------

def _compute_initial_mtm(
    spec:    PositionSpec,
    spots:   dict,
    vols:    dict,
    ois:     object,
    div:     object,
    verbose: bool = False,
) -> float:
    """
    Calcule le prix unitaire du produit a start_date (mtm0).
    quantity = nominal / mtm0 (Option B).
    Retourne NaN si le calcul echoue.
    """
    start_d = _parse_date(spec.start_date)
    spot_0  = _nearest_spot(spots, start_d)

    if spot_0 is None:
        if verbose:
            print(f"\n  [WARN] Pas de spot a {start_d} pour {spec.product_id}.")
        return float("nan")

    vol_0  = vols.get(start_d, _DEFAULT_VOL)
    snap_0 = build_snapshot(
        ticker         = spec.ticker,
        valuation_date = start_d,
        spot           = spot_0,
        ois            = ois,
        div            = div,
        realized_vol   = vol_0,
    )
    T0 = _parse_tenor_months(spec.maturity) / 12.0

    try:
        return _price_position(spec, snap_0, T0)
    except Exception as e:
        if verbose:
            print(f"\n  [WARN] mtm0 {spec.product_id}: {e}")
        return float("nan")


# ---------------------------------------------------------------------------
# Decomposition P&L
# ---------------------------------------------------------------------------

def _compute_pnl_attribution(df_pos: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les colonnes de decomposition P&L a df_pos.

    Formules
    --------
    pnl             = mtm_total(t) - mtm_total(t-1)
    pnl_delta       = delta(t-1) * dS * quantity
    pnl_gamma       = 0.5 * gamma(t-1) * dS**2 * quantity
    pnl_vega        = vega(t-1) * dvol * quantity
    pnl_theta       = theta(t-1) * dT * quantity
    pnl_unexplained = pnl - (pnl_delta + pnl_gamma + pnl_vega + pnl_theta)

    Chocs
    -----
    dS   = spot(t) - spot(t-1)
    dvol = realized_vol(t) - realized_vol(t-1)   (decimal)
    dT   = T_residual(t-1) - T_residual(t)       (annees, > 0)

    Notes
    -----
    - Premiere date de chaque position : NaN (pas de J-1).
    - Greques NaN (mode MC) : tous les composants P&L sont NaN.
    - Greques de J-1 utilisees (convention fin de journee).
    """
    groups: list = []

    for _pid, grp in df_pos.groupby(level="product_id"):
        grp = grp.copy().sort_index(level="date")

        qty = grp["quantity"]

        # Valeurs J-1
        spot_prev      = grp["spot"].shift(1)
        vol_prev       = grp["realized_vol"].shift(1)
        T_prev         = grp["T_residual"].shift(1)
        mtm_total_prev = grp["mtm_total"].shift(1)
        delta_prev     = grp["delta"].shift(1)
        gamma_prev     = grp["gamma"].shift(1)
        vega_prev      = grp["vega"].shift(1)
        theta_prev     = grp["theta"].shift(1)

        # Chocs
        dS   = grp["spot"]         - spot_prev
        dvol = grp["realized_vol"] - vol_prev
        dT   = T_prev              - grp["T_residual"]  # > 0

        # P&L total
        grp["pnl"] = grp["mtm_total"] - mtm_total_prev

        # Composants expliques
        grp["pnl_delta"] = delta_prev * dS              * qty
        grp["pnl_gamma"] = 0.5 * gamma_prev * (dS ** 2) * qty
        grp["pnl_vega"]  = vega_prev  * dvol            * qty
        grp["pnl_theta"] = theta_prev * dT              * qty

        # Residuel -- NaN si un composant est NaN (greques absentes ou premiere date)
        grp["pnl_unexplained"] = (
            grp["pnl"]
            - grp["pnl_delta"]
            - grp["pnl_gamma"]
            - grp["pnl_vega"]
            - grp["pnl_theta"]
        )

        groups.append(grp)

    return pd.concat(groups).sort_index()


# ---------------------------------------------------------------------------
# Moteur principal
# ---------------------------------------------------------------------------

class BacktestEngine:

    def __init__(
        self,
        positions:      List[PositionSpec],
        start_date:     Optional[str],
        end_date:       Optional[str],
        freq:           str  = "daily",
        aggregation:    str  = "quantity",
        weights:        Optional[Dict[str, float]] = None,
        compute_greeks: bool = True,
        verbose:        bool = True,
    ) -> None:
        self.positions      = positions
        self.freq           = freq
        self.aggregation    = aggregation
        self.weights        = weights or {}
        self.compute_greeks = compute_greeks
        self.verbose        = verbose

        all_starts = [_parse_date(p.start_date) for p in positions]
        all_ends   = [
            _maturity_date(_parse_date(p.start_date), p.maturity)
            for p in positions
        ]

        self.bt_start = _parse_date(start_date) if start_date else min(all_starts)
        self.bt_end   = _parse_date(end_date)   if end_date   else max(all_ends)

        self._mat_date: dict = {
            p.product_id: _maturity_date(_parse_date(p.start_date), p.maturity)
            for p in positions
        }
        self._start_d: dict = {
            p.product_id: _parse_date(p.start_date)
            for p in positions
        }

    # -----------------------------------------------------------------------
    # run()
    # -----------------------------------------------------------------------

    def run(self) -> BacktestResults:
        dates = _business_dates(self.bt_start, self.bt_end, self.freq)

        # -- 1. Pre-chargement par ticker ------------------------------------
        tickers = list({p.ticker for p in self.positions})

        spots_by_ticker: dict = {}
        for tk in tickers:
            if self.verbose:
                print(f"[Backtest] Fetch spots {tk} ...")
            spots_by_ticker[tk] = fetch_historical_spots(
                tk, self.bt_start, self.bt_end
            )

        vol_by_ticker: dict = {
            tk: compute_realized_vol(spots_by_ticker[tk])
            for tk in tickers
        }

        ois_by_ticker: dict = {}
        div_by_ticker: dict = {}
        for tk in tickers:
            if self.verbose:
                print(f"[Backtest] Fetch OIS / dividende {tk} ...")
            ois, div = fetch_live_market_data(tk, spot=100.0)
            ois_by_ticker[tk] = ois
            div_by_ticker[tk] = div

        # -- 2. Calcul des quantites (Option B) ------------------------------
        if self.verbose:
            print("[Backtest] Calcul des quantites initiales ...")

        qty_by_id: dict = {}
        for spec in self.positions:

            if spec.quantity is not None:
                qty_by_id[spec.product_id] = spec.quantity
                if self.verbose:
                    print(
                        f"  {spec.product_id}: quantity explicite = "
                        f"{spec.quantity:,.4f}"
                    )
                continue

            mtm_0 = _compute_initial_mtm(
                spec    = spec,
                spots   = spots_by_ticker[spec.ticker],
                vols    = vol_by_ticker[spec.ticker],
                ois     = ois_by_ticker[spec.ticker],
                div     = div_by_ticker[spec.ticker],
                verbose = self.verbose,
            )

            if math.isnan(mtm_0) or mtm_0 <= 0:
                # Fallback si pricing initial echoue
                spot_0 = _nearest_spot(
                    spots_by_ticker[spec.ticker],
                    self._start_d[spec.product_id],
                )
                fallback = spot_0 if spot_0 is not None else spec.spot
                qty = spec.nominal / fallback
                if self.verbose:
                    print(
                        f"  [WARN] {spec.product_id}: mtm0 invalide ({mtm_0}) -- "
                        f"fallback quantity = nominal / spot0 = {qty:,.2f}"
                    )
            else:
                qty = spec.nominal / mtm_0
                if self.verbose:
                    print(
                        f"  {spec.product_id}: "
                        f"mtm0 = {mtm_0:.4f} EUR | "
                        f"quantity = {qty:,.0f} unites"
                    )

            qty_by_id[spec.product_id] = qty

        # -- 3. Boucle principale --------------------------------------------
        rows: list = []

        for d in dates:
            if self.verbose:
                print(f"[Backtest] {d} ...", end="\r")

            for spec in self.positions:
                mat_date = self._mat_date[spec.product_id]
                start_d  = self._start_d[spec.product_id]

                if d < start_d or d > mat_date:
                    continue

                T = max((mat_date - d).days / 365.0, 0.0)

                spot_hist = _nearest_spot(
                    spots_by_ticker.get(spec.ticker, {}), d
                )
                if spot_hist is None:
                    continue

                realized_vol = vol_by_ticker.get(
                    spec.ticker, {}
                ).get(d, _DEFAULT_VOL)

                snap = build_snapshot(
                    ticker         = spec.ticker,
                    valuation_date = d,
                    spot           = spot_hist,
                    ois            = ois_by_ticker[spec.ticker],
                    div            = div_by_ticker[spec.ticker],
                    realized_vol   = realized_vol,
                )

                try:
                    mtm = _price_position(spec, snap, T)
                except Exception as e:
                    if self.verbose:
                        print(f"\n  [WARN] Price {spec.product_id} {d}: {e}")
                    mtm = float("nan")

                qty       = qty_by_id[spec.product_id]
                mtm_total = mtm * qty if not math.isnan(mtm) else float("nan")

                row: dict = {
                    "date":         d,
                    "product_id":   spec.product_id,
                    "spot":         spot_hist,
                    "T_residual":   round(T, 6),
                    "realized_vol": round(realized_vol, 6),
                    "mtm":          mtm,
                    "quantity":     qty,
                    "nominal":      spec.nominal,
                    "mtm_total":    mtm_total,
                    "delta":        float("nan"),
                    "gamma":        float("nan"),
                    "vega":         float("nan"),
                    "theta":        float("nan"),
                }

                # Greques -- mode AL uniquement
                if (
                    self.compute_greeks
                    and spec.mode == "AL"
                    and not math.isnan(mtm)
                    and T > 1.0 / 365.0
                ):
                    try:
                        g = _compute_greeks(spec, snap, T, mtm)
                        row.update(g)
                    except Exception as e:
                        if self.verbose:
                            print(
                                f"\n  [WARN] Greeks {spec.product_id} {d}: {e}"
                            )

                rows.append(row)

        if self.verbose:
            print()

        if not rows:
            raise RuntimeError(
                "Aucune donnee produite. Verifiez les tickers et les dates."
            )

        # -- 4. Construction de df_pos ---------------------------------------
        df_pos = (
            pd.DataFrame(rows)
            .assign(date=lambda df: pd.to_datetime(df["date"]))
            .set_index(["date", "product_id"])
            .sort_index()
        )

        # -- 5. Decomposition P&L --------------------------------------------
        df_pos = _compute_pnl_attribution(df_pos)

        # -- 6. Agregation portefeuille --------------------------------------
        df_port = self._aggregate(df_pos)

        return BacktestResults(positions=df_pos, portfolio=df_port)

    # -----------------------------------------------------------------------
    # _aggregate()
    # -----------------------------------------------------------------------

    def _aggregate(self, df_pos: pd.DataFrame) -> pd.DataFrame:
        """
        Agrege les positions au niveau portefeuille par date.

        portfolio_value = sum(mtm_total)
        greques         = sum(greek * quantity)   (sensibilite euros agregee)
        pnl_*           = sum(pnl_*) sur les positions actives

        mode custom_weights : greques ponderees par les poids fournis.
        """
        greek_cols = ["delta", "gamma", "vega", "theta"]
        pnl_cols   = [
            "pnl",
            "pnl_delta",
            "pnl_gamma",
            "pnl_vega",
            "pnl_theta",
            "pnl_unexplained",
        ]
        records: list = []

        for d, grp in df_pos.groupby(level="date"):

            quantities  = grp["quantity"].values.astype(float)
            mtm_totals  = grp["mtm_total"].values.astype(float)

            portfolio_value = float(np.nansum(mtm_totals))

            # Poids pour les greques
            if self.aggregation == "custom_weights":
                ids = grp.index.get_level_values("product_id")
                w   = np.array([self.weights.get(pid, 0.0) for pid in ids])
                s   = w.sum()
                greek_weights = w / s if s > 0 else np.ones(len(w)) / len(w)
            else:
                # Agregation naturelle : sensibilite euros = greek * qty
                greek_weights = quantities

            row: dict = {"date": d, "portfolio_value": portfolio_value}

            for col in greek_cols:
                if col in grp.columns:
                    vals     = grp[col].values.astype(float)
                    row[col] = float(np.nansum(vals * greek_weights))

            for col in pnl_cols:
                if col in grp.columns:
                    vals = grp[col].values.astype(float)
                    # NaN si toutes les positions sont NaN (premiere date)
                    row[col] = (
                        float(np.nansum(vals))
                        if not np.all(np.isnan(vals))
                        else float("nan")
                    )

            records.append(row)

        df_port = (
            pd.DataFrame(records)
            .set_index("date")
            .sort_index()
        )
        df_port.index = pd.to_datetime(df_port.index)
        return df_port
