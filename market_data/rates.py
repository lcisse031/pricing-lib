from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Dict, List, Sequence

import numpy as np
import requests
from bs4 import BeautifulSoup


_API_URL = "https://app.bluegamma.io/public/swap-rates/table/estr"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bluegamma.io/swap-rates/estr-swap-rates",
    "Accept": "text/html,application/xhtml+xml,*/*",
}
_DAY_COUNT = 360.0  # Act/360


def _tenor_to_years(tenor: str) -> float:
    """'1Y' → 1.0 | '6M' → 0.5 | '3W' → 0.0583 | '2D' → 0.00556"""
    n = int(tenor[:-1])
    unit = tenor[-1].upper()
    if unit == "D":
        return n / _DAY_COUNT
    if unit == "W":
        return n * 7.0 / _DAY_COUNT
    if unit == "M":
        return n * 30.0 / _DAY_COUNT
    if unit == "Y":
        return float(n)
    raise ValueError(f"Tenor inconnu : {tenor!r}")


def _bootstrap(rates: Dict[str, float]) -> Dict[float, float]:
    """Bootstrap discret → discount factors."""
    items = sorted(((_tenor_to_years(k), k, v) for k, v in rates.items()))
    dfs: Dict[float, float] = {0.0: 1.0}

    for T, _, r in items:
        if T < 1.0:
            dfs[T] = 1.0 / (1.0 + r * T)
        else:
            times_arr = np.array(sorted(dfs.keys()))
            logdf_arr = np.array([math.log(dfs[t]) for t in times_arr])
            payment_times = np.arange(1.0, T, 1.0)
            known_sum = sum(
                math.exp(float(np.interp(t, times_arr, logdf_arr)))
                for t in payment_times
            )
            dfs[T] = (1.0 - r * known_sum) / (1.0 + r)

    return dfs


class OISCurve:
    """Courbe OIS bootstrappée, interpolation log-linéaire."""

    def __init__(self, valuation_date: date, dfs: Dict[float, float]) -> None:
        self.valuation_date = valuation_date
        _sorted = sorted(dfs.items())
        self._times   = np.array([t for t, _ in _sorted], dtype=float)
        self._log_dfs = np.array([math.log(max(df, 1e-12)) for _, df in _sorted], dtype=float)

    # ── interface publique ────────────────────────────────────────────────────

    def df(self, target: date) -> float:
        """Discount factor vers `target`."""
        tau = self._tau(target)
        if tau <= 0.0:
            return 1.0
        return float(np.exp(np.interp(tau, self._times, self._log_dfs)))

    def df_tau(self, tau: float) -> float:
        """Discount factor pour une durée tau (années)."""
        if tau <= 0.0:
            return 1.0
        return float(np.exp(np.interp(tau, self._times, self._log_dfs)))

    def zero_rate(self, target: date) -> float:
        """Taux zéro continu Act/360 vers `target`."""
        tau = self._tau(target)
        if tau <= 0.0:
            return 0.0
        return -math.log(self.df(target)) / tau

    def zero_rate_tau(self, tau: float) -> float:
        """Taux zéro continu Act/360 pour une durée tau (années)."""
        if tau <= 0.0:
            return 0.0
        return -math.log(self.df_tau(tau)) / tau

    def forward_rate(self, t1: date, t2: date) -> float:
        """Taux forward continu Act/360 entre t1 et t2."""
        tau1 = self._tau(t1)
        tau2 = self._tau(t2)
        if tau2 <= tau1:
            raise ValueError("t2 doit être postérieur à t1")
        return (math.log(self.df(t1)) - math.log(self.df(t2))) / (tau2 - tau1)

    def df_many(self, dates: Sequence[date]) -> Dict[date, float]:
        """Vectorisé : {date: DF}."""
        return {d: self.df(d) for d in dates}

    def summary(self) -> None:
        """Affichage console de la courbe."""
        print(f"{'Date':<15} {'τ (ans)':>8} {'Rate (%)':>10} {'DF':>12}")
        print("─" * 50)
        for days in [1, 30, 90, 180, 365, 730, 1825, 3650]:
            d = self.valuation_date + timedelta(days=days)
            tau = self._tau(d)
            print(
                f"{str(d):<15} {tau:>8.4f} "
                f"{self.zero_rate(d)*100:>9.4f}% {self.df(d):>12.6f}"
            )

    # ── interne ───────────────────────────────────────────────────────────────

    def _tau(self, target: date) -> float:
        return (target - self.valuation_date).days / _DAY_COUNT

    def __repr__(self) -> str:
        return (
            f"OISCurve(valuation_date={self.valuation_date}, "
            f"tenors={len(self._times)})"
        )


def fetch_estr_rates() -> Dict[str, float]:
    """Télécharge les taux €STR depuis bluegamma.io. Retourne {tenor: rate}."""
    r = requests.get(_API_URL, headers=_HEADERS, timeout=15)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    rates: Dict[str, float] = {}

    for row in soup.select("tbody tr"):
        cells = row.select("td")
        if len(cells) < 2:
            continue
        raw_tenor = cells[0].get_text(strip=True)           # "ESTR 1Y Swap Rate"
        raw_rate  = cells[1].get_text(strip=True)           # "2.48%"
        tenor = raw_tenor.replace("ESTR", "").replace("Swap Rate", "").strip()
        try:
            rates[tenor] = float(raw_rate.replace("%", "").strip()) / 100.0
        except ValueError:
            continue

    if not rates:
        raise ValueError("Aucun taux €STR trouvé — vérifier la source.")
    return rates


def fetch_ois_curve(valuation_date: date | None = None) -> OISCurve:
    """Télécharge, bootstrappe et retourne une OISCurve prête à l'emploi."""
    if valuation_date is None:
        valuation_date = date.today()
    rates = fetch_estr_rates()
    dfs   = _bootstrap(rates)
    return OISCurve(valuation_date, dfs)


def flat_ois_curve(rate: float, valuation_date: date | None = None) -> OISCurve:
    """
    Courbe plate à taux fixe — utile pour les tests unitaires.

    >>> curve = flat_ois_curve(0.035)
    """
    if valuation_date is None:
        valuation_date = date.today()
    tenors = {
        "1D": rate, "1W": rate, "1M": rate, "3M": rate, "6M": rate,
        "1Y": rate, "2Y": rate, "3Y": rate, "5Y": rate, "10Y": rate,
    }
    return OISCurve(valuation_date, _bootstrap(tenors))
