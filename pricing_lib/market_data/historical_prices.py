"""
pricing_lib/market_data/historical_prices.py
─────────────────────────────────────────────────────────────────────────────
Historiques de prix des sous-jacents — Yahoo Finance.

Utilisé par :
  - backtest      → revalorisation historique
  - VaR           → distribution des chocs journaliers
  - stress test   → scénarios fondés sur historique

Classes / fonctions publiques
─────────────────────────────
fetch_history(ticker, start, end, interval)        → pd.Series de close
fetch_history_multi(tickers, start, end, interval) → pd.DataFrame
HistoricalPrices                                    — wrapper pratique
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd


# ─── mapping ticker court CAC 40 → ticker Yahoo ───────────────────────────────

_YF_SUFFIX = ".PA"  # Paris

_YF_OVERRIDES = {
    # cas spéciaux ou tickers déjà au format Yahoo
    "MT":   "MT.AS",   # ArcelorMittal Amsterdam
    "STLA": "STLA.PA",
    "URW":  "URW.AS",
    "AIR":  "AIR.PA",
    "EL":   "EL.PA",
    "ERF":  "ERF.PA",
}


def _to_yf(ticker: str) -> str:
    t = ticker.upper()
    if t in _YF_OVERRIDES:
        return _YF_OVERRIDES[t]
    if "." in t:
        return t
    return f"{t}{_YF_SUFFIX}"


# ─── fetch ────────────────────────────────────────────────────────────────────

def fetch_history(
    ticker:   str,
    start:    Union[str, date, datetime, None] = None,
    end:      Union[str, date, datetime, None] = None,
    interval: str = "1d",
    field:    str = "Close",
) -> pd.Series:
    """
    Historique de prix d'un sous-jacent.

    Paramètres
    ----------
    ticker   : code court (ex: "OR", "MC", "TTE") ou ticker Yahoo direct
    start    : date début (par défaut 2 ans avant aujourd'hui)
    end      : date fin (par défaut aujourd'hui)
    interval : "1d" (défaut), "1wk", "1mo"
    field    : "Close" (défaut) ou "Adj Close"

    Retourne
    --------
    pd.Series (DateTimeIndex, prix close), nommée par le ticker original.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance manquant. `pip install yfinance` — utilisé pour les historiques."
        ) from e

    yf_ticker = _to_yf(ticker)

    if end is None:
        end = datetime.utcnow().date()
    if start is None:
        start = (datetime.utcnow().date() - timedelta(days=730))

    df = yf.download(
        yf_ticker, start=start, end=end, interval=interval,
        progress=False, auto_adjust=False, threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Yahoo: aucune donnée pour {yf_ticker}")

    # robust column selection (yfinance peut renvoyer MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        if field in df.columns.get_level_values(0):
            ser = df[field].iloc[:, 0]
        else:
            ser = df.iloc[:, 0]
    else:
        ser = df[field] if field in df.columns else df.iloc[:, 0]

    ser = ser.dropna()
    ser.name = ticker.upper()
    ser.index = pd.to_datetime(ser.index).tz_localize(None)
    return ser


def fetch_history_multi(
    tickers:  Sequence[str],
    start:    Union[str, date, datetime, None] = None,
    end:      Union[str, date, datetime, None] = None,
    interval: str = "1d",
    field:    str = "Close",
) -> pd.DataFrame:
    """
    Historique multi-tickers — DataFrame indexé par date, colonnes = tickers.
    """
    series = []
    for t in tickers:
        try:
            s = fetch_history(t, start=start, end=end, interval=interval, field=field)
            series.append(s)
        except Exception as e:
            print(f"[historical_prices] {t}: {e}")
    if not series:
        raise RuntimeError("Aucun historique récupéré.")
    df = pd.concat(series, axis=1).sort_index()
    df = df.ffill().dropna(how="all")
    return df


# ─── wrapper ──────────────────────────────────────────────────────────────────

@dataclass
class HistoricalPrices:
    """
    Conteneur historique multi-tickers prêt à l'emploi.

    Méthodes utilitaires :
      - returns()         : log-rendements journaliers
      - simple_returns()  : rendements simples
      - rolling_vol(n)    : vol annualisée roulante
      - last(ticker)      : dernier prix connu
      - shocks(n)         : matrice (n×k) des n dernières variations relatives
    """
    prices: pd.DataFrame   # index = dates, columns = tickers

    @classmethod
    def fetch(
        cls,
        tickers:  Sequence[str],
        start:    Union[str, date, None] = None,
        end:      Union[str, date, None] = None,
        interval: str = "1d",
    ) -> "HistoricalPrices":
        df = fetch_history_multi(tickers, start=start, end=end, interval=interval)
        return cls(prices=df)

    @property
    def tickers(self) -> list:
        return list(self.prices.columns)

    @property
    def n_obs(self) -> int:
        return len(self.prices)

    def returns(self, log: bool = True) -> pd.DataFrame:
        if log:
            return np.log(self.prices / self.prices.shift(1)).dropna()
        return self.prices.pct_change().dropna()

    def rolling_vol(self, window: int = 21, annualize: bool = True) -> pd.DataFrame:
        r = self.returns(log=True)
        v = r.rolling(window).std()
        if annualize:
            v = v * np.sqrt(252.0)
        return v.dropna()

    def realized_vol(self, lookback: int = 252) -> "pd.Series":
        r = self.returns(log=True).tail(lookback)
        return r.std() * np.sqrt(252.0)

    def correlation(self, lookback: int = 252) -> pd.DataFrame:
        return self.returns(log=True).tail(lookback).corr()

    def last(self, ticker: Optional[str] = None):
        if ticker is None:
            return self.prices.iloc[-1]
        return float(self.prices[ticker.upper()].iloc[-1])

    def shocks(self, n: int = 252) -> pd.DataFrame:
        """
        n derniers chocs RELATIFS journaliers (S_t / S_{t-1} - 1).
        Utilisé par la VaR historique : shock_i appliqué au spot courant.
        """
        return self.prices.pct_change().dropna().tail(n)

    def __repr__(self) -> str:
        return (
            f"HistoricalPrices(tickers={self.tickers}, "
            f"n={self.n_obs}, "
            f"range=[{self.prices.index.min().date()} → {self.prices.index.max().date()}])"
        )
