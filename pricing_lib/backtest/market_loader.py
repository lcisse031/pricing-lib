"""
pricing_lib/backtest/market_loader.py
─────────────────────────────────────────────────────────────────────────────
Chargement des données de marché pour le backtest.

Approche :
- Spots historiques  : yfinance (un seul fetch bulk par ticker)
- Vol historique     : vol réalisée 30 jours glissants sur les log-rendements
- OIS / dividendes   : fetch live une seule fois par ticker (proxy acceptable)
- Snapshot par date  : VolSurface.flat(date, spot_hist, r, q, vol_réalisée)

On ne refait AUCUN appel HTTP par date — tout est pré-chargé.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from pricing_lib.market_data.market_snapshot import MarketSnapshot
from pricing_lib.market_data.rates import fetch_ois_curve, OISCurve
from pricing_lib.market_data.dividends import fetch_dividend, DividendCurve
from pricing_lib.market_data.vol_surface import VolSurface

# Mapping ticker CAC40 → symbole yfinance
_YF_MAP: dict[str, str] = {
    "GLE": "GLE.PA", "OR":  "OR.PA",  "MC":  "MC.PA",  "TTE": "TTE.PA",
    "BNP": "BNP.PA", "SAN": "SAN.PA", "AI":  "AI.PA",  "AIR": "AIR.PA",
    "AXA": "CS.PA",  "CA":  "ACA.PA", "DG":  "DG.PA",  "EN":  "ENGI.PA",
    "HO":  "HO.PA",  "KER": "KER.PA", "LR":  "LR.PA",  "ML":  "ML.PA",
    "MT":  "MT.AS",  "PUB": "PUB.PA", "RMS": "RMS.PA", "SAF": "SAF.PA",
    "SGO": "SGO.PA", "SU":  "SU.PA",  "URW": "URW.AS", "VIE": "VIE.PA",
    "VIV": "VIV.PA",
}

# Vol réalisée par défaut si pas assez de données (25%)
_DEFAULT_VOL = 0.25


def _yf_symbol(ticker: str) -> str:
    return _YF_MAP.get(ticker.upper(), ticker.upper() + ".PA")


def fetch_historical_spots(
    ticker: str,
    start: date,
    end: date,
) -> dict[date, float]:
    """
    Fetch cours de clôture ajustés via yfinance.
    Récupère start - 40 jours de plus pour avoir assez de données pour la vol réalisée.
    Retourne {date: spot}.
    """
    ticker = ticker.upper()
    sym    = _yf_symbol(ticker)
    start_ext = start - timedelta(days=40)

    df = yf.download(sym, start=start_ext.isoformat(), end=end.isoformat(),
                     auto_adjust=True, progress=False)
    if df.empty:
        df = yf.download(ticker, start=start_ext.isoformat(), end=end.isoformat(),
                         auto_adjust=True, progress=False)
    if df.empty:
        return {}

    # yfinance ≥ 0.2 peut retourner un MultiIndex sur les colonnes
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    return {d.date(): float(v) for d, v in close.items()}


def compute_realized_vol(
    spots: dict[date, float],
    window: int = 21,
) -> dict[date, float]:
    """
    Vol réalisée annualisée sur une fenêtre glissante de `window` jours ouvrés.
    Retourne {date: vol} pour chaque date ayant assez d'historique.
    """
    dates  = sorted(spots.keys())
    prices = np.array([spots[d] for d in dates])
    log_ret = np.diff(np.log(prices))

    result: dict[date, float] = {}
    for i, d in enumerate(dates[1:], start=1):
        if i < window:
            # Pas assez de données → vol par défaut
            result[d] = _DEFAULT_VOL
        else:
            vol = float(np.std(log_ret[i - window: i]) * np.sqrt(252))
            result[d] = vol if vol > 0.01 else _DEFAULT_VOL

    return result


def fetch_live_market_data(
    ticker: str,
    spot: float,
) -> Tuple[OISCurve, DividendCurve]:
    """
    Fetch OIS curve et dividende une seule fois par ticker.
    Retourne (ois_curve, div_curve).
    """
    ticker = ticker.upper()
    ois = fetch_ois_curve(date.today())

    try:
        dy  = fetch_dividend(ticker)
        div = DividendCurve.from_dividend_yield(dy)
    except Exception:
        div = DividendCurve(ticker, 0.0)

    return ois, div


def build_snapshot(
    ticker: str,
    valuation_date: date,
    spot: float,
    ois: OISCurve,
    div: DividendCurve,
    realized_vol: float,
) -> MarketSnapshot:
    """
    Construit un MarketSnapshot pour une date historique donnée.
    Utilise une surface plate calibrée sur la vol réalisée.
    """
    r_1y = ois.zero_rate_tau(1.0)
    surf  = VolSurface.flat(valuation_date, spot, r_1y, div.q, realized_vol)

    return MarketSnapshot(
        ticker         = ticker.upper(),
        valuation_date = valuation_date,
        spot           = spot,
        ois_curve      = ois,
        div_curve      = div,
        vol_surface    = surf,
    )
