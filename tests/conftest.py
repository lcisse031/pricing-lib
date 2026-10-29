"""
Fixtures communes : marchés SYNTHÉTIQUES (aucun appel réseau).
Tous les tests tournent hors ligne et sont reproductibles (seeds fixes).
"""
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pricing_lib.market_data.rates import flat_ois_curve
from pricing_lib.market_data.dividends import flat_dividend
from pricing_lib.market_data.vol_surface import VolSurface
from pricing_lib.market_data.market_snapshot import MarketSnapshot
from pricing_lib.models.heston import HestonParams

VAL_DATE = date(2026, 1, 5)
S0, R, Q, SIGMA = 100.0, 0.03, 0.01, 0.25


def make_snap(spot=S0, r=R, q=Q, sigma=SIGMA, d=VAL_DATE, ticker="TST"):
    return MarketSnapshot(
        ticker=ticker, valuation_date=d, spot=spot,
        ois_curve=flat_ois_curve(r, d), div_curve=flat_dividend(ticker, q),
        vol_surface=VolSurface.flat(d, spot, r, q, sigma),
    )


@pytest.fixture
def snap():
    return make_snap()


@pytest.fixture
def bs_like_heston():
    """Heston quasi Black-Scholes : vol-of-vol ~0, v0 = θ = σ² → doit redonner BS."""
    return HestonParams(kappa=5.0, theta=SIGMA**2, sigma_v=0.01, rho=-0.3, v0=SIGMA**2)


@pytest.fixture
def heston():
    return HestonParams(kappa=2.0, theta=0.0625, sigma_v=0.4, rho=-0.7, v0=0.0625)


def r_cont(snap, T):
    """Taux continu tel que exp(-r T) = DF OIS — celui qu'utilisent les pricers."""
    return -math.log(snap.ois_curve.df_tau(T)) / T


def gbm_terminal(S, r, q, sigma, T, n=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n // 2)
    Z = np.concatenate([Z, -Z])
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
