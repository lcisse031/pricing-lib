"""
Backtest hors ligne : spots synthétiques injectés à la place de yfinance,
OIS / dividendes plats. Vérifie les termes figés, l'attribution P&L et le
suivi des fixings d'autocall.
"""
import math
from datetime import date, timedelta

import numpy as np
import pytest

import pricing_lib.backtest.engine as eng
import pricing_lib.pricers.autocalls as ac
import pricing_lib.pricers.mc as mcmod
from pricing_lib.backtest_api import Backtest, PositionSpec
from pricing_lib.market_data.rates import flat_ois_curve
from pricing_lib.market_data.dividends import flat_dividend
from pricing_lib.models.heston import shift_heston_vol

START = date(2024, 1, 2)


def synthetic_spots(drift=0.0, vol=0.25, s0=100.0, days=500, seed=4, override=None):
    rng = np.random.default_rng(seed)
    out, s, d = {}, s0, START - timedelta(days=60)
    while len(out) < days:
        if d.weekday() < 5:
            out[d] = s
            s *= math.exp((drift - vol**2 / 2) / 252 + vol / math.sqrt(252) * rng.standard_normal())
        d += timedelta(days=1)
    for k, v in (override or {}).items():
        out[k] = v
    return out


@pytest.fixture
def offline(monkeypatch, heston):
    def patch(spots):
        monkeypatch.setattr(eng, "fetch_historical_spots", lambda tk, a, b: spots)
        monkeypatch.setattr(eng, "fetch_live_market_data",
                            lambda tk, spot: (flat_ois_curve(0.03, date.today()), flat_dividend(tk, 0.01)))

        def fake_params(snap, *a, **k):
            shift = getattr(snap, "vol_shift", None) or getattr(snap, "_vbump", None) or 0.0
            return shift_heston_vol(heston, shift)
        for m in (eng, ac, mcmod):
            monkeypatch.setattr(m, "get_heston_params", fake_params)
    return patch


def test_rc_terms_are_frozen(offline):
    offline(synthetic_spots())
    spec = PositionSpec("RC", "TST", "rc", 100.0, "02/01/2024", "12M", nominal=100_000,
                        params={"freq": "Q"})
    res = Backtest(spec, end_date="28/06/2024", freq="weekly", verbose=False)
    pos = res.positions
    assert pos["mtm_total"].iloc[0] == pytest.approx(100_000, rel=1e-9)
    assert pos["mtm"].std() > 0.5


def test_call_pnl_explain(offline):
    offline(synthetic_spots(seed=8))
    spec = PositionSpec("C", "TST", "call", 100.0, "02/01/2024", "6M", nominal=100_000,
                        params={"strike": 100.0})
    res = Backtest(spec, end_date="29/03/2024", freq="daily", verbose=False)
    p = res.positions.dropna(subset=["pnl_unexplained"])
    explained = p[["pnl_delta", "pnl_gamma", "pnl_vega", "pnl_vomma", "pnl_theta"]].sum(axis=1)
    # L'explication par les Greeks capte l'essentiel du P&L quotidien
    corr = np.corrcoef(explained, p["pnl"])[0, 1]
    assert corr > 0.98
    assert p["pnl_unexplained"].abs().mean() < 0.15 * p["pnl"].abs().mean()


def test_call_value_at_expiry_is_payoff(offline):
    spots = synthetic_spots(seed=9)
    offline(spots)
    spec = PositionSpec("C", "TST", "call", 100.0, "02/01/2024", "3M", nominal=100_000,
                        params={"strike": 100.0})
    res = Backtest(spec, freq="daily", verbose=False, compute_greeks=False)
    last = res.positions.iloc[-1]
    assert last["mtm"] == pytest.approx(max(last["spot"] - 100.0, 0.0), abs=0.05)


def test_autocall_redemption_is_tracked(offline):
    # Spot à 115 à la 1re date d'observation (02/04/2024) → rappel anticipé
    first_obs = date(2024, 4, 2)
    spots = synthetic_spots(seed=1, vol=0.05, override={first_obs: 115.0})
    offline(spots)
    spec = PositionSpec("PHX", "TST", "phoenix", 100.0, "02/01/2024", "24M",
                        nominal=100_000, mode="MC", n_paths=4_000, seed=5,
                        params={"barrier_coupon": 0.8, "barrier_recall": 1.0,
                                "capital_barrier": 0.6, "freq_months": 3, "coupon": 0.02})
    res = Backtest(spec, end_date="30/05/2024", freq="weekly", verbose=False)
    pos = res.positions.reset_index()
    after = pos[pos["date"] > str(first_obs)]
    assert len(after) > 0
    assert np.allclose(after["mtm"], 100.0 * 1.02)      # nominal + coupon encaissés
    assert after["delta"].isna().all()                   # plus de risque après rappel
