"""Stress / scénarios (termes figés, chocs de vol Heston) et courbe OIS."""
import math
import pytest

from pricing_lib.market_data.rates import _bootstrap, OISCurve, flat_ois_curve
from pricing_lib.risk.greeks import _BumpedMarket
from pricing_lib.risk.sensitivities import StressTest, StressScenario, _ClosurePricer
from pricing_lib.pricers import mc as mc_mod
from pricing_lib.pricers.mc import MCPricer, get_heston_params
from pricing_lib.pricer_api import _make_pricer_fn
from conftest import VAL_DATE, S0


def test_stress_rc_uses_fixed_coupon(snap):
    """Le coupon reste celui du marché de base dans chaque scénario."""
    fn = _make_pricer_fn("AL", "rc", snap, 1.0, VAL_DATE, None, None, freq="Q")
    df = StressTest(_ClosurePricer(fn), None, snap,
                    scenarios=[StressScenario("Base", 0, 0, 0),
                               StressScenario("Crash", -0.30, 0.15, 0)]).run()
    assert df.loc["Base", "Price"] == pytest.approx(100.0, abs=1e-4)
    assert df.loc["Crash", "PnL"] < -10


def test_stress_participation_fixed(snap):
    fn = _make_pricer_fn("AL", "outperformance", snap, 1.0, VAL_DATE, None, None, X=100)
    assert fn(_BumpedMarket(snap, spot=S0 * 0.8)) < fn(snap) - 15


def test_vol_shock_reaches_heston(snap, heston, monkeypatch):
    monkeypatch.setitem(mc_mod._HESTON_CACHE, (snap.ticker, snap.valuation_date), heston)
    bumped = _BumpedMarket(snap, sigma_bump=0.05)
    p = get_heston_params(bumped)
    assert math.sqrt(p.v0) == pytest.approx(math.sqrt(heston.v0) + 0.05)
    base = MCPricer(snap, 20_000, seed=1).bonus_certificate(110, 75, 1.0).price
    stressed = MCPricer(bumped, 20_000, seed=1).bonus_certificate(110, 75, 1.0).price
    assert stressed < base - 0.5           # plus de vol → barrière plus probable


def test_flat_curve_discount_factors():
    c = flat_ois_curve(0.03, VAL_DATE)
    assert c.df_tau(2.0) == pytest.approx(1 / (1.03) ** 2, rel=1e-3)


def test_bootstrap_reprices_swaps_with_missing_pillars():
    """5Y bootstrappé avec seulement 3Y en stock : le swap 5Y doit repricer au pair."""
    rates = {"1Y": 0.025, "2Y": 0.026, "3Y": 0.027, "5Y": 0.030, "10Y": 0.033}
    curve = OISCurve(VAL_DATE, _bootstrap(rates))
    for T, r in [(3, 0.027), (5, 0.030), (10, 0.033)]:
        annuity = sum(curve.df_tau(t) for t in range(1, T + 1))
        assert r * annuity + curve.df_tau(T) == pytest.approx(1.0, abs=1e-9)
