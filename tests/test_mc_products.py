"""
Produits MC (Heston) : avec un Heston quasi Black-Scholes, chaque produit doit
retrouver le prix analytique. Les écarts tolérés couvrent l'erreur MC et le
biais de monitoring discret (quotidien) des barrières.
"""
import pytest

from pricing_lib.pricers.analytical import AnalyticalPricer
from pricing_lib.pricers.mc import MCPricer

N = 40_000


@pytest.mark.parametrize("method,args,tol", [
    ("discount_certificate", (110, 1.0), 0.30),
    ("bonus_certificate", (110, 75, 1.0), 0.60),
    ("reverse_bonus_certificate", (90, 130, 1.0), 0.60),
])
def test_structures_vs_analytical(snap, bs_like_heston, method, args, tol):
    al = getattr(AnalyticalPricer(snap), method)(*args).price
    mc = getattr(MCPricer(snap, N, params=bs_like_heston, seed=2), method)(*args).price
    assert mc == pytest.approx(al, abs=tol)


@pytest.mark.parametrize("method,args,tol", [
    ("outperformance_certificate", (100, 1.0), 0.04),
    ("sprint_certificate", (100, 120, 1.0), 0.05),
    ("twin_win_certificate", (100, 70, 1.0), 0.10),
    ("airbag_certificate", (100, 80, 1.0), 0.06),
    ("garantie_certificate", (100, 3.0), 0.05),
])
def test_fair_participation_vs_analytical(snap, bs_like_heston, method, args, tol):
    al = getattr(AnalyticalPricer(snap), method)(*args).fair_participation
    mc = getattr(MCPricer(snap, N, params=bs_like_heston, seed=2), method)(*args).fair_participation
    assert mc == pytest.approx(al, abs=tol)


def test_rc_mc_vs_analytical(snap, bs_like_heston):
    al = AnalyticalPricer(snap).reverse_convertible(cap=100, T=1.0, freq="Q")
    mc = MCPricer(snap, N, params=bs_like_heston, seed=2).reverse_convertible(
        cap=100, T=1.0, coupon=al.fair_coupon, freq="Q")
    assert mc.price == pytest.approx(100.0, abs=0.3)


def test_reverse_bonus_pays_the_bonus(snap, bs_like_heston):
    """Sans toucher la barrière, le payoff vaut au moins 2·S0 − X."""
    res = MCPricer(snap, N, params=bs_like_heston, seed=2).reverse_bonus_certificate(90, 1e9, 1.0)
    al = AnalyticalPricer(snap).reverse_bonus_certificate(90, 1e9, 1.0)
    assert res.price == pytest.approx(al.price, abs=0.3)


def test_mc_exotic_greeks_consistent(snap, bs_like_heston):
    al = AnalyticalPricer(snap).discount_certificate(110, 1.0).greeks
    mc = MCPricer(snap, N, params=bs_like_heston, seed=2).discount_certificate(110, 1.0).greeks
    assert mc["delta"] == pytest.approx(al["delta"], abs=0.03)
    assert mc["gamma"] == pytest.approx(al["gamma"], abs=0.004)
    assert mc["vega"] == pytest.approx(al["vega"], abs=0.05)
