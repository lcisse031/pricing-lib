"""Phoenix / Athena : cohérence AL ↔ MC, probabilités, cours de vie, Greeks."""
from datetime import timedelta

import numpy as np
import pytest

from pricing_lib.pricers.autocalls import PhoenixPricer, AthenaPricer, barrier_level
from conftest import VAL_DATE, S0, make_snap

N = 30_000


def phoenix(snap, heston, mode="AL", coupon=0.0, **kw):
    return PhoenixPricer(snap, N, VAL_DATE, S0, 80, 100, 70, 3, 36, mode=mode,
                         coupon=coupon, params=heston, seed=3, **kw)


def athena(snap, heston, mode="AL", coupon=0.0, kg="no", **kw):
    return AthenaPricer(snap, N, VAL_DATE, S0, 100, 70, 3, 36, kg=kg, mode=mode,
                        coupon=coupon, params=heston, seed=3, **kw)


@pytest.mark.parametrize("factory", [phoenix, athena])
def test_fair_coupon_reprices_at_par(snap, heston, factory):
    al = factory(snap, heston).price()
    assert al.price == pytest.approx(100.0, abs=1e-6)
    assert 0.0 < al.fair_coupon < 0.10                  # coupon trimestriel plausible
    mc = factory(snap, heston, mode="MC", coupon=al.fair_coupon).price()
    assert mc.price == pytest.approx(100.0, abs=0.05)


def test_athena_probabilities_sum_to_one(snap, heston):
    res = athena(snap, heston).price()
    assert res.probabilities["Prob_recall"].sum() == pytest.approx(1.0, abs=1e-9)


def test_phoenix_probabilities_sum_to_one(snap, heston):
    res = phoenix(snap, heston).price()
    assert res.probabilities["Prob_recall"].sum() == pytest.approx(1.0, abs=1e-9)


def test_capital_guarantee_lowers_coupon(snap, heston):
    assert athena(snap, heston, kg="yes").price().fair_coupon < athena(snap, heston).price().fair_coupon


def test_coupon_monotonic_price(snap, heston):
    p1 = phoenix(snap, heston, "MC", 0.02).price().price
    p2 = phoenix(snap, heston, "MC", 0.03).price().price
    assert p2 > p1


def test_mid_life_valuation(heston):
    """6 mois après le départ, spot inchangé : 2 dates passées, 10 à venir."""
    d = VAL_DATE + timedelta(days=182)
    snap = make_snap(d=d)
    pr = athena(snap, heston, mode="MC", coupon=0.02)
    assert pr.k0 == 2 and pr.n_periods == 10 and pr.n_total == 12
    assert pr.obs_taus[0] > 0
    assert 80 < pr.price().price < 115


def test_spot_moves_price_and_reference_stays(snap, heston):
    """Le spot courant bouge le prix ; S0 (référence de perte) reste figé."""
    base = athena(snap, heston, "MC", 0.02).price().price
    down = athena(snap, heston, "MC", 0.02, spot=85.0).price().price
    assert down < base


def test_greeks_have_theta_and_sign(snap, heston):
    res = PhoenixPricer(snap, 20_000, VAL_DATE, S0, 80, 100, 70, 3, 36, mode="MC",
                        coupon=0.025, params=heston, seed=3, compute_greeks=True).price()
    g = res.greeks
    assert g["delta"] > 0          # le porteur est long du sous-jacent
    assert g["vega"] < 0           # et vendeur de vol (put down-and-in)
    assert abs(g["theta"]) > 1e-4


def test_barrier_level():
    assert barrier_level(0.7, 20) == pytest.approx(14.0)
    assert barrier_level(8.0, 10) == 8.0
    assert barrier_level(1.0, 50) == 50.0


def test_coupon_label_per_period(snap, heston):
    res = athena(snap, heston).price()
    assert res.coupon_periods_per_year == 4
    assert "par période" in repr(res)
