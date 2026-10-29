"""Heston : prix semi-fermé, schéma QE, chocs de vol."""
import math
import numpy as np
import pytest
from scipy.integrate import quad

from pricing_lib.models.heston import (
    heston_price, heston_cf, heston_simulate, shift_heston_vol, _heston_call_vec,
)
from conftest import S0, R, Q


def reference_call(K, T, p):
    a = (S0, p.v0, R, Q, T, p.kappa, p.theta, p.sigma_v, p.rho)
    phi_mi = heston_cf(-1j, *a)
    f1 = lambda u: (np.exp(-1j * u * math.log(K)) * heston_cf(u - 1j, *a) / (1j * u * phi_mi)).real
    f2 = lambda u: (np.exp(-1j * u * math.log(K)) * heston_cf(u, *a) / (1j * u)).real
    P1 = 0.5 + quad(f1, 1e-8, 500, limit=500)[0] / math.pi
    P2 = 0.5 + quad(f2, 1e-8, 500, limit=500)[0] / math.pi
    return S0 * math.exp(-Q * T) * P1 - K * math.exp(-R * T) * P2


@pytest.mark.parametrize("K,T", [(100, 1.0), (80, 0.1), (120, 0.1), (100, 0.05), (130, 2.0)])
def test_semi_closed_price_vs_quadrature(heston, K, T):
    assert heston_price("c", S0, K, T, R, Q, heston) == pytest.approx(reference_call(K, T, heston), abs=2e-4)


def test_vectorised_matches_scalar(heston):
    Ks = np.array([80., 95., 100., 110., 130.])
    vec = _heston_call_vec(S0, Ks, 1.0, R, Q, heston)
    sca = [heston_price("c", S0, K, 1.0, R, Q, heston) for K in Ks]
    assert np.allclose(vec, sca, atol=1e-8)


@pytest.mark.parametrize("T", [1.0, 5.0])
def test_qe_is_martingale(heston, T):
    ST = heston_simulate(S0, R, Q, T, heston, 200_000, seed=11)
    se = ST.std() / math.sqrt(len(ST))
    assert abs(ST.mean() - S0 * math.exp((R - Q) * T)) < 4 * se


def test_qe_mc_matches_semi_closed(heston):
    K, T = 95.0, 1.0
    ST = heston_simulate(S0, R, Q, T, heston, 300_000, seed=5)
    pay = np.maximum(ST - K, 0) * math.exp(-R * T)
    assert abs(pay.mean() - heston_price("c", S0, K, T, R, Q, heston)) < 4 * pay.std() / math.sqrt(len(pay)) + 0.02


def test_shift_heston_vol(heston):
    up = shift_heston_vol(heston, 0.05)
    assert math.sqrt(up.v0) == pytest.approx(math.sqrt(heston.v0) + 0.05)
    assert math.sqrt(up.theta) == pytest.approx(math.sqrt(heston.theta) + 0.05)
    assert shift_heston_vol(heston, 0.0) is heston
