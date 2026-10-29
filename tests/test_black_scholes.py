"""Vanilles : formules fermées, parité, Greeks, convergence MC."""
import math
import pytest
from scipy.stats import norm

from pricing_lib.pricers.analytical import AnalyticalPricer
from pricing_lib.pricers.mc import MCPricer
from pricing_lib.risk.greeks import compute_greeks_bs
from conftest import S0, Q, SIGMA, r_cont


def bs_call(S, K, T, r, q, s):
    d1 = (math.log(S / K) + (r - q + s * s / 2) * T) / (s * math.sqrt(T))
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - s * math.sqrt(T))


@pytest.mark.parametrize("K,T", [(80, 0.25), (100, 1.0), (125, 2.0)])
def test_call_matches_black_scholes(snap, K, T):
    r = r_cont(snap, T)
    assert AnalyticalPricer(snap).call(K, T).price == pytest.approx(bs_call(S0, K, T, r, Q, SIGMA), abs=1e-5)


@pytest.mark.parametrize("K,T", [(90, 0.5), (110, 3.0)])
def test_put_call_parity(snap, K, T):
    pr = AnalyticalPricer(snap)
    r = r_cont(snap, T)
    lhs = pr.call(K, T).price - pr.put(K, T).price
    rhs = S0 * math.exp(-Q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-5)


def test_closed_form_greeks_vs_finite_differences(snap):
    K, T = 105.0, 0.75
    r = r_cont(snap, T)
    g = compute_greeks_bs("c", S0, K, T, r, Q, SIGMA)
    h = 1e-3
    delta_fd = (bs_call(S0 + h, K, T, r, Q, SIGMA) - bs_call(S0 - h, K, T, r, Q, SIGMA)) / (2 * h)
    gamma_fd = (bs_call(S0 + h, K, T, r, Q, SIGMA) - 2 * bs_call(S0, K, T, r, Q, SIGMA)
                + bs_call(S0 - h, K, T, r, Q, SIGMA)) / h**2
    vega_fd = (bs_call(S0, K, T, r, Q, SIGMA + 1e-4) - bs_call(S0, K, T, r, Q, SIGMA - 1e-4)) / 2e-4 * 0.01
    theta_fd = -(bs_call(S0, K, T + 1e-4, r, Q, SIGMA) - bs_call(S0, K, T - 1e-4, r, Q, SIGMA)) / 2e-4 / 365
    assert g["delta"] == pytest.approx(delta_fd, abs=1e-5)
    assert g["gamma"] == pytest.approx(gamma_fd, abs=1e-4)
    assert g["vega"] == pytest.approx(vega_fd, abs=1e-5)
    assert g["theta"] == pytest.approx(theta_fd, abs=1e-5)


@pytest.mark.parametrize("qmc", [False, True])
def test_mc_gbm_converges_to_closed_form(snap, qmc):
    K, T = 100.0, 1.0
    al = AnalyticalPricer(snap).call(K, T)
    mc = MCPricer(snap, 200_000, seed=7, qmc=qmc).call(K, T)
    assert abs(mc.price - al.price) < max(4 * mc.mc_stderr, 0.01)


def test_mc_vanilla_gamma_is_not_noise(snap):
    """p_0 et les bumps partagent le même tirage."""
    K, T = 100.0, 1.0
    al = AnalyticalPricer(snap).call(K, T).greeks
    mc = MCPricer(snap, 100_000, seed=1).call(K, T).greeks
    assert mc["delta"] == pytest.approx(al["delta"], abs=0.01)
    assert mc["gamma"] == pytest.approx(al["gamma"], rel=0.15)
