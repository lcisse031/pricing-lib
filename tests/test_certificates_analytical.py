"""
Certificats en analytique : chaque formule fermée est comparée à son payoff
simulé (GBM exact à maturité) ou à une parité de structure.
"""
import math
import numpy as np
import pytest

from pricing_lib.pricers.core import pdts
from pricing_lib.pricers.analytical import AnalyticalPricer
from conftest import S0, Q, SIGMA, gbm_terminal, make_snap

R, T = 0.03, 1.0
ST = gbm_terminal(S0, R, Q, SIGMA, T)
DF = math.exp(-R * T)


def mc(payoff):
    x = payoff * DF
    return x.mean(), x.std() / math.sqrt(len(x))


def close(formula, payoff, k=4):
    m, se = mc(payoff)
    assert abs(formula - m) < k * se + 1e-3, (formula, m, se)


# ── Barrières : in + out = vanille (8 types, rebate nul) ────────────────────
@pytest.mark.parametrize("cp,H,X", [("c", 80, 100), ("c", 120, 100), ("p", 80, 100), ("p", 120, 100),
                                    ("c", 90, 85), ("p", 110, 115)])
def test_barrier_in_out_parity(cp, H, X):
    b = R - Q
    kind = "d" if H < S0 else "u"
    vin = pdts._barrier(f"{cp}{kind}i", S0, X, H, 0., T, R, b, SIGMA)
    vout = pdts._barrier(f"{cp}{kind}o", S0, X, H, 0., T, R, b, SIGMA)
    assert vin + vout == pytest.approx(pdts._gbs(cp, S0, X, T, R, b, SIGMA), abs=1e-8)


def test_down_and_out_put_vs_fine_grid_mc():
    """Put down-and-out : formule de Haug vs MC à pas fin avec correction de
    Broadie-Glasserman-Kou : un monitoring discret à H·exp(+0.5826 σ √dt)
    équivaut à un monitoring continu à H."""
    X, H, n_steps, n = 100.0, 80.0, 500, 200_000
    rng = np.random.default_rng(3)
    dt = T / n_steps
    logS = np.full(n, math.log(S0)); hit = np.zeros(n, bool)
    Hadj = H * math.exp(0.5826 * SIGMA * math.sqrt(dt))
    for _ in range(n_steps):
        logS += (R - Q - SIGMA**2 / 2) * dt + SIGMA * math.sqrt(dt) * rng.standard_normal(n)
        hit |= logS <= math.log(Hadj)
    pay = np.where(hit, 0.0, np.maximum(X - np.exp(logS), 0.0)) * DF
    f = pdts._barrier("pdo", S0, X, H, 0., T, R, R - Q, SIGMA)
    assert abs(f - pay.mean()) < 4 * pay.std() / math.sqrt(n) + 0.01


def test_barrier_already_crossed():
    b = R - Q
    assert pdts._barrier("pdo", 75, 100, 80, 0., T, R, b, SIGMA) == 0.0
    assert pdts._barrier("pdi", 75, 100, 80, 0., T, R, b, SIGMA) == pytest.approx(pdts._gbs("p", 75, 100, T, R, b, SIGMA))
    assert pdts._barrier("cuo", 125, 100, 120, 0., T, R, b, SIGMA) == 0.0


# ── Payoffs terminaux ───────────────────────────────────────────────────────
def test_discount():
    close(pdts.discount_certificate(S0, 110, T, R, Q, SIGMA), np.minimum(ST, 110))


@pytest.mark.parametrize("part", [1.5, 2.0, 3.0])
def test_sprint(part):
    X, cap = 100, 120
    pay = np.where(ST < X, ST, np.minimum(X + part * (ST - X), X + part * (cap - X)))
    close(pdts._sprint(S0, X, cap, T, R, Q, SIGMA, part), pay)


@pytest.mark.parametrize("part", [1.3, 2.5])
def test_capped_outperformance(part):
    X, cap = 100, 130
    pay = np.minimum(ST + (part - 1) * np.maximum(ST - X, 0), cap + (part - 1) * (cap - X))
    close(pdts._capped_outperformance(S0, X, cap, T, R, Q, SIGMA, part), pay)


def test_outperformance():
    close(pdts._outperformance(S0, 100, T, R, Q, SIGMA, 1.6), ST + 0.6 * np.maximum(ST - 100, 0))


def test_airbag():
    X, B, p = 100, 80, 0.7
    pay = np.where(ST >= X, X + p * (ST - X), np.where(ST >= B, X, X * ST / B))
    close(pdts._airbag(S0, X, B, T, R, Q, SIGMA, p), pay)


def test_garantie_is_scale_invariant():
    """La participation fair ne doit pas dépendre du niveau absolu du spot."""
    parts = []
    for spot in (22.5, 100.0, 450.0):
        res = AnalyticalPricer(make_snap(spot=spot)).garantie_certificate(spot, 5.0)
        assert res.price == pytest.approx(100.0, abs=1e-6)
        parts.append(res.fair_participation)
    assert max(parts) - min(parts) < 1e-6
    p = 0.6
    close(pdts._garantie(S0, S0, T, R, Q, SIGMA, p), 100 * (1 + p * np.maximum(ST / S0 - 1, 0)))


def test_reverse_convertible_payoff():
    cap, c = 100, 0.08
    v = pdts._rc(S0, cap, T, R, Q, SIGMA, 100., c)
    close(v, (100 * (1 + c) - np.maximum(cap - ST, 0) * 100 / cap))


# ── Produits « au pair » : participation / coupon fair ──────────────────────
@pytest.mark.parametrize("method,args,par", [
    ("outperformance_certificate", (100, T), S0),
    ("sprint_certificate", (100, 120, T), S0),
    ("twin_win_certificate", (100, 70, T), S0),
    ("airbag_certificate", (100, 80, T), S0),
    ("garantie_certificate", (100, 3.0), 100.0),
])
def test_fair_participation_prices_at_par(snap, method, args, par):
    res = getattr(AnalyticalPricer(snap), method)(*args)
    assert res.price == pytest.approx(par, abs=1e-4)
    # Termes figés : avec la même participation on retrouve le même prix
    res2 = getattr(AnalyticalPricer(snap), method)(*args, part=res.fair_participation)
    assert res2.price == pytest.approx(res.price, abs=1e-4)   # part arrondie à 1e-6


def test_rc_fair_coupon_and_fixed_coupon(snap):
    pr = AnalyticalPricer(snap)
    res = pr.reverse_convertible(cap=100, T=1.0, freq="Q")
    assert res.price == pytest.approx(100.0, abs=1e-5)
    assert 0.0 < res.fair_coupon < 0.5
    fixed = pr.reverse_convertible(cap=100, T=1.0, freq="Q", coupon=res.fair_coupon + 0.01)
    assert fixed.price == pytest.approx(101.0, abs=0.05)      # +1 % de coupon ≈ +1 pt
    assert fixed.greeks["rho"] < 0                              # jambe obligataire incluse


def test_rc_schedule_does_not_roll_mid_life():
    """En cours de vie, l'échéancier part de la date de départ (pas de glissement)."""
    from datetime import date
    from pricing_lib.pricers.analytical import _schedule
    s = _schedule(date(2025, 1, 15), 0.5, "Q", date(2025, 7, 20))
    assert [d.isoformat() for d in s][-2:] == ["2025-10-15", "2026-01-15"]
