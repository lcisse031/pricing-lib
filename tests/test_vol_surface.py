"""Construction de la surface : options OTM seules, structure par terme préservée, SSVI."""
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pricing_lib.market_data.vol_surface import VolSurface, _bs_price, _third_friday
from pricing_lib.market_data.ssvi import ssvi_from_grid, _observed_mask

VAL = date(2026, 1, 5)
S, R, Q = 100.0, 0.03, 0.01
TERM = {"2026-03": 0.20, "2026-06": 0.25, "2026-12": 0.30}   # vol par échéance


def synthetic_chain(corrupt_itm=True):
    rows = []
    for exp, vol in TERM.items():
        y, m = int(exp[:4]), int(exp[5:])
        T = (_third_friday(y, m) - VAL).days / 365.25
        F = S * math.exp((R - Q) * T)
        for K in np.arange(70, 132, 4.0):
            for typ, flag in (("C", 1), ("P", -1)):
                px = _bs_price(flag, S, K, T, R, Q, vol)
                itm = (typ == "C" and K < F) or (typ == "P" and K >= F)
                if corrupt_itm and itm:
                    px *= 1.25            # cotation ITM périmée / au tick
                rows.append({"expiry_date": exp, "strike": K, "type": typ,
                             "bid": px, "ask": px, "last": px, "mid": px})
    return pd.DataFrame(rows)


def test_itm_quotes_are_ignored():
    surf = VolSurface.from_option_chain(synthetic_chain(), VAL, S, R, Q)
    for exp, vol in TERM.items():
        y, m = int(exp[:4]), int(exp[5:])
        T = (_third_friday(y, m) - VAL).days / 365.25
        for K in (85.0, 100.0, 115.0):
            assert surf.vol(T, K) == pytest.approx(vol, abs=0.005)


def test_term_structure_is_preserved():
    """La vol ATM varie avec la maturité."""
    surf = VolSurface.from_option_chain(synthetic_chain(), VAL, S, R, Q)
    atm = [surf.vol(T, S) for T in surf.tenors]
    assert atm[-1] - atm[0] > 0.08


def test_observed_mask_drops_filled_cells():
    v = np.array([[0.30, 0.25, 0.25, 0.25],     # prolongement plat en strike
                  [0.31, 0.26, 0.24, 0.23],
                  [0.31, 0.26, 0.24, 0.23]])    # recopie de la maturité précédente
    m = _observed_mask(v)
    assert m[1].sum() >= 3 and not m[2].any()
    assert not m[0, 2] and not m[0, 3]


def test_ssvi_from_grid_is_arbitrage_free():
    surf = VolSurface.from_option_chain(synthetic_chain(), VAL, S, R, Q)
    ssvi = ssvi_from_grid(surf)
    rep = ssvi.no_arbitrage_report()
    assert rep["calendar_free"] and rep["butterfly_free"]
    T = surf.tenors[1]
    assert ssvi.atm_vol(T) == pytest.approx(surf.atm_vol(T), abs=0.01)
