from __future__ import annotations

import math
import threading
from datetime import date, datetime
from typing import List, Optional

from market_data.dividends import fetch_dividend, DividendCurve
from market_data.rates import fetch_ois_curve
from market_data.vol_surface import (
    fetch_option_chain, VolSurface, fetch_market_params, dupire_local_vol,
    _TICKER_MAP as _EURONEXT_MAPPING, _AMSTERDAM_TICKERS,
)
from market_data.market_snapshot import MarketSnapshot
from pricers.base import PricingResult
from pricers.analytical import AnalyticalPricer
from pricers.mc import MCPricer
from pricers.autocalls import PhoenixPricer, AthenaPricer
from pricing_lib.risk import (
    FiniteDiffGreeks, GreeksResult, _ClosurePricer,
    ScenarioAnalyzer, StressTest, STANDARD_STRESSES, StressScenario,
)

_SNAPSHOT_CACHE: dict = {}
_SNAPSHOT_LOCK  = threading.Lock()


def _parse_date(s: str) -> date:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Format de date non reconnu : {s!r}. Utiliser 'DD/MM/YYYY'.")


def _parse_tenor(s: str) -> float:
    """'3M'→0.25  '12M'→1.0  '60M'→5.0  '2Y'→2.0  '1Y'→1.0"""
    s = s.strip().upper()
    if s.endswith("M"):
        return int(s[:-1]) / 12.0
    if s.endswith("Y"):
        return float(s[:-1])
    if s.endswith("D"):
        return int(s[:-1]) / 365.0
    raise ValueError(f"Format de maturité non reconnu : {s!r}. Ex: '3M', '1Y'.")


def _parse_tenor_months(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("M"):
        return int(s[:-1])
    if s.endswith("Y"):
        return int(s[:-1]) * 12
    raise ValueError(f"Format de maturité non reconnu : {s!r}.")

def _to_euronext_symbol(ticker: str) -> str:
    ticker = ticker.upper()
    return _EURONEXT_MAPPING.get(ticker, ticker + "4")


def _build_snapshot(
    ticker: str,
    spot: float,
    valuation_date: date,
    verbose: bool = True,
) -> MarketSnapshot:
    """
    Récupère toutes les données de marché et construit le MarketSnapshot.
    Cache thread-safe : un seul fetch Euronext par (ticker, spot, date) par run.
    """
    ticker = ticker.upper()
    cache_key = (ticker, valuation_date, spot)

    with _SNAPSHOT_LOCK:
        if cache_key in _SNAPSHOT_CACHE:
            return _SNAPSHOT_CACHE[cache_key]

    if verbose:
        print(f"[Snapshot] Récupération données marché pour {ticker}...")

    # Taux OIS
    ois = fetch_ois_curve(valuation_date)
    if verbose:
        print(f"[Snapshot] OIS curve OK (r_1Y={ois.zero_rate_tau(1.0):.4%})")

    # Dividendes
    try:
        dy  = fetch_dividend(ticker)
        div = DividendCurve.from_dividend_yield(dy)
        if verbose:
            print(f"[Snapshot] Dividende OK (q={div.q:.4%})")
    except Exception as e:
        if verbose:
            print(f"[Snapshot] Dividende indisponible ({e}) — q=0")
        div = DividendCurve(ticker, 0.0)

    # Surface de volatilité — Euronext (exchange selon le ticker)
    r_1y         = ois.zero_rate_tau(1.0)
    euronext_sym = _to_euronext_symbol(ticker)
    exchange     = "DAMS" if ticker in _AMSTERDAM_TICKERS else "DPAR"
    try:
        chain = fetch_option_chain(euronext_sym, exchange)
        surf  = VolSurface.from_option_chain(chain, valuation_date, spot, r_1y, div.q)
        if verbose:
            # Affiche la vol locale ATM pour 3M et 1Y
            sig_3m = dupire_local_vol(surf, 0.25, spot)
            sig_1y = dupire_local_vol(surf, 1.0, spot)
            print(f"[Snapshot] VolSurface OK — {surf}")
            print(f"[Snapshot] σ_loc ATM: 3M={sig_3m:.2%}, 1Y={sig_1y:.2%}")
    except Exception as e:
        if verbose:
            print(f"[Snapshot] VolSurface indisponible ({e}) — surface plate 25%")
        surf = VolSurface.flat(valuation_date, spot, r_1y, div.q, 0.25)

    snap = MarketSnapshot(
        ticker=ticker,
        valuation_date=valuation_date,
        spot=spot,
        ois_curve=ois,
        div_curve=div,
        vol_surface=surf,
    )
    with _SNAPSHOT_LOCK:
        _SNAPSHOT_CACHE[cache_key] = snap
    return snap


def Call(
    mode: str,
    ticker: str,
    *args,
    ratio: float = 1.0,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    Pricing d'un Call vanille.

    AL : Call('AL', 'GLE', spot, strike, start_date, maturity)
    MC : Call('MC', 'GLE', n_paths, spot, strike, start_date, maturity)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, strike, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, strike, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).call(float(strike), T, ratio)
    else:
        result = MCPricer(snap, int(n_paths), seed=seed).call(float(strike), T, ratio)

    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def Put(
    mode: str,
    ticker: str,
    *args,
    ratio: float = 1.0,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    AL : Put('AL', 'GLE', spot, strike, start_date, maturity)
    MC : Put('MC', 'GLE', n_paths, spot, strike, start_date, maturity)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, strike, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, strike, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).put(float(strike), T, ratio)
    else:
        result = MCPricer(snap, int(n_paths), seed=seed).put(float(strike), T, ratio)

    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def Warrant(
    mode: str,
    ticker: str,
    *args,
    flag: str = "c",
    ratio: float = 1.0,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    AL : Warrant('AL', 'MC', spot, strike, start_date, maturity, flag='c', ratio=0.1)
    MC : Warrant('MC', 'MC', n_paths, spot, strike, start_date, maturity, ratio=0.1)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, strike, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, strike, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).warrant_product(flag, float(strike), T, ratio)
    else:
        fn = MCPricer(snap, int(n_paths), seed=seed)
        result = fn.call(float(strike), T, ratio) if flag == "c" else fn.put(float(strike), T, ratio)

    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def RC(
    mode: str,
    ticker: str,
    *args,
    freq: str = "A",
    nominal: float = 100.0,
    ratio: float = 1.0,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    Reverse Convertible avec fréquence de coupon.

    AL : RC('AL', 'GLE', spot, start_date, maturity, freq='Q')
    MC : RC('MC', 'GLE', n_paths, spot, coupon, start_date, maturity, freq='Q')

    Mode AL → retourne le coupon fair (price = 100%).
    Mode MC → coupon fourni explicitement en input.
    """
    mode = mode.upper()
    if mode == "AL":
        spot, start_str, tenor_str = args
        start = _parse_date(start_str)
        T     = _parse_tenor(tenor_str)
        snap  = _build_snapshot(ticker, float(spot), start)
        result = AnalyticalPricer(snap).reverse_convertible(
            cap=float(spot), T=T, freq=freq, nominal=nominal, ratio=ratio, start=start
        )
    else:
        n_paths, spot, coupon, start_str, tenor_str = args
        start = _parse_date(start_str)
        T     = _parse_tenor(tenor_str)
        snap  = _build_snapshot(ticker, float(spot), start)
        result = MCPricer(snap, int(n_paths), seed=seed).reverse_convertible(
            cap=float(spot), T=T, coupon=float(coupon),
            freq=freq, nominal=nominal, ratio=ratio, start=start,
        )

    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def BonusCertificate(
    mode: str,
    ticker: str,
    *args,
    ratio: float = 1.0,
    barrier_hit: bool = False,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    AL : BonusCertificate('AL', 'GLE', spot, bonus_level, barrier, start_date, maturity)
    MC : BonusCertificate('MC', 'GLE', n_paths, spot, bonus_level, barrier, start_date, maturity)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, X, B, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, X, B, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).bonus_certificate(float(X), float(B), T, ratio, barrier_hit)
    else:
        result = MCPricer(snap, int(n_paths), seed=seed).bonus_certificate(float(X), float(B), T, ratio)

    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def ReverseBonusCertificate(
    mode: str,
    ticker: str,
    *args,
    ratio: float = 1.0,
    barrier_hit: bool = False,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    AL : ReverseBonusCertificate('AL', 'GLE', spot, bonus_level, barrier, start_date, maturity)
    MC : ReverseBonusCertificate('MC', 'GLE', n_paths, spot, bonus_level, barrier, start_date, maturity)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, X, B, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, X, B, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).reverse_bonus_certificate(float(X), float(B), T, ratio, barrier_hit)
    else:
        result = MCPricer(snap, int(n_paths), seed=seed).reverse_bonus_certificate(float(X), float(B), T, ratio)

    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def DiscountCertificate(
    mode: str,
    ticker: str,
    *args,
    ratio: float = 1.0,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    AL : DiscountCertificate('AL', 'GLE', spot, cap, start_date, maturity)
    MC : DiscountCertificate('MC', 'GLE', n_paths, spot, cap, start_date, maturity)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, cap, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, cap, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).discount_certificate(float(cap), T, ratio)
    else:
        result = MCPricer(snap, int(n_paths), seed=seed).discount_certificate(float(cap), T, ratio)

    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def AirbagCertificate(
    mode: str,
    ticker: str,
    *args,
    ratio: float = 1.0,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    AL : AirbagCertificate('AL', 'GLE', spot, strike, barrier, start_date, maturity)
    MC : AirbagCertificate('MC', 'GLE', n_paths, spot, strike, barrier, start_date, maturity)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, X, B, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, X, B, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).airbag_certificate(float(X), float(B), T, ratio)
    else:
        result = MCPricer(snap, int(n_paths), seed=seed).airbag_certificate(float(X), float(B), T, ratio)
    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def GarantieCertificate(
    mode: str,
    ticker: str,
    *args,
    nominal: float = 100.0,
    ratio: float = 1.0,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    AL : GarantieCertificate('AL', 'GLE', spot, strike, start_date, maturity)
    MC : GarantieCertificate('MC', 'GLE', n_paths, spot, strike, start_date, maturity)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, X, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, X, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).garantie_certificate(float(X), T, nominal, ratio)
    else:
        result = MCPricer(snap, int(n_paths), seed=seed).garantie_certificate(float(X), T, nominal, ratio)
    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def OutperformanceCertificate(
    mode: str,
    ticker: str,
    *args,
    ratio: float = 1.0,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    AL : OutperformanceCertificate('AL', 'GLE', spot, strike, start_date, maturity)
    MC : OutperformanceCertificate('MC', 'GLE', n_paths, spot, strike, start_date, maturity)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, X, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, X, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).outperformance_certificate(float(X), T, ratio)
    else:
        result = MCPricer(snap, int(n_paths), seed=seed).outperformance_certificate(float(X), T, ratio)
    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def SprintCertificate(
    mode: str,
    ticker: str,
    *args,
    ratio: float = 1.0,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    AL : SprintCertificate('AL', 'GLE', spot, strike, cap, start_date, maturity)
    MC : SprintCertificate('MC', 'GLE', n_paths, spot, strike, cap, start_date, maturity)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, X, cap, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, X, cap, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).sprint_certificate(float(X), float(cap), T, ratio)
    else:
        result = MCPricer(snap, int(n_paths), seed=seed).sprint_certificate(float(X), float(cap), T, ratio)
    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def TwinWinCertificate(
    mode: str,
    ticker: str,
    *args,
    ratio: float = 1.0,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    AL : TwinWinCertificate('AL', 'GLE', spot, strike, barrier, start_date, maturity)
    MC : TwinWinCertificate('MC', 'GLE', n_paths, spot, strike, barrier, start_date, maturity)
    """
    mode = mode.upper()
    if mode == "AL":
        spot, X, B, start_str, tenor_str = args
        n_paths = None
    else:
        n_paths, spot, X, B, start_str, tenor_str = args

    start = _parse_date(start_str)
    T     = _parse_tenor(tenor_str)
    snap  = _build_snapshot(ticker, float(spot), start)

    if mode == "AL":
        result = AnalyticalPricer(snap).twin_win_certificate(float(X), float(B), T, ratio)
    else:
        result = MCPricer(snap, int(n_paths), seed=seed).twin_win_certificate(float(X), float(B), T, ratio)
    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def Phoenix(
    mode: str,
    ticker: str,
    n_paths: int,
    start_date: str,
    spot: float,
    coupon: float = 0.0,
    barrier_coupon: float = 0.80,
    barrier_recall: float = 1.00,
    capital_barrier: float = 0.70,
    freq_months: int = 3,
    maturity: str = "60M",
    kg: str = "no",
    autocall_start: int = 0,
    compute_greeks: bool = False,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    Phoenix / Autocall Monte Carlo Heston.

    mode='AL' : coupon fair calibré, prix ≈ 100 % (coupon ignoré)
        Phoenix('AL', 'GLE', 20000, '02/05/2026', 100,
                barrier_coupon=0.80, barrier_recall=1.00, capital_barrier=0.70,
                freq_months=3, maturity='60M', kg='no')

    mode='MC'   : prix en % pour un coupon donné, coupon_fair en info
        Phoenix('MC', 'GLE', 20000, '02/05/2026', 100, coupon=0.08,
                barrier_coupon=0.80, barrier_recall=1.00, capital_barrier=0.70,
                freq_months=3, maturity='60M', kg='no')

    Barrières en fraction du spot (ex: 0.80 = 80%) ou valeur absolue si ≥ 10.
    """
    start      = _parse_date(start_date)
    mat_months = _parse_tenor_months(maturity)
    snap       = _build_snapshot(ticker, float(spot), start)

    B_c = barrier_coupon  * spot if barrier_coupon  < 10 else barrier_coupon
    B_r = barrier_recall  * spot if barrier_recall  < 10 else barrier_recall
    B_k = capital_barrier * spot if capital_barrier < 10 else capital_barrier

    pricer = PhoenixPricer(
        snapshot=snap, n_paths=int(n_paths), start_date=start, S0=float(spot),
        barrier_coupon=B_c, barrier_recall=B_r, capital_barrier=B_k,
        freq_months=int(freq_months), maturity_months=int(mat_months),
        kg=kg, autocall_start=int(autocall_start),
        mode=mode, coupon=float(coupon), compute_greeks=compute_greeks, seed=seed,
    )
    result = pricer.price()
    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def Athena(
    mode: str,
    ticker: str,
    n_paths: int,
    start_date: str,
    spot: float,
    coupon: float = 0.0,
    barrier_recall: float = 1.00,
    capital_barrier: float = 0.70,
    freq_months: int = 3,
    maturity: str = "36M",
    kg: str = "no",
    compute_greeks: bool = False,
    seed: Optional[int] = None,
    show_vol: bool = False,
) -> PricingResult:
    """
    Athena Monte Carlo Heston.

    mode='AL' : coupon fair calibré, prix ≈ 100 % (coupon ignoré)
        Athena('AL', 'GLE', 20000, '02/05/2026', 100,
               barrier_recall=1.00, capital_barrier=0.70,
               freq_months=3, maturity='36M', kg='no')

    mode='MC'   : prix en % pour un coupon donné, coupon_fair en info
        Athena('MC', 'GLE', 20000, '02/05/2026', 100, coupon=0.08,
               barrier_recall=1.00, capital_barrier=0.70,
               freq_months=3, maturity='36M', kg='no')

    Barrières en fraction du spot (ex: 1.00 = 100%) ou valeur absolue si ≥ 10.
    """
    start      = _parse_date(start_date)
    mat_months = _parse_tenor_months(maturity)
    snap       = _build_snapshot(ticker, float(spot), start)

    B_r = barrier_recall  * spot if barrier_recall  < 10 else barrier_recall
    B_k = capital_barrier * spot if capital_barrier < 10 else capital_barrier

    pricer = AthenaPricer(
        snapshot=snap, n_paths=int(n_paths), start_date=start, S0=float(spot),
        barrier_recall=B_r, capital_barrier=B_k,
        freq_months=int(freq_months), maturity_months=int(mat_months),
        kg=kg, mode=mode, coupon=float(coupon), compute_greeks=compute_greeks, seed=seed,
    )
    result = pricer.price()
    if show_vol:
        print(result)
        snap.vol_surface.plot(ticker=ticker)
    return result


def build_snapshot(ticker: str, spot: float, start_date: str) -> MarketSnapshot:
    """
    Construit et retourne un MarketSnapshot (données de marché complètes).

    Utile pour utiliser directement les classes du module risk :
        snap = build_snapshot('GLE', 100, '02/05/2026')
        from pricing_lib.risk import FiniteDiffGreeks, StressTest
    """
    return _build_snapshot(ticker, float(spot), _parse_date(start_date))


def _make_pricer_fn(mode: str, product: str, snap: MarketSnapshot,
                    T: float, start, n_paths, seed, **pkwargs):
    """Retourne une closure f(market)->float pour n'importe quel produit."""
    product_l = product.lower().replace("-", "_").replace(" ", "_")
    ratio     = pkwargs.get("ratio", 1.0)
    rc_cap    = pkwargs.get("cap", snap.spot)

    def _fn(market: MarketSnapshot) -> float:
        if mode == "AL":
            p = AnalyticalPricer(market)
        else:
            p = MCPricer(market, int(n_paths), seed=seed)
        if product_l in ("call", "c"):
            return p.call(pkwargs["K"], T, ratio).price
        elif product_l in ("put", "p"):
            return p.put(pkwargs["K"], T, ratio).price
        elif product_l in ("rc", "reverse_convertible"):
            return p.reverse_convertible(cap=rc_cap, T=T,
                freq=pkwargs.get("freq", "A"), nominal=pkwargs.get("nominal", 100.0),
                ratio=ratio, start=start).price
        elif product_l in ("bonus", "bonus_certificate"):
            return p.bonus_certificate(pkwargs["X"], pkwargs["B"], T, ratio).price
        elif product_l in ("reverse_bonus", "reverse_bonus_certificate"):
            return p.reverse_bonus_certificate(pkwargs["X"], pkwargs["B"], T, ratio).price
        elif product_l in ("twin_win", "twin_win_certificate"):
            return p.twin_win_certificate(pkwargs["X"], pkwargs["B"], T, ratio).price
        elif product_l in ("airbag", "airbag_certificate"):
            return p.airbag_certificate(pkwargs["X"], pkwargs["B"], T, ratio).price
        elif product_l in ("discount", "discount_certificate"):
            return p.discount_certificate(pkwargs["X"], T, ratio).price
        elif product_l in ("sprint", "sprint_certificate"):
            return p.sprint_certificate(pkwargs["X"], pkwargs["cap_level"], T, ratio).price
        elif product_l in ("outperformance", "outperformance_certificate"):
            return p.outperformance_certificate(pkwargs["X"], T, ratio).price
        else:
            raise ValueError(f"product='{product}' non supporte. "
                "Options: call, put, rc, bonus, reverse_bonus, twin_win, airbag, discount.")
    return _fn

def Greeks(mode: str, ticker: str, *args,
           product: str = "call", seed: Optional[int] = None, **pkwargs) -> GreeksResult:
    """
    Greeks par differences finies centrees.
    AL : Greeks('AL', 'GLE', spot, start_date, maturity, product='call', K=71)
    AL : Greeks('AL', 'GLE', spot, start_date, maturity, product='rc', freq='Q')
    """
    mode = mode.upper()
    spot_val, start_str, tenor_str = (args if len(args)==3 else args[1:])
    n_paths = None if len(args)==3 else int(args[0])
    snap   = _build_snapshot(ticker, float(spot_val), _parse_date(start_str))
    T      = _parse_tenor(tenor_str)
    fn     = _make_pricer_fn(mode, product, snap, T, _parse_date(start_str), n_paths, seed, **pkwargs)
    return FiniteDiffGreeks(_ClosurePricer(fn), None, snap).all()

def Stress(mode: str, ticker: str, *args,
           product: str = "call", seed: Optional[int] = None,
           scenarios: Optional[List[StressScenario]] = None, **pkwargs) -> "pd.DataFrame":
    """
    Stress test multi-scenarios.
    AL : Stress('AL', 'GLE', spot, start_date, maturity, product='rc', freq='Q')
    AL : Stress('AL', 'GLE', spot, start_date, maturity, product='bonus', X=78, B=46)
    AL : Stress('AL', 'GLE', spot, start_date, maturity, product='twin_win', X=71, B=46)
    """
    mode = mode.upper()
    spot_val, start_str, tenor_str = (args if len(args)==3 else args[1:])
    n_paths = None if len(args)==3 else int(args[0])
    start  = _parse_date(start_str)
    snap   = _build_snapshot(ticker, float(spot_val), start)
    T      = _parse_tenor(tenor_str)
    fn     = _make_pricer_fn(mode, product, snap, T, start, n_paths, seed, **pkwargs)
    return StressTest(_ClosurePricer(fn), None, snap, scenarios=scenarios).run()

def ScenarioGrid(mode: str, ticker: str, *args,
                 product: str = "call", seed: Optional[int] = None,
                 spot_shocks: Optional[List[float]] = None,
                 vol_shocks:  Optional[List[float]] = None, **pkwargs) -> "pd.DataFrame":
    """
    Grille de prix spot x vol.
    AL : ScenarioGrid('AL', 'GLE', spot, start_date, maturity, product='rc', freq='Q',
                      spot_shocks=[-0.20,-0.10,0,0.10,0.20], vol_shocks=[-0.05,0,0.05])
    """
    mode = mode.upper()
    spot_val, start_str, tenor_str = (args if len(args)==3 else args[1:])
    n_paths = None if len(args)==3 else int(args[0])
    start  = _parse_date(start_str)
    snap   = _build_snapshot(ticker, float(spot_val), start)
    T      = _parse_tenor(tenor_str)
    _ss = spot_shocks if spot_shocks is not None else [-0.20,-0.10,0,+0.10,+0.20]
    _vs = vol_shocks  if vol_shocks  is not None else [-0.05, 0, +0.05]
    fn  = _make_pricer_fn(mode, product, snap, T, start, n_paths, seed, **pkwargs)
    return ScenarioAnalyzer(_ClosurePricer(fn), None, snap).spot_vol_grid(_ss, _vs)

# ── Exemple d'utilisation (exécuter directement : python -m pricing_lib.api) ──
if __name__ == "__main__":
    '''
    y = Phoenix('AL', 'GLE', 100000, '29/05/2026', 71,
            barrier_coupon=0.70,
            barrier_recall=1.00,
            capital_barrier=0.50,
            freq_months=3, maturity='60M', kg='no')
    #print(y)'''


Stress('AL', 'GLE', 100, 100, '02/05/2026', '3M', product='call')
# Reverse Convertible trimestriel
Stress('AL', 'GLE', 71, '29/05/2026', '12M', product='rc', freq='Q')

# Bonus Certificate (bonus 110%, barrière 65%)
Stress('AL', 'GLE', 71, '29/05/2026', '12M', product='bonus', X=78, B=46)

# Twin Win (strike ATM, barrière 65%)
Stress('AL', 'GLE', 71, '29/05/2026', '24M', product='twin_win', X=71, B=46)

# Airbag
df = Stress('AL', 'GLE', 71, '29/05/2026', '24M', product='airbag', X=71, B=50)


print(df)
