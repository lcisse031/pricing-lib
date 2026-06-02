from __future__ import annotations

import calendar
import math
import os
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import brentq
from scipy.stats import norm

# Local imports
from .rates import fetch_ois_curve
from .dividends import fetch_dividend


_EURONEXT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://live.euronext.com/fr/product/stock-options/",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "text/html, */*",
}

_MONTH_FR: Dict[str, str] = {
    "Jan": "01", "Fév": "02", "Mar": "03", "Avr": "04",
    "Mai": "05", "Jun": "06", "Jul": "07", "Aoû": "08",
    "Sep": "09", "Oct": "10", "Nov": "11", "Déc": "12",
    "Feb": "02", "Apr": "04", "May": "05", "Aug": "08", "Dec": "12",
}

_TICKER_MAP = {
    "GLE":   "GL4",    # Société Générale
    "AI":    "AI4",    # Air Liquide
    "AIR":   "EA4",   # Airbus
    "BNP":   "BN4",   # BNP Paribas
    "EN":    "EN4",    # Bouygues
    "CA":    "CA4",    # Carrefour
    "ACA":   "CR4",   # Crédit Agricole
    "BN":    "BN4",    # Danone

    "MC":    "MC4",    # LVMH
    "RMS":   "HE4",   # Hermès International
    "OR":    "OR4",    # L'Oréal
    "KER":   "KR4",   # Kering

    "SAF":   "SM4",   # Safran
    "HO":    "HO4",    # Thales
    "SU":    "SU4",    # Schneider Electric
    "SGO":   "SG4",   # Saint-Gobain
    "LR":    "LR4",    # Legrand
    "MT":    "MT9",    # ArcelorMittal
    "STMPA": "STM4",   # STMicroelectronics

    "CS":    "CS4",    # AXA

    "TTE":   "TO4",   # TotalEnergies
    "ENGI":  "ENGI4",  # Engie
    "VIE":   "VIE4",   # Veolia Environnement

    "SAN":   "ZA4",   # Sanofi
    "EL":    "EL4",    # EssilorLuxottica
    "ERF":   "ERF4",   # Eurofins Scientific

    "DG":    "DG4",    # Vinci
    "FGR":   "FGR4",   # Eiffage

    "STLAP": "STL4",   # Stellantis
    "ML":    "ML4",    # Michelin
    "RNO":   "RN4",   # Renault

    "DSY":   "DS4",   # Dassault Systèmes
    "CAP":   "CP4",   # Capgemini

    "RI":    "RI4",    # Pernod Ricard

    "ORA":   "FT4",   # Orange

    "BVI":   "BVI4",   # Bureau Veritas
    "PUB":   "PU4",   # Publicis Groupe
    "ENX":   "ENX4",   # Euronext

    "AC":    "AC4",    # Accor

    "URW":   "UB9",   # Unibail-Rodamco-Westfield
}

# Tickers listés sur Euronext Amsterdam plutôt que Paris
_AMSTERDAM_TICKERS = {"MT", "URW"}

_IV_BOUNDS = (1e-4, 5.0)


def fetch_market_params(ticker: str, valuation_date: Optional[date] = None) -> Tuple[float, float]:
    """Récupère (r, q) automatiquement depuis OIS + dividendes."""
    if valuation_date is None:
        valuation_date = date.today()

    try:
        ois_curve = fetch_ois_curve(valuation_date)
        r = ois_curve.zero_rate_tau(1.0)
    except Exception:
        r = 0.04

    try:
        div_yield = fetch_dividend(ticker)
        q = div_yield.recommended
    except Exception:
        q = 0.02

    return r, q


def fetch_option_chain(symbol: str, exchange: str, verbose: bool = True) -> pd.DataFrame:


    # ── Session commune — conserve les cookies entre requêtes ─────────────────
    session = requests.Session()
    session.headers.update(_EURONEXT_HEADERS)

    # Visite initiale pour établir la session (cookies JSESSIONID, etc.)
    _BASE_HEADERS_INIT = {
        **_EURONEXT_HEADERS,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        session.get(
            f"https://live.euronext.com/fr/product/stock-options/{symbol}-{exchange}",
            headers=_BASE_HEADERS_INIT,
            timeout=15,
        )
    except requests.RequestException:
        pass   # non bloquant — on tente quand même

    # ── 1. Récupération des maturités ─────────────────────────────────────────
    url_form = f"https://live.euronext.com/fr/ajax/getPricesOptionsForm/{symbol}/{exchange}"
    try:
        resp_form = session.get(url_form, timeout=15)
        resp_form.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Impossible de récupérer les maturités pour {symbol}/{exchange} : {e}")

    soup_form = BeautifulSoup(resp_form.text, "html.parser")
    maturites = [opt.get("value") for opt in soup_form.find_all("option") if opt.get("value")]

    if not maturites:
        raise ValueError(
            f"Aucune maturité trouvée pour {symbol}/{exchange}.\n"
            f"  HTTP {resp_form.status_code} — réponse ({len(resp_form.text)} chars) :\n"
            f"  {resp_form.text[:300]}"
        )

    if verbose:
        print(f"[Euronext] {symbol}/{exchange} — {len(maturites)} maturité(s) : {maturites}")

    # ── 2. Worker : scrape UNE maturité ──────────────────────────────────────
    url_data = f"https://live.euronext.com/fr/ajax/submitOptionsForm/stock-options/{symbol}/{exchange}"

    def scraper_maturite(mat: str) -> List[Dict]:
        try:
            resp = session.post(
                url_data,
                data={"md[]": mat, "ps": "999"},
                headers={
                    **_EURONEXT_HEADERS,
                    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                    "X-Requested-With": "XMLHttpRequest",
                },
                timeout=15,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            if verbose:
                print(f"[Euronext]   POST {mat} — erreur réseau : {e}")
            return []

        # Détection réponse vide ou non-HTML
        body = resp.text.strip()
        if not body or body.startswith("{") or body.startswith("["):
            if verbose:
                print(f"[Euronext]   POST {mat} — réponse inattendue ({resp.status_code}): {body[:120]}")
            return []

        soup  = BeautifulSoup(body, "html.parser")
        table = soup.find("table")
        if not table and verbose:
            print(f"[Euronext]   POST {mat} — pas de <table> dans la réponse ({len(body)} chars)")
            print(f"             Début réponse : {body[:200]}")

        records = []
        if table:
            rows = table.find_all("tr")[1:]
            if verbose:
                print(f"[Euronext]   {mat} — {len(rows)} ligne(s) trouvée(s)")
            _n_skip = 0
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 12:
                    continue
                try:
                    call_bid = _pf(cols[3].text.strip())
                    call_ask = _pf(cols[4].text.strip())
                    call_der = _pf(cols[2].text.strip())

                    strike_str = cols[6].text.strip()
                    strike = float(strike_str.replace(",", ".")) if strike_str else None
                    if strike is None:
                        continue

                    put_bid = _pf(cols[8].text.strip())
                    put_ask = _pf(cols[9].text.strip())
                    put_der = _pf(cols[10].text.strip())

                    call_mid = (call_bid + call_ask) / 2 if call_bid and call_ask else call_der
                    put_mid  = (put_bid  + put_ask)  / 2 if put_bid  and put_ask  else put_der

                    expiry = _parse_maturity_to_yyyy_mm(mat)

                    if call_mid and call_mid > 0:
                        records.append({
                            "maturity_label": mat, "expiry_date": expiry,
                            "strike": strike, "type": "C",
                            "bid": call_bid, "ask": call_ask,
                            "last": call_der, "mid": call_mid,
                        })
                    else:
                        _n_skip += 1
                    if put_mid and put_mid > 0:
                        records.append({
                            "maturity_label": mat, "expiry_date": expiry,
                            "strike": strike, "type": "P",
                            "bid": put_bid, "ask": put_ask,
                            "last": put_der, "mid": put_mid,
                        })
                    else:
                        _n_skip += 1
                except (ValueError, IndexError, AttributeError):
                    _n_skip += 1
                    continue

            if verbose and _n_skip > 0:
                print(f"[Euronext]   {mat} — {_n_skip} ligne(s) ignorée(s) (prix absents/colonnes?)")

        return records

    # ── 3. Parallélisation ────────────────────────────────────────────────────
    all_records: List[Dict] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scraper_maturite, m): m for m in maturites}
        for future in as_completed(futures):
            try:
                all_records.extend(future.result())
            except Exception as e:
                if verbose:
                    print(f"[Euronext]   future erreur : {e}")

    if not all_records:
        raise ValueError(
            f"Aucune donnée options pour {symbol}/{exchange}.\n"
            f"  Maturités tentées : {maturites}\n"
            f"  Vérifier : cookies session, format des maturités, structure du tableau."
        )

    df = pd.DataFrame(all_records)
    if verbose:
        print(f"[Euronext] {len(df)} lignes récupérées "
              f"({df['expiry_date'].nunique()} maturités, "
              f"{df['strike'].nunique()} strikes)")
    return df.sort_values(["expiry_date", "strike", "type"]).reset_index(drop=True)


def _bs_price(flag: int, S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Prix Black-Scholes (flag=+1 call, flag=-1 put)."""
    if T <= 0 or sigma <= 0:
        return max(flag * (S * math.exp(-q * T) - K * math.exp(-r * T)), 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return flag * (
        S * math.exp(-q * T) * norm.cdf(flag * d1)
        - K * math.exp(-r * T) * norm.cdf(flag * d2)
    )


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    tol: float = 1e-7,
) -> Optional[float]:
    """Vol implicite par inversion Black-Scholes (Brent)."""
    if T <= 0 or market_price <= 0:
        return None
    flag = 1 if option_type.upper() == "C" else -1

    intrinsic = max(flag * (S * math.exp(-q * T) - K * math.exp(-r * T)), 0.0)
    if market_price < intrinsic - 1e-6:
        return None

    def objective(sigma: float) -> float:
        return _bs_price(flag, S, K, T, r, q, sigma) - market_price

    try:
        lo, hi = _IV_BOUNDS
        if objective(lo) * objective(hi) > 0:
            return None
        return brentq(objective, lo, hi, xtol=tol, full_output=False)
    except (ValueError, RuntimeError):
        return None


class VolSurface:
    """Surface de volatilité implicite interpolée (T, K) → σ_i."""

    def __init__(
        self,
        valuation_date: date,
        spot: float,
        r: float,
        q: float,
        tenors: np.ndarray,
        strikes: np.ndarray,
        vols: np.ndarray,
    ) -> None:
        self.valuation_date = valuation_date
        self.spot = spot
        self.r = r
        self.q = q
        self.tenors = tenors
        self.strikes = strikes
        self.vols = vols

        # spline 2D
        if len(tenors) >= 4 and len(strikes) >= 4:
            self._spline = RectBivariateSpline(
                tenors, strikes, vols, kx=min(3, len(tenors)-1), ky=min(3, len(strikes)-1)
            )
            self._use_spline = True
        else:
            self._use_spline = False

    def __call__(self, T: float, K: float) -> float:
        return self.vol(T, K)

    def vol(self, T: float, K: float) -> float:
        """Volatilité implicite interpolée."""
        T_clip = float(np.clip(T, self.tenors[0], self.tenors[-1]))
        K_clip = float(np.clip(K, self.strikes[0], self.strikes[-1]))

        if self._use_spline:
            v = float(self._spline(T_clip, K_clip)[0][0])
        else:
            v = float(np.interp(
                K_clip, self.strikes,
                self.vols[np.argmin(np.abs(self.tenors - T_clip)), :]
            ))
        return max(v, 1e-4)

    def atm_vol(self, T: float) -> float:
        """Vol ATM (strike = forward)."""
        F = self.spot * math.exp((self.r - self.q) * T)
        return self.vol(T, F)

    def skew(self, T: float, dK: float = 5.0) -> float:
        """∂σ/∂K numérique autour de l'ATM."""
        F = self.spot * math.exp((self.r - self.q) * T)
        return (self.vol(T, F + dK) - self.vol(T, F - dK)) / (2 * dK)

    @classmethod
    def from_option_chain(
        cls,
        df: pd.DataFrame,
        valuation_date: date,
        spot: float,
        r: float,
        q: float,
    ) -> "VolSurface":
        """Construit une VolSurface depuis la chaîne d'options - SANS FILTRES INUTILES."""

        records: List[Dict] = []
        today = valuation_date

        for _, row in df.iterrows():
            expiry_str = str(row["expiry_date"])   # format "YYYY-MM"
            try:
                year, month = int(expiry_str[:4]), int(expiry_str[5:7])
                expiry_dt   = _third_friday(year, month)
            except (ValueError, IndexError):
                continue

            T = (expiry_dt - today).days / 365.25
            if T <= 0:
                continue

            K = float(row["strike"])
            opt_type = str(row["type"]).upper()

            # Récupère le prix
            mid = row.get("mid")
            if mid is not None and mid > 0:
                price = float(mid)
            else:
                bid = row.get("bid")
                ask = row.get("ask")
                if bid and ask and bid > 0 and ask > 0:
                    price = (float(bid) + float(ask)) / 2
                elif row.get("last") and float(row["last"]) > 0:
                    price = float(row["last"])
                else:
                    continue

            # Calcule VI
            iv = implied_vol(price, spot, K, T, r, q, opt_type)
            if iv is None:
                continue

            records.append({"T": T, "K": K, "iv": iv})

        if not records:
            raise ValueError("Surface vide — aucune vol implicite calculée.")

        ivdf = pd.DataFrame(records)

        tenors_raw = sorted(ivdf["T"].unique())
        strikes_raw = sorted(ivdf["K"].unique())

        if len(tenors_raw) < 2 or len(strikes_raw) < 2:
            raise ValueError(f"Surface trop pauvre ({len(tenors_raw)} tenor(s), {len(strikes_raw)} strike(s))")

        # Grid avec interpolation
        grid = ivdf.groupby(["T", "K"])["iv"].mean().unstack("K")
        grid = grid.reindex(index=tenors_raw).ffill()
        grid = grid.T.reindex(strikes_raw).ffill().bfill().T
        grid = grid.ffill().bfill()

        vols_array = grid.to_numpy(dtype=float)

        return cls(
            valuation_date=today,
            spot=spot,
            r=r,
            q=q,
            tenors=np.array(sorted(grid.index.tolist())),
            strikes=np.array(sorted(grid.columns.tolist())),
            vols=vols_array,
        )

    @classmethod
    def flat(cls, valuation_date: date, spot: float, r: float, q: float, sigma: float) -> "VolSurface":
        """Surface plate — utile pour les tests."""
        Ts = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
        Ks = np.linspace(spot * 0.5, spot * 1.5, 15)
        vols = np.full((len(Ts), len(Ks)), sigma)
        return cls(valuation_date, spot, r, q, Ts, Ks, vols)

    # ── Persistance disque ────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Sauvegarde la surface dans un fichier .npz (numpy compressed).
        Utilisé comme cache EOD pour fonctionner hors heures de marché.
        """
        np.savez_compressed(
            path,
            tenors=self.tenors,
            strikes=self.strikes,
            vols=self.vols,
            spot=np.array([self.spot]),
            r=np.array([self.r]),
            q=np.array([self.q]),
            valuation_date=np.array([self.valuation_date.isoformat()]),
            saved_at=np.array([datetime.now().isoformat()]),
        )

    @classmethod
    def load(cls, path: str) -> "VolSurface":
        """
        Reconstruit une VolSurface depuis un fichier .npz sauvegardé par save().
        """
        data = np.load(path, allow_pickle=False)
        val_date = date.fromisoformat(str(data["valuation_date"][0]))
        saved_at = str(data["saved_at"][0])
        surf = cls(
            valuation_date=val_date,
            spot=float(data["spot"][0]),
            r=float(data["r"][0]),
            q=float(data["q"][0]),
            tenors=data["tenors"],
            strikes=data["strikes"],
            vols=data["vols"],
        )
        print(f"[VolSurface] Cache chargé depuis {path}")
        print(f"             Sauvegardé le {saved_at[:16]} | spot={surf.spot:.2f} | "
              f"{len(surf.tenors)} tenors × {len(surf.strikes)} strikes")
        return surf

    def __repr__(self) -> str:
        return (
            f"VolSurface(spot={self.spot:.2f}, "
            f"tenors={self.tenors[0]:.2f}-{self.tenors[-1]:.2f}y, "
            f"strikes={self.strikes[0]:.1f}-{self.strikes[-1]:.1f})"
        )

    def to_df(self) -> pd.DataFrame:
        """
        DataFrame de la surface de vol implicite.
        Index   : maturités (label '3M', '1Y', ...)
        Colonnes: strikes
        Valeurs : IV en %
        """
        labels = [f"{int(round(T*12))}M" if T < 1 else f"{T:.1f}Y" for T in self.tenors]
        df = pd.DataFrame(
            np.round(self.vols * 100, 2),
            index=labels,
            columns=self.strikes,
        )
        df.index.name   = "Maturité"
        df.columns.name = "Strike"
        return df

    def plot(self, ticker: str = "") -> None:
        """
        Affiche 2 graphiques :
          - Smile       : IV (%) vs Strike,  une courbe par maturité
          - Term struct : IV (%) vs Maturité, une courbe par strike (80/90/100/110/120 % spot)
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        title_prefix = f"[{ticker}] " if ticker else ""

        # ── Smile : IV vs Strike ──────────────────────────────────────────────
        ax = axes[0]
        colors = cm.viridis(np.linspace(0, 0.9, len(self.tenors)))
        for i, T in enumerate(self.tenors):
            label = f"{int(round(T * 12))}M" if T < 1 else f"{T:.1f}Y"
            ax.plot(self.strikes, self.vols[i, :] * 100,
                    color=colors[i], label=label, lw=1.5)
        ax.axvline(self.spot, color="red", linestyle="--", lw=1.2,
                   label=f"Spot {self.spot:.1f}")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Vol implicite (%)")
        ax.set_title(f"{title_prefix}Smile de volatilité")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # ── Term structure : IV vs Maturité ───────────────────────────────────
        ax = axes[1]
        pcts   = [0.80, 0.90, 1.00, 1.10, 1.20]
        colors2 = cm.plasma(np.linspace(0, 0.85, len(pcts)))
        for pct, col in zip(pcts, colors2):
            K = float(np.clip(self.spot * pct, self.strikes[0], self.strikes[-1]))
            ivs = [self.vol(T, K) * 100 for T in self.tenors]
            ax.plot(self.tenors, ivs, color=col, marker="o", ms=4, lw=1.5,
                    label=f"{int(pct * 100)}% spot")
        ax.set_xlabel("Maturité (années)")
        ax.set_ylabel("Vol implicite (%)")
        ax.set_title(f"{title_prefix}Structure par terme")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── DataFrames filtrés : ±40 % autour du spot ────────────────────────
        s    = self.spot
        mask = (self.strikes >= s * 0.60) & (self.strikes <= s * 1.40)
        fk   = self.strikes[mask]
        fv   = self.vols[:, mask]
        lbl  = [f"{int(round(T*12))}M" if T < 1 else f"{T:.1f}Y" for T in self.tenors]

        def _make_df(k_mask):
            return pd.DataFrame(
                np.round(fv[:, k_mask] * 100, 2),
                index=lbl,
                columns=[f"{k:.1f}" for k in fk[k_mask]],
            ).rename_axis("Mat \\ K", axis="index")

        left  = _make_df(fk <= s)   # puts OTM / calls ITM
        right = _make_df(fk >  s)   # calls OTM / puts ITM

        sep = "─" * 64
        print(f"\n{sep}")
        print(f"  Surface IV (%) — {ticker}   [Spot = {s:.1f}]")
        print(f"{sep}")
        print(f"\n  ◀  Strikes ≤ spot  (puts OTM)\n")
        print(left.to_string())
        print(f"\n  ▶  Strikes > spot  (calls OTM)\n")
        print(right.to_string())
        print(f"\n{sep}\n")

        fig.tight_layout()
        plt.show()


def dupire_local_vol(
    surf: VolSurface,
    T: float,
    K: float,
    dT: float = 1.0 / 52,
    dK_frac: float = 0.005,
) -> float:
    """Vol locale σ_loc(T, K) — formule de Gatheral (Dupire en variance totale).

    Principe
    --------
    Travaille directement sur w(T, y) = σ²_IV(T, K) · T (variance totale implicite),
    où σ_IV est lu depuis la surface (déjà extraite par inversion Brent).
    Aucun aller-retour via des prix BS.

    Formule de Gatheral
    -------------------
        σ²_loc(T, y) = (∂w/∂T)|_y  /  g(y, w)

        g = 1 − (y/w)·(∂w/∂y) + ¼·(∂w/∂y)²·(−¼ − 1/w + y²/w²) + ½·∂²w/∂y²

        y = ln(K / F),   F = S · exp((r − q) · T)

    Différentielles
    ---------------
    ∂w/∂T  : centrée O(dT²) à y fixe — K s'ajuste pour garder ln(K/F) constant
    ∂w/∂y  = K · ∂w/∂K              (règle de la chaîne, y = ln K − ln F)
    ∂²w/∂y²= K²·∂²w/∂K² + K·∂w/∂K
    """
    S, r, q = surf.spot, surf.r, surf.q

    # ── variance totale implicite w(t, k) = σ²_IV · t ────────────────────────
    def w(t: float, k: float) -> float:
        k_clipped = float(np.clip(k, surf.strikes[0], surf.strikes[-1]))
        return surf.vol(t, k_clipped) ** 2 * t

    # ── forward et log-moneyness au point (T, K) ──────────────────────────────
    F   = S * math.exp((r - q) * T)
    y   = math.log(K / F)
    w0  = w(T, K)

    if w0 <= 1e-10:
        return surf.atm_vol(T)

    # ── ∂w/∂T à y fixe : K s'ajuste pour conserver ln(K/F(T)) = y ───────────
    half_dT = dT / 2.0
    T_up = T + half_dT
    T_dn = max(T - half_dT, 1e-4)
    eff_dT = T_up - T_dn

    K_up = S * math.exp((r - q) * T_up) * math.exp(y)   # même y à T_up
    K_dn = S * math.exp((r - q) * T_dn) * math.exp(y)   # même y à T_dn

    dw_dT = (w(T_up, K_up) - w(T_dn, K_dn)) / eff_dT

    # ── ∂w/∂K et ∂²w/∂K² à T fixe (différences centrées) ────────────────────
    dK     = K * dK_frac
    w_up_K = w(T, K + dK)
    w_dn_K = w(T, K - dK)

    dw_dK   = (w_up_K - w_dn_K) / (2.0 * dK)
    d2w_dK2 = (w_up_K - 2.0 * w0 + w_dn_K) / dK ** 2

    # ── passage en log-moneyness : d/dy = K·d/dK ─────────────────────────────
    dw_dy   = K * dw_dK
    d2w_dy2 = K ** 2 * d2w_dK2 + K * dw_dK

    # ── facteur g de Gatheral ─────────────────────────────────────────────────
    g = (1.0
         - (y / w0) * dw_dy
         + 0.25 * dw_dy ** 2 * (-0.25 - 1.0 / w0 + y ** 2 / w0 ** 2)
         + 0.5 * d2w_dy2)

    if g <= 1e-10 or dw_dT <= 0:
        return surf.atm_vol(T)

    local_var = dw_dT / g
    if local_var <= 0:
        return surf.atm_vol(T)

    return float(np.clip(math.sqrt(local_var), 0.01, 3.0))


def _cache_path(ticker: str) -> str:
    """Chemin du fichier cache pour un ticker donné."""
    cache_dir = os.path.join(os.path.dirname(__file__), ".vol_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{ticker.upper()}.npz")


def build_surface(ticker: str, spot: float) -> VolSurface:
    """Construit une VolSurface en UNE LIGNE."""
    t        = ticker.upper()
    euronext = _TICKER_MAP.get(t, t)
    exchange = "DAMS" if t in _AMSTERDAM_TICKERS else "DPAR"
    r, q     = fetch_market_params(ticker)
    cache    = _cache_path(t)

    # ── Tentative live ────────────────────────────────────────────────────────
    surf = None
    try:
        df   = fetch_option_chain(euronext, exchange)
        surf = VolSurface.from_option_chain(df, date.today(), spot, r, q)
        # Sauvegarde le cache si la surface est suffisamment riche
        if len(surf.tenors) >= 3 and len(surf.strikes) >= 5:
            surf.save(cache)
            print(f"[VolSurface] Cache mis à jour → {cache}")
    except Exception as e:
        print(f"[VolSurface] Données live indisponibles : {e}")

    # ── Fallback cache ────────────────────────────────────────────────────────
    if surf is None or len(surf.tenors) < 2 or len(surf.strikes) < 2:
        if os.path.exists(cache):
            print(f"[VolSurface] Marché fermé ou données insuffisantes — chargement du cache EOD.")
            surf = VolSurface.load(cache)
            # Met à jour spot/r/q avec les valeurs courantes
            surf.spot = spot
            surf.r    = r
            surf.q    = q
        else:
            print(f"[VolSurface] Aucun cache disponible pour {t} — surface plate 25%.")
            surf = VolSurface.flat(date.today(), spot, r, q, sigma=0.25)

    return surf


def compute_dupire_surface(
    surf: VolSurface,
    tenors: Optional[np.ndarray] = None,
    strikes: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute local vol surface via Dupire."""
    if tenors is None:
        tenors = surf.tenors
    if strikes is None:
        strikes = surf.strikes

    records = []
    for T in tenors:
        for K in strikes:
            loc_vol = dupire_local_vol(surf, T, K)
            records.append({"T": T, "K": K, "local_vol": loc_vol})

    return pd.DataFrame(records)


def _pf(s) -> Optional[float]:
    """Parse float from string."""
    if s is None:
        return None
    s = str(s).strip().replace(",", ".")
    if s in ("-", "", "—", "N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _third_friday(year: int, month: int) -> date:
    """
    Retourne le 3ème vendredi du mois — date d'échéance standard Euronext
    pour les options sur actions (STO).
    """
    weeks = calendar.monthcalendar(year, month)   # listes [Lun..Dim]
    # index 4 = vendredi ; on filtre les semaines où vendredi ≠ 0
    fridays = [w[4] for w in weeks if w[4] != 0]
    return date(year, month, fridays[2])          # 3ème vendredi


def _parse_maturity_to_yyyy_mm(s: str) -> str:
    """Parse maturity label → YYYY-MM."""
    s = s.strip()

    if "-" in s:
        parts = s.split("-")
        if len(parts) == 3:
            try:
                day, month, year = parts
                return f"{year}-{month.zfill(2)}"
            except (ValueError, TypeError):
                pass

    parts = s.split()
    if len(parts) >= 2:
        month_str = parts[0][:3].capitalize()
        year = parts[-1]
        month_num = _MONTH_FR.get(month_str, "??")
        return f"{year}-{month_num}"

    return s
