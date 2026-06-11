from datetime import date, datetime
import numpy as np
import yfinance as yf
from pricing_lib.market_data.vol_surface import (
    fetch_option_chain,
    fetch_market_params,
    implied_vol,
    VolSurface,
    dupire_local_vol,
    _parse_maturity_to_yyyy_mm,
)
import pandas as pd


TICKER_MAP = {
    "GLE": "GL1",
    "AI": "AI1",
    "AIR": "AIRFR",
    "BNP": "BNPP",
    "EN": "EN1",
    "CA": "CA1",
}


def get_spot(ticker: str) -> float:
    """Fetch spot depuis Yahoo Finance."""
    data = yf.download(f"{ticker}.PA", progress=False, period="1d")
    return float(data["Close"].iloc[-1])


def build_surface(ticker: str) -> VolSurface:
    """Build surface de vol pour un ticker donné."""

    print(f"\n🚀 Building vol surface for {ticker}")
    print("=" * 70)

    # 1. Map et fetch spot
    euronext = TICKER_MAP.get(ticker.upper(), ticker.upper())
    spot = get_spot(ticker)
    print(f"✅ Spot: {spot:.2f}€")

    # 2. Fetch params
    r, q = fetch_market_params(ticker)
    today = date.today()

    # 3. Fetch options
    print(f"📥 Fetching options for {euronext}...")
    df = fetch_option_chain(euronext, "DPAR")
    print(f"✅ {len(df)} options downloaded")

    # 4. Calcule VI pour CHAQUE option - SANS FILTRES
    print(f"🔢 Computing implied vols...")
    records = []

    for _, row in df.iterrows():
        expiry_str = str(row["expiry_date"])
        K = float(row["strike"])
        opt_type = str(row["type"]).upper()

        # Parse maturité
        try:
            expiry_dt = datetime.strptime(expiry_str + "-18", "%Y-%m-%d").date()
        except ValueError:
            continue

        T = (expiry_dt - today).days / 365.25
        if T <= 0:
            continue

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
        raise ValueError(f"❌ Aucune VI calculée pour {ticker}")

    print(f"✅ {len(records)} options avec VI valide")

    # 5. Build surface
    ivdf = pd.DataFrame(records)
    tenors = sorted(ivdf["T"].unique())
    strikes = sorted(ivdf["K"].unique())

    print(f"   {len(tenors)} maturités: {[f'{t:.2f}y' for t in tenors]}")
    print(f"   {len(strikes)} strikes: {[f'{k:.0f}' for k in strikes]}")

    # 6. Construit VolSurface
    grid = ivdf.groupby(["T", "K"])["iv"].mean().unstack("K")
    grid = grid.reindex(index=tenors).ffill()
    grid = grid.T.reindex(strikes).ffill().bfill().T
    grid = grid.ffill().bfill()

    vols_array = grid.to_numpy(dtype=float)

    surf = VolSurface(
        valuation_date=today,
        spot=spot,
        r=r,
        q=q,
        tenors=np.array(tenors),
        strikes=np.array(strikes),
        vols=vols_array,
    )

    print(f"\n✅ {surf}\n")
    return surf


def compute_dupire_surface(surf: VolSurface, tenors=None, strikes=None):
    """Compute local vol surface via Dupire formula."""

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


if __name__ == "__main__":
    # Build surface
    surf = build_surface("GLE")

    # Compute Dupire
    print("📊 Computing Dupire local vol surface...")
    dupire_df = compute_dupire_surface(surf)
    print(dupire_df.head(10))

    # Example queries
    print("\n📈 Example ATM vols:")
    for T in [0.25, 0.5, 1.0]:
        atm_vol = surf.atm_vol(T)
        print(f"   T={T:.2f}y: σ_ATM = {atm_vol:.2%}")

    print("\n✅ Done!")
