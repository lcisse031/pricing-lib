"""
pricing_lib/pricers/base.py
─────────────────────────────────────────────────────────────────────────────
Output standardisé pour tous les pricers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


@dataclass
class PricingResult:
    """
    Résultat de pricing unifié pour tous les produits.

    Attributs obligatoires
    ----------------------
    price   : prix du produit (en % du nominal pour RC/autocalls, en EUR pour certificats)
    greeks  : dict {delta, gamma, vega, theta, rho}
    ticker  : code sous-jacent
    product : nom du produit
    model   : "analytical" | "heston_mc"

    Attributs optionnels
    --------------------
    fair_coupon        : coupon fair annualisé (RC, Phoenix, Athena)
    fair_participation : taux de participation fair (Outperformance, Sprint, etc.)
    cashflows          : DataFrame des flux actualisés (RC fréquence, autocalls)
    probabilities      : DataFrame des probabilités par scénario (autocalls)
    """

    price:    float
    greeks:   Dict[str, float]
    ticker:   str = ""
    product:  str = ""
    model:    str = "analytical"

    fair_coupon:        Optional[float]        = None
    fair_participation: Optional[float]        = None
    cashflows:          Optional[pd.DataFrame] = None
    probabilities:      Optional[pd.DataFrame] = None
    bs_params:          Optional[Dict]         = None   # paramètres BS utilisés (analytique)

    def __repr__(self) -> str:
        sep = "═" * 55
        lines = [
            sep,
            f"  {self.product:<30} [{self.ticker}]",
            f"  Modèle : {self.model}",
            sep,
            f"  Prix              : {self.price:>10.4f}",
        ]
        if self.fair_coupon is not None:
            lines.append(f"  Coupon fair       : {self.fair_coupon*100:>9.3f}% p.a.")
        if self.fair_participation is not None:
            lines.append(f"  Participation fair: {self.fair_participation*100:>9.1f}%")

        if self.bs_params is not None:
            p = self.bs_params
            lines.append(f"  {'─'*51}")
            lines.append(f"  Paramètres Black-Scholes :")
            lines.append(f"    {'S':<6} : {p.get('S', 0):>10.4f}    (spot)")
            lines.append(f"    {'K':<6} : {p.get('K', 0):>10.4f}    (strike)")
            lines.append(f"    {'T':<6} : {p.get('T', 0):>10.6f}    (années)")
            lines.append(f"    {'r':<6} : {p.get('r', 0):>10.4%}    (taux OIS)")
            lines.append(f"    {'q':<6} : {p.get('q', 0):>10.4%}    (div. continu)")
            lines.append(f"    {'σ':<6} : {p.get('sigma', 0):>10.4%}    ({p.get('vol_type', 'vol')})")
            lines.append(f"    {'F':<6} : {p.get('F', 0):>10.4f}    (forward)")
            if 'd1' in p:
                lines.append(f"    {'d1':<6} : {p.get('d1', 0):>10.6f}")
                lines.append(f"    {'d2':<6} : {p.get('d2', 0):>10.6f}")

        lines.append(f"  {'─'*51}")
        lines.append(f"  Greeks :")
        for k, v in self.greeks.items():
            lines.append(f"    {k:>6} : {v:+.6f}")

        if self.cashflows is not None:
            lines.append(f"  {'─'*51}")
            lines.append("  Cashflows :")
            lines.append(self.cashflows.to_string(index=False, float_format="{:.6f}".format))

        if self.probabilities is not None:
            lines.append(f"  {'─'*51}")
            lines.append("  Probabilités :")
            lines.append(self.probabilities.to_string(index=False, float_format="{:.4f}".format))

        lines.append(sep)
        return "\n".join(lines)
