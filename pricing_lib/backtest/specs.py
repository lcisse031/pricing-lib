"""
pricing_lib/backtest/specs.py
-----------------------------------------------------------------------------
Specifications de position pour le moteur de backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


SUPPORTED_PRODUCTS = {
    "call", "put", "warrant",
    "rc", "reverse_convertible",
    "bonus", "bonus_certificate",
    "reverse_bonus", "reverse_bonus_certificate",
    "twin_win", "twin_win_certificate",
    "airbag", "airbag_certificate",
    "discount", "discount_certificate",
    "sprint", "sprint_certificate",
    "outperformance", "outperformance_certificate",
    "garantie", "garantie_certificate",
    "phoenix",
    "athena",
}


@dataclass
class PositionSpec:
    """
    Specification complete d'une position pour le backtest.

    Parametres
    ----------
    product_id   : identifiant unique dans le portefeuille
    ticker       : sous-jacent (ex: 'GLE', 'OR', 'MC', 'TTE')
    product_type : type de produit (voir SUPPORTED_PRODUCTS)
    spot         : prix spot initial S0 a start_date
    start_date   : date de depart  'DD/MM/YYYY'
    maturity     : maturite contractuelle '3M', '12M', '60M'...
    nominal      : montant investi dans le produit en euros (defaut 100)
    mode         : 'AL' (analytique) | 'MC' (Monte Carlo)
    n_paths      : trajectoires MC (ignore en mode AL)
    seed         : graine aleatoire MC pour reproductibilite
    quantity     : nb d'unites achetees du produit.
                   Si None, l'engine calcule automatiquement :
                   quantity = nominal / mtm0
                   ou mtm0 est le prix unitaire du produit a start_date.
                   Peut etre fourni explicitement pour override.
    params       : parametres produit-specifiques

    Params selon le type
    --------------------
    call / put           : strike (float)
    warrant              : strike (float), flag='c'|'p', ratio (float)
    rc                   : freq='Q'|'M'|'S'|'A', cap=spot, coupon (MC only)
    bonus / reverse_bonus: X (niveau bonus), B (barriere)
    twin_win             : X (strike), B (barriere basse)
    airbag               : X (strike), B (barriere)
    discount             : X (cap)
    sprint               : X (strike), cap_level (cap)
    outperformance       : X (strike)
    garantie             : X (strike)
    phoenix              : barrier_coupon, barrier_recall, capital_barrier,
                           freq_months, coupon (MC only), kg, autocall_start
    athena               : barrier_recall, capital_barrier,
                           freq_months, coupon (MC only), kg
    """

    product_id:   str
    ticker:       str
    product_type: str
    spot:         float
    start_date:   str
    maturity:     str
    nominal:      float          = 100.0
    mode:         str            = "AL"
    n_paths:      int            = 10_000
    seed:         Optional[int]  = None
    quantity:     Optional[float] = None
    params:       Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.product_type = self.product_type.lower().strip()
        self.mode         = self.mode.upper().strip()
        self.ticker       = self.ticker.upper().strip()

        if self.product_type not in SUPPORTED_PRODUCTS:
            raise ValueError(
                f"product_type '{self.product_type}' inconnu. "
                f"Valeurs acceptees : {sorted(SUPPORTED_PRODUCTS)}"
            )
        if self.mode not in ("AL", "MC"):
            raise ValueError(f"mode='{self.mode}' invalide. Utiliser 'AL' ou 'MC'.")
        if self.nominal <= 0:
            raise ValueError(f"nominal doit etre > 0, recu {self.nominal}.")
        if self.spot <= 0:
            raise ValueError(f"spot doit etre > 0, recu {self.spot}.")
        if self.quantity is not None and self.quantity <= 0:
            raise ValueError(f"quantity doit etre > 0, recu {self.quantity}.")

    @property
    def is_autocall(self) -> bool:
        return self.product_type in ("phoenix", "athena")

    @property
    def is_vanilla(self) -> bool:
        return self.product_type in ("call", "put", "warrant")

    def __repr__(self) -> str:
        qty_str = f"{self.quantity:.4f}" if self.quantity is not None else "auto"
        return (
            f"PositionSpec(id={self.product_id!r}, "
            f"type={self.product_type!r}, "
            f"ticker={self.ticker!r}, "
            f"spot={self.spot}, "
            f"nominal={self.nominal}, "
            f"quantity={qty_str}, "
            f"maturity={self.maturity!r}, "
            f"mode={self.mode!r})"
        )
