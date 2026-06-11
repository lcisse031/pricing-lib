"""
pricing_lib/backtest
────────────────────
Moteur de backtest pour produits structurés et dérivés vanilles.

Exports publics
───────────────
    PositionSpec     — spécification d'une position
    BacktestResults  — conteneur de résultats (.positions, .portfolio,
                       .portfolio)
    BacktestEngine   — moteur bas niveau (usage avancé)

La fonction de haut niveau `Backtest()` est dans pricing_lib/backtest_api.py.
"""

from pricing_lib.backtest.specs import PositionSpec, SUPPORTED_PRODUCTS
from pricing_lib.backtest.results import BacktestResults
from pricing_lib.backtest.engine import BacktestEngine

__all__ = [
    "PositionSpec",
    "SUPPORTED_PRODUCTS",
    "BacktestResults",
    "BacktestEngine",
]
