"""
Strategies - Trading Strategy Implementations

This package contains trading strategy implementations:

- base: Base class for all strategies
- flash_crash: Flash crash volatility strategy

Usage:
    from strategies.base import BaseStrategy, StrategyConfig
    from strategies.flash_crash import FlashCrashStrategy, FlashCrashConfig

    # Or discover all registered strategies:
    from strategies import get_strategy
    strategy_cls = get_strategy("flash_crash")
"""

import importlib
import pkgutil

from strategies.base import BaseStrategy, StrategyConfig
from strategies.flash_crash import FlashCrashStrategy, FlashCrashConfig
from strategies.trade_flow import TradeFlowStrategy, TradeFlowConfig
from strategies.event_burst import EventBurstStrategy, EventBurstConfig


def _discover_strategies():
    """Import all modules in strategies/ to trigger __init_subclass__ registration."""
    import strategies
    for _importer, modname, _ispkg in pkgutil.iter_modules(strategies.__path__):
        if modname != "base":
            importlib.import_module(f"strategies.{modname}")


def get_strategy(name: str) -> type:
    """Get a strategy class by name. Raises KeyError if not found."""
    _discover_strategies()
    return BaseStrategy.REGISTRY[name]


def list_strategies() -> dict:
    """Return {name: strategy_class} for all registered strategies."""
    _discover_strategies()
    return dict(BaseStrategy.REGISTRY)


__all__ = [
    "BaseStrategy",
    "StrategyConfig",
    "FlashCrashStrategy",
    "FlashCrashConfig",
    "TradeFlowStrategy",
    "TradeFlowConfig",
    "EventBurstStrategy",
    "EventBurstConfig",
    "get_strategy",
    "list_strategies",
]
