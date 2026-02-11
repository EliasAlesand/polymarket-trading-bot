"""
Strategy Base Class - Foundation for Trading Strategies

Provides:
- Base class for all trading strategies
- Common lifecycle methods (start, stop, run)
- Integration with lib components (MarketManager, PriceTracker, PositionManager)
- Logging and status display utilities
- Strategy registry with auto-discovery via __init_subclass__

Usage:
    from strategies.base import BaseStrategy, StrategyConfig

    class MyStrategy(BaseStrategy):
        name = "my_strategy"
        description = "My custom strategy"
        config_class = MyConfig  # Must be a StrategyConfig subclass

        async def on_book_update(self, snapshot):
            # Handle orderbook updates
            pass

        async def on_tick(self, prices):
            # Called each strategy tick
            pass
"""

import argparse
import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Optional, Dict, List, ClassVar, Type

from lib.console import LogBuffer, log
from lib.market_manager import MarketManager, MarketInfo
from lib.price_tracker import PriceTracker
from lib.position_manager import PositionManager, Position
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot, LastTradePrice


@dataclass
class StrategyConfig:
    """Base strategy configuration."""

    coin: str = "ETH"
    size: float = 5.0  # USDC size per trade
    max_positions: int = 1
    take_profit: float = 0.30  # 30% gain
    stop_loss: float = 0.25  # 25% loss

    def __post_init__(self):
        # Auto-convert if user passed percentages as whole numbers (e.g. 25 instead of 0.25)
        if self.take_profit > 1:
            self.take_profit = self.take_profit / 100
        if self.stop_loss > 1:
            self.stop_loss = self.stop_loss / 100

    # Market settings
    market_check_interval: float = 30.0
    auto_switch_market: bool = True

    # Price tracking
    price_lookback_seconds: int = 10
    price_history_size: int = 100

    # Display settings
    update_interval: float = 0.1
    order_refresh_interval: float = 30.0  # Seconds between order refreshes

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add common CLI arguments. Override to add strategy-specific args."""
        parser.add_argument("--coin", type=str, default="BTC",
                            choices=["BTC", "ETH", "SOL", "XRP"],
                            help="Coin to trade (default: BTC)")
        parser.add_argument("--size", type=float, default=1.0,
                            help="Trade size in USDC (default: 1.0)")
        parser.add_argument("--take-profit", type=float, default=0.30,
                            help="Take profit percentage, e.g. 0.30 = 30%% (default: 0.30)")
        parser.add_argument("--stop-loss", type=float, default=0.25,
                            help="Stop loss percentage, e.g. 0.25 = 25%% (default: 0.25)")
        parser.add_argument("--lookback", type=int, default=10,
                            help="Price lookback window in seconds (default: 10)")
        parser.add_argument("--debug", action="store_true",
                            help="Enable debug logging")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "StrategyConfig":
        """Create config from parsed CLI args. Override for strategy-specific args."""
        return cls(
            coin=args.coin.upper(),
            size=args.size,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            price_lookback_seconds=args.lookback,
        )


class BaseStrategy(ABC):
    """
    Base class for trading strategies.

    Provides common infrastructure:
    - MarketManager for WebSocket and market discovery
    - PriceTracker for price history
    - PositionManager for positions and TP/SL
    - Logging and status display

    Subclasses set `name`, `description`, and `config_class` to auto-register:

        class MyStrategy(BaseStrategy):
            name = "my_strategy"
            description = "My custom strategy"
            config_class = MyConfig
    """

    # Strategy registry - populated by __init_subclass__
    REGISTRY: ClassVar[Dict[str, Type["BaseStrategy"]]] = {}

    # Subclasses override these
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    config_class: ClassVar[Type[StrategyConfig]] = StrategyConfig

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            BaseStrategy.REGISTRY[cls.name] = cls

    def __init__(self, bot: TradingBot, config: StrategyConfig):
        """
        Initialize base strategy.

        Args:
            bot: TradingBot instance for order execution
            config: Strategy configuration
        """
        self.bot = bot
        self.config = config

        # Core components
        self.market = MarketManager(
            coin=config.coin,
            market_check_interval=config.market_check_interval,
            auto_switch_market=config.auto_switch_market,
        )

        self.prices = PriceTracker(
            lookback_seconds=config.price_lookback_seconds,
            max_history=config.price_history_size,
        )

        self.positions = PositionManager(
            take_profit=config.take_profit,
            stop_loss=config.stop_loss,
            max_positions=config.max_positions,
        )

        # State
        self.running = False
        self._status_mode = False

        # Logging
        self._log_buffer = LogBuffer(max_size=5)

        # Open orders cache (refreshed in background)
        self._cached_orders: List[dict] = []
        self._last_order_refresh: float = 0
        self._order_refresh_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.market.is_connected

    @property
    def current_market(self) -> Optional[MarketInfo]:
        """Get current market info."""
        return self.market.current_market

    @property
    def token_ids(self) -> Dict[str, str]:
        """Get current token IDs."""
        return self.market.token_ids

    @property
    def open_orders(self) -> List[dict]:
        """Get cached open orders."""
        return self._cached_orders

    def _refresh_orders_sync(self) -> List[dict]:
        """Refresh open orders synchronously (called via to_thread)."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.bot.get_open_orders())
            finally:
                loop.close()
        except Exception as e:
            log(f"Order refresh failed: {e}", "warning")
            return []

    async def _do_order_refresh(self) -> None:
        """Background task to refresh orders without blocking."""
        try:
            orders = await asyncio.to_thread(self._refresh_orders_sync)
            self._cached_orders = orders
        except Exception as e:
            log(f"Background order refresh failed: {e}", "warning")
        finally:
            self._order_refresh_task = None

    def _maybe_refresh_orders(self) -> None:
        """Schedule order refresh if interval has passed (fire-and-forget)."""
        now = time.time()
        if now - self._last_order_refresh > self.config.order_refresh_interval:
            # Don't start new refresh if one is already running
            if self._order_refresh_task is not None and not self._order_refresh_task.done():
                return
            self._last_order_refresh = now
            # Fire and forget - doesn't block main loop
            self._order_refresh_task = asyncio.create_task(self._do_order_refresh())

    def log(self, msg: str, level: str = "info") -> None:
        """
        Log a message.

        Args:
            msg: Message to log
            level: Log level (info, success, warning, error, trade)
        """
        if self._status_mode:
            self._log_buffer.add(msg, level)
        else:
            log(msg, level)

    async def start(self) -> bool:
        """
        Start the strategy.

        Returns:
            True if started successfully
        """
        self.running = True

        # Register callbacks on market manager
        @self.market.on_book_update
        async def handle_book(snapshot: OrderbookSnapshot):  # pyright: ignore[reportUnusedFunction]
            # Record price
            for side, token_id in self.token_ids.items():
                if token_id == snapshot.asset_id:
                    self.prices.record(side, snapshot.mid_price)
                    break

            # Delegate to subclass
            await self.on_book_update(snapshot)

        @self.market.on_trade
        async def handle_trade(trade: LastTradePrice):  # pyright: ignore[reportUnusedFunction]
            await self.on_trade(trade)

        @self.market.on_market_change
        def handle_market_change(old_slug: str, new_slug: str):  # pyright: ignore[reportUnusedFunction]
            self.log(f"Market changed: {old_slug} -> {new_slug}", "warning")
            self.prices.clear()
            self.on_market_change(old_slug, new_slug)

        @self.market.on_connect
        def handle_connect():  # pyright: ignore[reportUnusedFunction]
            self.log("WebSocket connected", "success")
            self.on_connect()

        @self.market.on_disconnect
        def handle_disconnect():  # pyright: ignore[reportUnusedFunction]
            self.log("WebSocket disconnected", "warning")
            self.on_disconnect()

        # Start market manager
        if not await self.market.start():
            self.running = False
            return False

        # Wait for initial data
        if not await self.market.wait_for_data(timeout=5.0):
            self.log("Timeout waiting for market data", "warning")

        return True

    async def stop(self) -> None:
        """Stop the strategy."""
        self.running = False

        # Cancel order refresh task if running
        if self._order_refresh_task is not None:
            self._order_refresh_task.cancel()
            try:
                await self._order_refresh_task
            except asyncio.CancelledError:
                pass
            self._order_refresh_task = None

        await self.market.stop()

    async def run(self) -> None:
        """Main strategy loop."""
        try:
            if not await self.start():
                self.log("Failed to start strategy", "error")
                return

            self._status_mode = True

            while self.running:
                # Get current prices
                prices = self._get_current_prices()

                # Call tick handler
                await self.on_tick(prices)

                # Check position exits
                await self._check_exits(prices)

                # Refresh orders in background (fire-and-forget)
                self._maybe_refresh_orders()

                # Update display
                self.render_status(prices)

                await asyncio.sleep(self.config.update_interval)

        except KeyboardInterrupt:
            self.log("Strategy stopped by user")
        finally:
            await self.stop()
            self._print_summary()

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices from market manager."""
        prices = {}
        for side in ["up", "down"]:
            price = self.market.get_mid_price(side)
            if price > 0:
                prices[side] = price
        return prices

    async def _check_exits(self, prices: Dict[str, float]) -> None:
        """Check and execute exits using best bid (actual sell price), not mid."""
        bid_prices = {}
        for side in ["up", "down"]:
            best_bid = self.market.get_best_bid(side)
            if best_bid > 0:
                bid_prices[side] = best_bid
            elif side in prices:
                bid_prices[side] = prices[side]

        exits = self.positions.check_all_exits(bid_prices)

        for position, exit_type, _pnl in exits:
            await self.execute_sell(position, prices.get(position.side, 0), exit_type=exit_type)

    async def execute_buy(self, side: str, current_price: float) -> bool:
        """
        Execute market buy order and wait for fill confirmation.

        Args:
            side: "up" or "down"
            current_price: Current market price

        Returns:
            True if order filled successfully
        """
        token_id = self.token_ids.get(side)
        if not token_id:
            self.log(f"No token ID for {side}", "error")
            return False

        # Use best ask (actual sell offer) instead of arbitrary +0.02 offset
        best_ask = self.market.get_best_ask(side)
        if best_ask <= 0 or best_ask >= 1.0:
            best_ask = current_price + 0.01  # fallback
        buy_price = min(best_ask, 0.99)
        # Round size UP so notional (size * price) stays >= config.size after
        # the Order class rounds size DOWN to 2 decimals
        size = math.ceil(self.config.size / buy_price * 100) / 100

        result = await self.bot.place_order(
            token_id=token_id,
            price=buy_price,
            size=size,
            side="BUY"
        )

        if not result.success:
            self.log(f"Buy failed: {result.message}", "error")
            return False

        # Check if order was immediately matched
        status = result.data.get("status", "")
        if status == "matched":
            # Already filled — use size from response to avoid extra API call
            actual_size = float(result.data.get("size_matched", 0)) or size
        else:
            filled_size = await self.bot.wait_for_fill(result.order_id, timeout=15.0)
            if filled_size <= 0:
                self.log("Order not filled, cancelling", "warning")
                await self.bot.cancel_order(result.order_id)
                return False
            actual_size = filled_size

            if actual_size < size:
                await self.bot.cancel_order(result.order_id)

        # Track actual execution price (buy_price), not mid price
        self.log(f"BUY {side.upper()} @ {buy_price:.4f} x{actual_size:.2f}", "success")
        self.positions.open_position(
            side=side,
            token_id=token_id,
            entry_price=buy_price,
            size=actual_size,
            order_id=result.order_id,
        )
        return True

    async def execute_sell(self, position: Position, current_price: float, exit_type: str = None) -> bool:
        """
        Execute sell order to close position.

        Verifies the buy order was actually filled before attempting to sell.

        Args:
            position: Position to close
            current_price: Current price
            exit_type: "take_profit", "stop_loss", or None (manual)

        Returns:
            True if sell order filled
        """
        # Use actual on-chain balance as the sell size (ground truth)
        on_chain_balance = await self.bot.get_token_balance(position.token_id)
        if on_chain_balance <= 0:
            if position.order_id:
                await self.bot.cancel_order(position.order_id)
            self.positions.close_position(position.id, realized_pnl=0)
            return False

        sell_size = on_chain_balance
        # Use best bid (actual buy offer) instead of arbitrary -0.02 offset
        best_bid = self.market.get_best_bid(position.side)
        if best_bid <= 0:
            best_bid = current_price - 0.01  # fallback
        sell_price = max(best_bid, 0.01)

        # Fee-adjusted PnL: subtract taker fees on both buy and sell legs
        fee_bps = self.bot._market_props_cache.get(position.token_id, {}).get("fee_rate_bps", 0)
        fee_rate = fee_bps / 10000  # e.g. 100 bps -> 0.01
        buy_fee = position.entry_price * sell_size * fee_rate
        sell_fee = sell_price * sell_size * fee_rate
        pnl = (sell_price - position.entry_price) * sell_size - buy_fee - sell_fee

        result = await self.bot.place_order(
            token_id=position.token_id,
            price=sell_price,
            size=sell_size,
            side="SELL"
        )

        if not result.success:
            self.log(f"Sell failed: {result.message}", "error")
            return False

        reason = ""
        if exit_type == "take_profit":
            reason = "TP "
        elif exit_type == "stop_loss":
            reason = "SL "
        level = "success" if pnl >= 0 else "warning"
        self.log(f"{reason}SELL {position.side.upper()} @ {sell_price:.4f} PnL: ${pnl:+.2f}", level)
        self.positions.close_position(position.id, realized_pnl=pnl)
        return True

    def _print_summary(self) -> None:
        """Print session summary."""
        self._status_mode = False
        print()
        stats = self.positions.get_stats()
        self.log("Session Summary:")
        self.log(f"  Trades: {stats['trades_closed']}")
        self.log(f"  Total PnL: ${stats['total_pnl']:+.2f}")
        self.log(f"  Win rate: {stats['win_rate']:.1f}%")

    # Abstract methods to implement in subclasses

    @abstractmethod
    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """
        Handle orderbook update.

        Called when new orderbook data is received.

        Args:
            snapshot: OrderbookSnapshot from WebSocket
        """
        pass

    @abstractmethod
    async def on_tick(self, prices: Dict[str, float]) -> None:
        """
        Handle strategy tick.

        Called on each iteration of the main loop.

        Args:
            prices: Current prices {side: price}
        """
        pass

    @abstractmethod
    def render_status(self, prices: Dict[str, float]) -> None:
        """
        Render status display.

        Called on each tick to update the display.

        Args:
            prices: Current prices
        """
        pass

    # Optional hooks (override as needed)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Called when market changes."""
        pass

    async def on_trade(self, trade: LastTradePrice) -> None:
        """Called when a trade executes on the market. Override to process trade flow."""
        pass

    def on_connect(self) -> None:
        """Called when WebSocket connects."""
        pass

    def on_disconnect(self) -> None:
        """Called when WebSocket disconnects."""
        pass
