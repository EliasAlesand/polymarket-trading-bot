"""
Flash Crash Strategy - Volatility Trading for 15-Minute Markets

This strategy monitors 15-minute Up/Down markets for sudden probability drops
and executes trades when probability crashes by a threshold within a lookback window.

Strategy Logic:
1. Auto-discover current 15-minute market for selected coin
2. Monitor orderbook prices in real-time via WebSocket
3. When either "Up" or "Down" probability drops by threshold:
   - Market buy the crashed side
4. Exit conditions:
   - Take profit: configurable (default +10 cents)
   - Stop loss: configurable (default -5 cents)

Usage:
    from strategies.flash_crash import FlashCrashStrategy, FlashCrashConfig

    strategy = FlashCrashStrategy(bot, config)
    await strategy.run()
"""

import argparse
from dataclasses import dataclass
from typing import Dict

from lib.console import Colors, format_countdown
from strategies.base import BaseStrategy, StrategyConfig
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


@dataclass
class FlashCrashConfig(StrategyConfig):
    """Flash crash strategy configuration."""

    drop_threshold: float = 0.30  # Absolute probability drop
    exit_before_expiry: int = 120  # Seconds before expiry to force-exit positions
    reverse: bool = False  # Momentum mode: buy opposite side of crash

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add flash crash specific CLI arguments."""
        super().add_args(parser)
        parser.add_argument("--drop", type=float, default=0.30,
                            help="Drop threshold as absolute probability change (default: 0.30)")
        parser.add_argument("--exit-before", type=int, default=120,
                            help="Exit positions N seconds before market expiry (default: 120)")
        parser.add_argument("--reverse", action="store_true",
                            help="Momentum mode: buy opposite side of crash")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "FlashCrashConfig":
        """Create config from parsed CLI args."""
        return cls(
            coin=args.coin.upper(),
            slug=getattr(args, "slug", ""),
            size=args.size,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            min_hold_seconds=args.min_hold,
            price_lookback_seconds=args.lookback,
            drop_threshold=args.drop,
            exit_before_expiry=args.exit_before,
            reverse=args.reverse,
        )


class FlashCrashStrategy(BaseStrategy):
    """
    Flash Crash Trading Strategy.

    Monitors 15-minute markets for sudden price drops and trades
    the volatility with defined take-profit and stop-loss levels.
    """

    name = "flash_crash"
    description = "Trade volatility on 15-minute markets by detecting sudden price drops"
    config_class = FlashCrashConfig

    def __init__(self, bot: TradingBot, config: FlashCrashConfig):
        """Initialize flash crash strategy."""
        super().__init__(bot, config)
        self.flash_config = config

        # Update price tracker with our threshold
        self.prices.drop_threshold = config.drop_threshold

    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Handle orderbook update - check for flash crashes."""
        pass  # Price recording is done in base class

    async def on_tick(self, prices: Dict[str, float]) -> None:
        """Check for flash crash on each tick."""
        # Check time-based exit before anything else
        await self._check_expiry_exit(prices)

        if not self.positions.can_open_position:
            return

        # Don't open new positions too close to expiry
        market = self.current_market
        if market:
            mins, secs = market.get_countdown()
            remaining = mins * 60 + secs if mins >= 0 else 999
            if remaining <= self.flash_config.exit_before_expiry:
                return

        # Detect flash crash
        event = self.prices.detect_flash_crash()
        if event:
            # In reverse (momentum) mode, buy the opposite side
            if self.flash_config.reverse:
                buy_side = self.negative_side if event.side == self.positive_side else self.positive_side
                self.log(
                    f"MOMENTUM: {event.side.upper()} crashed "
                    f"{event.drop:.2f} ({event.old_price:.2f} -> {event.new_price:.2f}) "
                    f"-> BUY {buy_side.upper()}",
                    "trade"
                )
            else:
                buy_side = event.side
                self.log(
                    f"FLASH CRASH: {event.side.upper()} "
                    f"drop {event.drop:.2f} ({event.old_price:.2f} -> {event.new_price:.2f})",
                    "trade"
                )

            current_price = prices.get(buy_side, 0)
            if current_price > 0:
                await self.execute_buy(buy_side, current_price)

    async def _check_expiry_exit(self, prices: Dict[str, float]) -> None:
        """Force-exit all positions if market is about to expire."""
        market = self.current_market
        if not market:
            return

        mins, secs = market.get_countdown()
        if mins < 0:
            return

        remaining = mins * 60 + secs
        if remaining > self.flash_config.exit_before_expiry:
            return

        # Close all open positions
        for pos in self.positions.get_all_positions():
            current = prices.get(pos.side, 0)
            if current <= 0:
                continue
            pnl = pos.get_pnl(current)
            self.log(
                f"EXPIRY EXIT: {pos.side.upper()} "
                f"({remaining}s left) PnL: ${pnl:+.2f}",
                "warning"
            )
            await self.execute_sell(pos, current)

    def render_status(self, prices: Dict[str, float]) -> None:
        """Render TUI status display."""
        lines = []

        # Header
        ws_status = f"{Colors.GREEN}WS{Colors.RESET}" if self.is_connected else f"{Colors.RED}REST{Colors.RESET}"
        countdown = self._get_countdown_str()
        stats = self.positions.get_stats()
        total_pnl = self.positions.get_total_pnl(prices)

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        mode_str = f"{Colors.YELLOW}MOMENTUM{Colors.RESET}" if self.flash_config.reverse else "REVERT"
        lines.append(
            f"{Colors.CYAN}[{self.market_label}]{Colors.RESET} [{ws_status}] [{mode_str}] "
            f"Ends: {countdown} | Trades: {stats['trades_closed']} "
            f"({stats['winning_trades']}W/{stats['losing_trades']}L) | "
            f"WR: {stats['win_rate']:.0f}% | PnL: ${total_pnl:+.2f}"
        )
        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # Orderbook display
        pos = self.positive_side
        neg = self.negative_side
        pos_ob = self.market.get_orderbook(pos)
        neg_ob = self.market.get_orderbook(neg)
        pos_label = pos.upper()
        neg_label = neg.upper()

        lines.append(f"{Colors.GREEN}{pos_label:^39}{Colors.RESET}|{Colors.RED}{neg_label:^39}{Colors.RESET}")
        lines.append(f"{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}|{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}")
        lines.append("-" * 80)

        # Get 5 levels
        pos_bids = pos_ob.bids[:5] if pos_ob else []
        pos_asks = pos_ob.asks[:5] if pos_ob else []
        neg_bids = neg_ob.bids[:5] if neg_ob else []
        neg_asks = neg_ob.asks[:5] if neg_ob else []

        for i in range(5):
            pb = f"{pos_bids[i].price:>9.4f} {pos_bids[i].size:>9.1f}" if i < len(pos_bids) else f"{'--':>9} {'--':>9}"
            pa = f"{pos_asks[i].price:>9.4f} {pos_asks[i].size:>9.1f}" if i < len(pos_asks) else f"{'--':>9} {'--':>9}"
            nb = f"{neg_bids[i].price:>9.4f} {neg_bids[i].size:>9.1f}" if i < len(neg_bids) else f"{'--':>9} {'--':>9}"
            na = f"{neg_asks[i].price:>9.4f} {neg_asks[i].size:>9.1f}" if i < len(neg_asks) else f"{'--':>9} {'--':>9}"
            lines.append(f"{pb} | {pa}|{nb} | {na}")

        lines.append("-" * 80)

        # Summary
        pos_mid = pos_ob.mid_price if pos_ob else prices.get(pos, 0)
        neg_mid = neg_ob.mid_price if neg_ob else prices.get(neg, 0)
        pos_spread = self.market.get_spread(pos)
        neg_spread = self.market.get_spread(neg)

        lines.append(
            f"Mid: {Colors.GREEN}{pos_mid:.4f}{Colors.RESET}  Spread: {pos_spread:.4f}           |"
            f"Mid: {Colors.RED}{neg_mid:.4f}{Colors.RESET}  Spread: {neg_spread:.4f}"
        )

        # Drop delta: current price vs price from lookback_seconds ago
        lookback = self.config.price_lookback_seconds
        pos_old = self.prices.get_price_at(pos, lookback)
        neg_old = self.prices.get_price_at(neg, lookback)
        pos_delta = (pos_mid - pos_old) if pos_old and pos_mid else 0
        neg_delta = (neg_mid - neg_old) if neg_old and neg_mid else 0
        pos_delta_color = Colors.RED if pos_delta <= -self.flash_config.drop_threshold else Colors.GREEN if pos_delta > 0 else ""
        neg_delta_color = Colors.RED if neg_delta <= -self.flash_config.drop_threshold else Colors.GREEN if neg_delta > 0 else ""
        pos_delta_str = f"{pos_delta_color}{pos_delta:+.4f}{Colors.RESET}" if pos_old else "  n/a "
        neg_delta_str = f"{neg_delta_color}{neg_delta:+.4f}{Colors.RESET}" if neg_old else "  n/a "

        lines.append(
            f"Delta({lookback}s): {pos_label}={pos_delta_str}  {neg_label}={neg_delta_str} | "
            f"Threshold: {self.flash_config.drop_threshold:.2f}"
        )

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # Open Orders section
        lines.append(f"{Colors.BOLD}Open Orders:{Colors.RESET}")
        if self.open_orders:
            for order in self.open_orders[:5]:  # Show max 5 orders
                side = order.get("side", "?")
                price = float(order.get("price", 0))
                size = float(order.get("original_size", order.get("size", 0)))
                filled = float(order.get("size_matched", 0))
                order_id = order.get("id", "")[:8]
                token = order.get("asset_id", "")
                token_side = "?"
                for s, tid in self.token_ids.items():
                    if token == tid:
                        token_side = s.upper()
                        break
                color = Colors.GREEN if side == "BUY" else Colors.RED
                lines.append(f"  {color}{side:4}{Colors.RESET} {token_side:4} @ {price:.4f} Size: {size:.1f} Filled: {filled:.1f} ID: {order_id}...")
        else:
            lines.append(f"  {Colors.CYAN}(no open orders){Colors.RESET}")

        # Positions
        lines.append(f"{Colors.BOLD}Positions:{Colors.RESET}")
        all_positions = self.positions.get_all_positions()
        if all_positions:
            for pos in all_positions:
                current = prices.get(pos.side, 0)
                pnl = pos.get_pnl(current)
                pnl_pct = pos.get_pnl_percent(current)
                hold_time = pos.get_hold_time()
                color = Colors.GREEN if pnl >= 0 else Colors.RED

                lines.append(
                    f"  {Colors.BOLD}{pos.side.upper():4}{Colors.RESET} "
                    f"Entry: {pos.entry_price:.4f} | Current: {current:.4f} | "
                    f"Size: {pos.size:.2f} shares (${pos.size * pos.entry_price:.2f}) | PnL: {color}${pnl:+.2f} ({pnl_pct:+.1f}%){Colors.RESET} | "
                    f"Hold: {hold_time:.0f}s"
                )
                lines.append(
                    f"       TP: {pos.take_profit_price:.4f} (+{self.config.take_profit*100:.0f}%) | "
                    f"SL: {pos.stop_loss_price:.4f} (-{self.config.stop_loss*100:.0f}%)"
                )
        else:
            lines.append(f"  {Colors.CYAN}(no open positions){Colors.RESET}")

        # Recent logs
        if self._log_buffer.messages:
            lines.append("-" * 80)
            lines.append(f"{Colors.BOLD}Recent Events:{Colors.RESET}")
            for msg in self._log_buffer.get_messages():
                lines.append(f"  {msg}")

        # Render
        output = "\033[H\033[J" + "\n".join(lines)
        print(output, flush=True)

    def _get_countdown_str(self) -> str:
        """Get formatted countdown string."""
        market = self.current_market
        if not market:
            return "--:--"

        mins, secs = market.get_countdown()
        return format_countdown(mins, secs)

    def on_market_change(self, old_slug: str, new_slug: str) -> None:
        """Handle market change - clear price history."""
        self.prices.clear()
