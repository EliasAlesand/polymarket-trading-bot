"""
Orderbook Imbalance Strategy - Volume Imbalance Trading for 15-Minute Markets

Monitors orderbook bid/ask volume ratio to detect sustained buying or selling
pressure, then trades in the direction of the dominant side.

Strategy Logic:
1. Calculate bid_volume / (bid_volume + ask_volume) using top N levels
2. When ratio exceeds threshold for N consecutive ticks, buy that side
3. Optionally require cross-confirmation (Up bid-heavy AND Down ask-heavy)

Usage:
    python apps/run.py orderbook_imbalance --coin ETH --threshold 0.65
    python apps/run.py orderbook_imbalance --coin BTC --depth 3 --cross-confirm
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Optional

from lib.console import Colors, format_countdown
from strategies.base import BaseStrategy, StrategyConfig
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot


@dataclass
class OrderbookImbalanceConfig(StrategyConfig):
    """Orderbook imbalance strategy configuration."""

    imbalance_threshold: float = 0.65  # Ratio above which to trigger (0.5-1.0)
    book_depth: int = 5  # Number of orderbook levels to consider
    confirmation_ticks: int = 3  # Consecutive ticks imbalance must persist
    require_cross_confirmation: bool = False  # Require both sides to agree
    cooldown_seconds: float = 30.0  # Seconds after trade before next signal

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add orderbook imbalance CLI arguments."""
        super().add_args(parser)
        parser.add_argument("--threshold", type=float, default=0.65,
                            help="Imbalance ratio to trigger (0.5-1.0, default: 0.65)")
        parser.add_argument("--depth", type=int, default=5,
                            help="Orderbook levels to consider (default: 5)")
        parser.add_argument("--confirmation", type=int, default=3,
                            help="Consecutive ticks above threshold to confirm (default: 3)")
        parser.add_argument("--cross-confirm", action="store_true",
                            help="Require both sides to agree on direction")
        parser.add_argument("--cooldown", type=float, default=30.0,
                            help="Seconds between trades (default: 30.0)")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "OrderbookImbalanceConfig":
        """Create config from parsed CLI args."""
        return cls(
            coin=args.coin.upper(),
            size=args.size,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            price_lookback_seconds=args.lookback,
            imbalance_threshold=args.threshold,
            book_depth=args.depth,
            confirmation_ticks=args.confirmation,
            require_cross_confirmation=args.cross_confirm,
            cooldown_seconds=args.cooldown,
        )


class OrderbookImbalanceStrategy(BaseStrategy):
    """
    Orderbook Imbalance Trading Strategy.

    Detects sustained bid/ask volume imbalance to predict short-term
    price direction on 15-minute Up/Down markets.
    """

    name = "orderbook_imbalance"
    description = "Trade sustained orderbook bid/ask imbalance on 15-minute markets"
    config_class = OrderbookImbalanceConfig

    def __init__(self, bot: TradingBot, config: OrderbookImbalanceConfig):
        """Initialize orderbook imbalance strategy."""
        super().__init__(bot, config)
        self.imb_config = config

        # Imbalance tracking
        self._imbalance: Dict[str, float] = {"up": 0.5, "down": 0.5}
        self._confirmation_count: Dict[str, int] = {"up": 0, "down": 0}
        self._last_trade_time: float = 0.0

    def _calculate_imbalance(self, snapshot: OrderbookSnapshot) -> float:
        """
        Calculate bid/ask imbalance ratio.

        Returns ratio in [0, 1]: >0.5 = bid-heavy, <0.5 = ask-heavy.
        """
        depth = self.imb_config.book_depth
        total_bid = sum(level.size for level in snapshot.bids[:depth])
        total_ask = sum(level.size for level in snapshot.asks[:depth])
        total = total_bid + total_ask
        if total == 0:
            return 0.5
        return total_bid / total

    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Update imbalance ratio when orderbook changes."""
        for side, token_id in self.token_ids.items():
            if token_id == snapshot.asset_id:
                self._imbalance[side] = self._calculate_imbalance(snapshot)
                break

    def _get_signal(self) -> Optional[str]:
        """Check if any side has a confirmed imbalance signal."""
        threshold = self.imb_config.imbalance_threshold
        required = self.imb_config.confirmation_ticks

        for side in ["up", "down"]:
            if self._confirmation_count[side] < required:
                continue

            if self.imb_config.require_cross_confirmation:
                other = "down" if side == "up" else "up"
                # Other side should be ask-heavy (below 1 - threshold)
                if self._imbalance.get(other, 0.5) > (1.0 - threshold):
                    continue

            return side

        return None

    async def on_tick(self, prices: Dict[str, float]) -> None:
        """Main decision loop — check for imbalance signals."""
        if not self.positions.can_open_position:
            return

        # Cooldown
        if time.time() - self._last_trade_time < self.imb_config.cooldown_seconds:
            return

        # Update confirmation counters
        threshold = self.imb_config.imbalance_threshold
        for side in ["up", "down"]:
            if self._imbalance.get(side, 0.5) >= threshold:
                self._confirmation_count[side] += 1
            else:
                self._confirmation_count[side] = 0

        # Check for confirmed signal
        buy_side = self._get_signal()
        if buy_side:
            current_price = prices.get(buy_side, 0)
            if current_price > 0:
                ratio = self._imbalance[buy_side]
                ticks = self._confirmation_count[buy_side]
                self.log(
                    f"IMBALANCE: {buy_side.upper()} ratio={ratio:.2f} "
                    f"(confirmed {ticks} ticks)",
                    "trade"
                )
                success = await self.execute_buy(buy_side, current_price)
                if success:
                    self._last_trade_time = time.time()
                    self._confirmation_count[buy_side] = 0

    def _format_imbalance_bar(self, ratio: float, width: int = 20) -> str:
        """Format imbalance ratio as a visual bar."""
        filled = int(ratio * width)
        bar = "=" * filled + " " * (width - filled)

        threshold = self.imb_config.imbalance_threshold
        if ratio >= threshold:
            color = Colors.GREEN
        elif ratio <= (1.0 - threshold):
            color = Colors.RED
        else:
            color = ""

        return f"{color}[{bar}]{Colors.RESET}"

    def render_status(self, prices: Dict[str, float]) -> None:
        """Render TUI status display."""
        lines = []

        # Header
        ws_status = f"{Colors.GREEN}WS{Colors.RESET}" if self.is_connected else f"{Colors.RED}REST{Colors.RESET}"
        countdown = self._get_countdown_str()
        stats = self.positions.get_stats()
        confirm_str = "+CROSS" if self.imb_config.require_cross_confirmation else ""

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        lines.append(
            f"{Colors.CYAN}[{self.config.coin}]{Colors.RESET} [{ws_status}] "
            f"[IMB>{self.imb_config.imbalance_threshold:.0%} x{self.imb_config.confirmation_ticks}{confirm_str}] "
            f"Ends: {countdown} | Trades: {stats['trades_closed']} "
            f"({stats['winning_trades']}W/{stats['losing_trades']}L) | "
            f"WR: {stats['win_rate']:.0f}% | PnL: ${stats['total_pnl']:+.2f}"
        )
        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # Orderbook display
        up_ob = self.market.get_orderbook("up")
        down_ob = self.market.get_orderbook("down")

        lines.append(f"{Colors.GREEN}{'UP':^39}{Colors.RESET}|{Colors.RED}{'DOWN':^39}{Colors.RESET}")
        lines.append(f"{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}|{'Bid':>9} {'Size':>9} | {'Ask':>9} {'Size':>9}")
        lines.append("-" * 80)

        up_bids = up_ob.bids[:5] if up_ob else []
        up_asks = up_ob.asks[:5] if up_ob else []
        down_bids = down_ob.bids[:5] if down_ob else []
        down_asks = down_ob.asks[:5] if down_ob else []

        for i in range(5):
            up_bid = f"{up_bids[i].price:>9.4f} {up_bids[i].size:>9.1f}" if i < len(up_bids) else f"{'--':>9} {'--':>9}"
            up_ask = f"{up_asks[i].price:>9.4f} {up_asks[i].size:>9.1f}" if i < len(up_asks) else f"{'--':>9} {'--':>9}"
            down_bid = f"{down_bids[i].price:>9.4f} {down_bids[i].size:>9.1f}" if i < len(down_bids) else f"{'--':>9} {'--':>9}"
            down_ask = f"{down_asks[i].price:>9.4f} {down_asks[i].size:>9.1f}" if i < len(down_asks) else f"{'--':>9} {'--':>9}"
            lines.append(f"{up_bid} | {up_ask}|{down_bid} | {down_ask}")

        lines.append("-" * 80)

        # Mid price / spread
        up_mid = up_ob.mid_price if up_ob else prices.get("up", 0)
        down_mid = down_ob.mid_price if down_ob else prices.get("down", 0)
        up_spread = self.market.get_spread("up")
        down_spread = self.market.get_spread("down")

        lines.append(
            f"Mid: {Colors.GREEN}{up_mid:.4f}{Colors.RESET}  Spread: {up_spread:.4f}           |"
            f"Mid: {Colors.RED}{down_mid:.4f}{Colors.RESET}  Spread: {down_spread:.4f}"
        )

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # Imbalance gauges
        lines.append(f"{Colors.BOLD}Imbalance (depth={self.imb_config.book_depth}):{Colors.RESET}")
        threshold = self.imb_config.imbalance_threshold
        required = self.imb_config.confirmation_ticks

        for side in ["up", "down"]:
            ratio = self._imbalance[side]
            count = self._confirmation_count[side]
            bar = self._format_imbalance_bar(ratio)
            confirmed = f"{Colors.GREEN}SIGNAL!{Colors.RESET}" if count >= required else f"{count}/{required}"
            label = f"{Colors.GREEN}UP  {Colors.RESET}" if side == "up" else f"{Colors.RED}DOWN{Colors.RESET}"
            lines.append(f"  {label} {bar} {ratio:.2f}  {confirmed}")

        # Cooldown status
        elapsed = time.time() - self._last_trade_time
        cooldown = self.imb_config.cooldown_seconds
        if self._last_trade_time > 0 and elapsed < cooldown:
            remaining_cd = cooldown - elapsed
            lines.append(f"  Cooldown: {remaining_cd:.0f}s remaining")

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # Open Orders
        lines.append(f"{Colors.BOLD}Open Orders:{Colors.RESET}")
        if self.open_orders:
            for order in self.open_orders[:5]:
                side = order.get("side", "?")
                price = float(order.get("price", 0))
                size = float(order.get("original_size", order.get("size", 0)))
                filled = float(order.get("size_matched", 0))
                order_id = order.get("id", "")[:8]
                token = order.get("asset_id", "")
                token_side = "UP" if token == self.token_ids.get("up") else "DOWN" if token == self.token_ids.get("down") else "?"
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
        """Handle market change — reset imbalance state."""
        self.prices.clear()
        self._imbalance = {"up": 0.5, "down": 0.5}
        self._confirmation_count = {"up": 0, "down": 0}
