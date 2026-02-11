"""
Trade Flow Strategy - Aggressor Flow Trading for 15-Minute Markets

Monitors executed trades on BOTH tokens to detect informed buying or
selling pressure via quote-based classification, combined directionally.

Key insight: Someone willing to cross the spread likely has information.
Both tokens are tracked and combined: buying UP or selling DOWN = bullish,
buying DOWN or selling UP = bearish.

Classification (quote-based):
    trade_price >= best_ask → taker bought (lifted the offer)
    trade_price <= best_bid → taker sold (hit the bid)
    in between → ambiguous, ignored

Directional combination:
    UP taker buy  + DOWN taker sell = bullish volume
    UP taker sell + DOWN taker buy  = bearish volume

Signal:
    flow_ratio = bullish_volume / (bullish_volume + bearish_volume)
    > threshold → buy UP   (informed flow pushing probability up)
    < 1-threshold → buy DOWN (informed flow pushing probability down)

Price guard: Never buy a token priced above 0.80 or below 0.20.
Near extremes, traders switch from information to valuation trading.

Usage:
    python apps/run.py trade_flow --coin ETH --threshold 0.70
    python apps/run.py trade_flow --coin BTC --window 30 --confirmation 5
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

from lib.console import Colors, format_countdown
from strategies.base import BaseStrategy, StrategyConfig
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot, LastTradePrice


@dataclass
class TradeFlowConfig(StrategyConfig):
    """Trade flow strategy configuration."""

    flow_threshold: float = 0.70  # Ratio above which to trigger (0.5-1.0)
    window_seconds: float = 60.0  # Rolling window for trade aggregation
    confirmation_ticks: int = 3  # Consecutive ticks above threshold
    max_price: float = 0.80  # Never buy token above this price
    min_price: float = 0.20  # Never buy token below this price
    cooldown_seconds: float = 30.0  # Seconds between trades

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add trade flow CLI arguments."""
        super().add_args(parser)
        parser.add_argument("--threshold", type=float, default=0.70,
                            help="Flow ratio to trigger (0.5-1.0, default: 0.70)")
        parser.add_argument("--window", type=float, default=60.0,
                            help="Rolling window in seconds (default: 60.0)")
        parser.add_argument("--confirmation", type=int, default=3,
                            help="Consecutive ticks above threshold (default: 3)")
        parser.add_argument("--max-price", type=float, default=0.80,
                            help="Never buy token above this price (default: 0.80)")
        parser.add_argument("--min-price", type=float, default=0.20,
                            help="Never buy token below this price (default: 0.20)")
        parser.add_argument("--cooldown", type=float, default=30.0,
                            help="Seconds between trades (default: 30.0)")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TradeFlowConfig":
        """Create config from parsed CLI args."""
        return cls(
            coin=args.coin.upper(),
            size=args.size,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            price_lookback_seconds=args.lookback,
            flow_threshold=args.threshold,
            window_seconds=args.window,
            confirmation_ticks=args.confirmation,
            max_price=args.max_price,
            min_price=args.min_price,
            cooldown_seconds=args.cooldown,
        )


class TradeFlowStrategy(BaseStrategy):
    """
    Trade Flow (Aggressor) Trading Strategy.

    Tracks executed trades on both UP and DOWN tokens, combines them
    directionally. Uses quote-based classification (best bid/ask) to
    determine aggressor direction.
    """

    name = "trade_flow"
    description = "Trade informed order flow (taker aggressor) on 15-minute markets"
    config_class = TradeFlowConfig

    def __init__(self, bot: TradingBot, config: TradeFlowConfig):
        """Initialize trade flow strategy."""
        super().__init__(bot, config)
        self.tf_config = config

        # Rolling trade window: deque of (timestamp, size, is_bullish)
        # Tracks both tokens, combined directionally:
        #   UP taker buy  → bullish    DOWN taker sell → bullish
        #   UP taker sell → bearish    DOWN taker buy  → bearish
        self._trades: deque = deque()

        # Computed each tick
        self._flow_ratio: float = 0.5  # bullish_vol / total
        self._bull_vol: float = 0.0
        self._bear_vol: float = 0.0
        self._total_vol: float = 0.0
        self._trade_count: int = 0
        self._ignored_count: int = 0  # trades in the spread (ambiguous)

        # Signal: "up" or "down" confirmation counter
        self._confirmation_count: Dict[str, int] = {"up": 0, "down": 0}
        self._last_trade_time: float = 0.0

        # Warmup: wait for rolling window to fill before accepting signals
        self._start_time: float = time.time()

    async def on_trade(self, trade: LastTradePrice) -> None:
        """Classify trade using quote-based method, both tokens combined.

        Directional combination (Polymarket has separate YES/NO orderbooks):
            UP taker buy  (lifted ask) → bullish
            UP taker sell (hit bid)    → bearish
            DOWN taker buy  (lifted ask) → bearish  (buying NO = bearish)
            DOWN taker sell (hit bid)    → bullish  (selling NO = bullish)

        Classification:
            price >= best_ask → taker bought (lifted offer)
            price <= best_bid → taker sold (hit bid)
            in between → ambiguous, skip
        """
        # Determine which token this trade is for
        side = None
        for s, token_id in self.token_ids.items():
            if token_id == trade.asset_id:
                side = s
                break
        if not side:
            return

        best_bid = self.market.get_best_bid(side)
        best_ask = self.market.get_best_ask(side)

        if best_bid <= 0 or best_ask >= 1.0 or best_bid >= best_ask:
            return  # no valid book

        if trade.price >= best_ask:
            is_taker_buy = True
        elif trade.price <= best_bid:
            is_taker_buy = False
        else:
            self._ignored_count += 1
            return  # in the spread — ambiguous

        # Convert to directional: is this trade bullish (pushing price UP)?
        if side == "up":
            is_bullish = is_taker_buy      # buying UP = bullish
        else:
            is_bullish = not is_taker_buy   # buying DOWN = bearish, selling DOWN = bullish

        self._trades.append((time.time(), trade.size, is_bullish))

    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Not used for signals — required by base class."""
        pass

    def _compute_flow(self) -> None:
        """Evict old trades and compute flow ratio."""
        cutoff = time.time() - self.tf_config.window_seconds

        # Evict expired trades
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()

        # Compute volumes
        bull_vol = sum(size for _, size, is_bull in self._trades if is_bull)
        bear_vol = sum(size for _, size, is_bull in self._trades if not is_bull)
        total = bull_vol + bear_vol

        self._bull_vol = bull_vol
        self._bear_vol = bear_vol
        self._total_vol = total
        self._trade_count = len(self._trades)
        self._flow_ratio = bull_vol / total if total > 0 else 0.5

    def _get_signal(self) -> Optional[str]:
        """Check if flow is confirmed in either direction.

        flow_ratio > threshold → buy UP (informed buying YES)
        flow_ratio < 1-threshold → buy DOWN (informed selling YES = buying NO)
        """
        threshold = self.tf_config.flow_threshold
        required = self.tf_config.confirmation_ticks

        if self._confirmation_count["up"] >= required:
            return "up"
        if self._confirmation_count["down"] >= required:
            return "down"

        return None

    @property
    def _warmup_remaining(self) -> float:
        """Seconds remaining in warmup period."""
        elapsed = time.time() - self._start_time
        remaining = self.tf_config.window_seconds - elapsed
        return max(0.0, remaining)

    @property
    def _is_warming_up(self) -> bool:
        """True if still in warmup period (waiting for window to fill)."""
        return self._warmup_remaining > 0

    async def on_tick(self, prices: Dict[str, float]) -> None:
        """Main decision loop — check for trade flow signals."""
        # Always compute flow (for display) even if we can't trade
        self._compute_flow()

        # Wait for rolling window to fill before accepting signals
        if self._is_warming_up:
            return

        if not self.positions.can_open_position:
            return

        # Cooldown
        if time.time() - self._last_trade_time < self.tf_config.cooldown_seconds:
            return

        # Update confirmation counters
        threshold = self.tf_config.flow_threshold

        # High ratio → informed buying on UP token → buy UP
        if self._flow_ratio >= threshold:
            self._confirmation_count["up"] += 1
        else:
            self._confirmation_count["up"] = 0

        # Low ratio → informed selling on UP token → buy DOWN
        if self._flow_ratio <= (1.0 - threshold):
            self._confirmation_count["down"] += 1
        else:
            self._confirmation_count["down"] = 0

        # Check for confirmed signal
        buy_side = self._get_signal()
        if not buy_side:
            return

        current_price = prices.get(buy_side, 0)
        if current_price <= 0:
            return

        # Price guard: don't buy near extremes
        if current_price > self.tf_config.max_price:
            self.log(
                f"BLOCKED: {buy_side.upper()} @ {current_price:.4f} > "
                f"{self.tf_config.max_price}",
                "warning"
            )
            self._confirmation_count[buy_side] = 0
            return

        if current_price < self.tf_config.min_price:
            self.log(
                f"BLOCKED: {buy_side.upper()} @ {current_price:.4f} < "
                f"{self.tf_config.min_price}",
                "warning"
            )
            self._confirmation_count[buy_side] = 0
            return

        ticks = self._confirmation_count[buy_side]
        self.log(
            f"FLOW: {buy_side.upper()} ratio={self._flow_ratio:.0%} "
            f"bull={self._bull_vol:.0f} bear={self._bear_vol:.0f} "
            f"({ticks} ticks)",
            "trade"
        )
        success = await self.execute_buy(buy_side, current_price)
        if success:
            self._last_trade_time = time.time()
            self._confirmation_count[buy_side] = 0

    def _format_flow_bar(self, ratio: float, width: int = 30) -> str:
        """Format flow ratio as a visual bar.

        Left = sell pressure (bearish), Right = buy pressure (bullish).
        Center (0.5) = neutral.
        """
        pos = int(ratio * width)
        threshold = self.tf_config.flow_threshold
        low_thresh = 1.0 - threshold

        bar_chars = []
        for i in range(width):
            bar_chars.append("=")
        bar = "".join(bar_chars[:pos]) + " " * (width - pos)

        if ratio >= threshold:
            color = Colors.GREEN
        elif ratio <= low_thresh:
            color = Colors.RED
        else:
            color = Colors.CYAN

        return f"{color}[{bar}]{Colors.RESET}"

    def render_status(self, prices: Dict[str, float]) -> None:
        """Render TUI status display."""
        lines = []

        # Header
        ws_status = f"{Colors.GREEN}WS{Colors.RESET}" if self.is_connected else f"{Colors.RED}REST{Colors.RESET}"
        countdown = self._get_countdown_str()
        stats = self.positions.get_stats()
        threshold = self.tf_config.flow_threshold

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        lines.append(
            f"{Colors.CYAN}[{self.config.coin}]{Colors.RESET} [{ws_status}] "
            f"[FLOW>{threshold:.0%} x{self.tf_config.confirmation_ticks} "
            f"{self.tf_config.window_seconds:.0f}s] "
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

        # Trade flow gauge — single bar for UP token
        required = self.tf_config.confirmation_ticks
        bar = self._format_flow_bar(self._flow_ratio)

        # Determine which direction is confirmed (if any)
        up_count = self._confirmation_count["up"]
        down_count = self._confirmation_count["down"]

        if up_count >= required:
            signal_str = f"{Colors.GREEN}>>> BUY UP{Colors.RESET}"
        elif down_count >= required:
            signal_str = f"{Colors.RED}<<< BUY DOWN{Colors.RESET}"
        elif up_count > 0:
            signal_str = f"UP {up_count}/{required}"
        elif down_count > 0:
            signal_str = f"DN {down_count}/{required}"
        else:
            signal_str = f"0/{required}"

        lines.append(
            f"{Colors.BOLD}Trade Flow{Colors.RESET} "
            f"({self.tf_config.window_seconds:.0f}s, "
            f"{self._trade_count} classified, {self._ignored_count} skipped):"
        )
        lines.append(
            f"  {Colors.RED}BEAR{Colors.RESET} {bar} {Colors.GREEN}BULL{Colors.RESET}  {self._flow_ratio:.0%}  "
            f"Bull:{self._bull_vol:>7.0f} Bear:{self._bear_vol:>7.0f}  {signal_str}"
        )

        # Price guard status
        for side in ["up", "down"]:
            price = prices.get(side, 0)
            if price > self.tf_config.max_price:
                label = "UP" if side == "up" else "DN"
                lines.append(
                    f"  {Colors.RED}! {label} @ {price:.4f} > "
                    f"{self.tf_config.max_price} (blocked){Colors.RESET}"
                )
            elif 0 < price < self.tf_config.min_price:
                label = "UP" if side == "up" else "DN"
                lines.append(
                    f"  {Colors.RED}! {label} @ {price:.4f} < "
                    f"{self.tf_config.min_price} (blocked){Colors.RESET}"
                )

        # Warmup status
        if self._is_warming_up:
            remaining = self._warmup_remaining
            lines.append(
                f"  {Colors.CYAN}Warming up... {remaining:.0f}s remaining "
                f"(collecting trade data){Colors.RESET}"
            )

        # Cooldown status
        elapsed = time.time() - self._last_trade_time
        cooldown = self.tf_config.cooldown_seconds
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
        """Handle market change — reset trade flow state."""
        self.prices.clear()
        self._trades = deque()
        self._flow_ratio = 0.5
        self._bull_vol = 0.0
        self._bear_vol = 0.0
        self._total_vol = 0.0
        self._trade_count = 0
        self._ignored_count = 0
        self._confirmation_count = {"up": 0, "down": 0}
        self._start_time = time.time()  # restart warmup for new market
