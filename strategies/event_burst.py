"""
Event Burst Strategy — 4-Filter Microstructure Signal

Detects sudden repricing events via 4 independent filters that must ALL
fire simultaneously. This catches moments when "something just happened"
before the market fully adjusts.

The 4 filters:
    1. Volume Burst:     trading volume spikes > 3x baseline
    2. Price Impact:     price moved > 1.5 points on small volume (informed)
    3. Liquidity Retreat: 40%+ of resting depth disappeared
    4. Spread Expansion:  spread widened > 1.8x average

Entry: All 4 must fire. Direction follows the price move.
Exit:  +4% profit OR 90 seconds OR liquidity recovers.

Usage:
    python apps/run.py event_burst --coin ETH --size 5
    python apps/run.py event_burst --coin BTC --burst-ratio 2.5 --impact 0.02
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from lib.console import Colors, format_countdown
from strategies.base import BaseStrategy, StrategyConfig
from src.bot import TradingBot
from src.websocket_client import OrderbookSnapshot, LastTradePrice


@dataclass
class EventBurstConfig(StrategyConfig):
    """Event burst strategy configuration."""

    # Filter thresholds
    burst_ratio: float = 3.0        # Volume must be Nx baseline
    burst_window: float = 10.0      # Recent volume window (seconds)
    baseline_window: float = 60.0   # Baseline volume window (seconds)
    min_price_impact: float = 0.015 # Min absolute price change (probability points)
    max_volume_ratio: float = 2.0   # Max volume for "informed" (vs crowd)
    liquidity_drop: float = 0.6     # Depth must drop below this ratio
    spread_expansion: float = 1.8   # Spread must be Nx average

    # Exit rules
    profit_target: float = 0.15     # +15% price move
    max_hold_seconds: float = 90.0  # Time-based exit
    liquidity_recovery: float = 0.8 # Depth recovery ratio to exit

    # Safety
    max_price: float = 0.85         # Never buy above
    min_price: float = 0.15         # Never buy below
    cooldown_seconds: float = 30.0  # Between trades
    warmup_seconds: float = 60.0    # Wait for data before trading

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add event burst CLI arguments."""
        super().add_args(parser)
        parser.add_argument("--burst-ratio", type=float, default=3.0,
                            help="Volume burst multiplier threshold (default: 3.0)")
        parser.add_argument("--burst-window", type=float, default=10.0,
                            help="Recent volume window in seconds (default: 10.0)")
        parser.add_argument("--baseline-window", type=float, default=60.0,
                            help="Baseline volume window in seconds (default: 60.0)")
        parser.add_argument("--impact", type=float, default=0.015,
                            help="Min price impact in probability points (default: 0.015)")
        parser.add_argument("--liq-drop", type=float, default=0.6,
                            help="Liquidity drop ratio threshold (default: 0.6)")
        parser.add_argument("--spread-exp", type=float, default=1.8,
                            help="Spread expansion multiplier (default: 1.8)")
        parser.add_argument("--profit", type=float, default=0.15,
                            help="Profit target as fraction (default: 0.15 = 15%%)")
        parser.add_argument("--max-hold", type=float, default=90.0,
                            help="Max hold time in seconds (default: 90)")
        parser.add_argument("--cooldown", type=float, default=30.0,
                            help="Seconds between trades (default: 30)")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EventBurstConfig":
        """Create config from parsed CLI args."""
        return cls(
            coin=args.coin.upper(),
            slug=getattr(args, "slug", ""),
            size=args.size,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss,
            min_hold_seconds=args.min_hold,
            price_lookback_seconds=args.lookback,
            burst_ratio=args.burst_ratio,
            burst_window=args.burst_window,
            baseline_window=args.baseline_window,
            min_price_impact=args.impact,
            liquidity_drop=args.liq_drop,
            spread_expansion=args.spread_exp,
            profit_target=args.profit,
            max_hold_seconds=args.max_hold,
            cooldown_seconds=args.cooldown,
        )


class EventBurstStrategy(BaseStrategy):
    """
    Event Burst Trading Strategy.

    Detects repricing events via 4 independent microstructure filters.
    All must fire simultaneously to enter. Exits on profit, timeout,
    or liquidity recovery.
    """

    name = "event_burst"
    description = "Trade repricing events via 4-filter microstructure signal"
    config_class = EventBurstConfig

    def __init__(self, bot: TradingBot, config: EventBurstConfig):
        """Initialize event burst strategy."""
        super().__init__(bot, config)
        self.eb_config = config

        # Trade tape: (timestamp, size) — all trades on both tokens
        self._trades: deque = deque()

        # Orderbook snapshots for UP token: (timestamp, mid, spread, depth)
        # depth = sum of top 3 bid sizes + top 3 ask sizes
        self._book_snapshots: deque = deque()

        # Current computed values (updated each tick for display)
        self._burst_ratio: float = 0.0
        self._price_change: float = 0.0
        self._liq_ratio: float = 1.0
        self._spread_ratio: float = 1.0
        self._baseline_vol: float = 0.0
        self._recent_vol: float = 0.0
        self._current_depth: float = 0.0
        self._direction: Optional[str] = None  # "up" or "down"

        # Filter states for display
        self._filter_states = {
            "burst": False,
            "impact": False,
            "liquidity": False,
            "spread": False,
        }

        # Entry state for custom exits
        self._entry_depth: float = 0.0

        # Timing
        self._start_time: float = time.time()
        self._last_trade_time: float = 0.0

    # --- Data collection ---

    async def on_trade(self, trade: LastTradePrice) -> None:
        """Record trade for volume burst detection."""
        # Count trades on both tokens for total activity
        for token_id in self.token_ids.values():
            if token_id == trade.asset_id:
                self._trades.append((time.time(), trade.size))
                return

    async def on_book_update(self, snapshot: OrderbookSnapshot) -> None:
        """Record orderbook state for impact, liquidity, and spread filters."""
        # Only track positive side token book (UP/YES)
        pos_token = self.token_ids.get(self.positive_side, "")
        if snapshot.asset_id != pos_token:
            return

        now = time.time()
        mid = snapshot.mid_price
        spread = snapshot.best_ask - snapshot.best_bid if snapshot.best_bid > 0 else 0.0

        # Depth: sum of top 3 levels on each side
        bid_depth = sum(level.size for level in snapshot.bids[:3])
        ask_depth = sum(level.size for level in snapshot.asks[:3])
        depth = bid_depth + ask_depth

        self._book_snapshots.append((now, mid, spread, depth))

    # --- Filter computations ---

    def _evict_old_data(self) -> None:
        """Remove data older than baseline window."""
        cutoff = time.time() - max(self.eb_config.baseline_window, 120.0)
        while self._trades and self._trades[0][0] < cutoff:
            self._trades.popleft()
        while self._book_snapshots and self._book_snapshots[0][0] < cutoff:
            self._book_snapshots.popleft()

    def _check_volume_burst(self) -> Tuple[bool, float]:
        """Filter 1: Is trading volume spiking?

        Returns (triggered, burst_ratio).
        """
        now = time.time()
        burst_cutoff = now - self.eb_config.burst_window
        baseline_cutoff = now - self.eb_config.baseline_window

        recent_vol = sum(
            size for ts, size in self._trades if ts >= burst_cutoff
        )
        baseline_vol = sum(
            size for ts, size in self._trades if ts >= baseline_cutoff
        )

        self._recent_vol = recent_vol
        self._baseline_vol = baseline_vol

        # Average baseline volume per burst_window
        num_windows = self.eb_config.baseline_window / self.eb_config.burst_window
        avg_per_window = baseline_vol / num_windows if num_windows > 0 else 0

        if avg_per_window <= 0:
            ratio = 0.0
        else:
            ratio = recent_vol / avg_per_window

        self._burst_ratio = ratio
        return (ratio >= self.eb_config.burst_ratio, ratio)

    def _check_price_impact(self) -> Tuple[bool, float, Optional[str]]:
        """Filter 2: Did price move significantly on small volume?

        Returns (triggered, abs_change, direction).
        direction is "up" if price increased, "down" if decreased.
        """
        now = time.time()
        lookback = 15.0  # Compare to 15 seconds ago

        # Current mid price
        current_mid = 0.0
        if self._book_snapshots:
            current_mid = self._book_snapshots[-1][1]

        # Mid price ~15s ago
        old_mid = 0.0
        target_time = now - lookback
        for ts, mid, _, _ in self._book_snapshots:
            if ts <= target_time:
                old_mid = mid
            else:
                break

        if current_mid <= 0 or old_mid <= 0:
            self._price_change = 0.0
            self._direction = None
            return (False, 0.0, None)

        change = current_mid - old_mid
        abs_change = abs(change)
        self._price_change = abs_change

        # Volume check: recent volume should be small (informed, not crowd)
        volume_ok = self._burst_ratio < self.eb_config.max_volume_ratio * self.eb_config.burst_ratio

        direction = self.positive_side if change > 0 else self.negative_side if change < 0 else None
        self._direction = direction

        triggered = abs_change >= self.eb_config.min_price_impact and direction is not None
        return (triggered, abs_change, direction)

    def _check_liquidity_retreat(self) -> Tuple[bool, float]:
        """Filter 3: Has orderbook depth dropped significantly?

        Compares current depth to depth ~20 seconds ago.
        Returns (triggered, liq_ratio).
        """
        now = time.time()
        lookback = 20.0

        # Current depth
        current_depth = 0.0
        if self._book_snapshots:
            current_depth = self._book_snapshots[-1][3]
        self._current_depth = current_depth

        # Depth ~20s ago
        old_depth = 0.0
        target_time = now - lookback
        for ts, _, _, depth in self._book_snapshots:
            if ts <= target_time:
                old_depth = depth
            else:
                break

        if old_depth <= 0:
            self._liq_ratio = 1.0
            return (False, 1.0)

        ratio = current_depth / old_depth
        self._liq_ratio = ratio
        return (ratio <= self.eb_config.liquidity_drop, ratio)

    def _check_spread_expansion(self) -> Tuple[bool, float]:
        """Filter 4: Has the spread widened significantly?

        Compares current spread to rolling average.
        Returns (triggered, spread_ratio).
        """
        now = time.time()
        baseline_cutoff = now - self.eb_config.baseline_window

        # Current spread
        current_spread = 0.0
        if self._book_snapshots:
            current_spread = self._book_snapshots[-1][2]

        # Average spread over baseline window
        spreads = [
            spread for ts, _, spread, _ in self._book_snapshots
            if ts >= baseline_cutoff and spread > 0
        ]

        if not spreads or current_spread <= 0:
            self._spread_ratio = 1.0
            return (False, 1.0)

        avg_spread = sum(spreads) / len(spreads)
        if avg_spread <= 0:
            self._spread_ratio = 1.0
            return (False, 1.0)

        ratio = current_spread / avg_spread
        self._spread_ratio = ratio
        return (ratio >= self.eb_config.spread_expansion, ratio)

    def _get_current_depth(self) -> float:
        """Get most recent orderbook depth."""
        if self._book_snapshots:
            return self._book_snapshots[-1][3]
        return 0.0

    # --- Warmup ---

    @property
    def _warmup_remaining(self) -> float:
        elapsed = time.time() - self._start_time
        return max(0.0, self.eb_config.warmup_seconds - elapsed)

    @property
    def _is_warming_up(self) -> bool:
        return self._warmup_remaining > 0

    # --- Main loop ---

    async def on_tick(self, prices: Dict[str, float]) -> None:
        """Main decision loop — check all 4 filters."""
        self._evict_old_data()

        # Always compute filters for display
        burst_ok, burst_val = self._check_volume_burst()
        impact_ok, impact_val, direction = self._check_price_impact()
        liq_ok, liq_val = self._check_liquidity_retreat()
        spread_ok, spread_val = self._check_spread_expansion()

        self._filter_states["burst"] = burst_ok
        self._filter_states["impact"] = impact_ok
        self._filter_states["liquidity"] = liq_ok
        self._filter_states["spread"] = spread_ok

        # Skip trading during warmup
        if self._is_warming_up:
            return

        if not self.positions.can_open_position:
            return

        # Cooldown
        if time.time() - self._last_trade_time < self.eb_config.cooldown_seconds:
            return

        # ALL 4 filters must fire
        if not (burst_ok and impact_ok and liq_ok and spread_ok):
            return

        if not direction:
            return

        buy_side = direction
        current_price = prices.get(buy_side, 0)
        if current_price <= 0:
            return

        # Price guard
        if current_price > self.eb_config.max_price:
            self.log(
                f"BLOCKED: {buy_side.upper()} @ {current_price:.4f} > "
                f"{self.eb_config.max_price}",
                "warning"
            )
            return

        if current_price < self.eb_config.min_price:
            self.log(
                f"BLOCKED: {buy_side.upper()} @ {current_price:.4f} < "
                f"{self.eb_config.min_price}",
                "warning"
            )
            return

        # All filters passed — enter
        self.log(
            f"BURST: {buy_side.upper()} vol={burst_val:.1f}x "
            f"impact={impact_val:.3f} liq={liq_val:.2f} spread={spread_val:.1f}x",
            "trade"
        )

        # Record depth at entry for liquidity recovery exit
        self._entry_depth = self._get_current_depth()

        success = await self.execute_buy(buy_side, current_price)
        if success:
            self._last_trade_time = time.time()

    # --- Custom exit logic ---

    async def _check_exits(self, prices: Dict[str, float]) -> None:
        """Custom exit: profit target, time limit, or liquidity recovery."""
        for position in self.positions.get_all_positions():
            price = prices.get(position.side, 0)
            if price <= 0:
                continue

            hold_time = position.get_hold_time()

            # Skip exits for positions still within minimum hold time
            if hold_time < self.config.min_hold_seconds:
                continue

            pnl_pct = position.get_pnl_percent(price) / 100  # convert to fraction
            current_depth = self._get_current_depth()

            exit_reason = None

            # Profit target
            if pnl_pct >= self.eb_config.profit_target:
                exit_reason = "PROFIT"

            # Stop loss (negative profit target)
            elif pnl_pct <= -self.eb_config.profit_target:
                exit_reason = "STOP"

            # Time limit
            elif hold_time >= self.eb_config.max_hold_seconds:
                exit_reason = "TIMEOUT"

            # Liquidity recovery
            elif (self._entry_depth > 0 and current_depth > 0 and
                  current_depth / self._entry_depth >= self.eb_config.liquidity_recovery):
                exit_reason = "LIQ_RECOVER"

            if exit_reason:
                self.log(
                    f"{exit_reason}: {position.side.upper()} "
                    f"PnL={pnl_pct:+.1%} hold={hold_time:.0f}s",
                    "trade"
                )
                await self.execute_sell(position, price, exit_type=exit_reason)

    # --- TUI ---

    def _format_filter_bar(self, value: float, threshold: float, width: int = 10, invert: bool = False) -> str:
        """Format a filter value as a visual bar.

        Args:
            value: Current value
            threshold: Trigger threshold
            width: Bar width in characters
            invert: If True, trigger is when value < threshold (liquidity drop)
        """
        if invert:
            # For liquidity: lower is more triggered
            if threshold <= 0:
                fill = 0
            else:
                fill = max(0, min(width, int((1.0 - value / 1.0) * width)))
            triggered = value <= threshold
        else:
            # For burst/impact/spread: higher is more triggered
            if threshold <= 0:
                fill = 0
            else:
                fill = max(0, min(width, int(value / threshold * width)))
            triggered = value >= threshold

        bar = "=" * fill + " " * (width - fill)
        color = Colors.GREEN if triggered else Colors.CYAN
        return f"{color}[{bar}]{Colors.RESET}"

    def render_status(self, prices: Dict[str, float]) -> None:
        """Render TUI status display."""
        lines = []

        # Header
        ws_status = f"{Colors.GREEN}WS{Colors.RESET}" if self.is_connected else f"{Colors.RED}REST{Colors.RESET}"
        countdown = self._get_countdown_str()
        stats = self.positions.get_stats()

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
        lines.append(
            f"{Colors.CYAN}[{self.market_label}]{Colors.RESET} [{ws_status}] "
            f"[EVENT BURST 4-filter] "
            f"Ends: {countdown} | Trades: {stats['trades_closed']} "
            f"({stats['winning_trades']}W/{stats['losing_trades']}L) | "
            f"WR: {stats['win_rate']:.0f}% | PnL: ${stats['total_pnl']:+.2f}"
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

        # Mid price / spread
        pos_mid = pos_ob.mid_price if pos_ob else prices.get(pos, 0)
        neg_mid = neg_ob.mid_price if neg_ob else prices.get(neg, 0)
        pos_spread = self.market.get_spread(pos)
        neg_spread = self.market.get_spread(neg)

        lines.append(
            f"Mid: {Colors.GREEN}{pos_mid:.4f}{Colors.RESET}  Spread: {pos_spread:.4f}           |"
            f"Mid: {Colors.RED}{neg_mid:.4f}{Colors.RESET}  Spread: {neg_spread:.4f}"
        )

        lines.append(f"{Colors.BOLD}{'='*80}{Colors.RESET}")

        # 4-Filter Dashboard
        lines.append(f"{Colors.BOLD}Filters (all must fire):{Colors.RESET}")

        # Filter 1: Volume Burst
        burst_bar = self._format_filter_bar(self._burst_ratio, self.eb_config.burst_ratio)
        burst_status = f"{Colors.GREEN}TRIGGERED{Colors.RESET}" if self._filter_states["burst"] else "waiting"
        lines.append(
            f"  Volume Burst:   {burst_bar} {self._burst_ratio:>5.1f}x / {self.eb_config.burst_ratio:.1f}x  "
            f"({burst_status})  "
            f"recent={self._recent_vol:.0f} base={self._baseline_vol:.0f}"
        )

        # Filter 2: Price Impact
        impact_bar = self._format_filter_bar(self._price_change, self.eb_config.min_price_impact)
        impact_status = f"{Colors.GREEN}TRIGGERED{Colors.RESET}" if self._filter_states["impact"] else "waiting"
        dir_str = f" {self._direction.upper()}" if self._direction else ""
        lines.append(
            f"  Price Impact:   {impact_bar} {self._price_change:>5.3f} / {self.eb_config.min_price_impact:.3f}  "
            f"({impact_status}){dir_str}"
        )

        # Filter 3: Liquidity Retreat
        liq_bar = self._format_filter_bar(self._liq_ratio, self.eb_config.liquidity_drop, invert=True)
        liq_status = f"{Colors.GREEN}TRIGGERED{Colors.RESET}" if self._filter_states["liquidity"] else "waiting"
        lines.append(
            f"  Liquidity Drop: {liq_bar} {self._liq_ratio:>5.2f} / {self.eb_config.liquidity_drop:.2f}  "
            f"({liq_status})  depth={self._current_depth:.0f}"
        )

        # Filter 4: Spread Expansion
        spread_bar = self._format_filter_bar(self._spread_ratio, self.eb_config.spread_expansion)
        spread_status = f"{Colors.GREEN}TRIGGERED{Colors.RESET}" if self._filter_states["spread"] else "waiting"
        lines.append(
            f"  Spread Expand:  {spread_bar} {self._spread_ratio:>5.1f}x / {self.eb_config.spread_expansion:.1f}x  "
            f"({spread_status})"
        )

        # All-fire status
        all_fired = all(self._filter_states.values())
        if all_fired:
            dir_label = self._direction.upper() if self._direction else "?"
            lines.append(
                f"  {Colors.GREEN}{Colors.BOLD}>>> ALL FILTERS FIRED — "
                f"SIGNAL: {dir_label} <<<{Colors.RESET}"
            )

        # Price guard status
        for side in self.token_ids:
            price = prices.get(side, 0)
            if price > self.eb_config.max_price:
                lines.append(
                    f"  {Colors.RED}! {side.upper()} @ {price:.4f} > "
                    f"{self.eb_config.max_price} (blocked){Colors.RESET}"
                )
            elif 0 < price < self.eb_config.min_price:
                lines.append(
                    f"  {Colors.RED}! {side.upper()} @ {price:.4f} < "
                    f"{self.eb_config.min_price} (blocked){Colors.RESET}"
                )

        # Warmup status
        if self._is_warming_up:
            remaining = self._warmup_remaining
            lines.append(
                f"  {Colors.CYAN}Warming up... {remaining:.0f}s remaining "
                f"(collecting data){Colors.RESET}"
            )

        # Cooldown status
        elapsed = time.time() - self._last_trade_time
        cooldown = self.eb_config.cooldown_seconds
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
                remaining_hold = max(0, self.eb_config.max_hold_seconds - hold_time)
                color = Colors.GREEN if pnl >= 0 else Colors.RED

                lines.append(
                    f"  {Colors.BOLD}{pos.side.upper():4}{Colors.RESET} "
                    f"Entry: {pos.entry_price:.4f} | Current: {current:.4f} | "
                    f"Size: {pos.size:.2f} | PnL: {color}${pnl:+.2f} ({pnl_pct:+.1f}%){Colors.RESET} | "
                    f"Hold: {hold_time:.0f}s | Exit in: {remaining_hold:.0f}s"
                )

                # Show exit conditions
                target_price = pos.entry_price * (1 + self.eb_config.profit_target)
                stop_price = pos.entry_price * (1 - self.eb_config.profit_target)
                liq_status = ""
                if self._entry_depth > 0 and self._current_depth > 0:
                    recovery = self._current_depth / self._entry_depth
                    liq_status = f" | Depth: {recovery:.0%}/{self.eb_config.liquidity_recovery:.0%}"
                lines.append(
                    f"       Target: {target_price:.4f} | Stop: {stop_price:.4f}{liq_status}"
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
        """Handle market change — reset all state."""
        self.prices.clear()
        self._trades = deque()
        self._book_snapshots = deque()
        self._burst_ratio = 0.0
        self._price_change = 0.0
        self._liq_ratio = 1.0
        self._spread_ratio = 1.0
        self._baseline_vol = 0.0
        self._recent_vol = 0.0
        self._current_depth = 0.0
        self._direction = None
        self._filter_states = {k: False for k in self._filter_states}
        self._entry_depth = 0.0
        self._start_time = time.time()
