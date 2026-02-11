#!/usr/bin/env python3
"""
Market Scanner - Discover live and upcoming Polymarket events.

Lists active events with volume, start time, live scores, and market slugs.
Useful for finding markets to trade with --slug.

Usage:
    python apps/scan.py                     # All active events
    python apps/scan.py --live              # Only live (in-progress) events
    python apps/scan.py --sport nba         # Filter by sport tag
    python apps/scan.py --min-volume 1000   # Minimum event volume
    python apps/scan.py --top 10            # Show top 10 by volume
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.console import Colors
from src.gamma_client import GammaClient


def format_volume(vol: float) -> str:
    """Format volume as human-readable string."""
    if vol >= 1_000_000:
        return f"${vol / 1_000_000:.1f}M"
    if vol >= 1_000:
        return f"${vol / 1_000:.1f}K"
    return f"${vol:.0f}"


def format_time_until(dt_str: str) -> str:
    """Format time until a datetime string as relative time."""
    if not dt_str:
        return "  --"
    try:
        dt_str = dt_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        now = datetime.now(timezone.utc)
        delta = dt - now

        total_secs = int(delta.total_seconds())
        if total_secs < 0:
            # Already started
            mins_ago = abs(total_secs) // 60
            if mins_ago < 60:
                return f"{mins_ago}m ago"
            hours_ago = mins_ago // 60
            if hours_ago < 24:
                return f"{hours_ago}h ago"
            return f"{hours_ago // 24}d ago"

        if total_secs < 3600:
            return f"in {total_secs // 60}m"
        if total_secs < 86400:
            return f"in {total_secs // 3600}h {(total_secs % 3600) // 60}m"
        return f"in {total_secs // 86400}d"
    except Exception:
        return "  --"


def parse_prices(market: dict) -> str:
    """Parse outcome prices into a compact display."""
    outcomes_raw = market.get("outcomes", "[]")
    prices_raw = market.get("outcomePrices", "[]")

    try:
        import json
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else outcomes_raw
        prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
    except Exception:
        return ""

    parts = []
    for i, (outcome, price) in enumerate(zip(outcomes, prices)):
        p = float(price)
        label = str(outcome)[:8]
        parts.append(f"{label}:{p:.0%}")

    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Scan Polymarket for live and upcoming events")
    parser.add_argument("--live", action="store_true", help="Only show live (in-progress) events")
    parser.add_argument("--sport", type=str, default="", help="Filter by sport tag slug (e.g., nba, nfl, mlb, nhl, soccer)")
    parser.add_argument("--min-volume", type=float, default=0, help="Minimum event volume in USD")
    parser.add_argument("--top", type=int, default=0, help="Show top N events by volume")
    parser.add_argument("--markets", action="store_true", help="Show individual markets under each event")
    parser.add_argument("--upcoming", action="store_true", help="Only show upcoming (not yet live) events")
    parser.add_argument("--limit", type=int, default=0, help="Max events to show (default: 0 = no limit)")
    args = parser.parse_args()

    gamma = GammaClient()

    # Build query params
    params = {
        "active": True,
        "closed": False,
        "order": "volume24hr",
        "ascending": False,
    }

    if args.sport:
        params["tag_slug"] = args.sport

    if args.min_volume > 0:
        params["volume_min"] = args.min_volume

    # Pass live/upcoming filters to the API so we don't miss results due to limit
    if args.live:
        params["live"] = True
    elif args.upcoming:
        params["live"] = False
        params["ended"] = False

    print(f"\n{Colors.BOLD}Scanning Polymarket events...{Colors.RESET}\n")

    # Paginate to collect all matching events
    page_size = 100
    events = []
    offset = 0
    while True:
        params["limit"] = page_size
        params["offset"] = offset
        batch = gamma.list_events(**params)
        if not batch:
            break
        events.extend(batch)
        if len(batch) < page_size or (args.limit and len(events) >= args.limit):
            break
        offset += page_size

    if args.limit:
        events = events[:args.limit]

    if not events:
        print(f"{Colors.RED}No events found.{Colors.RESET}")
        return

    # Sort by 24h volume descending
    events.sort(key=lambda e: float(e.get("volume24hr") or 0), reverse=True)

    if args.top > 0:
        events = events[:args.top]

    if not events:
        print(f"{Colors.YELLOW}No matching events.{Colors.RESET}")
        return

    # Display
    print(f"{Colors.BOLD}{'#':>3}  {'Status':<10} {'Volume(24h)':>11} {'Total Vol':>11} {'Start':>10}  Title{Colors.RESET}")
    print("-" * 100)

    for i, event in enumerate(events, 1):
        title = (event.get("title") or "Untitled")[:55]
        vol_24h = float(event.get("volume24hr") or 0)
        vol_total = float(event.get("volume") or 0)
        start_time = event.get("startTime") or ""
        is_live = event.get("live", False)
        is_ended = event.get("ended", False)
        score = event.get("score") or ""
        period = event.get("period") or ""
        game_status = event.get("gameStatus") or ""

        # Status display
        if is_live:
            status_parts = [f"{Colors.GREEN}LIVE{Colors.RESET}"]
            if score:
                status_parts.append(f" {score}")
            if period:
                status_parts.append(f" {period}")
            status = "".join(status_parts)
        elif is_ended:
            status = f"{Colors.DIM}ENDED{Colors.RESET}"
        else:
            status = f"{Colors.CYAN}UPCOMING{Colors.RESET}"

        time_str = format_time_until(start_time)

        print(
            f"{i:>3}  {status:<22} {format_volume(vol_24h):>11} {format_volume(vol_total):>11} "
            f"{time_str:>10}  {title}"
        )

        # Show individual markets if requested
        if args.markets:
            markets = event.get("markets") or []
            # Sort markets by volume
            markets.sort(key=lambda m: float(m.get("volumeNum") or 0), reverse=True)

            for mkt in markets:
                slug = mkt.get("slug") or "?"
                question = (mkt.get("question") or "")[:50]
                mkt_vol = float(mkt.get("volumeNum") or 0)
                accepting = mkt.get("acceptingOrders", False)
                prices = parse_prices(mkt)
                sport_type = mkt.get("sportsMarketType") or ""
                delay = mkt.get("secondsDelay") or 0

                status_icon = f"{Colors.GREEN}*{Colors.RESET}" if accepting else f"{Colors.DIM}-{Colors.RESET}"

                type_str = f" [{sport_type}]" if sport_type else ""
                delay_str = f" {delay}s" if delay else ""

                print(
                    f"     {status_icon} {format_volume(mkt_vol):>9}{type_str}{delay_str}  "
                    f"{prices}  {Colors.DIM}--slug {slug}{Colors.RESET}"
                )

            print()

    print("-" * 100)
    print(f"\n{Colors.DIM}Total: {len(events)} events{Colors.RESET}")

    if not args.markets:
        print(f"{Colors.DIM}Tip: Use --markets to see individual markets with slugs for trading{Colors.RESET}")

    print()


if __name__ == "__main__":
    main()
