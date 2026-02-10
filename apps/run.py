#!/usr/bin/env python3
"""
Unified Strategy Runner

Run any registered strategy by name with auto-discovered CLI arguments.

Usage:
    python apps/run.py flash_crash --coin ETH --size 5 --drop 0.25
    python apps/run.py flash_crash --coin BTC --reverse
    python apps/run.py --list
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path

# Suppress noisy logs
logging.getLogger("src.websocket_client").setLevel(logging.WARNING)
logging.getLogger("src.bot").setLevel(logging.WARNING)

# Auto-load .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.console import Colors
from src.bot import TradingBot
from src.config import Config
from strategies import list_strategies


def main():
    strategies = list_strategies()

    # Handle --list before argparse (so it works without a strategy name)
    if "--list" in sys.argv or "-l" in sys.argv:
        print(f"\n{Colors.BOLD}Available strategies:{Colors.RESET}\n")
        for name, cls in sorted(strategies.items()):
            print(f"  {Colors.CYAN}{name:<20}{Colors.RESET} {cls.description}")
        print(f"\nUsage: python apps/run.py <strategy> [options]")
        return

    if len(sys.argv) < 2 or sys.argv[1].startswith("-"):
        print(f"Usage: python apps/run.py <strategy> [options]")
        print(f"       python apps/run.py --list")
        print(f"\nAvailable: {', '.join(sorted(strategies.keys()))}")
        sys.exit(1)

    strategy_name = sys.argv[1]
    if strategy_name not in strategies:
        print(f"{Colors.RED}Unknown strategy: {strategy_name}{Colors.RESET}")
        print(f"Available: {', '.join(sorted(strategies.keys()))}")
        sys.exit(1)

    strategy_cls = strategies[strategy_name]
    config_cls = strategy_cls.config_class

    # Build argparse with strategy-specific args
    parser = argparse.ArgumentParser(
        description=strategy_cls.description,
        prog=f"python apps/run.py {strategy_name}",
    )
    config_cls.add_args(parser)
    args = parser.parse_args(sys.argv[2:])

    # Debug logging
    if getattr(args, "debug", False):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("src.websocket_client").setLevel(logging.DEBUG)

    # Check environment
    private_key = os.environ.get("POLY_PRIVATE_KEY")
    safe_address = os.environ.get("POLY_SAFE_ADDRESS")

    if not private_key or not safe_address:
        print(f"{Colors.RED}Error: POLY_PRIVATE_KEY and POLY_SAFE_ADDRESS must be set{Colors.RESET}")
        print("Set them in .env file or export as environment variables")
        sys.exit(1)

    # Create bot
    config = Config.from_env()
    bot = TradingBot(config=config, private_key=private_key)

    if not bot.is_initialized():
        print(f"{Colors.RED}Error: Failed to initialize bot{Colors.RESET}")
        sys.exit(1)

    # Create strategy config from CLI args
    strategy_config = config_cls.from_args(args)

    # Print configuration
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}  {strategy_cls.description}{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    print(f"Strategy: {strategy_name}")
    print(f"Coin: {strategy_config.coin} | Size: ${strategy_config.size:.2f}")
    print(f"TP: +{strategy_config.take_profit*100:.0f}% | SL: -{strategy_config.stop_loss*100:.0f}%")
    print()

    # Create and run
    strategy = strategy_cls(bot=bot, config=strategy_config)

    try:
        asyncio.run(strategy.run())
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
