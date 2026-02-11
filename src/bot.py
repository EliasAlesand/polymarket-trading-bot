"""
Trading Bot Module - Main Trading Interface

A production-ready trading bot for Polymarket with:
- Gasless transactions via Builder Program
- Encrypted private key storage
- Modular strategy support
- Comprehensive order management

Example:
    from src.bot import TradingBot

    # Initialize with config
    bot = TradingBot(config_path="config.yaml")

    # Or manually
    bot = TradingBot(
        safe_address="0x...",
        builder_creds=builder_creds,
        private_key="0x..."  # or use encrypted key
    )

    # Place an order
    result = await bot.place_order(
        token_id="123...",
        price=0.65,
        size=10,
        side="BUY"
    )
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum

from .config import Config, BuilderConfig
from .signer import OrderSigner, Order
from .client import ClobClient, RelayerClient, ApiCredentials
from .crypto import KeyManager, CryptoError, InvalidPasswordError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar("T")

class OrderSide(str, Enum):
    """Order side constants."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type constants."""
    GTC = "GTC"  # Good Till Cancelled
    GTD = "GTD"  # Good Till Date
    FOK = "FOK"  # Fill Or Kill


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    order_id: Optional[str] = None
    status: Optional[str] = None
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "OrderResult":
        """Create from API response."""
        success = response.get("success", False)
        error_msg = response.get("errorMsg", "")
        order_id = response.get("orderID") or response.get("orderId")

        return cls(
            success=success,
            order_id=order_id,
            status=response.get("status"),
            message=error_msg if not success else "Order placed successfully",
            data=response
        )


class TradingBotError(Exception):
    """Base exception for trading bot errors."""
    pass


class NotInitializedError(TradingBotError):
    """Raised when bot is not initialized."""
    pass


class TradingBot:
    """
    Main trading bot class for Polymarket.

    Provides a high-level interface for:
    - Order placement and cancellation
    - Position management
    - Trade history
    - Gasless transactions (with Builder Program)

    Attributes:
        config: Bot configuration
        signer: Order signer instance
        clob_client: CLOB API client
        relayer_client: Relayer API client (if gasless enabled)
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        safe_address: Optional[str] = None,
        builder_creds: Optional[BuilderConfig] = None,
        private_key: Optional[str] = None,
        encrypted_key_path: Optional[str] = None,
        password: Optional[str] = None,
        api_creds_path: Optional[str] = None,
        log_level: int = logging.INFO
    ):
        """
        Initialize trading bot.

        Can be initialized in multiple ways:

        1. From config file:
           bot = TradingBot(config_path="config.yaml")

        2. From Config object:
           bot = TradingBot(config=my_config)

        3. With manual parameters:
           bot = TradingBot(
               safe_address="0x...",
               builder_creds=builder_creds,
               private_key="0x..."
           )

        4. With encrypted key:
           bot = TradingBot(
               safe_address="0x...",
               encrypted_key_path="credentials/key.enc",
               password="mypassword"
           )

        Args:
            config_path: Path to config YAML file
            config: Config object
            safe_address: Safe/Proxy wallet address
            builder_creds: Builder Program credentials
            private_key: Raw private key (with 0x prefix)
            encrypted_key_path: Path to encrypted key file
            password: Password for encrypted key
            api_creds_path: Path to API credentials file
            log_level: Logging level
        """
        # Set log level
        logger.setLevel(log_level)

        # Load configuration
        if config_path:
            self.config = Config.load(config_path)
        elif config:
            self.config = config
        else:
            self.config = Config()

        # Override with provided parameters
        if safe_address:
            self.config.safe_address = safe_address
        if builder_creds:
            self.config.builder = builder_creds
            self.config.use_gasless = True

        # Initialize components
        self.signer: Optional[OrderSigner] = None
        self.clob_client: Optional[ClobClient] = None
        self.relayer_client: Optional[RelayerClient] = None
        self._api_creds: Optional[ApiCredentials] = None
        self._market_props_cache: Dict[str, Dict[str, Any]] = {}  # token_id -> {neg_risk, tick_size, fee_rate_bps}
        self._w3 = None  # Cached Web3 instance
        self._ctf_contract = None  # Cached CTF contract instance

        # Load private key
        if private_key:
            self.signer = OrderSigner(private_key)
        elif encrypted_key_path and password:
            self._load_encrypted_key(encrypted_key_path, password)

        # Load API credentials
        if api_creds_path:
            self._load_api_creds(api_creds_path)

        # Initialize API clients
        self._init_clients()

        # Auto-derive API credentials if we have a signer but no API creds
        if self.signer and not self._api_creds:
            self._derive_api_creds()

        logger.info(f"TradingBot initialized (gasless: {self.config.use_gasless})")

    def _load_encrypted_key(self, filepath: str, password: str) -> None:
        """Load and decrypt private key from encrypted file."""
        try:
            manager = KeyManager()
            private_key = manager.load_and_decrypt(password, filepath)
            self.signer = OrderSigner(private_key)
            logger.info(f"Loaded encrypted key from {filepath}")
        except FileNotFoundError:
            raise TradingBotError(f"Encrypted key file not found: {filepath}")
        except InvalidPasswordError:
            raise TradingBotError("Invalid password for encrypted key")
        except CryptoError as e:
            raise TradingBotError(f"Failed to load encrypted key: {e}")

    def _load_api_creds(self, filepath: str) -> None:
        """Load API credentials from file."""
        if os.path.exists(filepath):
            try:
                self._api_creds = ApiCredentials.load(filepath)
                logger.info(f"Loaded API credentials from {filepath}")
            except Exception as e:
                logger.warning(f"Failed to load API credentials: {e}")

    def _derive_api_creds(self) -> None:
        """Derive L2 API credentials from signer."""
        if not self.signer or not self.clob_client:
            return

        try:
            logger.info("Deriving L2 API credentials...")
            self._api_creds = self.clob_client.create_or_derive_api_key(self.signer)
            self.clob_client.set_api_creds(self._api_creds)
            logger.info("L2 API credentials derived successfully")
        except Exception as e:
            logger.warning(f"Failed to derive API credentials: {e}")
            logger.warning("Some API endpoints may not be accessible")

    def _init_clients(self) -> None:
        """Initialize API clients."""
        # CLOB client
        self.clob_client = ClobClient(
            host=self.config.clob.host,
            chain_id=self.config.clob.chain_id,
            signature_type=self.config.clob.signature_type,
            funder=self.config.safe_address,
            signer_address=self.signer.address if self.signer else "",
            api_creds=self._api_creds,
            builder_creds=self.config.builder if self.config.use_gasless else None,
        )

        # Relayer client (for gasless)
        if self.config.use_gasless:
            self.relayer_client = RelayerClient(
                host=self.config.relayer.host,
                chain_id=self.config.clob.chain_id,
                builder_creds=self.config.builder,
                tx_type=self.config.relayer.tx_type,
            )
            logger.info("Relayer client initialized (gasless enabled)")

    async def _run_in_thread(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a blocking call in a worker thread to avoid event loop stalls."""
        return await asyncio.to_thread(func, *args, **kwargs)

    def is_initialized(self) -> bool:
        """Check if bot is properly initialized."""
        return (
            self.signer is not None and
            self.config.safe_address and
            self.clob_client is not None
        )

    def require_signer(self) -> OrderSigner:
        """Get signer or raise if not initialized."""
        if not self.signer:
            raise NotInitializedError(
                "Signer not initialized. Provide private_key or encrypted_key."
            )
        return self.signer

    async def place_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: str,
        order_type: str = "GTC",
        fee_rate_bps: int = 0
    ) -> OrderResult:
        """
        Place a limit order.

        Args:
            token_id: Market token ID
            price: Price per share (0-1)
            size: Number of shares
            side: 'BUY' or 'SELL'
            order_type: Order type (GTC, GTD, FOK)
            fee_rate_bps: Fee rate in basis points

        Returns:
            OrderResult with order status
        """
        signer = self.require_signer()

        try:
            # Check market properties (cached + parallel fetch)
            cached = self._market_props_cache.get(token_id)
            if cached:
                neg_risk = cached["neg_risk"]
                tick_size = cached["tick_size"]
                if fee_rate_bps == 0:
                    fee_rate_bps = cached["fee_rate_bps"]
            else:
                # Fetch all 3 in parallel instead of sequentially
                results = await asyncio.gather(
                    self._run_in_thread(self.clob_client.get_neg_risk, token_id),
                    self._run_in_thread(self.clob_client.get_tick_size, token_id),
                    self._run_in_thread(self.clob_client.get_fee_rate_bps, token_id),
                )
                neg_risk, tick_size, market_fee = results
                self._market_props_cache[token_id] = {
                    "neg_risk": neg_risk,
                    "tick_size": tick_size,
                    "fee_rate_bps": market_fee,
                }
                if fee_rate_bps == 0:
                    fee_rate_bps = market_fee

            # Create order with proper rounding
            order = Order(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
                maker=self.config.safe_address,
                fee_rate_bps=fee_rate_bps,
                tick_size=tick_size,
            )

            # Sign order with correct exchange domain
            signed = signer.sign_order(order, neg_risk=neg_risk)

            # Submit to CLOB
            response = await self._run_in_thread(
                self.clob_client.post_order,
                signed,
                order_type,
            )

            logger.debug(
                f"Order placed: {side} {size}@{price} "
                f"(token: {token_id[:16]}...)"
            )

            return OrderResult.from_response(response)

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return OrderResult(
                success=False,
                message=str(e)
            )

    async def place_orders(
        self,
        orders: List[Dict[str, Any]],
        order_type: str = "GTC"
    ) -> List[OrderResult]:
        """
        Place multiple orders.

        Args:
            orders: List of order dictionaries with keys:
                - token_id: Market token ID
                - price: Price per share
                - size: Number of shares
                - side: 'BUY' or 'SELL'
            order_type: Order type (GTC, GTD, FOK)

        Returns:
            List of OrderResults
        """
        results = []
        for order_data in orders:
            result = await self.place_order(
                token_id=order_data["token_id"],
                price=order_data["price"],
                size=order_data["size"],
                side=order_data["side"],
                order_type=order_type,
            )
            results.append(result)

            # Small delay between orders to avoid rate limits
            await asyncio.sleep(0.1)

        return results

    async def cancel_order(self, order_id: str) -> OrderResult:
        """
        Cancel a specific order.

        Args:
            order_id: Order ID to cancel

        Returns:
            OrderResult with cancellation status
        """
        try:
            response = await self._run_in_thread(self.clob_client.cancel_order, order_id)
            logger.info(f"Order cancelled: {order_id}")
            return OrderResult(
                success=True,
                order_id=order_id,
                message="Order cancelled",
                data=response
            )
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                message=str(e)
            )

    async def cancel_all_orders(self) -> OrderResult:
        """
        Cancel all open orders.

        Returns:
            OrderResult with cancellation status
        """
        try:
            response = await self._run_in_thread(self.clob_client.cancel_all_orders)
            logger.info("All orders cancelled")
            return OrderResult(
                success=True,
                message="All orders cancelled",
                data=response
            )
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return OrderResult(success=False, message=str(e))

    async def cancel_market_orders(
        self,
        market: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> OrderResult:
        """
        Cancel orders for a specific market.

        Args:
            market: Condition ID of the market (optional)
            asset_id: Token/asset ID (optional)

        Returns:
            OrderResult with cancellation status
        """
        try:
            response = await self._run_in_thread(
                self.clob_client.cancel_market_orders,
                market,
                asset_id,
            )
            logger.info(f"Market orders cancelled (market: {market or 'all'}, asset: {asset_id or 'all'})")
            return OrderResult(
                success=True,
                message=f"Orders cancelled for market {market or 'all'}",
                data=response
            )
        except Exception as e:
            logger.error(f"Failed to cancel market orders: {e}")
            return OrderResult(success=False, message=str(e))

    async def wait_for_fill(
        self,
        order_id: str,
        timeout: float = 15.0,
    ) -> float:
        """
        Poll order status until matched/filled or timeout.

        Uses fast polling initially (0.3s) then backs off to 1s.

        Args:
            order_id: Order ID to check
            timeout: Max seconds to wait

        Returns:
            Filled size (float), or 0.0 if not filled
        """
        import time
        start = time.time()
        poll_interval = 0.3  # Start fast
        while time.time() - start < timeout:
            order_data = await self.get_order(order_id)
            if order_data:
                size_matched = order_data.get("size_matched", "0")
                original_size = order_data.get("original_size", "0")
                try:
                    matched = float(size_matched)
                    if matched > 0:
                        logger.debug(
                            f"Order {order_id} filled: "
                            f"{size_matched}/{original_size}"
                        )
                        return matched
                except (ValueError, TypeError):
                    pass
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 1.0)  # Backoff to 1s

        logger.warning(f"Order {order_id} not filled within {timeout}s")
        return 0.0

    async def get_filled_size(self, order_id: str) -> float:
        """
        Get the filled size for an order.

        Args:
            order_id: Order ID

        Returns:
            Filled size as float, or 0.0
        """
        order_data = await self.get_order(order_id)
        if order_data:
            try:
                return float(order_data.get("size_matched", "0"))
            except (ValueError, TypeError):
                pass
        return 0.0

    def _get_ctf_contract(self):
        """Get cached CTF contract instance (creates Web3 + contract once)."""
        if self._ctf_contract is None:
            from web3 import Web3
            import os

            rpc = os.environ.get("POLY_RPC_URL", "https://polygon-rpc.com")
            self._w3 = Web3(Web3.HTTPProvider(rpc))

            CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
            abi = [{"name": "balanceOf", "type": "function",
                    "stateMutability": "view",
                    "inputs": [{"name": "owner", "type": "address"},
                               {"name": "id", "type": "uint256"}],
                    "outputs": [{"name": "", "type": "uint256"}]}]

            self._ctf_contract = self._w3.eth.contract(address=CTF, abi=abi)
        return self._ctf_contract

    async def get_token_balance(self, token_id: str) -> float:
        """
        Get actual on-chain ERC-1155 token balance for the Safe wallet.

        This is the ground truth for how many shares can be sold.
        The CLOB's size_matched can differ due to fees/settlement.

        Args:
            token_id: The ERC-1155 token ID

        Returns:
            Balance in shares (float), or 0.0
        """
        try:
            ctf = self._get_ctf_contract()
            balance = await self._run_in_thread(
                ctf.functions.balanceOf(
                    self.config.safe_address, int(token_id)
                ).call
            )
            return balance / 1e6
        except Exception as e:
            logger.error(f"Failed to get token balance: {e}")
            return 0.0

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders.

        Returns:
            List of open orders
        """
        try:
            orders = await self._run_in_thread(self.clob_client.get_open_orders)
            logger.debug(f"Retrieved {len(orders)} open orders")
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details.

        Args:
            order_id: Order ID

        Returns:
            Order details or None
        """
        try:
            return await self._run_in_thread(self.clob_client.get_order, order_id)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def get_trades(
        self,
        token_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get trade history.

        Args:
            token_id: Optional token ID to filter
            limit: Maximum number of trades

        Returns:
            List of trades
        """
        try:
            trades = await self._run_in_thread(self.clob_client.get_trades, token_id, limit)
            logger.debug(f"Retrieved {len(trades)} trades")
            return trades
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    async def get_average_fill_price(self, order_id: str, token_id: str) -> Optional[float]:
        """
        Get the actual average fill price for a filled order by querying trades.

        The CLOB can fill orders at better prices than the limit price (price
        improvement), so the submitted price != actual execution price.

        Args:
            order_id: Order ID to look up
            token_id: Token ID (for filtering trades)

        Returns:
            Weighted average fill price, or None if not determinable
        """
        try:
            trades = await self.get_trades(token_id=token_id, limit=20)
            if not trades:
                return None

            total_size = 0.0
            total_notional = 0.0
            for trade in trades:
                if trade.get("taker_order_id") == order_id:
                    price = float(trade.get("price", 0))
                    size = float(trade.get("size", 0))
                    if price > 0 and size > 0:
                        total_notional += price * size
                        total_size += size

            if total_size > 0:
                avg_price = total_notional / total_size
                logger.debug(f"Average fill price for {order_id}: {avg_price:.4f}")
                return avg_price
            return None
        except Exception as e:
            logger.debug(f"Could not determine fill price: {e}")
            return None

    async def get_order_book(self, token_id: str) -> Dict[str, Any]:
        """
        Get order book for a token.

        Args:
            token_id: Market token ID

        Returns:
            Order book data
        """
        try:
            return await self._run_in_thread(self.clob_client.get_order_book, token_id)
        except Exception as e:
            logger.error(f"Failed to get order book: {e}")
            return {}

    async def get_market_price(self, token_id: str) -> Dict[str, Any]:
        """
        Get current market price for a token.

        Args:
            token_id: Market token ID

        Returns:
            Price data
        """
        try:
            return await self._run_in_thread(self.clob_client.get_market_price, token_id)
        except Exception as e:
            logger.error(f"Failed to get market price: {e}")
            return {}

    async def deploy_safe_if_needed(self) -> bool:
        """
        Deploy Safe proxy wallet if not already deployed.

        Returns:
            True if deployment was needed or successful
        """
        if not self.config.use_gasless or not self.relayer_client:
            logger.debug("Gasless not enabled, skipping Safe deployment")
            return False

        try:
            response = await self._run_in_thread(
                self.relayer_client.deploy_safe,
                self.config.safe_address,
            )
            logger.info(f"Safe deployment initiated: {response}")
            return True
        except Exception as e:
            logger.warning(f"Safe deployment failed (may already be deployed): {e}")
            return False



# Convenience function for quick initialization
def create_bot(
    config_path: str = "config.yaml",
    private_key: Optional[str] = None,
    encrypted_key_path: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs
) -> TradingBot:
    """
    Create a TradingBot instance with common options.

    Args:
        config_path: Path to config file
        private_key: Private key (with 0x prefix)
        encrypted_key_path: Path to encrypted key file
        password: Password for encrypted key
        **kwargs: Additional arguments for TradingBot

    Returns:
        Configured TradingBot instance
    """
    return TradingBot(
        config_path=config_path,
        private_key=private_key,
        encrypted_key_path=encrypted_key_path,
        password=password,
        **kwargs
    )
