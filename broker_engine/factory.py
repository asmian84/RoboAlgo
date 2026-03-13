"""
RoboAlgo — Broker Factory
Returns the configured broker adapter instance.

Configuration:
  Set BROKER env var to: "alpaca" | "ibkr" | "questrade"
  Default: "alpaca"

Usage:
    from broker_engine.factory import get_broker
    broker = get_broker()                    # uses BROKER env var
    broker = get_broker("ibkr")              # override
    account = broker.get_account()
    result  = broker.place_bracket_order("TQQQ", 100, "buy",
                                          limit_price=55.00,
                                          take_profit=62.50,
                                          stop_loss=52.00)
"""

import os
import logging
from broker_engine.base import BrokerBase

logger = logging.getLogger(__name__)

_SUPPORTED_BROKERS = ("alpaca", "ibkr", "questrade")


def get_broker(broker_name: str | None = None) -> BrokerBase:
    """
    Return a connected broker adapter instance.

    Args:
        broker_name: Override the BROKER env var. One of: alpaca, ibkr, questrade.

    Returns:
        Instantiated broker adapter implementing BrokerBase.

    Raises:
        ValueError:       Unknown broker name.
        EnvironmentError: Required env vars missing.
        RuntimeError:     SDK not installed.
        ConnectionError:  Cannot connect to broker (IBKR).
    """
    name = (broker_name or os.getenv("BROKER", "alpaca")).lower().strip()

    if name == "alpaca":
        from broker_engine.alpaca import AlpacaBroker
        broker = AlpacaBroker()
        logger.info(f"Broker: Alpaca ({'paper' if broker.is_paper else 'live'})")
        return broker

    elif name in ("ibkr", "ib", "interactive_brokers"):
        from broker_engine.ibkr import IBKRBroker
        broker = IBKRBroker()
        logger.info(f"Broker: IBKR ({'paper' if broker.is_paper else 'live'})")
        return broker

    elif name == "questrade":
        from broker_engine.questrade import QuestradeBroker
        broker = QuestradeBroker()
        logger.info(f"Broker: Questrade (live)")
        return broker

    else:
        raise ValueError(
            f"Unknown broker: '{name}'. "
            f"Supported brokers: {', '.join(_SUPPORTED_BROKERS)}"
        )
