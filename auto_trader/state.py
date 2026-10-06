"""
auto_trader.state — Redis-backed kill switch, run flag, and cycle lock.

Shared between the FastAPI process and the Celery worker so both sides agree
on whether autonomous trading is allowed to run right now.

`redis` is an optional dependency: it is imported lazily (inside
``_get_client``) so this module can be imported even in environments where
the package is not installed. When redis is unavailable — either because the
package is missing or because the server can't be reached — every function
degrades according to the fail-safe rule below instead of raising.

FAIL-SAFE (deliberate, do not invert): if redis is not importable or not
reachable, ``kill_switch_on()`` returns ``True`` — we refuse to trade because
the kill state is unknowable, so we assume the kill switch is engaged. The
only way to opt out of this fail-safe is to set
``ROBOALGO_AUTOTRADER_ALLOW_NO_REDIS=1``, which falls back to a process-local
in-memory store (useful for local dev / tests) and logs a WARNING every time
it's used.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from os import getenv
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_REDIS_URL_ENV = "ROBOALGO_REDIS_URL"
_ALLOW_NO_REDIS_ENV = "ROBOALGO_AUTOTRADER_ALLOW_NO_REDIS"
_DEFAULT_REDIS_URL = "redis://localhost:6379/0"

_NAMESPACE = "roboalgo:auto_trader:"
_KEY_KILL_SWITCH = _NAMESPACE + "kill_switch"
_KEY_KILL_SWITCH_META = _NAMESPACE + "kill_switch_meta"
_KEY_RUNNING = _NAMESPACE + "running"
_KEY_CYCLE_LOCK = _NAMESPACE + "cycle_lock"
_KEY_HEARTBEAT = _NAMESPACE + "heartbeat"

# Atomically delete the cycle lock only if it still holds our value — avoids
# releasing a lock that another cycle has since acquired.
_RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

# Cached client instance. None means "redis package unavailable" — this is
# only re-checked once per process since the package can't appear at
# runtime. A live connection failure (server down) is NOT cached here; each
# call attempts the command fresh so we recover automatically once redis
# comes back up.
_client = None
_client_init_attempted = False

# Fallback store — only ever touched when ROBOALGO_AUTOTRADER_ALLOW_NO_REDIS=1.
#
# This is FILE-backed, not process-local, and that is load-bearing. The control
# API and the Celery worker are separate processes: with a process-local dict,
# an operator hitting KILL in the API would flip a flag the worker never reads,
# so the UI would report a working kill switch while the worker kept trading.
# A file on the host at least makes the switch real for a single-host install.
#
# It is still NOT correct across multiple hosts — that is what Redis is for.
# `loop.py` additionally refuses to place non-dry-run orders while
# `redis_available()` is False, so this path cannot quietly arm live trading.
_FALLBACK_PATH = os.path.join(tempfile.gettempdir(), "roboalgo_auto_trader_state.json")


class _FileStore:
    """Dict-like façade over a JSON file, so every call site stays unchanged.

    Reads hit the file each time (another process may have written it); writes
    are atomic via tmp+replace. Any I/O error degrades to an empty store rather
    than raising — a state-file problem must never break a trading cycle.
    """

    def _read(self) -> Dict[str, Any]:
        try:
            with open(_FALLBACK_PATH, "r") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, ValueError, OSError):
            return {}

    def _write(self, data: Dict[str, Any]) -> None:
        try:
            tmp = f"{_FALLBACK_PATH}.tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, _FALLBACK_PATH)
        except OSError as exc:
            logger.error("auto_trader.state: fallback store write failed: %s", exc)

    def get(self, key: str, default: Any = None) -> Any:
        return self._read().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._read()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        data = self._read()
        data[key] = value
        self._write(data)

    def pop(self, key: str, default: Any = None) -> Any:
        data = self._read()
        value = data.pop(key, default)
        self._write(data)
        return value


_local_store = _FileStore()


def _allow_no_redis() -> bool:
    return getenv(_ALLOW_NO_REDIS_ENV) == "1"


def _redis_url() -> str:
    return getenv(_REDIS_URL_ENV, _DEFAULT_REDIS_URL)


def _get_client():
    """Lazily import redis and build a client.

    Returns None if the `redis` package is not installed. Construction alone
    does not prove the server is reachable — callers must handle connection
    errors when they actually issue a command.
    """
    global _client, _client_init_attempted
    if _client_init_attempted:
        return _client
    _client_init_attempted = True
    try:
        import redis  # lazy import — redis may not be installed
    except ImportError:
        logger.warning(
            "auto_trader.state: 'redis' package not installed; state will "
            "fail-safe (kill switch ON) unless %s=1",
            _ALLOW_NO_REDIS_ENV,
        )
        _client = None
        return None
    try:
        _client = redis.Redis.from_url(
            _redis_url(),
            socket_connect_timeout=1,
            socket_timeout=1,
            decode_responses=True,
        )
    except Exception as exc:  # pragma: no cover - defensive, from_url rarely raises
        logger.warning("auto_trader.state: failed to construct redis client: %s", exc)
        _client = None
    return _client


def redis_available() -> bool:
    """True if a live redis connection can be established right now."""
    client = _get_client()
    if client is None:
        return False
    try:
        return bool(client.ping())
    except Exception as exc:
        logger.warning("auto_trader.state: redis ping failed: %s", exc)
        return False


def _no_redis_warning(op: str) -> None:
    logger.warning(
        "auto_trader.state: %s operating without redis (%s=1); using "
        "process-local state",
        op,
        _ALLOW_NO_REDIS_ENV,
    )


def kill_switch_on() -> bool:
    client = _get_client()
    if client is not None:
        try:
            value = client.get(_KEY_KILL_SWITCH)
            return value == "1"
        except Exception as exc:
            logger.warning("auto_trader.state: kill_switch_on redis error: %s", exc)

    if _allow_no_redis():
        _no_redis_warning("kill_switch_on")
        return bool(_local_store.get(_KEY_KILL_SWITCH, False))

    logger.warning(
        "auto_trader.state: redis unavailable and %s not set; fail-safe "
        "kill switch engaged (assuming ON)",
        _ALLOW_NO_REDIS_ENV,
    )
    return True


def set_kill_switch(on: bool, reason: str = "", actor: str = "api") -> None:
    meta = {"on": bool(on), "reason": reason, "actor": actor, "ts": time.time()}
    client = _get_client()
    if client is not None:
        try:
            client.set(_KEY_KILL_SWITCH, "1" if on else "0")
            client.set(_KEY_KILL_SWITCH_META, json.dumps(meta))
            return
        except Exception as exc:
            logger.warning("auto_trader.state: set_kill_switch redis error: %s", exc)

    if _allow_no_redis():
        _no_redis_warning("set_kill_switch")
        _local_store[_KEY_KILL_SWITCH] = bool(on)
        _local_store[_KEY_KILL_SWITCH_META] = meta
        return

    logger.warning(
        "auto_trader.state: set_kill_switch dropped — redis unavailable and "
        "%s not set",
        _ALLOW_NO_REDIS_ENV,
    )


def running_enabled() -> bool:
    client = _get_client()
    if client is not None:
        try:
            value = client.get(_KEY_RUNNING)
            return value == "1"
        except Exception as exc:
            logger.warning("auto_trader.state: running_enabled redis error: %s", exc)

    if _allow_no_redis():
        _no_redis_warning("running_enabled")
        return bool(_local_store.get(_KEY_RUNNING, False))

    logger.warning(
        "auto_trader.state: redis unavailable and %s not set; assuming "
        "running=False (fail-safe)",
        _ALLOW_NO_REDIS_ENV,
    )
    return False


def set_running(on: bool, actor: str = "api") -> None:
    client = _get_client()
    if client is not None:
        try:
            client.set(_KEY_RUNNING, "1" if on else "0")
            return
        except Exception as exc:
            logger.warning("auto_trader.state: set_running redis error: %s", exc)

    if _allow_no_redis():
        _no_redis_warning("set_running")
        _local_store[_KEY_RUNNING] = bool(on)
        return

    logger.warning(
        "auto_trader.state: set_running(actor=%s) dropped — redis "
        "unavailable and %s not set",
        actor,
        _ALLOW_NO_REDIS_ENV,
    )


def acquire_cycle_lock(cycle_id: str, ttl: int = 900) -> bool:
    client = _get_client()
    if client is not None:
        try:
            return bool(client.set(_KEY_CYCLE_LOCK, cycle_id, nx=True, ex=ttl))
        except Exception as exc:
            logger.warning("auto_trader.state: acquire_cycle_lock redis error: %s", exc)

    if _allow_no_redis():
        _no_redis_warning("acquire_cycle_lock")
        now = time.time()
        existing = _local_store.get(_KEY_CYCLE_LOCK)
        if existing is not None:
            _, expires_at = existing
            if expires_at > now:
                return False
        _local_store[_KEY_CYCLE_LOCK] = (cycle_id, now + ttl)
        return True

    logger.warning(
        "auto_trader.state: acquire_cycle_lock refused — redis unavailable "
        "and %s not set",
        _ALLOW_NO_REDIS_ENV,
    )
    return False


def release_cycle_lock(cycle_id: str) -> None:
    client = _get_client()
    if client is not None:
        try:
            client.eval(_RELEASE_LOCK_SCRIPT, 1, _KEY_CYCLE_LOCK, cycle_id)
            return
        except Exception as exc:
            logger.warning("auto_trader.state: release_cycle_lock redis error: %s", exc)

    if _allow_no_redis():
        _no_redis_warning("release_cycle_lock")
        existing = _local_store.get(_KEY_CYCLE_LOCK)
        if existing is not None and existing[0] == cycle_id:
            _local_store.pop(_KEY_CYCLE_LOCK, None)
        return

    logger.warning(
        "auto_trader.state: release_cycle_lock no-op — redis unavailable "
        "and %s not set",
        _ALLOW_NO_REDIS_ENV,
    )


def current_lock() -> Optional[str]:
    client = _get_client()
    if client is not None:
        try:
            return client.get(_KEY_CYCLE_LOCK)
        except Exception as exc:
            logger.warning("auto_trader.state: current_lock redis error: %s", exc)

    if _allow_no_redis():
        _no_redis_warning("current_lock")
        existing = _local_store.get(_KEY_CYCLE_LOCK)
        if existing is None:
            return None
        value, expires_at = existing
        if expires_at <= time.time():
            _local_store.pop(_KEY_CYCLE_LOCK, None)
            return None
        return value

    logger.warning(
        "auto_trader.state: current_lock unknown — redis unavailable and "
        "%s not set",
        _ALLOW_NO_REDIS_ENV,
    )
    return None


def set_heartbeat(payload: dict) -> None:
    client = _get_client()
    if client is not None:
        try:
            client.set(_KEY_HEARTBEAT, json.dumps(payload))
            return
        except Exception as exc:
            logger.warning("auto_trader.state: set_heartbeat redis error: %s", exc)

    if _allow_no_redis():
        _no_redis_warning("set_heartbeat")
        _local_store[_KEY_HEARTBEAT] = dict(payload)
        return

    logger.warning(
        "auto_trader.state: set_heartbeat dropped — redis unavailable and "
        "%s not set",
        _ALLOW_NO_REDIS_ENV,
    )


def get_kill_switch_meta() -> Optional[dict]:
    """Why the kill switch is in its current state, and who set it.

    Returns ``{"on", "reason", "actor", "ts"}`` as written by
    :func:`set_kill_switch`, or ``None`` if it has never been set or the
    state store is unreachable. Purely informational — never gate trading on
    this, gate on :func:`kill_switch_on`, which fails safe.
    """
    client = _get_client()
    if client is not None:
        try:
            value = client.get(_KEY_KILL_SWITCH_META)
            return json.loads(value) if value is not None else None
        except Exception as exc:
            logger.warning("auto_trader.state: get_kill_switch_meta redis error: %s", exc)

    if _allow_no_redis():
        _no_redis_warning("get_kill_switch_meta")
        meta = _local_store.get(_KEY_KILL_SWITCH_META)
        if isinstance(meta, str):
            try:
                return json.loads(meta)
            except ValueError:
                return None
        return meta

    return None


def get_heartbeat() -> Optional[dict]:
    client = _get_client()
    if client is not None:
        try:
            value = client.get(_KEY_HEARTBEAT)
            return json.loads(value) if value is not None else None
        except Exception as exc:
            logger.warning("auto_trader.state: get_heartbeat redis error: %s", exc)

    if _allow_no_redis():
        _no_redis_warning("get_heartbeat")
        return _local_store.get(_KEY_HEARTBEAT)

    logger.warning(
        "auto_trader.state: get_heartbeat unknown — redis unavailable and "
        "%s not set",
        _ALLOW_NO_REDIS_ENV,
    )
    return None
