"""Read-only Alpaca market-data client: daily/intraday bars, the asset list, the market clock, corporate actions.

Plain HTTP (urllib, injected `fetch`), no SDK. It places no orders and reads no account or position data:
trading stays in broker_engine/alpaca.py (alpaca-py). Nothing here touches the database or config.settings.

CONTRACT USED (Alpaca REST v2; every request is a GET except the OAuth token POST)
  data    GET https://data.alpaca.markets/v2/stocks/bars
              symbols (comma list, <= 100), timeframe, start, end (RFC 3339 UTC), adjustment, feed, limit (<= 10000),
              sort=asc, page_token, asof (UNVERIFIED live); answer {"bars": {SYM: [{t,o,h,l,c,v,n,vw}]}, "next_page_token"}
  trading GET {paper-api|api}.alpaca.markets/v2/clock
          GET {paper-api|api}.alpaca.markets/v2/assets?status=&asset_class=&exchange=      (list of assets)
          GET {paper-api|api}.alpaca.markets/v2/assets/{symbol_or_id}                      (404 -> None)
          GET {paper-api|api}.alpaca.markets/v2/corporate_actions/announcements
              ca_types, since, until, symbol (UNVERIFIED live: path and ca_types casing)

AUTH (two schemes, credentials come from the environment only)
  keys   ALPACA_API_KEY + ALPACA_SECRET_KEY (the names broker_engine/alpaca.py uses) -> headers APCA-API-KEY-ID and
         APCA-API-SECRET-KEY. Paper-account keys are enough for market data.
  oauth  ALPACA_CLIENT_ID + ALPACA_CLIENT_SECRET -> client-credentials token from authx.alpaca.markets, sent as a
         Bearer header; used only when ALPACA_API_KEY is not set. UNVERIFIED live: it may belong to Alpaca's Broker
         API rather than personal accounts.
  Secrets are never logged, never stored, never put in a URL and never left in exception text (a response body
  echoed into an error is scrubbed first). ALPACA_PAPER=false (any case) selects the live trading host for the
  read-only asset and clock calls, the broker_engine/alpaca.py rule; anything else is the paper host.

PLANS AND LIMITS
  free  200 requests a minute, IEX real time, SIP history WITHOUT the latest 15 minutes, history from 2016.
  plus  10,000 requests a minute.
  The client throttles itself (free 2.5 requests/s, plus 10/s), retries 429 and 5xx with a 1/2/4 s backoff, and
  on the free plan trims `end` to 16 minutes before now for every feed (daily bars are unaffected).

DATA NOTES
  `adjustment` is ALWAYS sent (never rely on the API default) and recorded in `df.attrs`.
  The default feed "iex" is IEX-exchange trades only (a small slice of volume, different OHLC, gaps in small caps).
  Research should pass feed="sip". Never mix IEX bars with the survivor price cache.
  Data endpoints return nothing for OTC symbols (Alpaca staff statement), hence `exchange` is kept in assets().
  Alpaca maps symbols "as of" a date (default today): the optional `asof` argument is UNVERIFIED live.
  Dates are passed as 'YYYY-MM-DD' strings: that means 00:00:00Z at the start and 23:59:59Z at the end, so the
  end day's bar (stamped 04:00 or 05:00Z) is included. A pd.Timestamp is read as a timestamp instead.

UNVERIFIED LIVE: the OAuth client-credentials scheme for these APIs, CORPORATE_ACTIONS_PATH and the ca_types
casing, the `asof` parameter.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, Mapping, Optional
from urllib.parse import quote, urlencode, urlsplit

import pandas as pd

log = logging.getLogger(__name__)

DATA_BASE = "https://data.alpaca.markets"
TRADING_PAPER = "https://paper-api.alpaca.markets"
TRADING_LIVE = "https://api.alpaca.markets"
OAUTH_TOKEN_URL = "https://authx.alpaca.markets/v1/oauth2/token"       # UNVERIFIED for the trading/data APIs
CORPORATE_ACTIONS_PATH = "/v2/corporate_actions/announcements"          # UNVERIFIED live
ENV_KEY_ID, ENV_SECRET, ENV_PAPER = "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER"
ENV_CLIENT_ID, ENV_CLIENT_SECRET = "ALPACA_CLIENT_ID", "ALPACA_CLIENT_SECRET"
ENV_NAMES = (ENV_KEY_ID, ENV_SECRET, ENV_CLIENT_ID, ENV_CLIENT_SECRET, ENV_PAPER)
PLANS = {"free": 2.5, "plus": 10.0}            # requests per second we allow ourselves (Alpaca: 200/min, 10,000/min)
BACKOFF = (1, 2, 4)
ADJUSTMENTS = ("raw", "split", "dividend", "all")
FEEDS = ("iex", "sip", "otc")
MAX_CHUNK = 100
PAGE_LIMIT = 10000
FREE_DELAY = pd.Timedelta(minutes=16)
START_OF_DAY = "T00:00:00Z"
END_OF_DAY = "T23:59:59Z"
CA_WINDOW_DAYS = 90
CA_TYPES = ("split", "dividend", "merger", "spinoff")
CA_TYPE_PARAM = {"split": "Split", "dividend": "Dividend", "merger": "Merger", "spinoff": "Spinoff"}   # UNVERIFIED casing
TOKEN_SKEW_S = 30.0
DEFAULT_EXPIRES_S = 300.0
USER_AGENT = "RoboAlgo-research"
ASSET_COLUMNS = ["id", "symbol", "name", "exchange", "status", "tradable", "class"]
BAR_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "trades", "vwap"]
BAR_FIELDS = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "trades", "vw": "vwap"}
CA_COLUMNS = ["ca_type", "symbol", "ex_date", "record_date", "payable_date", "cash", "split_from", "split_to", "id", "target_symbol"]


class AlpacaError(RuntimeError):
    """Any failure talking to Alpaca."""


class AlpacaConfigError(AlpacaError):
    """Credentials are missing or incomplete in the environment."""


class AlpacaAuthError(AlpacaError):
    """Alpaca rejected the credentials or the data subscription (HTTP 401/403, or a failed token exchange)."""


class AlpacaRateLimited(AlpacaError):
    """HTTP 429 persisted after every backoff retry."""


class AlpacaServerError(AlpacaError):
    """HTTP 5xx (or a network failure) persisted after every backoff retry."""


def key_headers(key_id: str, secret: str) -> dict:
    return {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}


def bearer_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class Credentials:
    """The scheme ('keys' | 'oauth') and two secret strings; repr shows the scheme only."""
    __slots__ = ("scheme", "_ident", "_secret")

    def __init__(self, scheme: str, ident: str, secret: str):
        self.scheme, self._ident, self._secret = scheme, ident, secret

    def __repr__(self) -> str:
        return f"Credentials(scheme={self.scheme!r})"


def credentials(env: Optional[Mapping[str, str]] = None) -> Credentials:
    env = os.environ if env is None else env
    key, secret = (env.get(ENV_KEY_ID) or "").strip(), (env.get(ENV_SECRET) or "").strip()
    cid, csec = (env.get(ENV_CLIENT_ID) or "").strip(), (env.get(ENV_CLIENT_SECRET) or "").strip()
    if key:
        if not secret:
            raise AlpacaConfigError(f"{ENV_KEY_ID} is set but {ENV_SECRET} is not")
        return Credentials("keys", key, secret)
    if cid and csec:
        return Credentials("oauth", cid, csec)
    raise AlpacaConfigError(f"no Alpaca credentials: set {ENV_KEY_ID} and {ENV_SECRET} (or {ENV_CLIENT_ID} and {ENV_CLIENT_SECRET})")


def trading_base(env: Optional[Mapping[str, str]] = None) -> str:
    """Live host only when ALPACA_PAPER is exactly 'false' (any case), the broker_engine/alpaca.py rule; read-only calls only."""
    env = os.environ if env is None else env
    return TRADING_LIVE if env.get(ENV_PAPER, "true").lower() == "false" else TRADING_PAPER


def default_fetch(method: str, url: str, headers: dict, body: Optional[bytes] = None) -> tuple[int, bytes]:
    """The one real network call: (status, body). HTTP errors come back as (code, body), not exceptions."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT, **headers}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read()


def ca_windows(since, until, max_days: int = CA_WINDOW_DAYS) -> list[tuple[str, str]]:
    """Inclusive [a, b] windows of at most max_days calendar days covering since..until ('YYYY-MM-DD')."""
    lo, hi = pd.Timestamp(str(since)[:10]), pd.Timestamp(str(until)[:10])
    if lo > hi:
        raise ValueError("since is after until")
    out, a = [], lo
    while a <= hi:
        b = min(a + pd.Timedelta(days=max_days - 1), hi)
        out.append((a.date().isoformat(), b.date().isoformat()))
        a = b + pd.Timedelta(days=1)
    return out


def _symbols(symbols) -> list[str]:
    items = [symbols] if isinstance(symbols, str) else list(symbols)
    out = list(dict.fromkeys(str(s).strip().upper() for s in items))
    if not out or any((not s) or ("," in s) for s in out):
        raise ValueError("symbols must be a non-empty list of tickers without commas")
    return out


def _date_param(value, end: bool) -> pd.Timestamp:
    """'YYYY-MM-DD' -> 00:00:00Z (start) or 23:59:59Z (end, so that day's bar, stamped 04:00/05:00Z, is included).
    Longer strings / Timestamps are parsed as timestamps; tz-naive means UTC. Pass dates as strings."""
    s = str(value).strip()
    if len(s) == 10:
        return pd.Timestamp(s + (END_OF_DAY if end else START_OF_DAY))
    t = pd.Timestamp(s)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _rfc3339(t: pd.Timestamp) -> str:
    return t.tz_convert("UTC").floor("s").strftime("%Y-%m-%dT%H:%M:%SZ")


def _bars_frame(records: list, timeframe: str, adjustment: str, feed: str) -> pd.DataFrame:
    if not records:
        df = pd.DataFrame({c: pd.Series(dtype="datetime64[ns]" if c == "date" else ("object" if c == "symbol" else "float64")) for c in BAR_COLUMNS})
    else:
        raw = pd.DataFrame(records)
        t = pd.to_datetime(raw["t"], utc=True).dt.tz_localize(None)
        df = pd.DataFrame({"symbol": raw["symbol"].astype(str), "date": t.dt.normalize() if timeframe == "1Day" else t})
        for c in BAR_COLUMNS[2:]:
            df[c] = pd.to_numeric(raw[c], errors="coerce").astype(float)
        df = df.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    df.attrs.update({"adjustment": adjustment, "feed": feed, "timeframe": timeframe})
    return df


def _ca_frame(rows: list) -> pd.DataFrame:
    out = pd.DataFrame([{"ca_type": r.get("ca_type"),
                         "symbol": r.get("initiating_symbol") or r.get("target_symbol") or r.get("symbol"),
                         "ex_date": r.get("ex_date"), "record_date": r.get("record_date"), "payable_date": r.get("payable_date"),
                         "cash": r.get("cash"), "split_from": r.get("old_rate"), "split_to": r.get("new_rate"),
                         "id": r.get("id"), "target_symbol": r.get("target_symbol")} for r in rows], columns=CA_COLUMNS)
    for c in ("ex_date", "record_date", "payable_date"):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    for c in ("cash", "split_from", "split_to"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    out = out[~(out["id"].notna() & out["id"].duplicated(keep="first"))]
    return out.sort_values(["ex_date", "symbol"], kind="mergesort").reset_index(drop=True)


class AlpacaClient:
    """Read-only Alpaca data client: throttled, retrying, authenticated; it never stores credentials on disk."""

    def __init__(self, plan: str = "free", fetch: Optional[Callable] = None, now: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, utcnow: Optional[Callable[[], pd.Timestamp]] = None,
                 env: Optional[Mapping[str, str]] = None, rate: Optional[float] = None):
        if plan not in PLANS:
            raise ValueError(f"plan must be one of {sorted(PLANS)}")
        env = os.environ if env is None else env
        self._creds = credentials(env)
        self.plan = plan
        self.trading_base = trading_base(env)
        self.data_base = DATA_BASE
        self.rate = float(PLANS[plan] if rate is None else rate)
        if self.rate <= 0:
            raise ValueError("rate must be positive")
        self._min_interval = 1.0 / self.rate
        self._fetch = fetch or default_fetch
        self._now, self._sleep = now, sleep
        self._utcnow = utcnow or (lambda: pd.Timestamp.now(tz="UTC"))
        self._last = None
        self._token, self._token_expiry = None, 0.0
        self.requests = 0            # GET attempts, retries included
        self.token_requests = 0      # OAuth token POSTs

    def __repr__(self) -> str:
        return f"AlpacaClient(plan={self.plan!r}, scheme={self._creds.scheme!r}, trading_base={self.trading_base!r})"

    def _scrub(self, text: str) -> str:
        for s in (self._creds._secret, self._creds._ident, self._token):
            if s:
                text = text.replace(s, "***")
        return text

    def _throttle(self) -> None:
        t = self._now()
        if self._last is not None and t < self._last + self._min_interval:
            self._sleep(self._last + self._min_interval - t)
            t = self._now()
        self._last = t

    def _bearer(self) -> str:
        t = self._now()
        if self._token is not None and t < self._token_expiry:
            return self._token
        body = urlencode({"grant_type": "client_credentials", "client_id": self._creds._ident,
                          "client_secret": self._creds._secret}).encode()
        try:
            status, raw = self._fetch("POST", OAUTH_TOKEN_URL, {"Content-Type": "application/x-www-form-urlencoded"}, body)
        except OSError:
            status, raw = 599, b""
        self.token_requests += 1
        log.debug("POST %s -> HTTP %s", urlsplit(OAUTH_TOKEN_URL).path, status)
        if status != 200:
            raise AlpacaAuthError(f"Alpaca OAuth token request answered HTTP {status}: check {ENV_CLIENT_ID} and {ENV_CLIENT_SECRET}")
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError:
            data = {}
        token = data.get("access_token") if isinstance(data, dict) else None
        if not token:
            raise AlpacaAuthError("Alpaca OAuth token response has no access_token")
        expires = float(data.get("expires_in") or DEFAULT_EXPIRES_S)
        self._token, self._token_expiry = str(token), t + max(expires - TOKEN_SKEW_S, 0.0)
        return self._token

    def _auth_headers(self) -> dict:
        if self._creds.scheme == "keys":
            return key_headers(self._creds._ident, self._creds._secret)
        return bearer_headers(self._bearer())

    def _get(self, url: str, allow_404: bool = False):
        path = urlsplit(url).path
        attempt, refreshed = 0, False
        while True:
            self._throttle()
            try:
                status, body = self._fetch("GET", url, self._auth_headers(), None)
            except OSError:
                status, body = 599, b""
            self.requests += 1
            log.debug("GET %s -> HTTP %s", path, status)
            if status == 200:
                try:
                    return json.loads(body.decode("utf-8")) if body else None
                except ValueError:
                    raise AlpacaError(f"Alpaca sent a non-JSON body for {path}") from None
            if status == 401 and self._creds.scheme == "oauth" and not refreshed:
                refreshed, self._token = True, None
                continue
            if status in (401, 403):
                raise AlpacaAuthError(f"Alpaca answered HTTP {status} for {path}: check the credentials and the data subscription")
            if status == 404 and allow_404:
                return None
            if status == 429 or status >= 500:
                if attempt >= len(BACKOFF):
                    cls = AlpacaRateLimited if status == 429 else AlpacaServerError
                    raise cls(f"Alpaca answered HTTP {status} for {path} after {len(BACKOFF)} retries")
                self._sleep(BACKOFF[attempt])
                attempt += 1
                continue
            snippet = self._scrub((body or b"")[:300].decode("utf-8", "replace"))
            raise AlpacaError(f"Alpaca answered HTTP {status} for {path}: {snippet}")

    def clock(self) -> dict:
        """The market clock (is_open, next_open, next_close, timestamp)."""
        return self._get(f"{self.trading_base}/v2/clock") or {}

    def asset(self, symbol_or_id: str) -> Optional[dict]:
        """One asset; None when Alpaca answers 404."""
        return self._get(f"{self.trading_base}/v2/assets/{quote(str(symbol_or_id).strip(), safe='')}", allow_404=True)

    def assets(self, status: Optional[str] = "inactive", asset_class: str = "us_equity", exchange: Optional[str] = None) -> pd.DataFrame:
        """Assets as a frame with ASSET_COLUMNS (exchange kept so OTC rows can be told apart); status None means all."""
        if status not in (None, "active", "inactive"):
            raise ValueError("status must be None, 'active' or 'inactive'")
        params = [] if status is None else [("status", status)]
        params.append(("asset_class", asset_class))
        if exchange is not None:
            params.append(("exchange", exchange))
        data = self._get(f"{self.trading_base}/v2/assets?{urlencode(params)}") or []
        return pd.DataFrame([{c: r.get(c) for c in ASSET_COLUMNS} for r in data if isinstance(r, dict)], columns=ASSET_COLUMNS)

    def bars(self, symbols, start, end, timeframe: str = "1Day", adjustment: str = "split", feed: str = "iex",
             chunk: int = MAX_CHUNK, asof: Optional[str] = None) -> pd.DataFrame:
        """Bars for one or more symbols as a frame with BAR_COLUMNS (daily bars are stamped at midnight, naive UTC date).

        `adjustment` is always sent and recorded in df.attrs. On the free plan `end` is trimmed to 16 minutes before now.
        Symbols are sent in chunks of at most `chunk` (<= 100); pages are followed until the token runs out.
        """
        if adjustment not in ADJUSTMENTS:
            raise ValueError(f"adjustment must be one of {ADJUSTMENTS}")
        if feed not in FEEDS:
            raise ValueError(f"feed must be one of {FEEDS}")
        if not 1 <= int(chunk) <= MAX_CHUNK:
            raise ValueError(f"chunk must be 1..{MAX_CHUNK}")
        syms = _symbols(symbols)
        start_ts, end_ts = _date_param(start, False), _date_param(end, True)
        if self.plan == "free":
            end_ts = min(end_ts, self._utcnow() - FREE_DELAY)
        if start_ts > end_ts:
            raise ValueError("start is after end")
        records = []
        for k in range(0, len(syms), int(chunk)):
            group = syms[k:k + int(chunk)]
            token = None
            while True:
                params = [("symbols", ",".join(group)), ("timeframe", timeframe), ("start", _rfc3339(start_ts)),
                          ("end", _rfc3339(end_ts)),
                          ("adjustment", adjustment),
                          ("feed", feed), ("limit", str(PAGE_LIMIT)), ("sort", "asc")]
                if asof is not None:
                    params.append(("asof", str(asof)))
                if token is not None:
                    params.append(("page_token", token))
                data = self._get(f"{self.data_base}/v2/stocks/bars?{urlencode(params, safe=',:')}")
                for sym, rows in ((data or {}).get("bars") or {}).items():
                    for r in rows or []:
                        records.append({"symbol": sym, "t": r.get("t"), **{col: r.get(f) for f, col in BAR_FIELDS.items()}})
                prev = token
                token = (data or {}).get("next_page_token") or None
                if token is None:
                    break
                if token == prev:
                    raise AlpacaError("Alpaca repeated a page token; stopping")
        return _bars_frame(records, timeframe, adjustment, feed)

    def corporate_actions(self, since, until, symbols=None, types=CA_TYPES) -> pd.DataFrame:
        """Corporate-action announcements as a frame with CA_COLUMNS.

        UNVERIFIED live (path and ca_types casing). Ranges longer than 90 days are split into 90-day windows (the
        endpoint's documented limit). Announcements exist only since April 2020 and do NOT include delistings or
        symbol changes. `symbol` is the initiating symbol (for a merger, the acquirer), `target_symbol` the target.
        One request per symbol per window.
        """
        types = tuple(types)
        if not types or any(t not in CA_TYPE_PARAM for t in types):
            raise ValueError(f"types must be a non-empty subset of {tuple(CA_TYPE_PARAM)}")
        syms = [None] if symbols is None else _symbols(symbols)
        windows = ca_windows(since, until)
        ca_types = ",".join(CA_TYPE_PARAM[t] for t in types)
        rows = []
        for sym in syms:
            for a, b in windows:
                params = [("ca_types", ca_types), ("since", a), ("until", b)]
                if sym is not None:
                    params.append(("symbol", sym))
                data = self._get(f"{self.trading_base}{CORPORATE_ACTIONS_PATH}?{urlencode(params, safe=',')}") or []
                rows.extend(r for r in data if isinstance(r, dict))
        return _ca_frame(rows)
