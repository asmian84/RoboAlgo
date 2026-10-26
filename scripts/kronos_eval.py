"""
Pre-registered test of the open-source Kronos candlestick foundation model on US stocks.

    ~/.roboalgo/kronos-venv/bin/python scripts/kronos_eval.py          # full run, writes ~/.roboalgo/reports/
    ~/.roboalgo/kronos-venv/bin/python scripts/kronos_eval.py --smoke 20   # 20 stocks, 1 clean origin, timing
    ~/.roboalgo/kronos-venv/bin/python scripts/kronos_eval.py --device cpu --sample-count 1

RUN IT WITH THE KRONOS VENV. torch lives in ~/.roboalgo/kronos-venv, never in the project venv, so this
script (and only this script) runs under that interpreter. The tests use the project venv and a fake.

WHAT IT DOES. Loads Kronos-small and Kronos-Tokenizer-base (zero-shot, no fine-tuning) from a local clone
of github.com/shiyu-coder/Kronos, gives each model 400 daily bars ending at a month-end origin and asks for
20 future candles. The forecast return is the mean over 5 sample paths of the predicted close at day 20
divided by the last observed close, minus 1. It then measures the cross-sectional Spearman rank IC between
that forecast and the realized 20-day return over the top 500 US stocks by 60-day dollar volume, and
compares it with 12-1 momentum, 1-month reversal and a shuffled-rank placebo on the same names and dates.

PRE-REGISTRATION. The origins, universe, horizon, sampling parameters, baselines, clustering and verdict
rule are fixed in setup_engine/kronos_eval.py and were not tuned after any result was seen. Clean window:
origins from 2025-07-31 (Kronos's weights went public 2025-06-30 and its pretraining range is NOT
disclosed, so earlier US data may have been seen in training). Informational window: month ends 2023-01 to
2024-12, reported separately and never part of the verdict.

VERDICT. PASS only if the 95% lower bound of the Kronos mean IC is above 0 in the CLEAN window AND the 95%
lower bound of (Kronos IC - momentum IC) is above 0. UNDERPOWERED is printed when the clean window holds
fewer than 12 origins. Otherwise: no demonstrated edge.

COSTS. Decile spreads are gross. No transaction costs, no slippage, no borrow. A monthly top-decile
strategy would pay all of them.

BIASES. Survivors only: the price cache is today's listings, so delisted stocks are missing; over a
15-month window that bias is small but it flatters every mean and every IC slightly. About 14 clean
origins means every interval here is wide. Yahoo's split adjustment is retroactive and there is no
dividend adjustment.

WHAT IS NOT VERIFIED. Zero-shot only. This says nothing about a fine-tuned Kronos, another horizon,
another lookback, or another market. The sample is US large caps, where candle patterns are the most
arbitraged away.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from setup_engine import kronos_eval as K
from setup_engine import mtf
from setup_engine import trade_stats as TS
from setup_engine.screen import clean_daily

# ── fixed locations and run choices ───────────────────────────────────────────────────────────────

DEFAULT_HISTORY = mtf.DEFAULT_HISTORY
DEFAULT_KRONOS = Path.home() / ".roboalgo" / "vendor" / "Kronos"
OUT_DIR = Path.home() / ".roboalgo" / "reports"
KRONOS_CACHE = Path.home() / ".roboalgo" / "kronos"
FORECAST_CACHE = KRONOS_CACHE / "forecasts.parquet"
MODEL_REPO = "NeoQuasar/Kronos-small"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
MAX_CONTEXT = 512
CLIP = 5.0
CHUNK = 32                      # series per predict_batch call (memory, not accuracy)
SAMPLE_SEED = 20261020          # model sampling only; set so a rerun reproduces the same forecast
OHLCV = ("open", "high", "low", "close", "volume")
MENU = ("kronos", "momentum", "reversal", "placebo")


# ── small helpers ────────────────────────────────────────────────────────────────────────────────

def _jsonable(o):
    """Anything -> JSON types with no NaN/Infinity (strict parsers reject those)."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o).date())
    if isinstance(o, Path):
        return str(o)
    return o


def _batch_files(history) -> list[Path]:
    return sorted(Path(history).expanduser().glob("batch_*.parquet"))


def read_calendar(history) -> pd.DatetimeIndex:
    """Every trading day present in the cache (the union over batches of unique dates)."""
    import pyarrow.parquet as pq

    parts = []
    for f in _batch_files(history):
        d = pd.to_datetime(pq.read_table(f, columns=["date"]).to_pandas()["date"])
        parts.append(np.unique(d.values.astype("datetime64[ns]")))
    if not parts:
        raise ValueError(f"no batch files under {history}")
    return K.trading_days(np.concatenate(parts))


def data_end(history) -> pd.Timestamp:
    """The last date, read from the date column only."""
    import pyarrow.parquet as pq

    return pd.Timestamp(max(pd.to_datetime(pq.read_table(f, columns=["date"]).to_pandas()["date"]).max()
                            for f in _batch_files(history)))


def resolve_device(choice: str) -> str:
    """auto -> mps when available, else cpu. Never imports torch for 'cpu' fallback logic here."""
    if choice != "auto":
        return choice
    import torch

    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── pass 1: per-symbol, per-origin universe inputs and baselines ─────────────────────────────────

def _stats_path(data_end_ts, origins) -> Path:
    tag = f"{pd.Timestamp(data_end_ts).date()}_{len(origins)}"
    return KRONOS_CACHE / f"universe_stats_{tag}.parquet"


def collect_stats(history, origins, refresh: bool = False) -> pd.DataFrame:
    """Per (symbol, origin): bars, last close, 60-day dollar volume, 12-1 momentum, 1-month reversal.

    Only bars up to and including each origin are read, so nothing from the future can enter.
    Columns: symbol, origin, n_bars, last_close, dvol, mom, rev.
    """
    origins = pd.DatetimeIndex(origins).sort_values()
    path = _stats_path(data_end(history), origins)
    if path.exists() and not refresh:
        print(f"  reusing {path.name}")
        return pd.read_parquet(path)

    t0 = time.time()
    files = _batch_files(history)
    parts = []
    for k, f in enumerate(files, 1):
        raw = pd.read_parquet(f)
        raw["date"] = pd.to_datetime(raw["date"])
        for sym, g in raw.groupby("symbol", sort=False):
            d = clean_daily(g.set_index("date")[list(OHLCV)].astype("float64"))
            s = K.symbol_stats(d, origins)
            if len(s) == 0:
                continue
            s = s.assign(symbol=sym,
                         mom=[K.momentum(d, t) for t in s["origin"]],
                         rev=[K.reversal(d, t) for t in s["origin"]])
            parts.append(s)
        print(f"  stats {k}/{len(files)} batches ({time.time() - t0:.0f}s)", flush=True)
    out = pd.concat(parts, ignore_index=True)[["symbol", "origin", "n_bars", "last_close", "dvol", "mom", "rev"]]
    KRONOS_CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f"  {len(out):,} symbol-origin rows -> {path}")
    return out


def universes_by_origin(stats: pd.DataFrame, origins) -> dict:
    """{origin: [symbols]} the top-500 list at each origin, from the pass-1 table only."""
    out = {}
    for t in origins:
        g = stats[stats["origin"] == pd.Timestamp(t)]
        out[pd.Timestamp(t)] = K.select_universe(g, K.TOP_N)
    return out


# ── the model ───────────────────────────────────────────────────────────────────────────────────

def load_predictor(kronos_dir: Path, device: str):
    """Kronos-small + tokenizer, weights from safetensors only. torch is imported HERE, not at module load."""
    import torch
    from safetensors.torch import load_file

    kronos_dir = Path(kronos_dir).expanduser()
    if not (kronos_dir / "model" / "kronos.py").exists():
        raise FileNotFoundError(f"no Kronos clone at {kronos_dir} (expected model/kronos.py)")
    sys.path.insert(0, str(kronos_dir))
    from model.kronos import Kronos, KronosPredictor, KronosTokenizer     # noqa: E402  (path set just above)

    def load(cls, repo):
        from huggingface_hub import snapshot_download

        local = Path(snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"]))
        cfg = json.loads((local / "config.json").read_text())
        net = cls(**cfg)
        state = load_file(str(local / "model.safetensors"), device="cpu")   # safetensors: no pickle
        net.load_state_dict(state, strict=True)
        return net.eval()

    torch.manual_seed(SAMPLE_SEED)
    model = load(Kronos, MODEL_REPO).to(device)
    tokenizer = load(KronosTokenizer, TOKENIZER_REPO).to(device)
    pred = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT, clip=CLIP)
    return pred, {"model": MODEL_REPO, "tokenizer": TOKENIZER_REPO, "device": device,
                  "max_context": MAX_CONTEXT, "clip": CLIP}


def model_commit(kronos_dir: Path) -> dict:
    """The exact Kronos commit, read from the clone's git metadata without shelling out to git."""
    import subprocess

    try:
        sha = subprocess.run(["git", "-C", str(kronos_dir), "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        when = subprocess.run(["git", "-C", str(kronos_dir), "log", "-1", "--format=%ad", "--date=short"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha, when = "unknown", "unknown"
    return {"repo": "github.com/shiyu-coder/Kronos", "commit": sha, "commit_date": when}


def forecast_inputs(frame: pd.DataFrame, origin, lookback: int = K.LOOKBACK, horizon: int = K.HORIZON):
    """(df of the last ``lookback`` bars up to the origin, their timestamps, the next ``horizon`` business days)."""
    origin = pd.Timestamp(origin)
    d = frame[frame.index <= origin].tail(lookback)
    if len(d) < lookback:
        return None, None, None
    # a Series, not an Index: Kronos's calc_time_stamps calls .dt on it
    x_ts = pd.Series(pd.DatetimeIndex(d.index))
    y_ts = pd.Series(pd.bdate_range(origin, periods=horizon + 1)[1:])
    return d[list(OHLCV)].astype("float64"), x_ts, y_ts


def load_forecast_cache(sample_count: int, tag: str) -> pd.DataFrame:
    """Cached forecasts for the same model, commit and sampling configuration; anything else is recomputed."""
    cols = ["symbol", "origin", "forecast", "sample_count", "lookback", "horizon", "model"]
    if not FORECAST_CACHE.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(FORECAST_CACHE)
    df = df[(df["sample_count"] == sample_count) & (df["lookback"] == K.LOOKBACK)
            & (df["horizon"] == K.HORIZON) & (df["model"] == tag)]
    return df[cols].reset_index(drop=True)


def save_forecast_cache(df: pd.DataFrame) -> None:
    """Atomic write: a crash mid-run never leaves a half-written cache."""
    KRONOS_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = FORECAST_CACHE.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, FORECAST_CACHE)


def forecast_all(frames: dict, universes: dict, predictor_factory, sample_count: int, tag: str,
                 verbose: bool = True) -> pd.DataFrame:
    """Forecast every (symbol, origin) in the universes, reusing the cache. Returns the cache frame.

    ``predictor_factory`` is called only when at least one forecast is missing, so a fully cached rerun
    never loads torch or the weights.
    """
    cache = load_forecast_cache(sample_count, tag)
    have = {(r.symbol, pd.Timestamp(r.origin)) for r in cache.itertuples()}
    todo = {t: [s for s in syms if (s, pd.Timestamp(t)) not in have] for t, syms in universes.items()}
    todo = {t: s for t, s in todo.items() if s}
    print(f"  {len(have):,} forecasts cached, {sum(len(v) for v in todo.values()):,} to compute")
    if not todo:
        return cache

    predictor = predictor_factory()
    t0 = time.time()
    done = 0
    total = sum(len(v) for v in todo.values())
    rows = []
    for k, (origin, syms) in enumerate(sorted(todo.items()), 1):
        chunks = [syms[i:i + CHUNK] for i in range(0, len(syms), CHUNK)]
        origin_rows = []
        for ci, chunk in enumerate(chunks, 1):
            df_list, x_list, y_list, kept = [], [], [], []
            for sym in chunk:
                d, x_ts, y_ts = forecast_inputs(frames[sym], origin)
                if d is None:
                    continue
                df_list.append(d)
                x_list.append(x_ts)
                y_list.append(y_ts)
                kept.append(sym)
            if not df_list:
                continue
            preds = predictor.predict_batch(df_list, x_list, y_list, pred_len=K.HORIZON, T=K.TEMPERATURE,
                                            top_p=K.TOP_P, sample_count=sample_count, verbose=False)
            for sym, p, d in zip(kept, preds, df_list):
                fc = K.forecast_return(float(p["close"].iloc[-1]), float(d["close"].iloc[-1]))
                origin_rows.append({"symbol": sym, "origin": pd.Timestamp(origin), "forecast": fc,
                                    "sample_count": sample_count, "lookback": K.LOOKBACK,
                                    "horizon": K.HORIZON, "model": tag})
            done += len(kept)
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done else float("nan")
            if verbose:
                print(f"  [{k}/{len(todo)}] {pd.Timestamp(origin).date()} chunk {ci}/{len(chunks)} "
                      f"{done}/{total} names, {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)
        rows.extend(origin_rows)
        if origin_rows:
            cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["symbol", "origin"], keep="last")
            rows = []
            save_forecast_cache(cache)
    print(f"  forecasts done in {time.time() - t0:.0f}s")
    return cache


# ── metrics ─────────────────────────────────────────────────────────────────────────────────────

def evaluate_window(stats: pd.DataFrame, forecasts: pd.DataFrame, frames: dict, universes: dict,
                    origins) -> dict:
    """Per-origin IC and spreads for Kronos and the three baselines, then the clustered means."""
    fc = {(r.symbol, pd.Timestamp(r.origin)): float(r.forecast) for r in forecasts.itertuples()}
    mom = {(r.symbol, pd.Timestamp(r.origin)): r.mom for r in stats.itertuples()}
    rev = {(r.symbol, pd.Timestamp(r.origin)): r.rev for r in stats.itertuples()}

    per_origin = []
    for i, t in enumerate(origins):
        t = pd.Timestamp(t)
        members = [s for s in universes[t] if s in frames]
        realized = [K.realized_return(frames[s], t) for s in members]
        kf = [fc.get((s, t), np.nan) for s in members]
        mf = [mom.get((s, t), np.nan) for s in members]
        rf = [rev.get((s, t), np.nan) for s in members]
        pf = K.placebo_forecast(kf, seed=K.PLACEBO_SEED + i)
        fin = np.isfinite(np.asarray(realized, dtype=float))
        row = {"origin": t, "n": int(len(members)),
               "n_realized": int(fin.sum()),
               "median_forecast": float(np.median(np.asarray(kf, dtype=float)[np.isfinite(np.asarray(kf, dtype=float))]))
               if np.isfinite(np.asarray(kf, dtype=float)).any() else None,
               "median_realized": float(np.median(np.asarray(realized, dtype=float)[fin])) if fin.any() else None}
        for name, series in (("kronos", kf), ("momentum", mf), ("reversal", rf), ("placebo", pf)):
            m = K.per_origin(series, realized)
            row[f"ic_{name}"] = m["ic"]
            row[f"spread_{name}"] = m["spread"]
            row[f"top_minus_universe_{name}"] = m["top_minus_universe"]
            if name == "kronos":
                row["n_ic"] = m["n"]
                row["k"] = m["k"]
                row["top_kronos"] = m["top"]
                row["bottom_kronos"] = m["bottom"]
        per_origin.append(row)

    def pairs(key):
        xs, cl = [], []
        for r in per_origin:
            v = r.get(key)
            if v is not None and np.isfinite(v):
                xs.append(float(v))
                cl.append(int(TS.month_key(r["origin"])[0]))
        return xs, cl

    signals = {}
    for name in MENU:
        xs, cl = pairs(f"ic_{name}")
        names = [r["n_ic"] for r in per_origin if r.get("n_ic")]
        signals[name] = {**K.cluster_summary(xs, cl), "n_origins": len(xs),
                         "mean_names": float(np.mean(names)) if names else None}
    spreads = {}
    for name in MENU:
        xs, cl = pairs(f"spread_{name}")
        spreads[name] = {**K.cluster_summary(xs, cl), "n_origins": len(xs)}
    top_minus = {}
    for name in MENU:
        xs, cl = pairs(f"top_minus_universe_{name}")
        top_minus[name] = {**K.cluster_summary(xs, cl), "n_origins": len(xs)}

    ka, ca = pairs("ic_kronos")
    mb, cb = pairs("ic_momentum")
    diff = K.diff_summary(ka, ca, mb, cb)
    return {"n_origins": len(per_origin), "per_origin": per_origin, "signals": signals,
            "spreads": spreads, "top_minus_universe": top_minus,
            "kronos_minus_momentum": diff}


# ── output ──────────────────────────────────────────────────────────────────────────────────────

def _d(value) -> str:
    """A date as YYYY-MM-DD."""
    return str(pd.Timestamp(value).date())


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f}%"


def _fmt_ci(block, pct: bool = False) -> str:
    """'0.021 [-0.004, 0.047]' from a cluster_bootstrap dict."""
    def f(v):
        if v is None:
            return "n/a"
        return f"{v * 100:+.2f}%" if pct else f"{v:+.4f}"
    return f"{f(block.get('mean'))} [{f(block.get('lo'))}, {f(block.get('hi'))}]"


def render_markdown(result: dict) -> str:
    c, info = result["clean"], result["informational"]
    u = result["universe"]
    v = result["verdict"]
    lines = [
        f"# Kronos on US stocks: {result['generated']}",
        "",
        "A pre-registered, zero-shot test of the Kronos candlestick foundation model. Backend research only;",
        "nothing here is wired into the app, and no strategy changed as a result.",
        "",
        f"**Verdict: {v['verdict']}.**" + ("  (UNDERPOWERED: fewer than 12 clean origins.)" if v["underpowered"] else ""),
        "",
        "## Setup",
        "",
        f"* Model: `{result['kronos']['model']}` + `{result['kronos']['tokenizer']}`, zero-shot, no fine-tuning.",
        f"* Kronos commit: `{result['kronos']['commit']}` ({result['kronos']['commit_date']}).",
        f"* Weights loaded from safetensors only. Device `{result['kronos']['device']}`.",
        f"* Lookback {result['kronos']['lookback']} daily bars, horizon {result['kronos']['horizon']} trading days, "
        f"T={result['kronos']['T']}, top_p={result['kronos']['top_p']}, sample_count={result['kronos']['sample_count']}.",
        f"* Universe: top {result['kronos']['top_n']} US stocks by 60-day dollar volume at each origin, "
        f"last close >= ${result['kronos']['min_price']:.0f}, at least {result['kronos']['min_bars']} bars.",
        f"* Price cache ends {pd.Timestamp(result['data_end']).date()}.",
        f"* Clean origins: {len(c['per_origin'])} ({_d(result['origins']['clean'][0]) if result['origins']['clean'] else 'none'}"
        f" .. {_d(result['origins']['clean'][-1]) if result['origins']['clean'] else 'none'}).",
        f"* Informational origins (may have been seen in training): {len(info['per_origin'])}"
        f" ({_d(result['origins']['informational'][0]) if result['origins']['informational'] else 'none'}"
        f" .. {_d(result['origins']['informational'][-1]) if result['origins']['informational'] else 'none'}).",
        "",
        "## Clean window (decides the verdict)",
        "",
        "| signal | mean IC [95% CI] | origins | names per origin |",
        "|---|---|---|---|",
    ]
    for name in MENU:
        s = c["signals"][name]
        lines.append(f"| {name} | {_fmt_ci(s)} | {s.get('n_origins', 0)} | {s.get('mean_names') or 0:.0f} |")
    lines += ["", f"Kronos IC minus momentum IC: {_fmt_ci(c['kronos_minus_momentum'])} "
                  f"(n = {c['kronos_minus_momentum'].get('n_a', 0)} origins).", "",
              "| signal | top-decile minus bottom-decile | top decile minus universe mean |",
              "|---|---|---|"]
    for name in MENU:
        lines.append(f"| {name} | {_fmt_ci(c['spreads'][name], pct=True)} | "
                     f"{_fmt_ci(c['top_minus_universe'][name], pct=True)} |")
    lines += [
        "",
        "Gross. No transaction costs, no slippage, no borrow. A monthly top-decile strategy would pay all of",
        "them, and with ~500 large caps the turnover of a monthly rebalance is not free.",
        "",
        "### Verdict rule, applied mechanically",
        "",
        f"* Kronos mean IC lower bound above 0: {v.get('kronos_ic_lower_bound_above_zero')}",
        f"* Kronos IC minus momentum IC lower bound above 0: {v.get('beats_momentum_lower_bound_above_zero')}",
        f"* Clean origins: {v.get('n_clean_origins')} (UNDERPOWERED below {v.get('min_clean_origins', 12)}): "
        f"{v.get('underpowered')}",
        f"* Result: **{v.get('verdict')}**",
        "",
        "## Informational window (2023-01 to 2024-12, possibly in training)",
        "",
        "| signal | mean IC | 95% CI | origins |",
        "|---|---|---|---|",
    ]
    for name in MENU:
        s = info["signals"][name]
        lines.append(f"| {name} | {_fmt_ci(s)} | {s.get('n_origins', 0)} |")
    lines += ["", f"Kronos IC minus momentum IC: {_fmt_ci(info['kronos_minus_momentum'])} "
                  f"(n = {info['kronos_minus_momentum'].get('n_a', 0)} origins).", ""]
    ct = result["contamination"]
    lines += [
        "## Contamination diagnostic (informational, not part of the verdict)",
        "",
        f"* Kronos mean IC, 2023-2024 window: {_fmt_ci(info['signals']['kronos'])}.",
        f"* Kronos mean IC, clean window: {_fmt_ci(c['signals']['kronos'])}.",
        f"* Old window minus clean window: {_fmt_ci(ct['info_minus_clean'])}.",
        f"* {ct['note']}",
        "",
        "## Per origin",
        "",
        "| window | origin | names | median forecast | median realized | IC Kronos | IC momentum | IC reversal | IC placebo | Kronos top-bottom |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, block in (("clean", c), ("info", info)):
        for r in block["per_origin"]:
            lines.append("| {w} | {o} | {n} | {mf} | {mr} | {k} | {m} | {v} | {p} | {s} |".format(
                w=label, o=_d(r["origin"]), n=r["n"],
                mf=_pct(r.get("median_forecast")), mr=_pct(r.get("median_realized")),
                k="n/a" if r["ic_kronos"] is None else f"{r['ic_kronos']:+.3f}",
                m="n/a" if r["ic_momentum"] is None else f"{r['ic_momentum']:+.3f}",
                v="n/a" if r["ic_reversal"] is None else f"{r['ic_reversal']:+.3f}",
                p="n/a" if r["ic_placebo"] is None else f"{r['ic_placebo']:+.3f}",
                s="n/a" if r["spread_kronos"] is None else f"{r['spread_kronos'] * 100:+.2f}%"))
    lines += ["", "## Caveats", ""] + [f"* {x}" for x in result["caveats"]] + [""]
    return "\n".join(lines)


def write_outputs(result: dict, base: Path) -> dict:
    """Write <base>.json (strict: no NaN) and <base>.md, each through a temp file and os.replace."""
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {"json": base.with_suffix(".json"), "md": base.with_suffix(".md")}
    payload = json.dumps(_jsonable(result), indent=1, allow_nan=False)
    for key, text in (("json", payload), ("md", render_markdown(result))):
        tmp = paths[key].with_suffix(paths[key].suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, paths[key])
    return paths


# ── main ───────────────────────────────────────────────────────────────────────────────────────

def caveats() -> list[str]:
    return [
        "Survivors only: the price cache is today's listings, so delisted stocks are absent. Over a 15-month "
        "clean window the bias is small, but every mean and every IC here is flattered a little.",
        "About 14 clean origins: each month-clustered interval rests on 14 observations, so the intervals are "
        "wide and the test is weak. A zero inside the interval is not proof of no edge either.",
        "Zero-shot only: no fine-tuning, no prompt engineering, one horizon (20 days) and one lookback (400 bars).",
        "No transaction costs, slippage or borrow. Decile spreads are gross.",
        "Yahoo's split adjustment is retroactive and there is no dividend adjustment.",
        "The model input is the 400 bars up to each origin close; nothing after the origin is read. Forecasts "
        "are sampled, not deterministic; sampling is seeded for reproducibility but a different seed would move "
        "the numbers slightly.",
        "US large caps only, the part of the market where candle patterns are most arbitraged away, so a null "
        "result here does not rule out an edge elsewhere.",
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    ap.add_argument("--kronos-dir", default=str(DEFAULT_KRONOS))
    ap.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    ap.add_argument("--sample-count", type=int, default=K.SAMPLE_COUNT)
    ap.add_argument("--out", default=None, help=f"output base path (default {OUT_DIR}/kronos_eval_<date>)")
    ap.add_argument("--smoke", type=int, nargs="?", const=20, default=None,
                    help="run N stocks at the first clean origin only, print timing, write no report")
    ap.add_argument("--refresh-stats", action="store_true", help="rebuild the universe statistics cache")
    args = ap.parse_args(argv)

    t_start = time.time()
    history = Path(args.history).expanduser()
    end = data_end(history)
    calendar = read_calendar(history)
    windows = K.origin_windows(calendar)
    print(f"cache ends {end.date()}, {len(calendar):,} trading days")
    print(f"origins: {len(windows['clean'])} clean, {len(windows['informational'])} informational, "
          f"{len(windows['immature'])} immature")

    origins = sorted(set(windows["clean"]) | set(windows["informational"]))
    if not origins:
        print("no origins with 20 forward trading days: nothing to test")
        return 1

    print("pass 1: universe inputs and baselines")
    stats = collect_stats(history, origins, refresh=args.refresh_stats)
    universes = universes_by_origin(stats, origins)
    empty = [t for t, s in universes.items() if not s]
    if empty:
        print(f"  {len(empty)} origins have an empty universe and are dropped")
    universes = {t: s for t, s in universes.items() if s}
    origins = [t for t in origins if t in universes]
    clean = [t for t in windows["clean"] if t in universes]
    info = [t for t in windows["informational"] if t in universes]

    if args.smoke:
        if not clean:
            print("no clean origin in the cache: nothing to smoke test")
            return 1
        clean = clean[:1]
        picks = universes[clean[0]][:args.smoke]
        universes = {clean[0]: picks}
        print(f"smoke: {len(picks)} stocks at {clean[0].date()}")

    members = sorted({s for syms in universes.values() for s in syms})
    print(f"pass 2: loading {len(members)} symbols of daily bars")
    frames = mtf.load_daily(history, members)
    frames = {k: v for k, v in frames.items() if len(v)}
    print(f"  {len(frames)} frames")

    meta = model_commit(Path(args.kronos_dir))
    tag = f"{MODEL_REPO}@{meta['commit'][:8]}"

    def factory():
        return load_predictor(Path(args.kronos_dir), resolve_device(args.device))[0]

    print("pass 3: forecasting")
    t_fc = time.time()
    forecasts = forecast_all(frames, universes, factory, args.sample_count, tag)
    fc_seconds = time.time() - t_fc

    if args.smoke:
        picks = set(universes[clean[0]])
        n = int(((forecasts["origin"] == pd.Timestamp(clean[0])) & forecasts["symbol"].isin(picks)).sum())
        per = fc_seconds / max(n, 1)
        total_names = len(set(windows["clean"]) | set(windows["informational"])) * K.TOP_N
        print(f"smoke done: {n} forecasts at {clean[0].date()} in {fc_seconds:.0f}s "
              f"({per:.3f}s per name)")
        print(f"projected full run: {total_names:,} forecasts, about {per * total_names / 60:.0f} minutes")
        return 0

    print("pass 4: metrics")
    res_clean = evaluate_window(stats, forecasts, frames, universes, clean)
    res_info = evaluate_window(stats, forecasts, frames, universes, info)
    verdict = K.evaluate_verdict(res_clean["signals"]["kronos"], res_clean["kronos_minus_momentum"],
                                 len(res_clean["per_origin"]))

    ka, ca = [], []
    for r in res_info["per_origin"]:
        if r["ic_kronos"] is not None and np.isfinite(r["ic_kronos"]):
            ka.append(r["ic_kronos"])
            ca.append(int(TS.month_key(r["origin"])[0]))
    kb, cb = [], []
    for r in res_clean["per_origin"]:
        if r["ic_kronos"] is not None and np.isfinite(r["ic_kronos"]):
            kb.append(r["ic_kronos"])
            cb.append(int(TS.month_key(r["origin"])[0]))
    info_minus_clean = K.diff_summary(ka, ca, kb, cb)
    lo = info_minus_clean.get("lo")
    note = ("The old window is not clearly higher: the difference interval straddles zero."
            if (lo is None or lo <= 0) else
            "The old window is clearly higher, which is consistent with training-data leakage. Nothing more "
            "than that can be said from these numbers.")

    result = {
        "generated": str(date.today()),
        "data_end": end,
        "kronos": {**meta, "model": MODEL_REPO, "tokenizer": TOKENIZER_REPO,
                   "device": resolve_device(args.device),
                   "lookback": K.LOOKBACK, "horizon": K.HORIZON, "T": K.TEMPERATURE, "top_p": K.TOP_P,
                   "sample_count": args.sample_count, "top_n": K.TOP_N, "min_price": K.MIN_PRICE,
                   "min_bars": K.MIN_BARS, "seeded": SAMPLE_SEED},
        "origins": {"clean": clean, "informational": info},
        "universe": {str(t.date()): len(s) for t, s in universes.items()},
        "clean": res_clean,
        "informational": res_info,
        "verdict": verdict,
        "contamination": {"info_minus_clean": info_minus_clean,
                          "clean_mean_ic": res_clean["signals"]["kronos"].get("mean"),
                          "info_mean_ic": res_info["signals"]["kronos"].get("mean"), "note": note},
        "caveats": caveats(),
    }
    base = Path(args.out) if args.out else OUT_DIR / f"kronos_eval_{date.today()}"
    paths = write_outputs(result, base)
    print()
    print(f"verdict: {verdict['verdict']}" + ("  UNDERPOWERED" if verdict["underpowered"] else ""))
    for name in MENU:
        print(f"  {name:9s} mean IC {_fmt_ci(res_clean['signals'][name])}  "
              f"(n={res_clean['signals'][name].get('n_origins', 0)})")
    print(f"  kronos - momentum {_fmt_ci(res_clean['kronos_minus_momentum'])}")
    print(f"wrote {paths['json']}")
    print(f"wrote {paths['md']}")
    print(f"total {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
