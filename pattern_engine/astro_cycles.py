"""Financial Astrology & Gann Cycles — Market timing via planetary positions.

Implements the following from the BlueprintResearch/Gann-and-Financial-Astrology-Indicators
methodology, reimplemented in Python using the `ephem` astronomical library:

1. Bradley Siderograph — Donald Bradley's 1947 planetary aspect timing index.
   Formula: P = L + D + M where:
     L = long-term outer-planet aspect contributions (Jupiter-Saturn, etc.)
     D = declination parallel/contraparallel contributions
     M = mid-term inner-planet aspect contributions
   Market turns often occur within 4 weeks of a Bradley turning point.

2. Planetary Retrograde Detection — periods when planets appear to move backward.
   Mercury retrograde is most market-relevant (communication, contracts, tech).

3. Planetary Ingress — when planets change zodiac signs (every 30° of ecliptic).
   Associated with sector rotation and trend changes in financial astrology.

4. Lunar Phase Markers — New Moon (accumulation) and Full Moon (distribution).
   Research shows increased volatility around full moons and market turns near new moons.

5. Square of Nine Planetary Levels — map current planetary longitude to Gann SQ9 price grid.
   Adds astrological price targets to our existing Gann fan analysis.

6. Aspect Table — all current major planetary aspects with orb precision.
   Clustered strong aspects = high-energy turning-point windows.

Reference: https://github.com/BlueprintResearch/Gann-and-Financial-Astrology-Indicators
Algorithms: Bradley (1947), Meeus "Astronomical Algorithms" (1998)
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

import ephem
import numpy as np
import pandas as pd


# ── Planets used in calculations ─────────────────────────────────────────────
_PLANETS: dict[str, Any] = {
    "Sun":     ephem.Sun,
    "Moon":    ephem.Moon,
    "Mercury": ephem.Mercury,
    "Venus":   ephem.Venus,
    "Mars":    ephem.Mars,
    "Jupiter": ephem.Jupiter,
    "Saturn":  ephem.Saturn,
    "Uranus":  ephem.Uranus,
    "Neptune": ephem.Neptune,
}

# ── Bradley Siderograph aspect valencies ─────────────────────────────────────
# Based on Donald Bradley (1947) "Stock Market Prediction"
# +1 = harmonious, -1 = discordant
_ASPECT_VALENCY: dict[float, int] = {
    0.0:   0,   # Conjunction — valency depends on planet pair (handled separately)
    60.0:  1,   # Sextile
    90.0: -1,   # Square
    120.0: 1,   # Trine
    180.0: -1,  # Opposition
}
_ASPECT_ORBS: dict[float, float] = {
    0.0:   10.0,
    60.0:  8.0,
    90.0:  8.0,
    120.0: 10.0,
    180.0: 10.0,
}

# Conjunction valencies by planet pair
# (pair is sorted alphabetically to avoid duplicates)
_CONJUNCTION_VALENCY: dict[frozenset, int] = {
    frozenset({"Jupiter", "Neptune"}):  1,
    frozenset({"Jupiter", "Venus"}):    1,
    frozenset({"Jupiter", "Moon"}):     1,
    frozenset({"Jupiter", "Mercury"}):  1,
    frozenset({"Jupiter", "Saturn"}):  -1,
    frozenset({"Jupiter", "Mars"}):     1,
    frozenset({"Jupiter", "Sun"}):      1,
    frozenset({"Jupiter", "Uranus"}):   1,
    frozenset({"Saturn",  "Mars"}):    -1,
    frozenset({"Saturn",  "Sun"}):     -1,
    frozenset({"Saturn",  "Moon"}):    -1,
    frozenset({"Saturn",  "Venus"}):   -1,
    frozenset({"Saturn",  "Mercury"}): -1,
    frozenset({"Saturn",  "Uranus"}):  -1,
    frozenset({"Saturn",  "Neptune"}): -1,
    frozenset({"Mars",    "Sun"}):      0,
    frozenset({"Mars",    "Moon"}):    -1,
    frozenset({"Mars",    "Venus"}):    1,
    frozenset({"Sun",     "Venus"}):    1,
    frozenset({"Sun",     "Moon"}):     0,
    frozenset({"Sun",     "Mercury"}):  0,
    frozenset({"Venus",   "Moon"}):     1,
    frozenset({"Venus",   "Mercury"}):  1,
    frozenset({"Uranus",  "Neptune"}):  0,
}


def _ephem_date(d: date | str) -> ephem.Date:
    """Convert a Python date or date string to ephem.Date."""
    if isinstance(d, str):
        d = date.fromisoformat(d[:10])
    return ephem.Date(d.strftime("%Y/%m/%d"))


def _planet_longitude(planet_cls, dt: ephem.Date) -> float:
    """Get ecliptic longitude in degrees for a planet at a given date."""
    p = planet_cls()
    p.compute(dt, epoch=dt)
    return float(ephem.Ecliptic(p, epoch=dt).lon) * 180.0 / math.pi % 360.0


def _planet_lat(planet_cls, dt: ephem.Date) -> float:
    """Get ecliptic latitude (declination) in degrees."""
    p = planet_cls()
    p.compute(dt, epoch=dt)
    return float(ephem.Ecliptic(p, epoch=dt).lat) * 180.0 / math.pi


def _planet_speed(planet_cls, dt: ephem.Date, days: float = 0.5) -> float:
    """Approximate daily speed (degrees/day) by finite difference."""
    lon1 = _planet_longitude(planet_cls, ephem.Date(dt - days))
    lon2 = _planet_longitude(planet_cls, ephem.Date(dt + days))
    diff = (lon2 - lon1 + 540) % 360 - 180  # handle 360° wraparound
    return diff / (2 * days)


def _aspect_angle(lon1: float, lon2: float) -> float:
    """Shortest angular separation between two ecliptic longitudes."""
    diff = abs(lon1 - lon2) % 360
    return min(diff, 360 - diff)


def _bradley_score_for_date(dt: ephem.Date) -> float:
    """Compute the Bradley Siderograph contribution for a single date.

    Based on Bradley (1947): sum of all significant planetary aspect contributions.
    Each aspect contributes its valency × weight (based on exactness of the aspect).
    """
    # Get current longitudes for all planets
    longitudes: dict[str, float] = {}
    declinations: dict[str, float] = {}
    for name, cls in _PLANETS.items():
        try:
            longitudes[name] = _planet_longitude(cls, dt)
            declinations[name] = _planet_lat(cls, dt)
        except Exception:
            pass

    names = list(longitudes.keys())
    total = 0.0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1, p2 = names[i], names[j]
            lon1 = longitudes[p1]
            lon2 = longitudes[p2]
            angle = _aspect_angle(lon1, lon2)

            for aspect_deg, valency in _ASPECT_VALENCY.items():
                orb = _ASPECT_ORBS[aspect_deg]
                diff = abs(angle - aspect_deg)
                if diff > orb:
                    continue

                # Weight: linear from 1.0 (exact) to 0.0 (at orb edge)
                weight = 1.0 - diff / orb

                if aspect_deg == 0.0:  # Conjunction
                    pair = frozenset({p1, p2})
                    v = _CONJUNCTION_VALENCY.get(pair, 0)
                else:
                    v = valency

                total += v * weight

            # Declination parallels (same declination = parallel, opposite = contraparallel)
            dec1 = declinations.get(p1, 0)
            dec2 = declinations.get(p2, 0)
            par_diff = abs(abs(dec1) - abs(dec2))
            if par_diff < 1.5:  # within 1.5° = parallel
                # Both same sign = parallel (harmonious like conjunction)
                pair = frozenset({p1, p2})
                v = _CONJUNCTION_VALENCY.get(pair, 0)
                total += v * (1.0 - par_diff / 1.5) * 0.5  # half-weight for declination

            contra_diff = abs(abs(dec1) + abs(dec2) - 0)
            if abs(dec1) > 0 and abs(dec2) > 0:
                sign_opp = (dec1 > 0) != (dec2 > 0)
                if sign_opp and par_diff < 1.5:  # contraparallel (opposite sign, same magnitude)
                    total -= v * (1.0 - par_diff / 1.5) * 0.5

    return total


def _compute_bradley_series(start_date: date, end_date: date) -> list[dict[str, Any]]:
    """Compute daily Bradley Siderograph scores from start to end date."""
    series = []
    current = start_date
    while current <= end_date:
        try:
            dt = _ephem_date(current)
            score = _bradley_score_for_date(dt)
            series.append({"date": str(current), "score": round(score, 4)})
        except Exception:
            pass
        current += timedelta(days=1)
    return series


def _find_bradley_turning_points(series: list[dict]) -> list[dict[str, Any]]:
    """Find local maxima and minima in the Bradley series (turning points)."""
    if len(series) < 5:
        return []
    scores = [s["score"] for s in series]
    dates = [s["date"] for s in series]
    turnings = []
    for i in range(2, len(scores) - 2):
        s = scores[i]
        # Local max: higher than neighbors on both sides
        if s > scores[i - 1] and s > scores[i - 2] and s > scores[i + 1] and s > scores[i + 2]:
            turnings.append({"date": dates[i], "type": "high", "score": s})
        # Local min: lower than neighbors on both sides
        elif s < scores[i - 1] and s < scores[i - 2] and s < scores[i + 1] and s < scores[i + 2]:
            turnings.append({"date": dates[i], "type": "low", "score": s})
    return turnings


def _get_moon_phases(start_date: date, end_date: date) -> list[dict[str, Any]]:
    """Find new moons and full moons in the date range."""
    phases = []
    current = start_date
    while current <= end_date:
        try:
            dt = _ephem_date(current)
            sun = ephem.Sun(dt)
            moon = ephem.Moon(dt)
            sun.compute(dt)
            moon.compute(dt)

            sun_lon = float(ephem.Ecliptic(sun, epoch=dt).lon) * 180 / math.pi % 360
            moon_lon = float(ephem.Ecliptic(moon, epoch=dt).lon) * 180 / math.pi % 360

            angle = (moon_lon - sun_lon) % 360

            # New Moon: angle near 0°
            if angle < 5 or angle > 355:
                phases.append({"date": str(current), "phase": "New Moon", "angle": round(angle, 1)})
            # Full Moon: angle near 180°
            elif 175 < angle < 185:
                phases.append({"date": str(current), "phase": "Full Moon", "angle": round(angle, 1)})
            # First Quarter
            elif 85 < angle < 95:
                phases.append({"date": str(current), "phase": "First Quarter", "angle": round(angle, 1)})
            # Last Quarter
            elif 265 < angle < 275:
                phases.append({"date": str(current), "phase": "Last Quarter", "angle": round(angle, 1)})
        except Exception:
            pass
        current += timedelta(days=1)
    return phases


def _get_retrograde_periods(
    planet_cls, planet_name: str, start_date: date, end_date: date,
) -> list[dict[str, Any]]:
    """Detect retrograde periods (negative daily speed) for a planet."""
    periods: list[dict[str, Any]] = []
    in_retro = False
    retro_start = None
    current = start_date
    while current <= end_date:
        try:
            dt = _ephem_date(current)
            speed = _planet_speed(planet_cls, dt)
            if speed < 0 and not in_retro:
                in_retro = True
                retro_start = current
            elif speed >= 0 and in_retro:
                in_retro = False
                if retro_start:
                    periods.append({
                        "planet": planet_name,
                        "start": str(retro_start),
                        "end": str(current),
                        "duration_days": (current - retro_start).days,
                    })
                    retro_start = None
        except Exception:
            pass
        current += timedelta(days=7 if planet_name in ("Jupiter", "Saturn", "Uranus", "Neptune") else 1)
    # Close open retrograde at end_date
    if in_retro and retro_start:
        periods.append({
            "planet": planet_name,
            "start": str(retro_start),
            "end": str(end_date),
            "duration_days": (end_date - retro_start).days,
            "ongoing": True,
        })
    return periods


_ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


def _longitude_to_sign(lon: float) -> str:
    sign_idx = int(lon / 30.0) % 12
    return _ZODIAC_SIGNS[sign_idx]


def _get_planetary_ingress(
    planet_cls, planet_name: str, start_date: date, end_date: date,
) -> list[dict[str, Any]]:
    """Detect when a planet changes zodiac signs (crosses a 30° boundary)."""
    ingresses = []
    current = start_date
    step = timedelta(days=7 if planet_name in ("Jupiter", "Saturn", "Uranus", "Neptune") else 1)
    prev_sign = None

    while current <= end_date:
        try:
            dt = _ephem_date(current)
            lon = _planet_longitude(planet_cls, dt)
            sign = _longitude_to_sign(lon)
            if prev_sign is not None and sign != prev_sign:
                ingresses.append({
                    "planet": planet_name,
                    "date": str(current),
                    "from_sign": prev_sign,
                    "to_sign": sign,
                    "longitude": round(lon, 2),
                })
            prev_sign = sign
        except Exception:
            pass
        current += step
    return ingresses


def _sq9_planetary_levels(current_price: float, lon_deg: float) -> list[dict[str, Any]]:
    """Map a planetary longitude to Square of Nine price levels.

    Method: The planetary longitude (0-360°) maps to a spiral position
    on the Square of Nine. For each harmonic (×1, ×2, ×4), project the
    price level where that longitude aligns with the square.

    Implementation based on: Gann SQ9 planetary degrees methodology.
    """
    if current_price <= 0:
        return []

    levels = []
    n = math.sqrt(current_price)

    # Map longitude to SQ9 step (each 90° = 0.25 in sqrt space)
    base_step = lon_deg / 360.0  # fraction of full rotation
    for harmonic in range(0, 5):  # 5 harmonics (planetary aspects)
        k = base_step + harmonic
        price = (n + k * 0.25) ** 2
        if price > 0 and abs(price - current_price) / current_price < 0.5:
            levels.append({
                "price": round(price, 4),
                "harmonic": harmonic,
                "longitude": round(lon_deg, 2),
                "label": f"SQ9 {lon_deg:.0f}°h{harmonic}",
            })
    return levels


def _compute_planetary_series(
    start_date: date, end_date: date, step_days: int = 2
) -> dict[str, list[list]]:
    """Compute geocentric ecliptic longitudes for key planets over a date range.

    Returns: {planet_name: [[date_str, longitude_degrees], ...]}
    Used for the "Path of Planets" sub-chart (Blueprint Research methodology).
    """
    planets = {
        "Sun":     ephem.Sun,
        "Moon":    ephem.Moon,
        "Mercury": ephem.Mercury,
        "Venus":   ephem.Venus,
        "Mars":    ephem.Mars,
        "Jupiter": ephem.Jupiter,
        "Saturn":  ephem.Saturn,
    }
    series: dict[str, list[list]] = {name: [] for name in planets}
    current = start_date
    step = timedelta(days=step_days)
    while current <= end_date:
        dt = _ephem_date(current)
        date_str = str(current)
        for name, cls in planets.items():
            try:
                lon = _planet_longitude(cls, dt)
                series[name].append([date_str, round(lon, 2)])
            except Exception:
                pass
        current += step
    return series


def _get_major_aspects_today(ref_date: date) -> list[dict[str, Any]]:
    """Get all major planetary aspects for today with orbs."""
    dt = _ephem_date(ref_date)
    longitudes: dict[str, float] = {}
    for name, cls in _PLANETS.items():
        try:
            longitudes[name] = _planet_longitude(cls, dt)
        except Exception:
            pass

    aspects = []
    names = list(longitudes.keys())
    aspect_names = {0.0: "Conjunction", 60.0: "Sextile", 90.0: "Square",
                    120.0: "Trine", 180.0: "Opposition"}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1, p2 = names[i], names[j]
            angle = _aspect_angle(longitudes[p1], longitudes[p2])
            for asp_deg, orb in _ASPECT_ORBS.items():
                diff = abs(angle - asp_deg)
                if diff <= orb:
                    aspects.append({
                        "planet1": p1,
                        "planet2": p2,
                        "aspect": aspect_names[asp_deg],
                        "degrees": round(asp_deg, 0),
                        "orb": round(diff, 2),
                        "exact": diff < 1.0,
                        "valency": _CONJUNCTION_VALENCY.get(frozenset({p1, p2}), 0)
                        if asp_deg == 0 else _ASPECT_VALENCY[asp_deg],
                    })
    aspects.sort(key=lambda x: x["orb"])
    return aspects


# ── Main detector function ────────────────────────────────────────────────────

def detect(symbol: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect financial astrology signals and return as pattern entries.

    Returns a list of pattern dicts for the service pipeline. Each entry
    represents an astro signal (Bradley turning point, retrograde, moon phase,
    planetary ingress, etc.) that can be rendered on the chart.
    """
    if len(df) < 20:
        return []

    try:
        results = []
        today = date.today()

        # Use the date range from the price data
        df_dates = df["date"].astype(str).tolist()
        try:
            first_date = date.fromisoformat(df_dates[0][:10])
            last_date = date.fromisoformat(df_dates[-1][:10])
        except Exception:
            first_date = today - timedelta(days=365)
            last_date = today

        # ── 1. Bradley Siderograph ────────────────────────────────────────
        # Compute for the past 90 days + next 90 days
        bradley_start = max(first_date, today - timedelta(days=90))
        bradley_end   = today + timedelta(days=90)

        bradley_series = _compute_bradley_series(bradley_start, bradley_end)
        turning_points = _find_bradley_turning_points(bradley_series)

        # Create Bradley pattern entry with the siderograph line data
        if bradley_series:
            # Build overlay line from the siderograph scores
            # Normalize scores to a price range centered around current price
            if len(df) > 0:
                last_close = float(df["close"].astype(float).iloc[-1])
                scores = [s["score"] for s in bradley_series]
                score_min = min(scores)
                score_max = max(scores)
                score_range = max(score_max - score_min, 0.01)
                # Map score range to ±5% of current price
                price_range = last_close * 0.05
                # Convert Bradley to overlay line for chart display
                date_to_idx = {str(d)[:10]: i for i, d in enumerate(df_dates)}
                bradley_points = []
                for s in bradley_series:
                    d_str = s["date"][:10]
                    idx = date_to_idx.get(d_str)
                    if idx is not None:
                        norm_score = (s["score"] - score_min) / score_range
                        price = last_close * (0.97 + norm_score * 0.06)  # ±3% band
                        bradley_points.append([idx, round(price, 4)])

            # Find upcoming turning points (within 60 days)
            upcoming = [
                tp for tp in turning_points
                if date.fromisoformat(tp["date"]) > today
                and date.fromisoformat(tp["date"]) <= today + timedelta(days=60)
            ]
            # Find most recent past turning point
            past = [
                tp for tp in turning_points
                if date.fromisoformat(tp["date"]) <= today
            ]
            recent_tp = past[-1] if past else None

            # Build Bradley pattern entry
            bradley_entry: dict[str, Any] = {
                "pattern_name": "Bradley Siderograph",
                "pattern_category": "astro",
                "status": "FORMING",
                "direction": "neutral",
                "confidence": 60.0,
                "breakout_level": None,
                "invalidation_level": None,
                "target": None,
                "points": bradley_points[:50] if bradley_points else [],  # limit for perf
                "bradley_series": bradley_series,
                "bradley_turning_points": turning_points,
                "upcoming_turning_points": upcoming,
            }
            if recent_tp:
                # Last turn direction
                recent_tp_date = date.fromisoformat(recent_tp["date"])
                days_since = (today - recent_tp_date).days
                if days_since <= 14:  # within 2 weeks = active signal
                    bradley_entry["status"] = "READY"
                    bradley_entry["confidence"] = 70.0
                    bradley_entry["direction"] = "bullish" if recent_tp["type"] == "low" else "bearish"
            results.append(bradley_entry)

            # Also add individual upcoming turning points as separate entries
            for tp in upcoming:
                tp_date = date.fromisoformat(tp["date"])
                days_until = (tp_date - today).days
                status = "FORMING" if days_until > 14 else "READY"
                direction = "bullish" if tp["type"] == "low" else "bearish"
                results.append({
                    "pattern_name": f"Bradley {tp['type'].title()} Turn",
                    "pattern_category": "astro",
                    "status": status,
                    "direction": direction,
                    "confidence": round(max(50.0, 70.0 - days_until * 0.5), 1),
                    "breakout_level": None,
                    "invalidation_level": None,
                    "target": None,
                    "points": [],
                    "turn_date": tp["date"],
                    "turn_type": tp["type"],
                    "days_until": days_until,
                })

        # ── 2. Mercury Retrograde ─────────────────────────────────────────
        retro_start = today - timedelta(days=120)
        retro_end   = today + timedelta(days=120)
        merc_retros = _get_retrograde_periods(ephem.Mercury, "Mercury", retro_start, retro_end)

        for retro in merc_retros:
            retro_start_d = date.fromisoformat(retro["start"])
            retro_end_d   = date.fromisoformat(retro["end"])
            is_active = retro_start_d <= today <= retro_end_d
            is_upcoming = retro_start_d > today
            if is_active or is_upcoming:
                status = "FORMING" if is_active else "READY"
                days_until_start = max(0, (retro_start_d - today).days)
                results.append({
                    "pattern_name": "Mercury Retrograde",
                    "pattern_category": "astro",
                    "status": status,
                    "direction": "bearish",  # bearish effect on clarity/contracts
                    "confidence": 68.0 if is_active else 60.0,
                    "breakout_level": None,
                    "invalidation_level": None,
                    "target": None,
                    "points": [],
                    "retro_start": retro["start"],
                    "retro_end": retro["end"],
                    "retro_ongoing": retro.get("ongoing", False),
                    "days_until": days_until_start,
                })

        # ── 3. Moon Phases (last 60 days + next 30 days) ─────────────────
        moon_start = today - timedelta(days=60)
        moon_end   = today + timedelta(days=30)
        moon_phases = _get_moon_phases(moon_start, moon_end)

        for phase in moon_phases:
            phase_date = date.fromisoformat(phase["date"])
            days_delta = (phase_date - today).days
            if -7 <= days_delta <= 7:  # only show recent/upcoming phases
                is_new_moon  = phase["phase"] == "New Moon"
                is_full_moon = phase["phase"] == "Full Moon"
                direction = "bullish" if is_new_moon else ("bearish" if is_full_moon else "neutral")
                status = "READY" if abs(days_delta) <= 2 else "FORMING"
                # Find corresponding bar index
                idx = _date_to_bar_idx(df, str(phase_date))
                results.append({
                    "pattern_name": phase["phase"],
                    "pattern_category": "astro",
                    "status": status,
                    "direction": direction,
                    "confidence": 62.0,
                    "breakout_level": None,
                    "invalidation_level": None,
                    "target": None,
                    "points": [[idx, float(df["close"].astype(float).iloc[-1])]] if idx is not None else [],
                    "phase_date": phase["date"],
                    "days_delta": days_delta,
                })

        # ── 4. Major Planetary Ingress (next 90 days) ──────────────────────
        ingress_planets = {
            "Mercury": ephem.Mercury,
            "Venus": ephem.Venus,
            "Mars": ephem.Mars,
            "Jupiter": ephem.Jupiter,
            "Saturn": ephem.Saturn,
        }
        for name, cls in ingress_planets.items():
            ingresses = _get_planetary_ingress(cls, name, today, today + timedelta(days=90))
            for ing in ingresses[:3]:  # max 3 per planet
                ing_date = date.fromisoformat(ing["date"])
                days_until = (ing_date - today).days
                if days_until <= 0:
                    continue
                weight = {"Mercury": 65, "Venus": 62, "Mars": 68, "Jupiter": 72, "Saturn": 75}
                results.append({
                    "pattern_name": f"{name} Enters {ing['to_sign']}",
                    "pattern_category": "astro",
                    "status": "FORMING" if days_until > 14 else "READY",
                    "direction": "neutral",
                    "confidence": float(weight.get(name, 62)),
                    "breakout_level": None,
                    "invalidation_level": None,
                    "target": None,
                    "points": [],
                    "ingress_date": ing["date"],
                    "from_sign": ing["from_sign"],
                    "to_sign": ing["to_sign"],
                    "days_until": days_until,
                })

        # ── 5. Current Aspect Table ────────────────────────────────────────
        today_aspects = _get_major_aspects_today(today)
        # Summarize: count exact aspects and overall energy level
        bullish_aspects = sum(1 for a in today_aspects if a["valency"] > 0 and a["orb"] < 3.0)
        bearish_aspects = sum(1 for a in today_aspects if a["valency"] < 0 and a["orb"] < 3.0)

        if bullish_aspects > 0 or bearish_aspects > 0:
            direction = "bullish" if bullish_aspects > bearish_aspects else (
                "bearish" if bearish_aspects > bullish_aspects else "neutral"
            )
            conf = 55.0 + max(bullish_aspects, bearish_aspects) * 3.0
            results.append({
                "pattern_name": "Planetary Aspect Cluster",
                "pattern_category": "astro",
                "status": "READY",
                "direction": direction,
                "confidence": round(min(conf, 82.0), 1),
                "breakout_level": None,
                "invalidation_level": None,
                "target": None,
                "points": [],
                "aspects": today_aspects[:10],  # top 10 by orb precision
                "bullish_aspects": bullish_aspects,
                "bearish_aspects": bearish_aspects,
                "aspect_date": str(today),
            })

        # ── 6. Square of Nine Planetary Price Levels ───────────────────────
        if len(df) > 0:
            last_close = float(df["close"].astype(float).iloc[-1])
            dt = _ephem_date(today)
            key_planets = {
                "Sun": ephem.Sun,
                "Jupiter": ephem.Jupiter,
                "Saturn": ephem.Saturn,
                "Mars": ephem.Mars,
            }
            all_sq9_levels = []
            for name, cls in key_planets.items():
                try:
                    lon = _planet_longitude(cls, dt)
                    levels = _sq9_planetary_levels(last_close, lon)
                    for lv in levels:
                        lv["planet"] = name
                        all_sq9_levels.append(lv)
                except Exception:
                    pass

            if all_sq9_levels:
                # Sort by proximity to current price
                all_sq9_levels.sort(key=lambda x: abs(x["price"] - last_close))
                results.append({
                    "pattern_name": "Planetary SQ9 Levels",
                    "pattern_category": "astro",
                    "status": "FORMING",
                    "direction": "neutral",
                    "confidence": 60.0,
                    "breakout_level": None,
                    "invalidation_level": None,
                    "target": None,
                    "points": [],
                    "sq9_planetary_levels": all_sq9_levels[:12],  # closest 12
                })

        # ── 7. Planetary Degrees Series (Path of Planets sub-chart) ───────────
        # Compute daily geocentric longitudes for the visible chart range
        planet_start = max(first_date, today - timedelta(days=365))
        planet_end   = today + timedelta(days=30)
        planet_series = _compute_planetary_series(planet_start, planet_end, step_days=2)

        results.append({
            "pattern_name": "Planetary Degrees",
            "pattern_category": "astro",
            "status": "READY",
            "direction": "neutral",
            "confidence": 50.0,
            "breakout_level": None,
            "invalidation_level": None,
            "target": None,
            "points": [],
            "planet_series": planet_series,
        })

        # ── 8. Raw Bradley score series for the standalone sub-chart ──────────
        # Already computed above — attach raw [[date, score]] pairs separately
        # so the sub-chart can use its own price scale (not price-normalized)
        raw_bradley_pts = [[s["date"], s["score"]] for s in bradley_series]
        # Attach to the existing Bradley Siderograph entry
        for r in results:
            if r.get("pattern_name") == "Bradley Siderograph":
                r["raw_bradley_series"] = raw_bradley_pts
                break

        return results

    except Exception as exc:
        import logging
        logging.getLogger("pattern_engine.astro_cycles").warning("Astro detect failed: %s", exc)
        return []


def _date_to_bar_idx(df: pd.DataFrame, d: str) -> int | None:
    """Find the bar index for a given date string."""
    try:
        dates = df["date"].astype(str).tolist()
        d_short = d[:10]
        for i, dd in enumerate(dates):
            if str(dd)[:10] == d_short:
                return i
        # If not found, find nearest
        target = date.fromisoformat(d_short)
        closest = min(
            range(len(dates)),
            key=lambda i: abs((date.fromisoformat(str(dates[i])[:10]) - target).days),
        )
        return closest
    except Exception:
        return None


# ── Standalone testing ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    # Test with fake price data
    test_df = pd.DataFrame({
        "date": [(date.today() - timedelta(days=i)).isoformat() for i in range(200, 0, -1)],
        "open":  [100.0] * 200,
        "high":  [105.0] * 200,
        "low":   [95.0]  * 200,
        "close": [102.0] * 200,
        "volume":[1000.0] * 200,
    })
    results = detect("TEST", test_df)
    print(f"Found {len(results)} astro signals")
    for r in results:
        print(f"  {r['pattern_name']:35s} | {r['status']:10s} | {r['direction']:8s} | conf={r['confidence']:.0f}")
