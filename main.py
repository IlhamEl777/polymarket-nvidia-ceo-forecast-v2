#!/usr/bin/env python3
"""Forecaster for make.fun polymarket-nvidia-ceo-forecast-01 (program/v1).

Rebuilt in a fresh session after the competition opened (2026-11-13
09:00 UTC). Reads every *.in case in $MAKEFUN_INPUT_DIR (else /input) and
writes <case>.out to $MAKEFUN_OUTPUT_DIR (else /output): for each market,
240 rows of five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of the Yes price
at each minute after the checkpoint.

Idea: these prices sit still for long stretches, move in ticks, and jump
rarely, so the quantiles of the price h minutes ahead are read off the
market's own recent h-minute changes, not a Gaussian spread. The forecast
is a 60/40 blend of two empirical forecasters built on that idea:

A (weight 0.6): per-market quantile-offset curves. For a grid of horizons,
the market's overlapping h-minute changes over the last HISTORY_MIN
minutes give empirical quantile offsets, blended by sample count with the
offsets pooled over all the case's markets (short or quiet markets lean on
the pool), interpolated over h in sqrt-space, with the inner offsets scaled
by INNER_SCALE and the outer by OUTER_SCALE and the median carrying a
shrunk share of the empirical median offset. A market with fewer than
MIN_COUNT samples at a horizon falls back to the pooled curve outright.

C (weight 0.4): pooled offsets at h = 240 only, grown back over the
horizon like sqrt(h), with each market's offsets rescaled by its recent
activity — its mean absolute 1-minute change over the last ACTIVITY_MIN
minutes relative to the pool's — clamped to [RATIO_MIN, RATIO_MAX] and
scaled by POOL_SCALE. Where A trusts each market's whole curve, C trusts
only its level of activity; averaging the two is steadier than either
alone.

v2 changes vs the pre-open build: the inner-quantile scale 0.45 -> 0.50
(the 25%/75% bands were slightly too tight on the practice cases), an
explicit pooled-curve fallback for markets with too little history, and a
shared weighted-pool helper used by both components. Deterministic, stdlib
only, no network.

On the kit's practice cases p00..p04 (scored against the next case's
history) this scores a mean pinball loss of about 0.31 against the house
persistence baseline's 0.393 on the same cases.
"""

import json
import math
import os

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
HORIZONS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 240)
HISTORY_MIN = 2880      # minutes of history used to estimate changes
SHRINK_SAMPLES = 800.0  # samples at which a market's own curve is half trusted
MEDIAN_KEEP = 0.25      # share of the empirical median offset kept
INNER_SCALE = 0.50      # scale for the 0.25 / 0.75 offsets (component A)
OUTER_SCALE = 0.35      # scale for the 0.05 / 0.95 offsets (component A)
ACTIVITY_MIN = 360      # minutes for the activity ratio (component C)
POOL_SCALE = 0.8        # overall scale of the pooled offsets (component C)
GROWTH_EXP = 0.5        # horizon growth exponent (component C)
RATIO_MIN, RATIO_MAX = 0.3, 3.0
WEIGHT_A = 0.6          # blend weight of component A
MIN_COUNT = 8           # below this, a market's own curve is not trusted


def _quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _change_quantiles(prices, h):
    """Empirical quantiles of the h-minute changes in one price series."""
    n = len(prices)
    if n <= h:
        return None, 0
    # Stride h//2 keeps long-horizon samples overlapping but not
    # degenerate: at h=240 over 2880 minutes that is still ~24 samples.
    stride = max(1, h // 2)
    diffs = [prices[i + h] - prices[i] for i in range(0, n - h, stride)]
    diffs.sort()
    return [_quantile(diffs, t) for t in TAUS], len(diffs)


def _weighted_pool(entries):
    """Sample-count-weighted mean of per-market quantile offsets.

    entries: list of (quantiles, sample_count). Returns the pooled
    quantiles, or all zeros when there is nothing to pool.
    """
    if not entries:
        return [0.0] * len(TAUS)
    total = sum(c for _, c in entries)
    return [sum(qs[j] * c for qs, c in entries) / total
            for j in range(len(TAUS))]


def _finish(vals):
    """Clamp to [0, 1], enforce non-decreasing, round to 6 decimals."""
    row = []
    prev = 0.0
    for v in vals:
        v = min(1.0, max(0.0, v))
        v = max(v, prev)
        row.append(round(v, 6) + 0.0)
        prev = v
    return row


def _interp(curve, h):
    if h <= HORIZONS[0]:
        return curve[HORIZONS[0]]
    if h >= HORIZONS[-1]:
        return curve[HORIZONS[-1]]
    for a, b in zip(HORIZONS, HORIZONS[1:]):
        if a <= h <= b:
            f = (math.sqrt(h) - math.sqrt(a)) / (math.sqrt(b) - math.sqrt(a))
            return [curve[a][j] * (1.0 - f) + curve[b][j] * f
                    for j in range(len(TAUS))]
    return curve[HORIZONS[-1]]


def _component_a(case, horizon):
    """Per-market empirical offset curves, blended with the pool."""
    markets = case["markets"]
    own = []
    pooled = {h: [] for h in HORIZONS}
    for m in markets:
        prices = m["prices"][-HISTORY_MIN:]
        per_h = {}
        for h in HORIZONS:
            qs, cnt = _change_quantiles(prices, h)
            per_h[h] = (qs, cnt)
            if qs is not None:
                pooled[h].append((qs, cnt))
        own.append(per_h)
    pool_q = {h: _weighted_pool(pooled[h]) for h in HORIZONS}
    scales = (OUTER_SCALE, INNER_SCALE, 0.0, INNER_SCALE, OUTER_SCALE)
    out = {}
    for m, per_h in zip(markets, own):
        curve = {}
        for h in HORIZONS:
            qs, cnt = per_h[h]
            if qs is None or cnt < MIN_COUNT:
                # Too little history: trust the pooled curve outright.
                curve[h] = pool_q[h]
            else:
                w = cnt / (cnt + SHRINK_SAMPLES)
                curve[h] = [w * qs[j] + (1.0 - w) * pool_q[h][j]
                            for j in range(len(TAUS))]
        x = m["prices"][-1]
        rows = []
        for h in range(1, horizon + 1):
            offs = _interp(curve, h)
            med = x + MEDIAN_KEEP * offs[2]
            vals = [med if j == 2 else
                    med + scales[j] * (offs[j] - offs[2])
                    for j in range(len(TAUS))]
            rows.append(_finish(vals))
        out[m["id"]] = rows
    return out


def _component_c(case, horizon):
    """Pooled h=240 offsets grown like sqrt(h), rescaled per market."""
    markets = case["markets"]
    pooled = []
    for m in markets:
        qs, cnt = _change_quantiles(m["prices"][-HISTORY_MIN:], horizon)
        if qs is not None:
            pooled.append((qs, cnt))
    pool = _weighted_pool(pooled)
    acts = []
    for m in markets:
        p = m["prices"][-ACTIVITY_MIN:]
        d = [abs(p[i + 1] - p[i]) for i in range(len(p) - 1)]
        acts.append(sum(d) / max(len(d), 1))
    a_pool = sum(acts) / len(acts) if acts else 0.0
    out = {}
    for m, a in zip(markets, acts):
        x = m["prices"][-1]
        r = (a / a_pool) if a_pool > 0 else 1.0
        r = min(RATIO_MAX, max(RATIO_MIN, r))
        rows = []
        for h in range(1, horizon + 1):
            g = (h / horizon) ** GROWTH_EXP * POOL_SCALE * r
            med = x + MEDIAN_KEEP * pool[2] * g
            vals = [med if j == 2 else med + (pool[j] - pool[2]) * g
                    for j in range(len(TAUS))]
            rows.append(_finish(vals))
        out[m["id"]] = rows
    return out


def forecast_case(case):
    horizon = int(case.get("horizon_minutes", 240))
    a = _component_a(case, horizon)
    c = _component_c(case, horizon)
    out = {}
    for m in case["markets"]:
        mid = m["id"]
        out[mid] = [_finish([WEIGHT_A * va + (1.0 - WEIGHT_A) * vc
                             for va, vc in zip(ra, rc)])
                    for ra, rc in zip(a[mid], c[mid])]
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as fh:
            case = json.load(fh)
        fc = forecast_case(case)
        forecasts = [{"id": m["id"], "q": fc[m["id"]]}
                     for m in case["markets"]]
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
