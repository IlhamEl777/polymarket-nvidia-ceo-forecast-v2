# Polymarket NVIDIA CEO Forecaster — v2

Entry for make.fun's **polymarket-nvidia-ceo-forecast-01**: forecast the Yes
price of every market of Polymarket's "Next CEO of NVIDIA?" event, minute by
minute, 4 hours ahead — five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) per
market per minute, scored by mean pinball loss.

`main.py` (Python 3, standard library only, deterministic, no network)
reads each case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one
`.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Method

These prices sit still for long stretches, move in ticks, and jump rarely,
so the quantiles of the price *h* minutes ahead are read off the market's
own recent *h*-minute changes instead of a Gaussian spread. The forecast is
a 60/40 blend of two empirical forecasters built on that idea:

- **A — per-market offset curves.** For a grid of horizons, the market's
  overlapping h-minute changes over the last 2880 minutes give empirical
  quantile offsets, blended by sample count with the offsets pooled over
  all the case's markets (short or quiet markets lean on the pool),
  interpolated over h in sqrt-space, with the inner offsets scaled by 0.50
  and the outer by 0.35, and the median carrying a shrunk share (0.25) of
  the empirical median offset. Markets with fewer than 8 samples at a
  horizon fall back to the pooled curve outright.
- **C — pooled activity rescaling.** Pooled offsets at h = 240 only, grown
  back over the horizon like sqrt(h), with each market's offsets rescaled
  by its recent activity — its mean absolute 1-minute change over the last
  360 minutes relative to the pool's — clamped to [0.3, 3.0] and scaled
  by 0.8.

## Backtest

On the kit's practice cases p00..p04 (scored against the next case's
history), mean pinball loss ≈ **0.31**, vs a naive fixed-band forecaster at
≈ 0.372 and the house persistence baseline at ≈ 0.393 on the same cases.
Reproduce locally with the dev harness from the first build
(`harness.py model <path-to-main.py>`).

## v2 note

Rebuilt in a fresh session after the competition opened (2026-11-13
09:00 UTC). Changes vs the pre-open build: inner-quantile scale 0.45 →
0.50, and an explicit pooled-curve fallback for markets with too little
history.
