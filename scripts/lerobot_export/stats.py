"""Incremental mean / std / min / max accumulator across episodes.

Operates on:
  - 1-D arrays (state, action, scalar features) → per-channel stats
  - HWC uint8 video frames → per-channel pixel stats (3 channels)
"""
from __future__ import annotations

import numpy as np


class Accum:
    """Welford-style accumulator returning mean, std, min, max."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)
        self.min = np.full(shape, np.inf, dtype=np.float64)
        self.max = np.full(shape, -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray):
        # x: [N, *shape] — collapse leading dim into incremental Welford
        n = x.shape[0]
        if n == 0:
            return
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0, ddof=0)
        if self.count == 0:
            self.mean = batch_mean.astype(np.float64)
            self.m2 = (batch_var * n).astype(np.float64)
        else:
            delta = batch_mean - self.mean
            tot = self.count + n
            self.mean = self.mean + delta * (n / tot)
            self.m2 = self.m2 + batch_var * n + (delta**2) * (self.count * n / tot)
        self.count += n
        self.min = np.minimum(self.min, x.min(axis=0))
        self.max = np.maximum(self.max, x.max(axis=0))

    def finalize(self) -> dict:
        var = self.m2 / max(self.count, 1)
        return {
            "mean": self.mean.astype(np.float32).tolist() if self.mean.ndim else float(self.mean),
            "std": np.sqrt(var).astype(np.float32).tolist() if var.ndim else float(np.sqrt(var)),
            "min": self.min.astype(np.float32).tolist() if self.min.ndim else float(self.min),
            "max": self.max.astype(np.float32).tolist() if self.max.ndim else float(self.max),
            "count": [self.count],
        }
