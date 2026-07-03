"""
Benchmark: original DeterministicEMGD vs FastDEMGD.

Compares wall-clock time on the default (graph-build dominated) path and on the
active-reduction path, and reports a reduction-quality metric (mean/max distance
from every original point to its nearest kept point -- lower is better coverage).

Run:  python3 benchmark.py
Requires: numpy, scipy  (matplotlib optional, not used here).
"""

import random
import time

import numpy as np
from scipy.spatial import cKDTree

import DeterministicEMGD as ORIG
import FastDEMGD as FAST


def make_blobs(n, seed=1):
    """Clustered 2-D data with varied density (so reduction actually triggers)."""
    random.seed(seed)
    centers = [(-25, -25), (25, 25), (0, 20), (20, -15), (-18, 8)]
    pts = []
    for _ in range(n):
        cx, cy = random.choice(centers)
        s = random.choice([0.3, 1.0, 4.0, 10.0])
        pts.append((cx + random.gauss(0, s), cy + random.gauss(0, s)))
    return pts


def coverage(original, reduced):
    if not reduced:
        return float("inf"), float("inf")
    tree = cKDTree(np.asarray(reduced, float))
    d, _ = tree.query(np.asarray(original, float), k=1)
    return float(d.mean()), float(d.max())


def orig_reduce_path(points, multiplier, percentage=0.70):
    """Drive the original method through a real reduction (its O(N^2) path)."""
    obj = ORIG.Deterministic_EMGD(list(points), "euclidean")
    obj.__createBuckets__(25)
    obj.__createGraphKNN__(5, 1)
    target = int(len(obj.set) * percentage)
    while len(obj.set) > target:
        if not obj.__EMGD__(multiplier):
            break
    return obj.set


def main():
    print("== Default path (graph build; default multiplier removes ~nothing) ==")
    print(f"{'N':>8} {'orig (s)':>10} {'fast (s)':>10} {'speedup':>9}")
    for n in [1000, 2000, 4000, 8000, 16000]:
        s = make_blobs(n)
        t = time.perf_counter(); ORIG.DeterministicEMGD(list(s), 0.80); to = time.perf_counter() - t
        t = time.perf_counter(); FAST.DeterministicEMGD(list(s), 0.80); tf = time.perf_counter() - t
        print(f"{n:>8} {to:>10.3f} {tf:>10.4f} {to / tf:>8.1f}x")

    print("\n== Active reduction path (multiplier=-2 forces heavy removal) ==")
    print(f"{'N':>8} {'orig (s)':>10} {'fast (s)':>10} {'speedup':>9}")
    for n in [500, 1000, 2000, 4000]:
        s = make_blobs(n)
        t = time.perf_counter(); orig_reduce_path(s, -2); to = time.perf_counter() - t
        t = time.perf_counter(); FAST.DeterministicEMGD(list(s), 0.70, multiplier=-2); tf = time.perf_counter() - t
        print(f"{n:>8} {to:>10.3f} {tf:>10.4f} {to / tf:>8.1f}x")

    print("\n== Reduction quality of FastDEMGD (coverage of original by kept set) ==")
    print(f"{'mult':>5} {'kept':>7} {'removed':>8} {'mean_cov':>9} {'max_cov':>9}")
    s = make_blobs(4000, seed=7)
    for mult in [0, -1, -2]:
        kept, removed = FAST.DeterministicEMGD(list(s), 0.70, multiplier=mult)
        mean_c, max_c = coverage(s, kept)
        print(f"{mult:>5} {len(kept):>7} {len(removed):>8} {mean_c:>9.3f} {max_c:>9.3f}")

    print("\n== Large-N stress (FastDEMGD only; original is infeasible here) ==")
    print(f"{'N':>8} {'fast (s)':>10} {'kept':>8} {'removed':>8}")
    for n in [50000, 100000, 200000]:
        s = make_blobs(n)
        t = time.perf_counter()
        kept, removed = FAST.DeterministicEMGD(list(s), 0.70, multiplier=-1)
        tf = time.perf_counter() - t
        print(f"{n:>8} {tf:>10.3f} {len(kept):>8} {len(removed):>8}")


if __name__ == "__main__":
    main()
