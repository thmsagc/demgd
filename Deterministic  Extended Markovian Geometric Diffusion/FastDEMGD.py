"""
Fast Deterministic Extended Markovian Geometric Diffusion (Fast DEMGD)
======================================================================

Reimplementation of ``DeterministicEMGD`` focused on **speed** and on the
**quality of the data reduction**, using vectorised NumPy operations, an exact
k-nearest-neighbours search via a KD-tree (``scipy.spatial.cKDTree``) and a
sparse Markov transition matrix (``scipy.sparse``).

It is a drop-in replacement for the original ``DeterministicEMGD`` function:
the public signature and the ``(reducedSet, removedSet)`` return value are the
same, so existing scripts (e.g. ``testDEMGD.py``) keep working by simply
importing from this module instead.

Why it is dramatically faster
-----------------------------
The original code has two algorithmic bottlenecks:

1. **Graph construction** did a brute-force KNN over hand-made 1-D "buckets"
   with pure-Python distance loops and ``list.insert`` (O(degree) shifts).
2. **The reduction/deletion path is O(N^2)**: every removed instance triggers
   ``__indexCorrector__`` (renumbers *all* edges of *all* vertices), a full
   ``__findDiagonal__`` rebuild, and a ``del self.set[index]`` list shift.

This version replaces the KNN with an exact KD-tree query (O(N log N)), keeps
the diffusion operator as a sparse matrix, and expresses the whole importance
computation with vectorised array math. The reduction is done by recomputing
the (cheap) diffusion importance on the shrinking point set each pass, which
also removes the quirks of the original incremental graph repair.

Mathematical model (unchanged intent)
--------------------------------------
* Build a symmetric KNN graph (each point linked to its ``k`` nearest
  neighbours plus a self-loop).
* Weight edges with the Gaussian kernel ``w = exp(-d^2)`` (Euclidean ``d``).
* Row-normalise into a Markov transition matrix ``P = D^{-1} W``.
* Importance of a node ``i`` is ``P_ii * degree_i``.
* In each pass, points whose importance falls below
  ``average - multiplier * std`` are candidates for removal; a non-maximal
  suppression step keeps, among adjacent candidates, only the least important
  one (matching the original marking logic). Iterate until the target size is
  reached or a pass removes nothing.
"""

import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


def _build_diffusion(X, k):
    """Build the sparse Markov diffusion operator and per-node importance.

    Parameters
    ----------
    X : (n, d) float ndarray
        The current point set.
    k : int
        Number of nearest neighbours (excluding the point itself).

    Returns
    -------
    importance : (n,) ndarray
        ``P_ii * degree_i`` for every node.
    W : scipy.sparse.csr_matrix
        The symmetric kernel-weighted adjacency (with self-loops).
    """
    n = len(X)
    # +1 because the query returns the point itself (distance 0) as the
    # nearest neighbour, which plays the role of the self-loop.
    kk = min(k + 1, n)

    tree = cKDTree(X)
    dist, idx = tree.query(X, k=kk, workers=-1)
    # cKDTree.query drops the last axis when kk == 1; normalise the shape.
    if kk == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    rows = np.repeat(np.arange(n), kk)
    cols = idx.ravel()
    weights = np.exp(-(dist.ravel() ** 2))

    W = csr_matrix((weights, (rows, cols)), shape=(n, n))
    # Union-symmetrise: an edge exists if either endpoint has the other in its
    # KNN. Weights are symmetric, so ``maximum`` just fills the missing side.
    W = W.maximum(W.T)
    W.sort_indices()

    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0  # isolated node guard (cannot normally happen)

    p_diag = W.diagonal() / row_sums          # P_ii (self-transition prob.)
    degree = np.diff(W.indptr)                # nonzeros per row (incl. self)
    importance = p_diag * degree
    return importance, W


def _suppress_and_mark(importance, W, multiplier):
    """Select the instances to remove in one EMGD pass.

    Reproduces the original non-maximal-suppression: iterate candidates in
    index order; when a candidate is processed, any already-marked neighbour is
    unmarked and, among the candidate and those neighbours, the least important
    one is (re)marked. Only candidate rows are touched, so this is cheap.
    """
    avg = importance.mean()
    std = importance.std()  # population std (ddof=0), matches the original
    threshold = avg - std * multiplier

    candidates = np.where(importance < threshold)[0]
    if candidates.size == 0:
        return candidates  # empty -> caller stops

    marked = np.zeros(importance.size, dtype=bool)
    indptr, indices = W.indptr, W.indices
    for i in candidates:
        least = i
        imp_i = importance[i]
        for j in indices[indptr[i]:indptr[i + 1]]:
            if marked[j]:
                if importance[j] < imp_i:
                    least = j
                marked[j] = False
        marked[least] = True

    return np.where(marked)[0]


def DeterministicEMGD(inputSet, percentage=0.75, maxPerBucket=25, k=5,
                      propagation=1, multiplier=1, distanceMetric="euclidean"):
    """Fast drop-in replacement for the original ``DeterministicEMGD``.

    ``maxPerBucket`` and ``propagation`` are accepted for signature
    compatibility but are not needed by the KD-tree based implementation.
    ``distanceMetric`` is kept for compatibility; the kernel always uses the
    Euclidean distance, exactly like the original graph construction.

    Returns
    -------
    (reducedSet, removedSet) : tuple(list, list)
        The kept points and the removed points, as lists of tuples.
    """
    if not isinstance(inputSet, list):
        warnings.warn("ERROR: The input set must be of the List type.")
        return
    if not isinstance(percentage, float):
        try:
            percentage = float(percentage)
        except (TypeError, ValueError):
            warnings.warn("ERROR: The param percentage must be a float.")
            return
    if percentage > 1 or percentage < 0:
        warnings.warn("ERROR: percentage must be within [0, 1]. Using 0.75.")
        percentage = 0.75
    if not isinstance(k, int) or k < 1:
        warnings.warn("ERROR: k must be a positive int. Using 5.")
        k = 5

    X = np.asarray(inputSet, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n0 = len(X)
    if n0 == 0:
        warnings.warn("ERROR: Empty set.")
        return [], []
    if k >= n0:
        k = n0 - 1
        warnings.warn("Warning: k changed implicitly to " + str(k) + ".")

    original_points = [tuple(row) for row in X]
    ids = np.arange(n0)                 # ids[j] = original index of current row j
    target = int(n0 * percentage)
    removed_ids = []

    cur = X
    while len(cur) > target and len(cur) > 1:
        importance, W = _build_diffusion(cur, min(k, len(cur) - 1))
        marked = _suppress_and_mark(importance, W, multiplier)
        if marked.size == 0:
            break
        removed_ids.extend(ids[marked].tolist())
        keep = np.ones(len(cur), dtype=bool)
        keep[marked] = False
        cur = cur[keep]
        ids = ids[keep]

    reduced_set = [original_points[i] for i in ids]
    removed_set = [original_points[i] for i in removed_ids]
    return reduced_set, removed_set
