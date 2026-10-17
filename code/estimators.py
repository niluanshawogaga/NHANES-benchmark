"""Weighted estimators with stratified-cluster jackknife standard errors.

Shared by the task builder (ground truth) and the evaluator (scoring), so the
benchmark's reference values and the scoring path can never drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def as_float(series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def wmean(y: np.ndarray, w: np.ndarray) -> float:
    mask = ~(np.isnan(y) | np.isnan(w))
    if mask.sum() == 0 or w[mask].sum() <= 0:
        return float("nan")
    return float(np.sum(w[mask] * y[mask]) / np.sum(w[mask]))


def wprop(y: np.ndarray, w: np.ndarray) -> float:
    return wmean(y.astype(float), w)


def wquantile(y: np.ndarray, w: np.ndarray, q: float = 0.5) -> float:
    """Weighted quantile as the smallest value whose cumulative weight reaches q.

    The interpolating definition is not used: many survey items are recorded in
    whole units, so tied values carry most of the weight and interpolation makes
    the result depend on the order of tied rows.
    """
    mask = ~(np.isnan(y) | np.isnan(w))
    y, w = y[mask], w[mask]
    if len(y) == 0:
        return float("nan")
    order = np.argsort(y, kind="stable")
    y = y[order]
    cw = np.cumsum(w[order]) / np.sum(w)
    idx = int(np.searchsorted(cw, q, side="left"))
    return float(y[min(idx, len(y) - 1)])


def wcorr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(w))
    if mask.sum() < 3:
        return float("nan")
    x, y, w = x[mask], y[mask], w[mask]
    w = w / w.sum()
    mx, my = np.sum(w * x), np.sum(w * y)
    cxy = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cxy / np.sqrt(vx * vy))


def kish(w: np.ndarray) -> float:
    w = w[~np.isnan(w)]
    if len(w) == 0 or np.sum(w ** 2) <= 0:
        return float("nan")
    return float(np.sum(w) ** 2 / np.sum(w ** 2))


def _estimate(df: pd.DataFrame, spec: dict, weight_col: str, weighted: bool) -> float:
    """Evaluate one estimand specification on a data frame."""
    weight = as_float(df[weight_col]) if weighted else np.ones(len(df))
    subgroup = spec.get("subgroup")
    if subgroup:
        if "spec" in subgroup:            # nested task-builder form
            subgroup = subgroup["spec"]
        if subgroup["op"] == "in":
            keep = None
            keep = np.isin(as_float(df[subgroup["col"]]), subgroup["vals"])
        elif subgroup["op"] == "range":
            v = as_float(df[subgroup["col"]])
            keep = (v >= subgroup["lo"]) & (v < subgroup["hi"])
        else:
            raise ValueError(subgroup["op"])
        df = df[keep]
        weight = weight[keep]
    kind = spec["kind"]
    if kind == "proportion":
        return wprop(as_float(df[spec["target"]]), weight)
    if kind == "mean":
        return wmean(as_float(df[spec["target"]]), weight)
    if kind == "median":
        return wquantile(as_float(df[spec["target"]]), weight, 0.5)
    if kind == "correlation":
        xname = spec["x"]["name"] if isinstance(spec["x"], dict) else spec["x"]
        yname = spec["y"]["name"] if isinstance(spec["y"], dict) else spec["y"]
        return wcorr(as_float(df[xname]), as_float(df[yname]), weight)
    raise ValueError(kind)


def estimate_with_se(df: pd.DataFrame, spec: dict, weight_col: str,
                     strata_col: str = "SDMVSTRA", psu_col: str = "SDMVPSU") -> dict:
    """Weighted and unweighted point estimates plus a delete-one-PSU jackknife SE."""
    theta = _estimate(df, spec, weight_col, True)
    unweighted = _estimate(df, spec, weight_col, False)
    se = float("nan")
    if strata_col in df.columns and psu_col in df.columns:
        reps: list[float] = []
        pair_cols = list(zip(df[strata_col].to_numpy(), df[psu_col].to_numpy()))
        keys = set(pair_cols)
        for key in sorted(keys, key=lambda k: (str(k[0]), str(k[1]))):
            drop = np.array([p != key for p in pair_cols])
            if drop.sum() < 10:
                continue
            reps.append(_estimate(df[drop].reset_index(drop=True), spec, weight_col, True))
        reps = np.array([r for r in reps if not np.isnan(r)])
        if len(reps) > 1:
            n_h = {}
            for h, _ in keys:
                n_h[h] = n_h.get(h, 0) + 1
            mean_rep = reps.mean()
            var = 0.0
            for key in sorted(keys, key=lambda k: (str(k[0]), str(k[1]))):
                h = key[0]
                drop = np.array([p != key for p in pair_cols])
                if drop.sum() < 10:
                    continue
                r = _estimate(df[drop].reset_index(drop=True), spec, weight_col, True)
                if not np.isnan(r):
                    var += (n_h[h] - 1) / n_h[h] * (r - mean_rep) ** 2
            se = float(np.sqrt(var))
    w = as_float(df[weight_col])
    return {"weighted": theta, "unweighted": unweighted, "se": se,
            "kish_ess": kish(w), "n_analysis": int(len(df))}
