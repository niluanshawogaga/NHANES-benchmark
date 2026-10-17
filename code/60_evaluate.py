"""Score every run, decompose the weighted-observation gap and emit tables.

    python 60_evaluate.py

Reads results/generations/*.jsonl plus the task files (which carry the reference
values) and writes the metric tables used by the manuscript figures and tables.
Every number in the paper is traceable to the files written here.
"""

from __future__ import annotations

import argparse
import csv as csv_module
import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (ABS_TOL, GEN_DIR, PROC_DIR, RESULT_DIR, ensure_dirs)  # noqa: E402

WEIGHT_COL = {"interview": "WTINT2YR", "exam": "WTMEC2YR", "dietary": "WTDRD1"}
BOOT = 2000


def tolerance(spec: dict) -> float:
    fam, kind = spec["family"], spec["kind"]
    truth = abs(spec["truth_weighted"])
    if fam == "proportion":
        return ABS_TOL["proportion"]
    if fam == "mean":
        return max(0.02, ABS_TOL["mean"] * truth)
    if fam == "difference":
        return max(ABS_TOL["difference"], ABS_TOL["mean"] * truth) if kind == "mean" \
            else ABS_TOL["difference"]
    if fam == "association":
        return ABS_TOL["association"]
    return float("nan")


def kendall(order_a: list[str], order_b: list[str]) -> float:
    pos_b = {k: i for i, k in enumerate(order_b)}
    keys = [k for k in order_a if k in pos_b]
    pairs = concordant = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            pairs += 1
            concordant += (pos_b[keys[i]] - pos_b[keys[j]]) * (order_a.index(keys[i])
                                                              - order_a.index(keys[j])) > 0
    return 2 * concordant / pairs - 1 if pairs else float("nan")


def load_tasks(tag: str) -> dict:
    with open(os.path.join(PROC_DIR, f"tasks_{tag}.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    out = {}
    for spec in payload["tasks"]:
        out[spec["task_id"]] = spec
    return out


def iter_runs() -> list[dict]:
    runs = []
    for path in sorted(glob.glob(os.path.join(GEN_DIR, "*.jsonl"))):
        name = os.path.basename(path)[:-6]
        parts = name.split("__")
        if len(parts) < 4 or parts[1] == "oracle":
            continue
        model, mode, tag, level = parts[0], parts[1], parts[2], parts[3]
        variant = parts[4] if len(parts) > 4 else "base"
        runs.append({"model": model, "mode": mode, "tag": tag, "level": level,
                     "variant": variant, "file": path})
    return runs


def latest_records(path: str) -> dict:
    records = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("api_error"):
                continue
            records[rec["task_id"]] = rec
    return records


def score_record(rec: dict, spec: dict) -> dict:
    level = rec.get("level", "s120")
    truth = spec["slices"][level]
    out = {"task_id": spec["task_id"], "model": rec["model"], "mode": rec["mode"],
           "level": level, "variant": rec.get("variant", "base"), "family": spec["family"],
           "kind": spec["kind"], "n_analysis": truth.get("n_analysis"),
           "kish_ess": truth.get("kish_ess"), "weight_class": spec["weight_class"]}
    if rec.get("skipped"):
        out.update({"answer": None, "hit": None, "failure": "not_applicable"})
        return out
    if rec.get("forced_failure") and rec.get("parsed") is None:
        out.update({"answer": None, "hit": False, "failure": rec["forced_failure"],
                    "abs_err": None, "design_share": None})
        return out
    # Gate variants: the consistency gate can be re-decided post hoc from the two
    # saved implementations, so its threshold is analysed instead of guessed.
    factor = rec.get("gate_tol_factor")
    if factor is not None:
        second = rec.get("second")
        first = rec.get("parsed")
        tol_now = tolerance({**spec, "truth_weighted": truth.get("weighted")})
        if first is None or second is None or not np.isfinite(first) or not np.isfinite(second):
            rec = dict(rec, parsed=None, forced_failure="gated_refusal")
            out.update({"answer": None, "hit": False, "failure": "gated_refusal",
                        "abs_err": None, "design_share": None})
            return out
        agree = abs(first - second) <= factor * tol_now
        rec = dict(rec, parsed=first if agree else None,
                   forced_failure=None if agree else "gated_refusal")
    if spec["family"] == "ranking":
        order = truth["order"]
        pred = rec.get("parsed")
        out["answer"] = None
        if not isinstance(pred, list):
            out.update({"hit": False, "failure": "no_answer", "tau": None})
        else:
            out.update({"hit": pred == order, "failure": None if pred == order else "ranking_mismatch",
                        "tau": kendall(pred, order)})
        return out
    answer = rec.get("parsed")
    if isinstance(answer, float) and not np.isfinite(answer):
        answer = None
    w = truth.get("weighted")
    u = truth.get("unweighted")
    tol = tolerance({**spec, "truth_weighted": w})
    out.update({"truth_weighted": w, "truth_unweighted": u, "tolerance": tol,
                "gap": (w - u) if (w is not None and u is not None) else None})
    if answer is None:
        failure = "no_answer"
        if rec.get("exec") and rec["exec"].get("rejected"):
            failure = "forbidden_construct"
        elif rec.get("exec") is not None and not rec["exec"].get("ok", True):
            failure = "execution_error"
        out.update({"answer": None, "hit": False, "failure": failure,
                    "abs_err": None, "design_share": None})
        return out
    abs_err = abs(answer - w)
    gap = (w - u)
    design_share = None
    if gap is not None and abs(gap) > max(tol / 3, 1e-9):
        # Projection of the model's answer onto the segment between the
        # design-correct value (0) and the unweighted value (1).  One means the
        # answer behaves exactly like an unweighted statistic, zero means it
        # matches the design-correct value.
        design_share = float(np.clip((answer - w) / (u - w), -1.0, 2.0))
    correct_weight = WEIGHT_COL[spec["weight_class"]]
    mentioned = rec.get("weight_mentions") or []
    spec_weight = rec.get("spec_weight")
    if spec_weight:
        mentioned = mentioned + [spec_weight]
    wrong_weight = (bool(mentioned) and correct_weight not in mentioned)
    no_weight = (not mentioned) and rec["mode"] in ("code", "sql", "fw")
    hit = abs_err <= tol
    if hit:
        failure = None
    elif design_share is not None and design_share > 0.6 and abs(answer - u) <= tol:
        failure = "unweighted_route"
    elif wrong_weight:
        failure = "wrong_weight_variable"
    elif no_weight:
        failure = "no_weight_used"
    else:
        failure = "weighted_but_inaccurate"
    out.update({"answer": answer, "hit": bool(hit), "failure": failure,
                "abs_err": abs_err, "rel_err": abs_err / abs(w) if w else None,
                "design_share": design_share,
                "used_correct_weight": (correct_weight in mentioned) if mentioned else None,
                "any_weight_mentioned": bool(mentioned)})
    return out


def bootstrap_ci(values: list[float], rng: random.Random, n: int = BOOT) -> tuple[float, float]:
    values = [v for v in values if v is not None]
    if len(values) < 5:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def summarise(rows: list[dict], key_fn) -> list[dict]:
    rng = random.Random(20261015)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        scored = [r for r in items if r.get("hit") is not None]
        hits = [1.0 if r["hit"] else 0.0 for r in scored]
        errs = [r["abs_err"] for r in scored if r.get("abs_err") is not None]
        shares = [r["design_share"] for r in scored if r.get("design_share") is not None]
        lo, hi = bootstrap_ci(hits, rng)
        out.append({
            "key": key if not isinstance(key, tuple) else list(key),
            "n": len(items), "n_scored": len(scored),
            "hit_rate": round(float(np.mean(hits)), 4) if hits else None,
            "hit_ci_low": round(lo, 4) if lo == lo else None,
            "hit_ci_high": round(hi, 4) if hi == hi else None,
            "median_abs_err": round(float(np.median(errs)), 4) if errs else None,
            "median_design_share": round(float(np.median(shares)), 4) if shares else None,
            "unweighted_route_rate": round(float(np.mean(
                [1.0 if r.get("failure") == "unweighted_route" else 0.0 for r in scored])), 4)
                if scored else None,
            "no_answer_rate": round(float(np.mean(
                [1.0 if r.get("failure") in ("no_answer", "execution_error",
                                             "forbidden_construct") else 0.0
                 for r in scored])), 4) if scored else None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    ensure_dirs()
    tasks = load_tasks(args.tag)
    rows = []
    for run in iter_runs():
        if run["tag"] != args.tag:
            continue
        for task_id, rec in latest_records(run["file"]).items():
            spec = tasks.get(task_id)
            # Runs recorded against a superseded slice level are skipped rather
            # than scored against the wrong reference values.
            if not spec or rec.get("level") not in spec["slices"]:
                continue
            # Records written before the estimand fingerprint existed cannot be
            # verified for the association family, whose sampling pool changed
            # between builds, so those are skipped instead of silently mismatched.
            if rec.get("spec_hash") is None and run["level"] != "s120" \
                    and spec["family"] == "association":
                continue
            if rec.get("spec_hash") and spec.get("spec_hash") \
                    and rec["spec_hash"] != spec["spec_hash"]:
                continue
            rows.append(score_record(rec, spec))
            if rec.get("mode") == "fw" and not rec.get("skipped") \
                    and spec["family"] != "ranking":
                # The framework run yields three ablation rungs: the specification
                # step alone is scored as fw_s1, the first implementation as
                # fw_impl, and the post-gate answer as fw_gated.
                impl = dict(rec, mode="fw_impl")
                rows.append(score_record(impl, spec))
                gated = dict(rec, mode="fw_gated", parsed=rec.get("gated_answer"),
                             forced_failure=None if rec.get("gated_answer") is not None
                             else "gated_refusal")
                rows.append(score_record(gated, spec))
                for factor in (0.5, 1.0, 2.0):
                    variant_row = dict(rec, mode=f"fw_gate@{factor:g}tol",
                                       gate_tol_factor=factor)
                    rows.append(score_record(variant_row, spec))
    with open(os.path.join(RESULT_DIR, f"by_task_{args.tag}.jsonl"), "w",
              encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    overall = summarise(rows, lambda r: (r["model"], r["mode"], r["variant"], r["level"]))
    by_family = summarise(rows, lambda r: (r["model"], r["mode"], r["family"]))
    for name, table in (("main_table", overall), ("family_table", by_family)):
        path = os.path.join(RESULT_DIR, f"{name}_{args.tag}.csv")
        keys = sorted({k for row in table for k in row})
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv_module.writer(fh)
            writer.writerow(keys)
            for row in table:
                writer.writerow([row.get(k, "") for k in keys])
        print("written:", path)
    taxonomy: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row.get("failure"):
            taxonomy[(row["model"], row["mode"])][row["failure"]] += 1
    path = os.path.join(RESULT_DIR, f"error_taxonomy_{args.tag}.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv_module.writer(fh)
        writer.writerow(["model", "mode", "failure", "count"])
        for (model, mode), counts in sorted(taxonomy.items()):
            for failure, count in sorted(counts.items()):
                writer.writerow([model, mode, failure, count])
    print("written:", path)
    runs = sorted({(r["model"], r["mode"], r["variant"]) for r in rows})
    print(f"scored {len(rows)} records over {len(runs)} runs from {len(tasks)} tasks")
    for row in overall:
        print(f"  {row['key']}: n={row['n_scored']} hit={row['hit_rate']} "
              f"[{row['hit_ci_low']},{row['hit_ci_high']}]"
              f" medAE={row['median_abs_err']} design_share={row['median_design_share']}")


if __name__ == "__main__":
    main()
