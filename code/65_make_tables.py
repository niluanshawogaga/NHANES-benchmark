"""Aggregate the scored records into the tables the manuscript prints.

    python 65_make_tables.py

Writes one CSV per table under results/paper/.  The Word builder reads these
files, so no number in the manuscript is transcribed by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PROC_DIR, RESULT_DIR, ensure_dirs  # noqa: E402

OUT_DIR = os.path.join(RESULT_DIR, "paper")
# These models were evaluated on the 100-task subset only, as Section VI-B of
# the manuscript states.  Their runs also contain partial main-level records
# from an interrupted batch, which must not enter the tables: mixing the two
# scopes would compare different question sets.
SUBSET_ONLY_MODELS = {"qwen2.5-7b", "phi-3.5-mini", "mistral-7b"}
PROTOCOL_ORDER = ["direct", "code", "sql", "fw_spec", "fw_impl", "fw_gated"]
LABELS = {"direct": "Direct", "code": "Code", "sql": "SQL", "fw_spec": "Spec+free code",
          "fw_impl": "Spec+constrained code", "fw_gated": "Framework + gate"}


def load_rows(tag: str) -> list[dict]:
    path = os.path.join(RESULT_DIR, f"by_task_{tag}.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def base_only(rows: list[dict]) -> list[dict]:
    """Main tables use the base condition; wording and serialisation variants are
    the robustness axis and must not leak into the headline numbers."""
    return [r for r in rows if r.get("variant", "base") == "base"]


def ci(values: list[float], n: int = 2000, seed: int = 20261015) -> tuple[float, float]:
    values = [v for v in values if v is not None]
    if len(values) < 5:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in range(len(values)))
                   / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def hit_rate(rows: list[dict]) -> tuple[int, float, float, float]:
    hits = [1.0 if r["hit"] else 0.0 for r in rows if r.get("hit") is not None]
    if not hits:
        return 0, float("nan"), float("nan"), float("nan")
    lo, hi = ci(hits)
    return len(hits), float(np.mean(hits)), lo, hi


def framework_cells_ready(rows: list[dict], model: str) -> bool:
    """True when a model's framework ladder has been run to completion.

    The ladder runs on the 100-task subset, whose non-ranking part is 79 tasks,
    so a model that is half way through those runs has a count far below the
    others.  Publishing such a row would report a rate over a handful of tasks
    as if it were comparable with the rest of the table.
    """
    modes = ("fw_impl", "fw_gated", "fw_gate@0.5tol", "fw_gate@1tol", "fw_gate@2tol")
    counts = [len([r for r in rows if r["model"] == model and r["mode"] == mode
                   and r["level"] == "s120" and r.get("hit") is not None])
              for mode in modes]
    largest = max(counts) if counts else 0
    if largest and largest < 75:
        print("  note: %s has %d scored framework tasks, not the full 79; its "
              "framework cells are left empty" % (model, largest))
        return False
    return True


def write(name: str, header: list[str], rows: list[list]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print("written:", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    ensure_dirs()
    rows = load_rows(args.tag)
    all_rows = rows                       # includes wording and format variants
    repl_rows: list[dict] = []
    if args.tag == "main":
        try:
            repl_rows = load_rows("repl")
        except FileNotFoundError:
            repl_rows = []
    base_rows = base_only(rows)
    import json as _json
    with open(os.path.join(PROC_DIR, "tasks_main.json"), encoding="utf-8") as fh:
        subset_ids = {t["task_id"] for t in _json.load(fh)["tasks"] if t.get("in_subset")}
    before = len(base_rows)
    base_rows = [r for r in base_rows
                 if r["model"] not in SUBSET_ONLY_MODELS or r["task_id"] in subset_ids]
    print("scope filter: %d -> %d base rows (subset-only models restricted to %d tasks)"
          % (before, len(base_rows), len(subset_ids)))
    rows = base_rows
    models = sorted({r["model"] for r in rows})

    # ---- Table III: accuracy by model and protocol on the main extract level
    table = []
    for model in models:
        line = [model]
        ready = framework_cells_ready(rows, model)
        for mode in ("direct", "code", "sql", "fw_impl", "fw_gated"):
            if mode.startswith("fw") and not ready:
                line.append("-")
                continue
            subset = [r for r in rows if r["model"] == model and r["mode"] == mode
                      and r["level"] == "s120"]
            n, hit, lo, hi = hit_rate(subset)
            line.append(f"{hit:.3f}" if n else "-")
        table.append(line)
    write("T3_accuracy.csv",
          ["Model", "Direct", "Code", "SQL", "Spec+constrained code", "Framework+gate"],
          table)

    # ---- Table IV: design-omission share
    table = []
    for model in models:
        for mode in ("direct", "code", "sql"):
            subset = [r for r in rows if r["model"] == model and r["mode"] == mode
                      and r["level"] == "s120" and r.get("design_share") is not None]
            if len(subset) < 20:
                continue
            shares = np.array([r["design_share"] for r in subset])
            table.append([model, LABELS[mode], f"{np.median(shares):.2f}",
                          f"{np.percentile(shares, 25):.2f}",
                          f"{np.percentile(shares, 75):.2f}", len(subset)])
    write("T4_gap.csv",
          ["Model", "Protocol", "median delta", "p25", "p75", "n"], table)

    # ---- Table V: failure taxonomy, share of scored tasks
    table = []
    for model in models:
        for mode in ("direct", "code"):
            subset = [r for r in rows if r["model"] == model and r["mode"] == mode
                      and r["level"] == "s120" and r.get("hit") is not None]
            if len(subset) < 20:
                continue
            counts: dict[str, int] = defaultdict(int)
            for r in subset:
                counts[r.get("failure") or "hit"] += 1
            total = sum(counts.values())
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
            table.append([model, LABELS[mode]] +
                         [f"{k} ({100*v/total:.0f}%)" for k, v in top])
    write("T5_taxonomy.csv",
          ["Model", "Protocol", "most frequent", "second", "third", "fourth", "fifth"],
          table)

    # ---- Table VI: same-scale cross-vendor comparison (subset)
    table = []
    for model in ["qwen2.5-7b", "phi-3.5-mini", "mistral-7b", "qwen2.5-3b"]:
        # A configuration with no run at all stays out of the table: a row of
        # dashes reads as a zero rather than as an experiment that was not done.
        if not any(r["model"] == model and r["mode"] in ("direct", "code", "sql")
                   for r in rows):
            continue
        line = [model]
        for mode in ("direct", "code", "sql"):
            subset = [r for r in rows if r["model"] == model and r["mode"] == mode
                      and r["level"] == "s120"]
            n, hit, lo, hi = hit_rate(subset)
            line.append(f"{hit:.3f}" if n >= 20 else "-")
        table.append(line)
    write("T6_vendor.csv", ["Model", "Direct", "Code", "SQL"], table)

    # ---- Table VII: framework ladder
    table = []
    for model in models:
        # The framework stages were run on the two API configurations only.
        if not any(r["model"] == model and r["mode"] in ("fw_spec", "fw_impl", "fw_gated")
                   for r in rows):
            continue
        if not framework_cells_ready(rows, model):
            continue
        line = [model]
        for mode in ["code", "fw_spec", "fw_impl", "fw_gated"]:
            subset = [r for r in rows if r["model"] == model and r["mode"] == mode
                      and r["level"] == "s120"]
            n, hit, lo, hi = hit_rate(subset)
            line.append(f"{hit:.3f}" if n >= 20 else "-")
        table.append(line)
    write("T7_ablation.csv",
          ["Model", "Code", "Spec + free code", "Spec + constrained code",
           "Spec + constrained code + gate"], table)

    # ---- Table VIII: gate threshold sensitivity
    table = []
    for model in models:
        if not any(r["model"] == model and r["mode"].startswith("fw_gate")
                   for r in rows):
            continue
        if not framework_cells_ready(rows, model):
            continue
        line = [model]
        for mode in ["fw_impl", "fw_gate@0.5tol", "fw_gate@1tol", "fw_gate@2tol"]:
            subset = [r for r in rows if r["model"] == model and r["mode"] == mode
                      and r["level"] == "s120"]
            n, hit, lo, hi = hit_rate(subset)
            line.append(f"{hit:.3f}" if n >= 20 else "-")
        table.append(line)
    write("T8_gate.csv",
          ["Model", "No gate", "Gate 0.5x tol", "Gate 1x tol", "Gate 2x tol"], table)

    # ---- Table IX: robustness
    table = []
    for label, model, mode, level, variant in [
            ("Replication cycle, direct", "deepseek-flash", "direct", "repl", "base"),
            ("Replication cycle, code", "deepseek-flash", "code", "repl", "base"),
            ("Extract 250 rows, direct", "deepseek-flash", "direct", "s250", "base"),
            ("Extract 250 rows, code", "deepseek-flash", "code", "s250", "base"),
            ("Extract 500 rows, direct", "deepseek-flash", "direct", "s500", "base"),
            ("Extract 500 rows, code", "deepseek-flash", "code", "s500", "base"),
            ("Paraphrased question, direct", "deepseek-flash", "direct", "s120", "paraphrase"),
            ("JSON serialisation, direct", "deepseek-flash", "direct", "s120", "fmt_json"),
            ("Markdown format, direct", "deepseek-flash", "direct", "s120", "fmt_md")]:
        pool = repl_rows if level == "repl" else all_rows
        wanted_level = "s120" if level == "repl" else level
        subset = [r for r in pool if r["model"] == model and r["mode"] == mode
                  and r["level"] == wanted_level and r.get("variant", "base") == variant]
        n, hit, lo, hi = hit_rate(subset)
        table.append([label, n, f"{hit:.3f}" if n else "-",
                      f"{lo:.3f}" if n else "-", f"{hi:.3f}" if n else "-"])
    write("T9_robustness.csv",
          ["Condition", "n", "hit rate", "95% low", "95% high"], table)


if __name__ == "__main__":
    main()
