"""Reference pipeline: the design-aware program a careful analyst would write.

Used for two purposes, both documented in the paper:
  * validating the benchmark (the reference program, executed through the same
    sandbox the models use, must reproduce the stored ground truth);
  * providing the "reference pipeline" row in the results tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P  # noqa: E402
from config import GEN_DIR, PROC_DIR, SLICE_DIR, ensure_dirs  # noqa: E402

BIN_EXPR = {
    "obesity": "(df['BMXBMI'] >= 30)",
    "overweight": "(df['BMXBMI'] >= 25)",
    "abdominal_obesity": "(((df['RIAGENDR'] == 1) & (df['BMXWAIST'] >= 102)) | "
                         "((df['RIAGENDR'] == 2) & (df['BMXWAIST'] >= 88)))",
    "current_smoker": "((df['SMQ020'] == 1) & (df['SMQ040'] == 1))",
    "former_smoker": "((df['SMQ020'] == 1) & (df['SMQ040'] == 3))",
    "short_sleep": "(df['SLD012'] < 7)",
    "inactive_leisure": "((df['PAQ650'] == 2) & (df['PAQ665'] == 2))",
    "low_income": "(df['INDFMPIR'] < 1.3)",
    "college_graduate": "(df['DMDEDUC2'] == 5)",
    "heavy_drinker": "(df['ALQ130'] >= 2)",
    "poor_diet_selfreport": "df['DBQ700'].isin([4, 5])",
}
CONT_COL = {
    "bmi": "BMXBMI", "waist_cm": "BMXWAIST", "weight_kg": "BMXWT", "height_cm": "BMXHT",
    "age_years": "RIDAGEYR", "income_pir": "INDFMPIR", "sleep_hours": "SLD012",
    "energy_kcal": "DR1TKCAL", "protein_g": "DR1TPROT", "carb_g": "DR1TCARB",
    "sugar_g": "DR1TSUGR", "fiber_g": "DR1TFIBE", "sodium_mg": "DR1TSODI",
    "vitc_mg": "DR1TVC", "calcium_mg": "DR1TCALC",
}

HEADER = """import numpy as np
import pandas as pd
df = pd.read_csv(DATA_PATH)


def wmean(y, w):
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    keep = ~(np.isnan(y) | np.isnan(w))
    return float(np.sum(w[keep] * y[keep]) / np.sum(w[keep]))


def wmedian(y, w):
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    keep = ~(np.isnan(y) | np.isnan(w))
    y, w = y[keep], w[keep]
    order = np.argsort(y, kind="stable")
    y = y[order]
    cw = np.cumsum(w[order]) / np.sum(w)
    idx = int(np.searchsorted(cw, 0.5, side="left"))
    return float(y[min(idx, len(y) - 1)])


def wcorr(x, y, w):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y) | np.isnan(w))
    x, y, w = x[keep], y[keep], w[keep]
    w = w / w.sum()
    mx, my = np.sum(w * x), np.sum(w * y)
    return float(np.sum(w * (x - mx) * (y - my)) /
                 np.sqrt(np.sum(w * (x - mx) ** 2) * np.sum(w * (y - my) ** 2)))


"""


def subgroup_expr(spec: dict, s: dict) -> str:
    if s["op"] == "in":
        vals = ", ".join(str(v) for v in s["vals"])
        return f"df['{s['col']}'].isin([{vals}])"
    return f"((df['{s['col']}'] >= {s['lo']}) & (df['{s['col']}'] < {s['hi']}))"


def value_expr(spec: dict) -> str:
    if spec["kind"] == "proportion":
        return BIN_EXPR[spec["target"]]
    return f"df['{CONT_COL[spec['target']]}']"


def reference_code(spec: dict) -> str:
    weight_col = {"interview": "WTINT2YR", "exam": "WTMEC2YR", "dietary": "WTDRD1"}[
        spec["weight_class"]]
    fam = spec["family"]
    body = [HEADER, f"w = df['{weight_col}']"]
    if fam == "proportion":
        sub = subgroup_expr(spec, spec["subgroup"]["spec"]) if spec.get("subgroup") else "df.index == df.index"
        body.append(f"keep = {sub}")
        body.append(f"y = ({value_expr(spec)}).astype(float)")
        body.append("answer = wmean(y[keep], w[keep])")
    elif fam == "mean":
        sub = subgroup_expr(spec, spec["subgroup"]["spec"]) if spec.get("subgroup") else "df.index == df.index"
        fn = "wmean" if spec["kind"] == "mean" else "wmedian"
        body.append(f"keep = {sub}")
        body.append(f"answer = {fn}({value_expr(spec)}[keep], w[keep])")
    elif fam == "difference":
        a, b = spec["levels"]
        fn = "wmean" if spec["kind"] == "mean" else "wmean"
        body.append(f"ya = ({value_expr(spec)}).astype(float)")
        for tag, lvl in (("a", a), ("b", b)):
            body.append(f"keep_{tag} = {subgroup_expr(spec, lvl['spec'])}")
            body.append(f"v_{tag} = wmean(ya[keep_{tag}], w[keep_{tag}])")
        body.append("answer = v_a - v_b")
    elif fam == "association":
        xcol = (BIN_EXPR[spec["x"]["name"]] + ".astype(float)"
                if spec["x"]["name"] in BIN_EXPR else f"df['{CONT_COL[spec['x']['name']]}']")
        ycol = (BIN_EXPR[spec["y"]["name"]] + ".astype(float)"
                if spec["y"]["name"] in BIN_EXPR else f"df['{CONT_COL[spec['y']['name']]}']")
        body.append(f"answer = wcorr({xcol}, {ycol}, w)")
    else:
        raise ValueError(fam)
    return "\n".join(body) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="main")
    ap.add_argument("--level", default="s500")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None, help="output file name suffix")
    args = ap.parse_args()
    ensure_dirs()
    with open(os.path.join(PROC_DIR, f"tasks_{args.tasks}.json"), encoding="utf-8") as fh:
        tasks = json.load(fh)["tasks"]
    tasks = [t for t in tasks if t["family"] != "ranking"]
    if args.limit:
        tasks = tasks[:args.limit]
    # A larger extract level exists for the subset only; skip the rest instead of
    # failing, so one command can validate whichever levels are present.
    tasks = [t for t in tasks if args.level in t["slices"]]
    rows, ok = [], 0
    for spec in tasks:
        csv_path, _ = P.materialize(spec, args.level, SLICE_DIR)
        code = reference_code(spec)
        res = P.run_code(code, csv_path)
        truth = spec["slices"][args.level]["weighted"]
        got = res.get("answer")
        if got is None or truth is None or truth != truth:
            diff = None
        else:
            diff = abs(got - truth)
            if diff < 1e-6:
                ok += 1
        rows.append({"task_id": spec["task_id"], "family": spec["family"],
                     "kind": spec["kind"], "weighted_truth": truth, "reference": got,
                     "abs_diff": diff, "ok": res.get("ok"), "error": res.get("error")})
    print(f"reference pipeline reproduced {ok}/{len(rows)} ground-truth values exactly")
    bad = [r for r in rows if r["abs_diff"] is None or r["abs_diff"] >= 1e-6]
    for r in bad[:8]:
        print("  mismatch:", r)
    out = os.path.join(GEN_DIR, f"reference__oracle__{args.tasks}__{args.level}"
                                f"{('__' + args.tag) if args.tag else ''}.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("written:", out)


if __name__ == "__main__":
    main()
