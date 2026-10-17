"""Build the estimation benchmark: 300 tasks with programmatic ground truth.

    python 20_build_tasks.py --cycle 2017-2018 --n 300
    python 20_build_tasks.py --cycle 2015-2016 --n 100 --tag repl

Each task carries a machine-readable estimand specification, a natural-language
question, a data slice, and the weighted / unweighted reference values computed
on that very slice (so every paradigm is judged against the same data).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (ABS_TOL, EXTRA_SLICE_LEVELS, MAIN_SLICE, PROC_DIR, RANDOM_SEED,
                    SLICE_DIR, SLICE_LEVELS, SUBSET_SIZE, TASK_FAMILIES, ensure_dirs)
from estimators import as_float, estimate_with_se

WEIGHT_COLS = ["WTINT2YR", "WTMEC2YR", "WTDRD1"]
DESIGN_COLS = ["SDMVPSU", "SDMVSTRA"]
WEIGHT_BY_CLASS = {"interview": "WTINT2YR", "exam": "WTMEC2YR", "dietary": "WTDRD1"}

# ---- raw source columns exposed to the models -------------------------------
BIN_SOURCE = {
    "obesity": ["BMXBMI"], "overweight": ["BMXBMI"],
    "abdominal_obesity": ["BMXWAIST", "RIAGENDR"],
    "current_smoker": ["SMQ020", "SMQ040"], "former_smoker": ["SMQ020", "SMQ040"],
    "short_sleep": ["SLD012"], "inactive_leisure": ["PAQ650", "PAQ665"],
    "low_income": ["INDFMPIR"], "college_graduate": ["DMDEDUC2"],
    "heavy_drinker": ["ALQ130", "ALQ111"], "poor_diet_selfreport": ["DBQ700"],
}
CONT_SOURCE = {
    "bmi": "BMXBMI", "waist_cm": "BMXWAIST", "weight_kg": "BMXWT", "height_cm": "BMXHT",
    "age_years": "RIDAGEYR", "income_pir": "INDFMPIR", "sleep_hours": "SLD012",
    "energy_kcal": "DR1TKCAL", "protein_g": "DR1TPROT", "carb_g": "DR1TCARB",
    "sugar_g": "DR1TSUGR", "fiber_g": "DR1TFIBE", "sodium_mg": "DR1TSODI",
    "vitc_mg": "DR1TVC", "calcium_mg": "DR1TCALC",
}
BIN_WEIGHT = {
    "obesity": "exam", "overweight": "exam", "abdominal_obesity": "exam",
    "current_smoker": "interview", "former_smoker": "interview",
    "short_sleep": "interview", "inactive_leisure": "interview",
    "low_income": "interview", "college_graduate": "interview",
    "heavy_drinker": "interview", "poor_diet_selfreport": "interview",
}
CONT_WEIGHT = {
    "bmi": "exam", "waist_cm": "exam", "weight_kg": "exam", "height_cm": "exam",
    "age_years": "interview", "income_pir": "interview", "sleep_hours": "interview",
    "energy_kcal": "dietary", "protein_g": "dietary", "carb_g": "dietary",
    "sugar_g": "dietary", "fiber_g": "dietary", "sodium_mg": "dietary",
    "vitc_mg": "dietary", "calcium_mg": "dietary",
}

# short verb phrases for question wording
BIN_PHRASE = {
    "obesity": "are obese (body mass index at or above 30.0 kg/m2)",
    "overweight": "are overweight or obese (body mass index at or above 25.0 kg/m2)",
    "abdominal_obesity": "have abdominal obesity (waist at or above 102 cm for men "
                         "and 88 cm for women)",
    "current_smoker": "currently smoke cigarettes",
    "former_smoker": "are former smokers (smoked at least 100 cigarettes in life "
                     "but do not smoke now)",
    "short_sleep": "usually sleep less than 7 hours on weekdays",
    "inactive_leisure": "report no moderate or vigorous leisure-time physical activity",
    "low_income": "live in families with an income to poverty ratio below 1.3",
    "college_graduate": "have completed college or above",
    "heavy_drinker": "average two or more alcoholic drinks per drinking day",
    "poor_diet_selfreport": "rate their own diet as fair or poor",
}

SUBGROUPS = {
    "sex": {"label": "sex", "levels": [
        {"id": "men", "phrase": "men", "spec": {"col": "RIAGENDR", "op": "in", "vals": [1]}},
        {"id": "women", "phrase": "women", "spec": {"col": "RIAGENDR", "op": "in", "vals": [2]}},
    ]},
    "race": {"label": "race and Hispanic origin", "levels": [
        {"id": "mexican_american", "phrase": "Mexican American adults",
         "spec": {"col": "RIDRETH1", "op": "in", "vals": [1]}},
        {"id": "other_hispanic", "phrase": "other Hispanic adults",
         "spec": {"col": "RIDRETH1", "op": "in", "vals": [2]}},
        {"id": "nh_white", "phrase": "non-Hispanic White adults",
         "spec": {"col": "RIDRETH1", "op": "in", "vals": [3]}},
        {"id": "nh_black", "phrase": "non-Hispanic Black adults",
         "spec": {"col": "RIDRETH1", "op": "in", "vals": [4]}},
        {"id": "other_race", "phrase": "adults of other race including multi-racial",
         "spec": {"col": "RIDRETH1", "op": "in", "vals": [5]}},
    ]},
    "education": {"label": "education", "levels": [
        {"id": "lt_hs", "phrase": "adults with less than high school education",
         "spec": {"col": "DMDEDUC2", "op": "in", "vals": [1, 2]}},
        {"id": "hs", "phrase": "adults with a high school diploma or GED",
         "spec": {"col": "DMDEDUC2", "op": "in", "vals": [3]}},
        {"id": "college_plus", "phrase": "adults with some college education or above",
         "spec": {"col": "DMDEDUC2", "op": "in", "vals": [4, 5]}},
    ]},
    "income": {"label": "family income to poverty ratio", "levels": [
        {"id": "pir_lt_1p3", "phrase": "adults with an income to poverty ratio below 1.3",
         "spec": {"col": "INDFMPIR", "op": "range", "lo": 0.0, "hi": 1.3}},
        {"id": "pir_1p3_2p5", "phrase": "adults with an income to poverty ratio from 1.3 "
                                        "to below 2.5",
         "spec": {"col": "INDFMPIR", "op": "range", "lo": 1.3, "hi": 2.5}},
        {"id": "pir_2p5_4", "phrase": "adults with an income to poverty ratio from 2.5 "
                                      "to below 4.0",
         "spec": {"col": "INDFMPIR", "op": "range", "lo": 2.5, "hi": 4.0}},
        {"id": "pir_ge_4", "phrase": "adults with an income to poverty ratio of 4.0 or above",
         "spec": {"col": "INDFMPIR", "op": "range", "lo": 4.0, "hi": 99.0}},
    ]},
    "age_group": {"label": "age group", "levels": [
        {"id": "age_20_29", "phrase": "adults aged 20-29",
         "spec": {"col": "RIDAGEYR", "op": "range", "lo": 20, "hi": 30}},
        {"id": "age_30_44", "phrase": "adults aged 30-44",
         "spec": {"col": "RIDAGEYR", "op": "range", "lo": 30, "hi": 45}},
        {"id": "age_45_59", "phrase": "adults aged 45-59",
         "spec": {"col": "RIDAGEYR", "op": "range", "lo": 45, "hi": 60}},
        {"id": "age_60p", "phrase": "adults aged 60 or older",
         "spec": {"col": "RIDAGEYR", "op": "range", "lo": 60, "hi": 200}},
    ]},
    "marital": {"label": "marital status", "levels": [
        {"id": "married", "phrase": "adults who are married or living with a partner",
         "spec": {"col": "DMDMARTL", "op": "in", "vals": [1, 6]}},
        {"id": "formerly_married", "phrase": "adults who are widowed, divorced or separated",
         "spec": {"col": "DMDMARTL", "op": "in", "vals": [2, 3, 4]}},
        {"id": "never_married", "phrase": "adults who never married",
         "spec": {"col": "DMDMARTL", "op": "in", "vals": [5]}},
    ]},
}


def question_for(spec: dict) -> str:
    fam = spec["family"]
    subgroup = spec.get("subgroup")
    def low(text: str) -> str:
        return text[:1].lower() + text[1:] if len(text) > 1 and text[1].islower() else text
    if fam == "proportion":
        who = BIN_PHRASE[spec["target"]]
        tail = "Report the proportion as a number between 0 and 1."
        if subgroup:
            return f"Among {subgroup['phrase']}, what proportion {who}? {tail}"
        return (f"What proportion of US non-institutionalized adults aged 20 or older "
                f"{who}? {tail}")
    if fam == "mean":
        stat = "mean" if spec["kind"] == "mean" else "median"
        if subgroup:
            return (f"Among {subgroup['phrase']}, what is the population {stat} of "
                    f"{low(spec['label'])}? Report the value in {spec['unit']}.")
        return (f"What is the population {stat} of {low(spec['label'])} among US "
                f"non-institutionalized adults aged 20 or older? Report the value in "
                f"{spec['unit']}.")
    if fam == "difference":
        a, b = spec["levels"]
        if spec["kind"] == "proportion":
            return (f"Among US non-institutionalized adults aged 20 or older, what is the "
                    f"difference between {a['phrase']} and {b['phrase']} in the proportion who "
                    f"{BIN_PHRASE[spec['target']]}? Report the first group minus the second "
                    f"group as a difference in proportions.")
        return (f"Among US non-institutionalized adults aged 20 or older, what is the "
                f"difference in mean {low(spec['label'])} between {a['phrase']} and "
                f"{b['phrase']}? "
                f"Report the first group minus the second group in {spec['unit']}.")
    if fam == "ranking":
        levels = ", ".join(lv["phrase"] for lv in spec["all_levels"])
        what = (f"the proportion who {BIN_PHRASE[spec['target']]}" if spec["kind"] == "proportion"
                else f"the mean {low(spec['label'])}")
        return (f"Among US non-institutionalized adults aged 20 or older, rank these groups by "
                f"{what} from highest to lowest: {levels}. Answer with the group names in order, "
                f"separated by ' > '.")
    if fam == "association":
        return (f"Among US non-institutionalized adults aged 20 or older with both measurements, "
                f"what is the population correlation between {low(spec['x']['label'])} and "
                f"{low(spec['y']['label'])}? Report a number between -1 and 1.")
    raise ValueError(fam)


def build_estimand_pool() -> dict:
    pool: dict[str, list[dict]] = {"proportion": [], "mean": [], "difference": [],
                                   "ranking": [], "association": []}
    for target in BIN_SOURCE:
        pool["proportion"].append({"family": "proportion", "kind": "proportion",
                                   "target": target, "weight_class": BIN_WEIGHT[target],
                                   "subgroup": None})
        for attr, info in SUBGROUPS.items():
            for lvl in info["levels"]:
                pool["proportion"].append({"family": "proportion", "kind": "proportion",
                                           "target": target,
                                           "weight_class": BIN_WEIGHT[target],
                                           "subgroup": {"attribute": attr, **lvl}})
    for var in CONT_SOURCE:
        for kind in ("mean", "median"):
            pool["mean"].append({"family": "mean", "kind": kind, "target": var,
                                 "weight_class": CONT_WEIGHT[var], "subgroup": None})
            for attr, info in SUBGROUPS.items():
                for lvl in info["levels"]:
                    pool["mean"].append({"family": "mean", "kind": kind, "target": var,
                                         "weight_class": CONT_WEIGHT[var],
                                         "subgroup": {"attribute": attr, **lvl}})
    attrs = ["sex", "race", "education", "income", "age_group", "marital"]
    for attr in attrs:
        levels = SUBGROUPS[attr]["levels"]
        pairs = [(levels[i], levels[j]) for i in range(len(levels))
                 for j in range(i + 1, len(levels))]
        for target in BIN_SOURCE:
            for a, b in pairs:
                pool["difference"].append({"family": "difference", "kind": "proportion",
                                           "target": target, "weight_class": BIN_WEIGHT[target],
                                           "attribute": attr, "levels": [a, b]})
        for var in ("bmi", "age_years", "sleep_hours", "energy_kcal"):
            for a, b in pairs:
                pool["difference"].append({"family": "difference", "kind": "mean",
                                           "target": var, "weight_class": CONT_WEIGHT[var],
                                           "attribute": attr, "levels": [a, b]})
    for attr in attrs:
        if len(SUBGROUPS[attr]["levels"]) < 3:
            continue
        for target in BIN_SOURCE:
            pool["ranking"].append({"family": "ranking", "kind": "proportion",
                                    "target": target, "weight_class": BIN_WEIGHT[target],
                                    "attribute": attr, "all_levels": SUBGROUPS[attr]["levels"]})
        for var in ("bmi", "age_years", "sleep_hours", "energy_kcal", "income_pir"):
            pool["ranking"].append({"family": "ranking", "kind": "mean", "target": var,
                                    "weight_class": CONT_WEIGHT[var], "attribute": attr,
                                    "all_levels": SUBGROUPS[attr]["levels"]})
    # Association estimands only pair variables that share one weight class, so
    # the design-correct weight is unambiguous.
    groups = {
        "exam": ["bmi", "waist_cm", "weight_kg", "height_cm"],
        "interview": ["age_years", "income_pir", "sleep_hours"],
        "dietary": ["energy_kcal", "protein_g", "carb_g", "sugar_g", "fiber_g",
                    "sodium_mg", "vitc_mg", "calcium_mg"],
    }
    for _, members in groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                x, y = members[i], members[j]
                pool["association"].append({
                    "family": "association", "kind": "correlation",
                    "x": {"name": x, "binary": False}, "y": {"name": y, "binary": False},
                    "weight_class": CONT_WEIGHT[x]})
    circular = {("low_income", "income_pir"), ("short_sleep", "sleep_hours")}
    interview_bin = ["current_smoker", "former_smoker", "short_sleep", "inactive_leisure",
                     "low_income", "college_graduate", "heavy_drinker", "poor_diet_selfreport"]
    for b in interview_bin:
        for c in groups["interview"]:
            if (b, c) in circular:
                continue
            pool["association"].append({
                "family": "association", "kind": "correlation",
                "x": {"name": b, "binary": True}, "y": {"name": c, "binary": False},
                "weight_class": "interview"})
    for b, c in [("obesity", "height_cm"), ("overweight", "height_cm"),
                 ("abdominal_obesity", "height_cm"), ("obesity", "waist_cm")]:
        pool["association"].append({
            "family": "association", "kind": "correlation",
            "x": {"name": b, "binary": True}, "y": {"name": c, "binary": False},
            "weight_class": "exam"})
    return pool


def attach_labels(spec: dict, meta: dict) -> dict:
    """Add human-facing labels and the exposed/analysis column lists."""
    cont, bina = meta["continuous"], meta["binary"]
    cols = set(WEIGHT_COLS + DESIGN_COLS + ["SEQN"])
    fam = spec["family"]
    if spec["kind"] == "proportion":
        cols.update(BIN_SOURCE[spec["target"]])
        spec["label"] = bina[spec["target"]]["label"]
    elif spec["kind"] in ("mean", "median"):
        cols.add(CONT_SOURCE[spec["target"]])
        spec["label"] = cont[spec["target"]]["label"]
        spec["unit"] = cont[spec["target"]]["unit"]
    elif spec["kind"] == "correlation":
        for key in ("x", "y"):
            name = spec[key]["name"]
            if name in CONT_SOURCE:
                spec[key]["label"] = cont[name]["label"]
                cols.add(CONT_SOURCE[name])
            else:
                spec[key]["label"] = (f"{bina[name]['label']} status "
                                      f"(1 if the respondent matches, 0 otherwise)")
                cols.update(BIN_SOURCE[name])
    if spec.get("subgroup"):
        cols.add(spec["subgroup"]["spec"]["col"])
    if fam in ("difference", "ranking"):
        for lvl in (spec.get("levels") or spec.get("all_levels") or []):
            cols.add(lvl["spec"]["col"])
    spec["slice_columns"] = sorted(cols)
    return spec


def analysis_mask(df: pd.DataFrame, spec: dict) -> np.ndarray:
    """Rows that form the analysis sample for this estimand."""
    fam = spec["family"]
    mask = np.ones(len(df), dtype=bool)
    if spec.get("subgroup"):
        s = spec["subgroup"]["spec"]
        mask &= subgroup_mask(df, s)
    if fam in ("difference", "ranking"):
        levels = spec.get("levels") or spec.get("all_levels")
        union = np.zeros(len(df), dtype=bool)
        for lvl in levels:
            union |= subgroup_mask(df, lvl["spec"])
        mask &= union
    if fam == "association":
        for key in ("x", "y"):
            mask &= df[spec[key]["name"]].notna().to_numpy()
    else:
        mask &= df[spec["target"]].notna().to_numpy()
    weight_col = WEIGHT_BY_CLASS[spec["weight_class"]]
    mask &= df[weight_col].notna().to_numpy()
    return mask


def subgroup_mask(df: pd.DataFrame, s: dict) -> np.ndarray:
    if s["op"] == "in":
        return np.isin(as_float(df[s["col"]]), s["vals"])
    v = as_float(df[s["col"]])
    return (v >= s["lo"]) & (v < s["hi"])


def truth_for(df: pd.DataFrame, spec: dict) -> dict:
    """Reference values for one estimand on one data slice."""
    weight_col = WEIGHT_BY_CLASS[spec["weight_class"]]
    fam = spec["family"]
    if fam == "ranking":
        values = {}
        for lvl in spec["all_levels"]:
            sub = {**spec, "family": "mean" if spec["kind"] == "mean" else "proportion",
                   "subgroup": {"attribute": spec["attribute"], **lvl}}
            mask = subgroup_mask(df, lvl["spec"]) & df[spec["target"]].notna().to_numpy()
            values[lvl["id"]] = estimate_with_se(df[mask].reset_index(drop=True), sub, weight_col)
        order = sorted(values, key=lambda k: -(values[k]["weighted"] if
                                               not np.isnan(values[k]["weighted"]) else -9e9))
        return {"per_level": {k: v["weighted"] for k, v in values.items()},
                "per_level_unweighted": {k: v["unweighted"] for k, v in values.items()},
                "order": order, "n_analysis": int(len(df))}
    if fam == "difference":
        out = {}
        for key, lvl in (("a", spec["levels"][0]), ("b", spec["levels"][1])):
            sub = {"family": "mean" if spec["kind"] == "mean" else "proportion",
                   "kind": spec["kind"], "target": spec["target"],
                   "subgroup": {"attribute": spec["attribute"], **lvl}}
            mask = subgroup_mask(df, lvl["spec"]) & df[spec["target"]].notna().to_numpy()
            out[key] = estimate_with_se(df[mask].reset_index(drop=True), sub, weight_col)
        return {"weighted": out["a"]["weighted"] - out["b"]["weighted"],
                "unweighted": out["a"]["unweighted"] - out["b"]["unweighted"],
                "se": float(np.sqrt(out["a"]["se"] ** 2 + out["b"]["se"] ** 2))
                if not (np.isnan(out["a"]["se"]) or np.isnan(out["b"]["se"])) else float("nan"),
                "component": {k: {"weighted": v["weighted"], "unweighted": v["unweighted"],
                                  "se": v["se"], "n_analysis": v["n_analysis"]}
                              for k, v in out.items()},
                "n_analysis": int(len(df))}
    res = estimate_with_se(df, spec, weight_col)
    res["weight_class_column"] = weight_col
    return res


def build(cycle: str, n_per_family: int, tag: str, mirror: bool = False) -> None:
    ensure_dirs()
    df = pd.read_parquet(os.path.join(PROC_DIR, f"analysis_{cycle}.parquet"))
    with open(os.path.join(PROC_DIR, f"variables_{cycle}.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    rng = random.Random(RANDOM_SEED)
    if mirror:
        # Same estimands as the main benchmark subset, recomputed on this cycle.
        with open(os.path.join(PROC_DIR, "tasks_main.json"), encoding="utf-8") as fh:
            src = json.load(fh)
        keep_keys = lambda t: {k: v for k, v in t.items()
                               if k not in ("slices", "in_subset", "task_id", "question")}
        tasks = [keep_keys(t) for t in src["tasks"] if t["in_subset"]]
        print(f"mirroring {len(tasks)} estimands from the main subset")
    else:
        pool = build_estimand_pool()
        tasks = []
        for family, _ in TASK_FAMILIES:
            candidates = pool[family]
            picked = rng.sample(candidates, min(n_per_family, len(candidates)))
            for spec in picked:
                spec = attach_labels(json.loads(json.dumps(spec)), meta)
                spec["cycle"] = cycle
                tasks.append(spec)
    for i, spec in enumerate(tasks):
        spec["task_id"] = f"{cycle[:4]}-{i+1:03d}"
        spec["question"] = question_for(spec)
        fingerprint = {k: spec[k] for k in sorted(spec)
                       if k not in ("question", "slice_columns", "slices")}
        spec["spec_hash"] = hashlib.sha1(
            json.dumps(fingerprint, sort_keys=True, default=str).encode()).hexdigest()[:16]

    subset_ids = ({t["task_id"] for t in tasks} if mirror
                  else {t["task_id"] for t in rng.sample(tasks, min(SUBSET_SIZE, len(tasks)))})
    records = []
    for spec in tasks:
        rec = {k: v for k, v in spec.items()}
        rec["in_subset"] = spec["task_id"] in subset_ids
        rec["slices"] = {}
        mask = analysis_mask(df, spec)
        pool_df = df[mask].reset_index(drop=True)
        levels = [MAIN_SLICE] + (EXTRA_SLICE_LEVELS if rec["in_subset"] else [])
        for level in levels:
            size = SLICE_LEVELS[level]
            if len(pool_df) > size:
                idx = random.Random(f"{spec['task_id']}|{level}").sample(range(len(pool_df)), size)
                idx.sort()
                sl = pool_df.iloc[idx].reset_index(drop=True)
            else:
                sl = pool_df.copy()
            exposed = sl[[c for c in spec["slice_columns"] if c in sl.columns]].copy()
            exposed.to_parquet(os.path.join(SLICE_DIR, f"{spec['task_id']}_{level}.parquet"),
                               index=False)
            truth = truth_for(sl, spec)
            truth["slice_level"] = level
            truth["slice_rows"] = int(len(sl))
            rec["slices"][level] = {"file": f"{spec['task_id']}_{level}.parquet", **truth}
        records.append(rec)

    out = os.path.join(PROC_DIR, f"tasks_{tag}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"cycle": cycle, "tag": tag, "weight_by_class": WEIGHT_BY_CLASS,
                   "abs_tol": ABS_TOL, "n_tasks": len(records), "tasks": records},
                  fh, ensure_ascii=False, indent=1)
    fam_count = {}
    for r in records:
        fam_count[r["family"]] = fam_count.get(r["family"], 0) + 1
    print(f"tasks={len(records)} subset={len(subset_ids)} -> {out}")
    print("  by family:", fam_count)
    print("  slice rows (main):",
          sorted({r["slices"][MAIN_SLICE]["slice_rows"] for r in records})[:8], "...")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", default="2017-2018")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--mirror", action="store_true",
                    help="reuse the main benchmark's subset estimands on this cycle")
    args = ap.parse_args()
    build(args.cycle, args.n, args.tag, args.mirror)


if __name__ == "__main__":
    main()
