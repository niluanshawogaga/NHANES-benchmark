"""Build the respondent-level analysis tables for both NHANES cycles.

    python 10_build_data.py --cycle 2017-2018
    python 10_build_data.py --all

Writes data/processed/analysis_<cycle>.parquet and data/processed/variables.json.
The JSON file is the single source of truth for variable metadata used by every
later stage (task construction, prompts, tables).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (ADULT_MIN_AGE, CYCLES, DOWNLOAD_BASE, MODULE_STEMS, PROC_DIR,
                    RAW_DIR, SRC_NHANES, ensure_dirs)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# Raw columns carried into the analysis table.  ``valid`` lists the admissible
# codes for categorical items (CDC refused / don't-know codes are dropped).
RAW_PULL = {
    "DEMO": ["RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "DMDMARTL", "INDFMPIR",
             "WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"],
    "BMX": ["BMXBMI", "BMXWAIST", "BMXWT", "BMXHT"],
    "SLQ": ["SLD012"],
    "SMQ": ["SMQ020", "SMQ040", "SMD030"],
    "ALQ": ["ALQ111", "ALQ130"],
    "PAQ": ["PAQ605", "PAQ620", "PAQ650", "PAQ665"],
    "DBQ": ["DBQ700"],
    "DR1TOT": ["DR1DRSTZ", "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR",
               "DR1TFIBE", "DR1TSODI", "DR1TVC", "DR1TCALC", "WTDRD1"],
}

CATE_VALID = {
    "RIAGENDR": {1, 2},
    "RIDRETH1": {1, 2, 3, 4, 5},
    "DMDEDUC2": {1, 2, 3, 4, 5},
    "DMDMARTL": {1, 2, 3, 4, 5, 6},
    "SMQ020": {1, 2},
    "SMQ040": {1, 2, 3},
    "ALQ111": {1, 2},
    "PAQ605": {1, 2},
    "PAQ620": {1, 2},
    "PAQ650": {1, 2},
    "PAQ665": {1, 2},
    "DBQ700": {1, 2, 3, 4, 5},
    "DR1DRSTZ": {1, 2, 3, 4, 5},
}
RANGE_VALID = {
    "SLD012": (1, 14),
    "ALQ130": (0, 15),
    "SMD030": (0, 100),
    "RIDAGEYR": (0, 80),
    "INDFMPIR": (0.0, 5.0),
}

# A few questionnaire items were renamed between cycles; map the canonical
# (2017-2018) name onto the equivalent column of the other cycle.
ALIAS_BY_CYCLE = {
    "2015-2016": {"ALQ111": "ALQ101"},
}

VALUE_LABELS = {
    "sex": {"1": "Male", "2": "Female"},
    "race": {"1": "Mexican American", "2": "Other Hispanic",
             "3": "Non-Hispanic White", "4": "Non-Hispanic Black",
             "5": "Other race, including multi-racial"},
    "education": {"lt_hs": "Less than high school", "hs": "High school graduate or GED",
                  "college": "Some college or above"},
    "income_q": {"1": "PIR quartile 1 (lowest)", "2": "PIR quartile 2",
                 "3": "PIR quartile 3", "4": "PIR quartile 4 (highest)"},
    "age_group": {"20-29": "20-29 years", "30-44": "30-44 years",
                  "45-59": "45-59 years", "60+": "60 years or older"},
    "marital": {"married": "Married or living with partner",
                "formerly": "Widowed, divorced or separated",
                "never": "Never married"},
}

# name -> (label, unit, weight class, min age, source columns that must be present)
CONTINUOUS = {
    "bmi": ("Body mass index", "kg/m2", "exam", ADULT_MIN_AGE, ["BMXBMI"]),
    "waist_cm": ("Waist circumference", "cm", "exam", ADULT_MIN_AGE, ["BMXWAIST"]),
    "weight_kg": ("Body weight", "kg", "exam", ADULT_MIN_AGE, ["BMXWT"]),
    "height_cm": ("Standing height", "cm", "exam", ADULT_MIN_AGE, ["BMXHT"]),
    "age_years": ("Age at interview", "years", "interview", 0, ["RIDAGEYR"]),
    "income_pir": ("Family income to poverty ratio", "ratio", "interview",
                   ADULT_MIN_AGE, ["INDFMPIR"]),
    "sleep_hours": ("Usual sleep hours on weekdays", "hours", "interview",
                    ADULT_MIN_AGE, ["SLD012"]),
    "energy_kcal": ("Total energy intake, day 1 recall", "kcal", "dietary",
                    ADULT_MIN_AGE, ["DR1TKCAL", "DR1DRSTZ"]),
    "protein_g": ("Total protein intake, day 1 recall", "g", "dietary",
                  ADULT_MIN_AGE, ["DR1TPROT", "DR1DRSTZ"]),
    "carb_g": ("Total carbohydrate intake, day 1 recall", "g", "dietary",
               ADULT_MIN_AGE, ["DR1TCARB", "DR1DRSTZ"]),
    "sugar_g": ("Total sugars intake, day 1 recall", "g", "dietary",
                ADULT_MIN_AGE, ["DR1TSUGR", "DR1DRSTZ"]),
    "fiber_g": ("Total dietary fiber intake, day 1 recall", "g", "dietary",
                ADULT_MIN_AGE, ["DR1TFIBE", "DR1DRSTZ"]),
    "sodium_mg": ("Total sodium intake, day 1 recall", "mg", "dietary",
                  ADULT_MIN_AGE, ["DR1TSODI", "DR1DRSTZ"]),
    "vitc_mg": ("Total vitamin C intake, day 1 recall", "mg", "dietary",
                ADULT_MIN_AGE, ["DR1TVC", "DR1DRSTZ"]),
    "calcium_mg": ("Total calcium intake, day 1 recall", "mg", "dietary",
                   ADULT_MIN_AGE, ["DR1TCALC", "DR1DRSTZ"]),
}

# name -> (label, weight class, needs, rule)
BINARY_RULES = {
    "obesity": ("Obesity (BMI at or above 30.0)", "exam", ["BMXBMI"],
                lambda d: d["BMXBMI"] >= 30),
    "overweight": ("Overweight or obesity (BMI at or above 25.0)", "exam", ["BMXBMI"],
                   lambda d: d["BMXBMI"] >= 25),
    "abdominal_obesity": ("Abdominal obesity (waist at or above 102 cm for men, "
                          "88 cm for women)", "exam", ["BMXWAIST", "RIAGENDR"],
                          lambda d: np.where(d["RIAGENDR"] == 1,
                                             d["BMXWAIST"] >= 102, d["BMXWAIST"] >= 88)),
    "current_smoker": ("Currently smokes cigarettes", "interview",
                       ["SMQ020"],
                       lambda d: (d["SMQ020"] == 1) & (d["SMQ040"] == 1)),
    "former_smoker": ("Smoked at least 100 cigarettes but does not smoke now",
                      "interview", ["SMQ020"],
                      lambda d: (d["SMQ020"] == 1) & (d["SMQ040"] == 3)),
    "short_sleep": ("Usual weekday sleep below 7 hours", "interview", ["SLD012"],
                    lambda d: d["SLD012"] < 7),
    "inactive_leisure": ("No vigorous and no moderate leisure-time physical activity",
                         "interview", ["PAQ650", "PAQ665"],
                         lambda d: (d["PAQ650"] == 2) & (d["PAQ665"] == 2)),
    "low_income": ("Family income below 1.3 times the poverty threshold",
                   "interview", ["INDFMPIR"], lambda d: d["INDFMPIR"] < 1.3),
    "college_graduate": ("Completed college or above", "interview", ["DMDEDUC2"],
                         lambda d: d["DMDEDUC2"] == 5),
    "heavy_drinker": ("Two or more alcoholic drinks per day on average",
                      "interview", ["ALQ111"], lambda d: d["ALQ130"] >= 2),
    "poor_diet_selfreport": ("Self-rated diet fair or poor", "interview", ["DBQ700"],
                             lambda d: d["DBQ700"].isin([4, 5])),
}

SUBGROUPS = {
    "sex": ("sex", "Sex"),
    "race": ("race", "Race and Hispanic origin"),
    "education": ("education", "Education (adults 20+)"),
    "income_q": ("income_q", "Family income to poverty ratio quartile"),
    "age_group": ("age_group", "Age group"),
    "marital": ("marital", "Marital status"),
}


def fetch_cycle(cycle: str, respect_existing: bool = True) -> str:
    """Download the cycle's modules into data/raw/<cycle>/ (returns the folder)."""
    meta = CYCLES[cycle]
    out = os.path.join(RAW_DIR, cycle)
    os.makedirs(out, exist_ok=True)
    local_src = os.path.join(SRC_NHANES, "raw") if SRC_NHANES else ""
    for stem in MODULE_STEMS:
        name = f"{stem}_{meta['suffix']}.XPT"
        dst = os.path.join(out, name)
        if respect_existing and os.path.exists(dst) and os.path.getsize(dst) > 50_000:
            continue
        reuse = os.path.join(local_src, name) if local_src else ""
        if reuse and os.path.exists(reuse) and os.path.getsize(reuse) > 50_000:
            shutil.copyfile(reuse, dst)
            print(f"[local] {name} reused from {local_src}", flush=True)
            continue
        url = DOWNLOAD_BASE.format(year=meta["year"], stem=f"{stem}_{meta['suffix']}")
        print(f"[download] {url}", flush=True)
        with requests.get(url, headers=HEADERS, stream=True, timeout=300) as r:
            r.raise_for_status()
            tmp = dst + ".part"
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            os.replace(tmp, dst)
        print(f"[download] {name} {os.path.getsize(dst)/1e6:.1f} MB", flush=True)
    return out


def read_module(folder: str, stem: str, suffix: str, cols: list[str],
                aliases: dict | None = None) -> pd.DataFrame:
    path = os.path.join(folder, f"{stem}_{suffix}.XPT")
    df = pd.read_sas(path, format="xport")
    for canonical, alt in (aliases or {}).items():
        if canonical not in df.columns and alt in df.columns:
            df[canonical] = df[alt]
            print(f"  note: {stem}_{suffix} uses {alt} for {canonical}")
    keep = ["SEQN"] + [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  note: {stem}_{suffix} lacks {missing}")
    return df[keep].copy()


def clean_codes(df: pd.DataFrame) -> pd.DataFrame:
    for col, valid in CATE_VALID.items():
        if col in df.columns:
            df.loc[~df[col].isin(list(valid)), col] = np.nan
    for col, (lo, hi) in RANGE_VALID.items():
        if col in df.columns:
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def build_table(cycle: str) -> pd.DataFrame:
    folder = fetch_cycle(cycle)
    suffix = CYCLES[cycle]["suffix"]
    aliases = ALIAS_BY_CYCLE.get(cycle, {})
    parts = {stem: read_module(folder, stem, suffix, cols, aliases)
             for stem, cols in RAW_PULL.items()}
    base = parts["DEMO"]
    for stem in MODULE_STEMS:
        if stem == "DEMO":
            continue
        extra = parts[stem].drop(columns=["SEQN"], errors="ignore")
        overlap = [c for c in extra.columns if c in base.columns]
        extra = extra.drop(columns=overlap)
        if not extra.empty:
            base = base.merge(parts[stem][["SEQN"]].join(extra), on="SEQN", how="left")
    df = clean_codes(base)
    df["cycle"] = cycle

    # ---- subgroup variables
    df["sex"] = df["RIAGENDR"].map({1.0: "1", 2.0: "2"})
    df["race"] = df["RIDRETH1"].map({float(k): k for k in range(1, 6)})
    edu_map = {1: "lt_hs", 2: "lt_hs", 3: "hs", 4: "college", 5: "college"}
    df["education"] = df["DMDEDUC2"].map({float(k): v for k, v in edu_map.items()})
    mar_map = {1: "married", 6: "married", 2: "formerly", 3: "formerly",
               4: "formerly", 5: "never"}
    df["marital"] = df["DMDMARTL"].map({float(k): v for k, v in mar_map.items()})
    adult = df["RIDAGEYR"] >= ADULT_MIN_AGE
    pir = df["INDFMPIR"].where(adult)
    df["income_q"] = pd.Series(index=df.index, dtype="object")
    if pir.notna().sum() > 100:
        q = pd.qcut(pir, 4, labels=["1", "2", "3", "4"])
        df["income_q"] = q.astype("object")
    bins = [-0.1, 29, 44, 59, 200]
    labs = ["20-29", "30-44", "45-59", "60+"]
    df["age_group"] = pd.cut(df["RIDAGEYR"].where(adult), bins=bins, labels=labs).astype("object")

    # ---- continuous estimands
    rename = {"bmi": "BMXBMI", "waist_cm": "BMXWAIST", "weight_kg": "BMXWT",
              "height_cm": "BMXHT", "age_years": "RIDAGEYR", "income_pir": "INDFMPIR",
              "sleep_hours": "SLD012", "energy_kcal": "DR1TKCAL", "protein_g": "DR1TPROT",
              "carb_g": "DR1TCARB", "sugar_g": "DR1TSUGR", "fiber_g": "DR1TFIBE",
              "sodium_mg": "DR1TSODI", "vitc_mg": "DR1TVC", "calcium_mg": "DR1TCALC"}
    for new, src in rename.items():
        df[new] = df[src] if src in df.columns else np.nan
    if "DR1DRSTZ" in df.columns:
        unreliable = df["DR1DRSTZ"] != 1
        for new in ("energy_kcal", "protein_g", "carb_g", "sugar_g", "fiber_g",
                    "sodium_mg", "vitc_mg", "calcium_mg"):
            df.loc[unreliable, new] = np.nan

    # ---- binary estimands
    for name, (_, _, needs, rule) in BINARY_RULES.items():
        if any(col not in df.columns for col in needs):
            print(f"  skip {name}: missing {[c for c in needs if c not in df.columns]}")
            df[name] = np.nan
            continue
        ok = np.ones(len(df), dtype=bool)
        for col in needs:
            ok &= df[col].notna().to_numpy()
        ok &= (df["RIDAGEYR"] >= ADULT_MIN_AGE).to_numpy()
        values = rule(df)
        df[name] = np.where(ok, np.asarray(values, dtype=float), np.nan)
    return df


def build_variables_meta(df: pd.DataFrame) -> dict:
    meta = {"continuous": {}, "binary": {}, "subgroups": {}, "weights": {}}
    for name, (label, unit, wclass, min_age, needs) in CONTINUOUS.items():
        ok = df["RIDAGEYR"] >= min_age
        for c in needs:
            ok &= df[c].notna()
        meta["continuous"][name] = {
            "label": label, "unit": unit, "weight_class": wclass,
            "min_age": int(min_age), "analysis_n": int((ok & df[name].notna()).sum())}
    for name, (label, wclass, needs, _) in BINARY_RULES.items():
        meta["binary"][name] = {
            "label": label, "weight_class": wclass, "min_age": int(ADULT_MIN_AGE),
            "analysis_n": int(df[name].notna().sum()),
            "prevalence_unweighted": round(float(df[name].mean()), 4)}
    for name, (col, label) in SUBGROUPS.items():
        vals = [v for v in df[col].dropna().unique().tolist()]
        vals = sorted(vals, key=lambda v: str(v))
        meta["subgroups"][name] = {"column": col, "label": label, "levels": vals,
                                   "level_labels": VALUE_LABELS.get(name, {}),
                                   "n_by_level": {str(v): int((df[col] == v).sum())
                                                  for v in vals}}
    for key, col in (("interview", "WTINT2YR"), ("exam", "WTMEC2YR"), ("dietary", "WTDRD1")):
        meta["weights"][key] = {"column": col, "n_nonmissing": int(df[col].notna().sum()),
                                "sum_millions": round(float(df[col].sum() / 1e6), 1)}
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    cycles = list(CYCLES) if args.all else [args.cycle or list(CYCLES)[0]]
    merged_meta = None
    for cycle in cycles:
        print(f"=== {cycle} ===")
        df = build_table(cycle)
        out = os.path.join(PROC_DIR, f"analysis_{cycle}.parquet")
        df.to_parquet(out, index=False)
        meta = build_variables_meta(df)
        meta["cycle"] = cycle
        meta["n_rows"] = int(len(df))
        with open(os.path.join(PROC_DIR, f"variables_{cycle}.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        print(f"rows={len(df)} -> {out}")
        print("  binaries:", ", ".join(f"{k}={v['analysis_n']}" for k, v in
                                       list(meta["binary"].items())[:6]))
        if merged_meta is None:
            merged_meta = meta
    if merged_meta is not None:
        with open(os.path.join(PROC_DIR, "variables.json"), "w", encoding="utf-8") as fh:
            json.dump(merged_meta, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
