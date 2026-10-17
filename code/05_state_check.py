"""Pre-flight audit: verify everything a resumed run depends on.

    python 05_state_check.py

Prints one line per check with OK / MISSING / STALE so a resumed session can
see immediately what is safe to build on.  Read-only: it never writes results.
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CYCLES, GEN_DIR, MAIN_SLICE, MODEL_DIR, PROC_DIR,  # noqa: E402
                    ROBUST_SLICE, SLICE_DIR, TASK_FAMILIES)

OK, BAD, WARN = "OK  ", "MISS", "WARN"
lines: list[tuple[str, str]] = []


def check(flag: str, text: str) -> None:
    lines.append((flag, text))


def main() -> None:
    # ---- data tables
    for cycle in CYCLES:
        path = os.path.join(PROC_DIR, f"analysis_{cycle}.parquet")
        if not os.path.exists(path):
            check(BAD, f"analysis table {cycle} missing")
            continue
        import pandas as pd
        df = pd.read_parquet(path)
        needed = ["WTINT2YR", "WTMEC2YR", "SDMVPSU", "SDMVSTRA", "obesity", "bmi"]
        missing = [c for c in needed if c not in df.columns]
        check(OK if not missing else BAD,
              f"analysis {cycle}: {len(df)} rows"
              + (f", missing {missing}" if missing else ", weights/design/derived present"))
    # ---- tasks and ground truth
    for tag, count in (("main", 300), ("repl", 100)):
        path = os.path.join(PROC_DIR, f"tasks_{tag}.json")
        if not os.path.exists(path):
            check(BAD, f"tasks_{tag}.json missing")
            continue
        payload = json.load(open(path, encoding="utf-8"))
        tasks = payload["tasks"]
        fam: dict[str, int] = {}
        for t in tasks:
            fam[t["family"]] = fam.get(t["family"], 0) + 1
        subset = sum(1 for t in tasks if t.get("in_subset"))
        levels = sorted({lv for t in tasks for lv in t["slices"]})
        no_truth = sum(1 for t in tasks if t["slices"][MAIN_SLICE].get("weighted") is None
                       and t["family"] != "ranking")
        flag = OK if len(tasks) == count and no_truth == 0 else BAD
        check(flag, f"tasks_{tag}: {len(tasks)} tasks {fam} subset={subset} "
                    f"levels={levels} missing_truth={no_truth}")
    # ---- slices
    for level, expect in ((MAIN_SLICE, 400), (ROBUST_SLICE, 200)):
        files = glob.glob(os.path.join(SLICE_DIR, f"*_{level}.parquet"))
        check(OK if len(files) >= expect * 0.9 else WARN,
              f"slice parquet {level}: {len(files)} files (expected about {expect})")
    # ---- oracle validation (the checkpoint that licenses everything else)
    for tag, want in (("main", 240), ("repl", 79)):
        path = os.path.join(GEN_DIR, f"reference__oracle__{tag}__{MAIN_SLICE}.jsonl")
        if not os.path.exists(path):
            check(BAD, f"oracle validation for {tag}/{MAIN_SLICE} missing")
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        exact = sum(1 for r in rows if r.get("abs_diff") is not None and r["abs_diff"] < 1e-6)
        check(OK if exact == len(rows) == want else BAD,
              f"oracle {tag}/{MAIN_SLICE}: {exact}/{len(rows)} ground-truth values reproduced exactly")
    # ---- model runs
    usable = {}
    for path in glob.glob(os.path.join(GEN_DIR, "*.jsonl")):
        name = os.path.basename(path)[:-6]
        if name.startswith("reference__"):
            continue
        parts = name.split("__")
        if len(parts) < 4:
            continue
        level = parts[3]
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        fresh = sum(1 for r in rows if not r.get("api_error"))
        usable[name] = (len(rows), fresh, level)
    by_level: dict[str, list[str]] = {}
    for name, (total, fresh, level) in usable.items():
        by_level.setdefault(level, []).append(f"{name}({fresh}/{total})")
    for level, names in sorted(by_level.items()):
        flag = OK if level in (MAIN_SLICE, ROBUST_SLICE) else WARN
        check(flag, f"model runs at level {level}: {len(names)} files -> "
                    + ", ".join(names[:6]) + (" ..." if len(names) > 6 else ""))
    if not usable:
        check(BAD, "no model runs recorded yet")
    # ---- environment
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    check(OK if key else BAD, f"DEEPSEEK_API_KEY {'present' if key else 'MISSING'} "
                              f"(len={len(key)})")
    total_gb = 0.0
    if os.path.isdir(MODEL_DIR):
        for root, _, files in os.walk(MODEL_DIR):
            for f in files:
                try:
                    total_gb += os.path.getsize(os.path.join(root, f)) / 1e9
                except OSError:
                    pass
    check(OK if total_gb > 1 else WARN, f"local model cache: {total_gb:.1f} GB in {MODEL_DIR}")
    from config import LOCAL_MODELS
    local_3b = next(m.get("path", "") for m in LOCAL_MODELS if m["id"] == "qwen2.5-3b")
    check(OK if local_3b and os.path.isdir(local_3b) else WARN,
          f"Qwen2.5-3B snapshot {'available' if local_3b and os.path.isdir(local_3b) else 'not configured'}")
    # ---- scripts
    here = os.path.dirname(os.path.abspath(__file__))
    for script in ("10_build_data.py", "20_build_tasks.py", "oracle.py", "prompts.py",
                   "40_run_api.py", "50_run_local.py", "60_evaluate.py", "70_figures.py",
                   "91_verify_claims.py"):
        check(OK if os.path.exists(os.path.join(here, script)) else BAD, f"script {script}")

    width = max(len(t) for _, t in lines)
    for flag, text in lines:
        print(f"[{flag}] {text:<{width}}")
    bad = sum(1 for f, _ in lines if f == BAD)
    warn = sum(1 for f, _ in lines if f == WARN)
    print(f"\n{bad} blocking, {warn} advisory")
    return


if __name__ == "__main__":
    main()
