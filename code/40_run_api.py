"""Run the benchmark against the API models.

    python 40_run_api.py --model deepseek-flash --mode direct --tasks main --level s500
    python 40_run_api.py --model deepseek-flash --mode fw --tasks main --level s500 --subset

``--mode`` is one of direct | code | sql | fw_s1 | fw (specification, constrained
code, independent recomputation and the consistency gate).  Every record is
appended to a jsonl file keyed by task id, so runs are resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P  # noqa: E402
from config import (GEN_DIR, PROC_DIR, SLICE_DIR, SRC_CODEBOOK,  # noqa: E402
                    ensure_dirs)

API_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
TIMEOUT = 90
# Thinking is disabled so that both API models run under the same protocol; the
# cap is generous enough that a long answer is never cut mid-sentence.
THINKING = {"type": "disabled"}
MAX_TOKENS = {"direct": 900, "code": 1400, "sql": 800, "fw_s1": 800,
              "fw_s2": 1400, "fw_full": 1400, "fw_spec": 1400}
# The endpoint rejects with HTTP 503 when it is saturated (that response arrives
# in milliseconds) and occasionally stalls on long contexts.  Short, repeated
# retries ride out the former; the circuit breaker below stops the run from
# burning an hour on the latter.
BACKOFF = [1, 2, 3, 5, 8, 13, 20, 30, 45]
CIRCUIT_AFTER = 30
CIRCUIT_SLEEP = 300
_LOCK = threading.Lock()
_STATS = {"calls": 0, "ok": 0, "consecutive_fail": 0}


def call_model(model: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, "stream": False,
            "thinking": THINKING}
    last = None
    for attempt, wait in enumerate(BACKOFF + [None]):
        started = time.time()
        try:
            r = requests.post(API_BASE.rstrip("/") + "/chat/completions",
                              headers={"Authorization": f"Bearer {API_KEY}",
                                       "Content-Type": "application/json"},
                              json=body, timeout=TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                text = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage", {})
                if not text.strip():
                    last = "empty completion (reasoning may have consumed the budget)"
                else:
                    with _LOCK:
                        _STATS["calls"] += 1
                        _STATS["ok"] += 1
                        _STATS["consecutive_fail"] = 0
                    return {"text": text, "usage": usage,
                            "latency": time.time() - started, "error": None}
            else:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        with _LOCK:
            _STATS["calls"] += 1
            _STATS["consecutive_fail"] += 1
            tripped = _STATS["consecutive_fail"] >= CIRCUIT_AFTER
            if tripped:
                _STATS["consecutive_fail"] = 0
        if tripped:
            print(f"  [circuit breaker] {CIRCUIT_AFTER} consecutive failures, "
                  f"pausing {CIRCUIT_SLEEP}s", flush=True)
            time.sleep(CIRCUIT_SLEEP)
        if wait is None:
            break
        time.sleep(wait + RANDOM_JITTER())
    return {"text": "", "usage": {}, "latency": None, "error": last, "api_error": True}


def RANDOM_JITTER() -> float:
    import random
    return random.uniform(0, 2.0)


def weight_mentions(text: str) -> list[str]:
    return [c for c in ("WTINT2YR", "WTMEC2YR", "WTDRD1") if c and c in (text or "")]


def load_tasks(tag: str, subset: bool, limit: int | None) -> list[dict]:
    path = os.path.join(PROC_DIR, f"tasks_{tag}.json")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    tasks = payload["tasks"]
    if subset:
        tasks = [t for t in tasks if t.get("in_subset")]
    if limit:
        tasks = tasks[:limit]
    return tasks


_CODEBOOK: dict | None = None


def codebook() -> dict:
    """Variable labels and value labels, taken from the CDC codebook."""
    global _CODEBOOK
    if _CODEBOOK is None:
        with open(SRC_CODEBOOK, encoding="utf-8") as fh:
            _CODEBOOK = json.load(fh)
    return _CODEBOOK


def outcome_for(paradigm: str, text: str, spec: dict, csv_path: str, db_path: str) -> dict:
    """Turn a raw completion into a scored answer."""
    if spec["family"] == "ranking":
        return {"parsed": P.parse_ranking(text, [lv["id"] for lv in spec["all_levels"]]),
                "exec": None, "form": "text"}
    if paradigm == "direct":
        return {"parsed": P.parse_numeric(text), "exec": None, "form": "text"}
    if paradigm == "code":
        code = P.extract_block(text, "python")
        if not code:
            return {"parsed": None, "exec": {"ok": False, "error": "no code block"},
                    "form": "code", "code": None}
        res = P.run_code(code, csv_path)
        return {"parsed": res.get("answer"), "exec": res, "form": "code", "code": code}
    if paradigm == "sql":
        sql = P.extract_block(text, "sql")
        if not sql:
            return {"parsed": None, "exec": {"ok": False, "error": "no sql block"},
                    "form": "sql", "code": None}
        res = P.run_sql(sql, db_path)
        return {"parsed": res.get("answer"), "exec": res, "form": "sql", "code": sql}
    raise ValueError(paradigm)


def out_path(model: str, mode: str, tag: str, level: str, variant: str) -> str:
    name = f"{model}__{mode}__{tag}__{level}"
    if variant != "base":
        name += f"__{variant}"
    return os.path.join(GEN_DIR, name + ".jsonl")


def completed(path: str) -> set[str]:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if rec.get("api_error"):
                        continue
                    if (rec.get("skipped") or rec.get("spec_json")
                            or rec.get("raw") or rec.get("exec") is not None):
                        done.add(rec["task_id"])
                except json.JSONDecodeError:
                    continue
    return done


def append(path: str, record: dict) -> None:
    with _LOCK:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_one(model: str, mode: str, spec: dict, level: str, variant: str) -> dict:
    csv_path, db_path = P.materialize(spec, level, SLICE_DIR)
    rec = {"task_id": spec["task_id"], "model": model, "mode": mode, "level": level,
           "variant": variant, "family": spec["family"], "kind": spec["kind"],
           "spec_hash": spec.get("spec_hash"),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if spec["family"] == "ranking" and mode != "direct":
        rec.update({"skipped": True, "parsed": None,
                    "reason": "an ordering cannot be returned as a single numeric cell"})
        return rec
    cb = codebook()
    if mode == "fw_spec":
        # Ablation rung between free code and the constrained framework:
        # the specification is supplied, but no design constraint is imposed.
        p1 = P.build_prompt(spec, cb, "fw_s1", csv_path, variant)
        r1 = call_model(model, p1, MAX_TOKENS["fw_s1"])
        specj = P.extract_json(r1["text"]) or {}
        spec2 = dict(spec)
        spec2["spec_json"] = json.dumps(specj, ensure_ascii=False)
        p2 = P.build_prompt(spec2, cb, "fw_spec", csv_path, variant)
        r2 = call_model(model, p2, MAX_TOKENS["fw_spec"])
        code = P.extract_block(r2["text"], "python") or ""
        first = P.run_code(code, csv_path) if code else {"ok": False,
                                                        "error": "no code block"}
        rec.update({"raw": r1["text"][:2000], "raw2": r2["text"][:6000],
                    "code": code, "exec": first, "parsed": first.get("answer"),
                    "spec_json": specj, "spec_weight": str(specj.get("weight_variable", "")),
                    "usage": [r1["usage"], r2["usage"]],
                    "api_error": bool(r1.get("api_error") or r2.get("api_error")),
                    "weight_mentions": weight_mentions(
                        code + " " + str(specj.get("weight_variable", "")))})
        return rec
    if mode == "fw":
        # stage 1: specification
        p1 = P.build_prompt(spec, cb, "fw_s1", csv_path, variant)
        r1 = call_model(model, p1, MAX_TOKENS["fw_s1"])
        specj = P.extract_json(r1["text"]) or {}
        rec["spec_json"] = specj
        rec["spec_weight"] = str(specj.get("weight_variable", ""))
        spec2 = dict(spec)
        spec2["spec_json"] = json.dumps(specj, ensure_ascii=False)
        p2 = P.build_prompt(spec2, cb, "fw_s2", csv_path, variant)
        r2 = call_model(model, p2, MAX_TOKENS["fw_s2"])
        code = P.extract_block(r2["text"], "python") or ""
        first = P.run_code(code, csv_path) if code else {"ok": False,
                                                        "error": "no code block"}
        if not first.get("ok"):
            # one repair attempt, echoing the failure
            spec3 = dict(spec2)
            spec3["spec_json"] = spec2["spec_json"] + (
                "\n\n# The previous implementation failed with: "
                + str(first.get("error"))[:200])
            p2b = P.build_prompt(spec3, cb, "fw_s2", csv_path, variant)
            r2b = call_model(model, p2b, MAX_TOKENS["fw_s2"])
            code = P.extract_block(r2b["text"], "python") or ""
            first = P.run_code(code, csv_path) if code else first
            rec["repaired"] = True
        spec4 = dict(spec2)
        spec4["first_answer"] = first.get("answer")
        p3 = P.build_prompt(spec4, cb, "fw_full", csv_path, variant)
        r3 = call_model(model, p3, MAX_TOKENS["fw_full"])
        code2 = P.extract_block(r3["text"], "python") or ""
        second = P.run_code(code2, csv_path) if code2 else {"ok": False,
                                                            "error": "no code block"}
        rec.update({
            "raw": r1["text"][:4000], "raw2": r2["text"][:6000], "raw3": r3["text"][:6000],
            "code": code, "code2": code2, "exec": first, "exec2": second,
            "parsed": first.get("answer"), "second": second.get("answer"),
            "api_error": bool(r1.get("api_error") or r2.get("api_error") or r3.get("api_error")),
            "usage": [r1["usage"], r2["usage"], r3["usage"]],
            "weight_mentions": weight_mentions(code + " " + rec["spec_weight"]),
        })
        a, b = rec["parsed"], rec["second"]
        agree = (a is not None and b is not None
                 and abs(a - b) <= max(1e-9, 0.15 * max(abs(a), abs(b))))
        rec["gate_agree"] = bool(agree)
        rec["gated_answer"] = a if agree else None
        return rec

    prompt = P.build_prompt(spec, cb, mode, csv_path if mode != "sql" else db_path, variant)
    res = call_model(model, prompt, MAX_TOKENS[mode])
    out = outcome_for(mode, res["text"], spec, csv_path, db_path)
    rec.update({"raw": res["text"][:8000], "parsed": out.get("parsed"),
                "exec": out.get("exec"), "code": out.get("code"),
                "form": out.get("form"), "usage": res.get("usage"),
                "latency": res.get("latency"), "error": res.get("error"),
                "api_error": bool(res.get("api_error")),
                "weight_mentions": weight_mentions(res["text"]),
                "prompt_chars": len(prompt)})
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["direct", "code", "sql", "fw_s1", "fw", "fw_spec"])
    ap.add_argument("--tasks", default="main")
    ap.add_argument("--level", default="s500")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    ensure_dirs()
    if not API_KEY:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    tasks = load_tasks(args.tasks, args.subset, args.limit)
    path = out_path(args.model, args.mode, args.tasks, args.level, args.variant)
    done = completed(path)
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"{args.model} / {args.mode} / {args.tasks} / {args.level}: "
          f"{len(tasks)} tasks, {len(done)} done, {len(todo)} to run")
    if not todo:
        return
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, args.model, args.mode, t, args.level, args.variant): t
                   for t in todo}
        for i, fut in enumerate(futures, 1):
            spec = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {"task_id": spec["task_id"], "model": args.model, "mode": args.mode,
                       "level": args.level, "variant": args.variant,
                       "error": f"{type(exc).__name__}: {exc}", "parsed": None}
            append(path, rec)
            if i % 25 == 0 or i == len(futures):
                rate = i / max(1e-6, time.time() - started)
                print(f"  {i}/{len(futures)}  {rate:.2f} rec/s  "
                      f"elapsed {time.time()-started:.0f}s", flush=True)
    print("written:", path)


if __name__ == "__main__":
    main()
