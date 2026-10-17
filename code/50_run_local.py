"""Run the benchmark against locally hosted open-weight models.

    python 50_run_local.py --model phi-3.5-mini --mode direct --tasks main --level s500 --subset

Weights are loaded in 4-bit (NF4) and generation is greedy, matching the API
protocol (temperature 0, no sampling).  Records are appended to the same jsonl
layout as the API runner, so runs resume and are scored identically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts as P  # noqa: E402
from config import (GEN_DIR, LOCAL_MODELS, MODEL_DIR, PROC_DIR, SLICE_DIR,  # noqa: E402
                    SRC_CODEBOOK, ensure_dirs)

# Models that ship remote code (Phi-3.5) make transformers build a dynamic module
# cache; the default location under the user profile is not writable here, so
# point it at the project directory before transformers is imported.
os.environ.setdefault("HF_MODULES_CACHE", os.path.join(MODEL_DIR, "hf_modules"))
# The hub is unreachable from the sandbox; without this, every model load spends
# about fifteen seconds on connection retries before falling back to the cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# The framework rungs use the same token budgets as the API runner, so a local
# record and an API record describe the same protocol.
MAX_NEW = {"direct": 450, "code": 800, "sql": 500,
           "fw_s1": 800, "fw_s2": 1400, "fw_full": 1400, "fw_spec": 1400}
_CACHE: dict = {}


def load_model(model_id: str):
    if model_id in _CACHE:
        return _CACHE[model_id]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    entry = next(m for m in LOCAL_MODELS if m["id"] == model_id)
    # Reuse an already-downloaded snapshot when one is registered, so a local
    # model can be evaluated even when the hub is unreachable.
    source = entry.get("path") if os.path.isdir(entry.get("path", "")) else entry["repo"]
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
    # Some checkpoints ship a remote implementation written against an older
    # transformers cache API.  Prefer the built-in implementation and fall back
    # to the remote code without a key/value cache, which the remote Phi-3
    # implementation requires on current transformers.
    use_cache = True
    try:
        tok = AutoTokenizer.from_pretrained(source, cache_dir=MODEL_DIR,
                                            trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            source, cache_dir=MODEL_DIR, device_map="cuda", quantization_config=quant,
            trust_remote_code=False)
    except Exception as exc:
        print(f"  built-in load failed ({type(exc).__name__}); using remote code "
              f"without the key/value cache", flush=True)
        use_cache = False
        tok = AutoTokenizer.from_pretrained(source, cache_dir=MODEL_DIR,
                                            trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            source, cache_dir=MODEL_DIR, device_map="cuda", quantization_config=quant,
            trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.eval()
    revision = "unknown"
    snap = os.path.join(MODEL_DIR, "models--" + entry["repo"].replace("/", "--"), "snapshots")
    if os.path.isdir(snap):
        subs = sorted(os.listdir(snap))
        if subs:
            revision = subs[-1][:12]
    _CACHE[model_id] = (tok, model, entry, revision, use_cache)
    return _CACHE[model_id]


def generate(tok, model, prompts: list[str], max_new: int, use_cache: bool = True) -> list[str]:
    import torch
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             use_cache=use_cache, pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    return tok.batch_decode(gen, skip_special_tokens=True)


def weight_mentions(text: str) -> list[str]:
    return [c for c in ("WTINT2YR", "WTMEC2YR", "WTDRD1") if c in (text or "")]


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def completed(path: str) -> set[str]:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("skipped") or rec.get("raw") or rec.get("exec") is not None:
                    done.add(rec["task_id"])
    return done


def outcome_for(mode: str, text: str, spec: dict, csv_path: str, db_path: str) -> dict:
    if spec["family"] == "ranking":
        return {"parsed": P.parse_ranking(text, [lv["id"] for lv in spec["all_levels"]]),
                "exec": None, "code": None}
    if mode == "direct":
        return {"parsed": P.parse_numeric(text), "exec": None, "code": None}
    if mode == "code":
        code = P.extract_block(text, "python")
        if not code:
            return {"parsed": None, "exec": {"ok": False, "error": "no code block"}, "code": None}
        res = P.run_code(code, csv_path)
        return {"parsed": res.get("answer"), "exec": res, "code": code}
    if mode == "sql":
        sql = P.extract_block(text, "sql")
        if not sql:
            return {"parsed": None, "exec": {"ok": False, "error": "no sql block"}, "code": None}
        res = P.run_sql(sql, db_path)
        return {"parsed": res.get("answer"), "exec": res, "code": sql}
    raise ValueError(mode)


def run_framework_spec(tok, model, todo, codebook, args, path: str, revision,
                       use_cache: bool) -> None:
    """Ablation rung between free code and the constrained framework.

    The specification is supplied exactly as the framework supplies it, but no
    design constraint is imposed on the program, so the difference against the
    constrained rung isolates what the constraint itself buys.  Mirrors the API
    runner's fw_spec branch.
    """
    started = time.time()
    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        refs, p1 = [], []
        for spec in chunk:
            csv_path, _ = P.materialize(spec, args.level, SLICE_DIR)
            refs.append((spec, csv_path))
            p1.append(P.build_prompt(spec, codebook, "fw_s1", csv_path, "base"))
        texts1 = generate(tok, model, p1, MAX_NEW["fw_s1"], use_cache)
        specs1 = [P.extract_json(t) or {} for t in texts1]

        p2 = []
        for (spec, csv_path), specj in zip(refs, specs1):
            s2 = dict(spec)
            s2["spec_json"] = json.dumps(specj, ensure_ascii=False)
            p2.append(P.build_prompt(s2, codebook, "fw_spec", csv_path, "base"))
        texts2 = generate(tok, model, p2, MAX_NEW["fw_spec"], use_cache)

        for i, (spec, csv_path) in enumerate(refs):
            code = P.extract_block(texts2[i], "python") or ""
            first = P.run_code(code, csv_path) if code else {"ok": False,
                                                            "error": "no code block"}
            rec = {"task_id": spec["task_id"], "model": args.model, "mode": "fw_spec",
                   "level": args.level, "variant": "base", "family": spec["family"],
                   "kind": spec["kind"], "spec_hash": spec.get("spec_hash"),
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "quantization": "4-bit NF4", "revision": revision,
                   "spec_json": specs1[i],
                   "spec_weight": str(specs1[i].get("weight_variable", "")),
                   "raw": texts1[i][:2000], "raw2": texts2[i][:6000],
                   "code": code, "exec": first, "parsed": first.get("answer"),
                   "prompt_chars": len(p2[i]),
                   "weight_mentions": weight_mentions(
                       code + " " + str(specs1[i].get("weight_variable", "")))}
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done = min(start + args.batch, len(todo))
        if done % 20 < args.batch or done == len(todo):
            elapsed = time.time() - started
            print(f"  {done}/{len(todo)}  {elapsed:.0f}s  {elapsed/max(done,1):.1f}s per task",
                  flush=True)
    print("written:", path)


def run_framework(tok, model, todo, codebook, args, path: str, revision,
                  use_cache: bool) -> None:
    """Design-aware framework for locally served models.

    Mirrors the API runner stage for stage, so the two paths write the same
    record fields and 60_evaluate.py can derive the ablation rungs from either:
    a specification, a constrained implementation with one repair attempt, and
    an independent recomputation whose agreement with the first decides whether
    the framework answers at all.
    """
    if args.mode == "fw_spec":
        run_framework_spec(tok, model, todo, codebook, args, path, revision,
                           use_cache)
        return
    started = time.time()
    done = 0
    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        refs, p1 = [], []
        for spec in chunk:
            csv_path, _ = P.materialize(spec, args.level, SLICE_DIR)
            refs.append((spec, csv_path))
            p1.append(P.build_prompt(spec, codebook, "fw_s1", csv_path, "base"))
        texts1 = generate(tok, model, p1, MAX_NEW["fw_s1"], use_cache)
        specs1 = [P.extract_json(t) or {} for t in texts1]

        # stage 2: constrained implementation, with one repair attempt
        p2, specs2 = [], []
        for (spec, csv_path), specj in zip(refs, specs1):
            s2 = dict(spec)
            s2["spec_json"] = json.dumps(specj, ensure_ascii=False)
            specs2.append(s2)
            p2.append(P.build_prompt(s2, codebook, "fw_s2", csv_path, "base"))
        texts2 = generate(tok, model, p2, MAX_NEW["fw_s2"], use_cache)
        firsts, codes = [], []
        for (spec, csv_path), text in zip(refs, texts2):
            code = P.extract_block(text, "python") or ""
            firsts.append(P.run_code(code, csv_path) if code
                          else {"ok": False, "error": "no code block"})
            codes.append(code)

        # one repair attempt for the implementations that did not run
        repair_at = [i for i, f in enumerate(firsts) if not f.get("ok")]
        if repair_at:
            p2b = []
            for i in repair_at:
                spec, csv_path = refs[i]
                s3 = dict(specs2[i])
                s3["spec_json"] = specs2[i]["spec_json"] + (
                    "\n\n# The previous implementation failed with: "
                    + str(firsts[i].get("error"))[:200])
                p2b.append(P.build_prompt(s3, codebook, "fw_s2", csv_path, "base"))
            repaired = generate(tok, model, p2b, MAX_NEW["fw_s2"], use_cache)
            for i, text in zip(repair_at, repaired):
                code = P.extract_block(text, "python") or ""
                if code:
                    codes[i] = code
                    firsts[i] = P.run_code(code, refs[i][1])

        # stage 3: an independent second implementation
        p3 = []
        for (spec, csv_path), s2, first in zip(refs, specs2, firsts):
            s4 = dict(s2)
            s4["first_answer"] = first.get("answer")
            p3.append(P.build_prompt(s4, codebook, "fw_full", csv_path, "base"))
        texts3 = generate(tok, model, p3, MAX_NEW["fw_full"], use_cache)
        seconds = []
        for (spec, csv_path), text in zip(refs, texts3):
            code2 = P.extract_block(text, "python") or ""
            seconds.append(P.run_code(code2, csv_path) if code2
                           else {"ok": False, "error": "no code block"})

        for i, (spec, csv_path) in enumerate(refs):
            a, b = firsts[i].get("answer"), seconds[i].get("answer")
            agree = (a is not None and b is not None
                     and abs(a - b) <= max(1e-9, 0.15 * max(abs(a), abs(b))))
            rec = {"task_id": spec["task_id"], "model": args.model, "mode": "fw",
                   "level": args.level, "variant": "base", "family": spec["family"],
                   "kind": spec["kind"], "spec_hash": spec.get("spec_hash"),
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "quantization": "4-bit NF4", "revision": revision,
                   "spec_json": specs1[i],
                   "spec_weight": str(specs1[i].get("weight_variable", "")),
                   "raw": texts1[i][:4000], "raw2": texts2[i][:6000],
                   "raw3": texts3[i][:6000],
                   "code": codes[i], "code2": P.extract_block(texts3[i], "python") or "",
                   "exec": firsts[i], "exec2": seconds[i],
                   "parsed": a, "second": b, "gate_agree": bool(agree),
                   "gated_answer": a if agree else None,
                   "prompt_chars": len(p3[i]),
                   "weight_mentions": weight_mentions(
                       codes[i] + " " + str(specs1[i].get("weight_variable", "")))}
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done = min(start + args.batch, len(todo))
        if done % 20 < args.batch or done == len(todo):
            elapsed = time.time() - started
            print(f"  {done}/{len(todo)}  {elapsed:.0f}s  {elapsed/max(done,1):.1f}s per task",
                  flush=True)
    print("written:", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["direct", "code", "sql", "fw", "fw_spec"])
    ap.add_argument("--tasks", default="main")
    ap.add_argument("--level", default="s500")
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()
    ensure_dirs()
    tasks = load_json(os.path.join(PROC_DIR, f"tasks_{args.tasks}.json"))["tasks"]
    if args.subset:
        tasks = [t for t in tasks if t.get("in_subset")]
    if args.limit:
        tasks = tasks[:args.limit]
    if args.mode != "direct":
        tasks = [t for t in tasks if t["family"] != "ranking"]
    path = os.path.join(GEN_DIR, f"{args.model}__{args.mode}__{args.tasks}__{args.level}.jsonl")
    done = completed(path)
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"{args.model} / {args.mode}: {len(tasks)} tasks, {len(done)} done, {len(todo)} to run",
          flush=True)
    if not todo:
        return
    codebook = load_json(SRC_CODEBOOK)
    tok, model, entry, revision, use_cache = load_model(args.model)
    print(f"  loaded {entry['repo']} rev={revision}", flush=True)
    if args.mode in ("fw", "fw_spec"):
        run_framework(tok, model, todo, codebook, args, path, revision, use_cache)
        return
    started = time.time()
    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        prompts, refs = [], []
        for spec in chunk:
            csv_path, db_path = P.materialize(spec, args.level, SLICE_DIR)
            ref = csv_path if args.mode != "sql" else db_path
            prompts.append(P.build_prompt(spec, codebook, args.mode, ref, "base"))
            refs.append((spec, csv_path, db_path))
        texts = generate(tok, model, prompts, MAX_NEW[args.mode], use_cache)
        for (spec, csv_path, db_path), text in zip(refs, texts):
            out = outcome_for(args.mode, text, spec, csv_path, db_path)
            rec = {"task_id": spec["task_id"], "model": args.model, "mode": args.mode,
                   "level": args.level, "variant": "base", "family": spec["family"],
                   "kind": spec["kind"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "spec_hash": spec.get("spec_hash"),
                   "quantization": "4-bit NF4", "revision": revision,
                   "raw": text[:8000], "parsed": out["parsed"], "exec": out["exec"],
                   "code": out["code"], "weight_mentions": weight_mentions(text),
                   "prompt_chars": len(prompts[0])}
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n = min(start + args.batch, len(todo))
        if n % 10 < args.batch or n == len(todo):
            elapsed = time.time() - started
            print(f"  {n}/{len(todo)}  {elapsed:.0f}s  {elapsed/max(n,1):.1f}s per task",
                  flush=True)
    print("written:", path)


if __name__ == "__main__":
    main()
