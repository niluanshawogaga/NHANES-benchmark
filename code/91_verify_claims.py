"""Recompute every numeric claim in the manuscript from the raw records.

    python 91_verify_claims.py

90_check_numbers.py only asks whether a value can be found somewhere under
results/.  This script asks the harder question: is the claimed quantity what
the records actually say?  Each check prints the claim, the recomputed value,
and a verdict, so a failure points at one sentence rather than at "the tables".
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GEN_DIR, PROC_DIR, RESULT_DIR  # noqa: E402

WEIGHT_COL = {"interview": "WTINT2YR", "exam": "WTMEC2YR", "dietary": "WTDRD1"}
SUBSET_ONLY = {"qwen2.5-7b", "phi-3.5-mini", "mistral-7b"}
RESULTS: list[tuple[bool, str, str, str]] = []


def check(claim: str, recomputed: str, ok: bool) -> None:
    RESULTS.append((ok, claim, recomputed, "PASS" if ok else "FAIL"))


def rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    scored = rows(os.path.join(RESULT_DIR, "by_task_main.jsonl"))
    repl = rows(os.path.join(RESULT_DIR, "by_task_repl.jsonl"))
    tasks = json.load(open(os.path.join(PROC_DIR, "tasks_main.json"),
                           encoding="utf-8"))["tasks"]
    subset = {t["task_id"] for t in tasks if t.get("in_subset")}
    weight_of = {t["task_id"]: WEIGHT_COL.get(t.get("weight_class", ""), "")
                 for t in tasks}
    base = [r for r in scored if r.get("variant", "base") == "base"
            and r.get("level") == "s120"
            and not (r["model"] in SUBSET_ONLY and r["task_id"] not in subset)]
    hit_of: dict[tuple, float] = {}
    for model in sorted({r["model"] for r in base}):
        for mode in ("direct", "code", "sql", "fw_impl", "fw_gated"):
            sel = [r for r in base if r["model"] == model and r["mode"] == mode
                   and r.get("hit") is not None]
            if sel:
                hit_of[(model, mode)] = float(np.mean([1.0 if r["hit"] else 0.0
                                                       for r in sel]))

    # --- abstract and conclusion -------------------------------------------
    direct = [v for (m, mo), v in hit_of.items() if mo == "direct"]
    check("direct answering spans 1.0% to 14.3%",
          "%.3f to %.3f" % (min(direct), max(direct)),
          abs(min(direct) - 0.010) < 5e-4 and abs(max(direct) - 0.143) < 5e-4)
    # The sentence is about the two API configurations; the open-weight ones
    # start lower and therefore multiply by more, which the paper must not
    # present as the general range.
    api = ("deepseek-flash", "deepseek-v4-pro")
    local = ("qwen2.5-7b", "phi-3.5-mini", "qwen2.5-3b")

    def folds(models) -> list[float]:
        out = []
        for m in models:
            if (m, "direct") not in hit_of:
                continue
            for mo in ("code", "sql"):
                if (m, mo) in hit_of:
                    out.append(hit_of[(m, mo)] / hit_of[(m, "direct")])
        return out

    check("raising accuracy four- to six-fold (API configurations)",
          "%.1f to %.1f fold" % (min(folds(api)), max(folds(api))),
          4.0 <= min(folds(api)) and max(folds(api)) <= 6.0)
    # The manuscript qualifies the range as applying to the two API
    # configurations; for the record, the open-weight range is wider.
    print("        (note: open-weight configurations improve "
          "%.1f to %.1f fold)" % (min(folds(local)), max(folds(local))))

    # --- prompt sizes and tokens -------------------------------------------
    prompt: dict[tuple, list[int]] = defaultdict(list)
    tokens = responses = 0
    for path in sorted(glob.glob(os.path.join(GEN_DIR, "*.jsonl"))):
        name = os.path.basename(path)
        if not name.startswith(api):
            continue                        # interface cost is what the paper counts
        repl_file = "__repl__" in name      # the replication cycle has shorter prompts
        for rec in rows(path):
            usage = rec.get("usage")
            for u in (usage if isinstance(usage, list) else [usage]):
                if isinstance(u, dict) and "total_tokens" in u:
                    responses += 1
                    tokens += float(u["total_tokens"] or 0)
            if (not repl_file and rec.get("variant", "base") == "base"
                    and rec.get("level") == "s120" and rec.get("prompt_chars")):
                prompt[(rec["model"], rec["mode"])].append(rec["prompt_chars"])
    direct_chars = statistics.mean([v for k, vs in prompt.items() if k[1] == "direct"
                                    for v in vs])
    code_chars = statistics.mean([v for k, vs in prompt.items() if k[1] == "code"
                                  for v in vs])
    sql_chars = statistics.mean([v for k, vs in prompt.items() if k[1] == "sql"
                                 for v in vs])
    check("prompt of code generation is about an eighth of the direct prompt",
          "%.0f vs %.0f characters (1/%.1f)" % (code_chars, direct_chars,
                                                direct_chars / code_chars),
          7.0 <= direct_chars / code_chars <= 9.0)
    check("means of 10,893 / 1,361 / 1,292 characters",
          "%.0f / %.0f / %.0f" % (direct_chars, code_chars, sql_chars),
          abs(direct_chars - 10893) < 3 and abs(code_chars - 1361) < 3
          and abs(sql_chars - 1292) < 3)
    stored = 0
    for path in sorted(glob.glob(os.path.join(GEN_DIR, "*.jsonl"))):
        if os.path.basename(path).startswith(api):
            stored += sum(1 for line in open(path, encoding="utf-8") if line.strip())
    scored_rows = len([r for r in base if r["model"] in api and r.get("hit") is not None])
    check("3,200 stored responses, 2,666 of them scored, 1.16e7 tokens",
          "%d stored, %d scored, %.2fe7 tokens"
          % (stored, scored_rows, tokens / 1e7),
          stored == 3200 and scored_rows == 2666 and abs(tokens / 1e7 - 1.16) < 0.02)

    # --- design-omission share --------------------------------------------
    for model, expected in (("deepseek-flash", 0.83), ("deepseek-v4-pro", 0.28),
                            ("qwen2.5-7b", 1.40)):
        shares = [r["design_share"] for r in base if r["model"] == model
                  and r["mode"] == "direct" and r.get("design_share") is not None]
        got = float(np.median(shares))
        check("median design-omission share of %s under direct answering" % model,
              "%.2f" % got, abs(got - expected) < 0.005)

    # --- the gap in points -------------------------------------------------
    seen: set[str] = set()
    gaps = []
    for r in scored:
        if (r["family"] != "proportion" or r.get("level") != "s120"
                or r["task_id"] in seen or r.get("truth_weighted") is None):
            continue
        seen.add(r["task_id"])
        gaps.append(abs(r["truth_weighted"] - r["truth_unweighted"]))
    check("ignoring the design moves a population proportion by 4.5 points on "
          "average and by as much as 14.8",
          "n=%d, mean %.1f points, max %.1f points, median %.1f, p90 %.1f"
          % (len(gaps), 100 * np.mean(gaps), 100 * max(gaps),
             100 * np.median(gaps), 100 * np.percentile(gaps, 90)),
          abs(100 * np.mean(gaps) - 4.5) < 0.1 and abs(100 * max(gaps) - 14.8) < 0.1
          and 3.0 <= 100 * np.median(gaps) <= 4.0
          and 9.0 <= 100 * np.percentile(gaps, 90) <= 10.0)

    # --- the delta bands quoted in Section VII-B ---------------------------
    # These two bands are computed over every scored answer of the code and SQL
    # protocols, across all five configurations, at the baseline condition of
    # the main extract level (the scope the manuscript's sentence describes).
    flat = [r for r in scored if r.get("variant", "base") == "base"
            and r["level"] == "s120"]
    code_answers = [r for r in flat if r["mode"] == "code"
                    and r.get("design_share") is not None and r.get("answer") is not None]
    sql_answers = [r for r in flat if r["mode"] == "sql"
                   and r.get("design_share") is not None and r.get("answer") is not None]
    hi = [r for r in code_answers if r["design_share"] >= 0.7]
    lo = [r for r in code_answers if r["design_share"] <= 0.3]
    unweighted = lambda rs: float(np.mean([abs(r["answer"] - r["truth_unweighted"])
                                           <= r["tolerance"] for r in rs]))
    # "mentions a weight column anywhere in the generated program" is the
    # weaker test the sentence describes, not "used the correct weight".
    weighted = lambda rs: float(np.mean([bool(r.get("any_weight_mentioned")) for r in rs]))
    check("183 code answers with delta >= 0.7: 68% on the unweighted value, 26% "
          "name a weight column (all scored code answers, any configuration)",
          "n=%d, %.0f%%, %.0f%%" % (len(hi), 100 * unweighted(hi), 100 * weighted(hi)),
          len(hi) == 183 and abs(unweighted(hi) - 0.68) < 0.01
          and abs(weighted(hi) - 0.26) < 0.01)
    subset_code = [r for r in base if r["model"] == "deepseek-flash"
                   and r["mode"] == "code" and r["task_id"] in subset]
    # the same 79 tasks means the non-ranking part of the subset: ranking tasks
    # are scored under direct answering only, so including them would compare
    # different question sets.
    subset_direct = [r for r in base if r["model"] == "deepseek-flash"
                     and r["mode"] == "direct" and r["task_id"] in subset
                     and r["family"] != "ranking"]
    def rate(sel):
        hits = [1.0 if r["hit"] else 0.0 for r in sel if r.get("hit") is not None]
        return len(hits), (float(np.mean(hits)) if hits else float("nan"))

    n_inline, inline_hit = rate([r for r in scored if r["model"] == "deepseek-flash"
                                 and r["mode"] == "code" and r.get("variant") == "inline"])
    n_c, code_hit = rate(subset_code)
    n_d, direct_hit = rate(subset_direct)
    check("the control arm moves the same 79 tasks from 59.5% to 88.6%, where "
          "direct answering reaches 11.4%",
          "code %.1f%%, inline %.1f%%, direct %.1f%%"
          % (100 * code_hit, 100 * inline_hit, 100 * direct_hit),
          abs(100 * code_hit - 59.5) < 0.2 and abs(100 * inline_hit - 88.6) < 0.2
          and abs(100 * direct_hit - 11.4) < 0.2)
    check("341 code answers with delta <= 0.3: 12% and 92%",
          "n=%d, %.0f%%, %.0f%%" % (len(lo), 100 * unweighted(lo), 100 * weighted(lo)),
          len(lo) == 341 and abs(unweighted(lo) - 0.12) < 0.01
          and abs(weighted(lo) - 0.92) < 0.01)
    shi = [r for r in sql_answers if r["design_share"] >= 0.7]
    slo = [r for r in sql_answers if r["design_share"] <= 0.3]
    check("text-to-SQL shows the same pattern (51% and 45%, against 13% and 95%)",
          "hi n=%d, %.0f%%, %.0f%% | lo n=%d, %.0f%%, %.0f%%"
          % (len(shi), 100 * unweighted(shi), 100 * weighted(shi),
             len(slo), 100 * unweighted(slo), 100 * weighted(slo)),
          abs(unweighted(shi) - 0.51) < 0.01 and abs(weighted(shi) - 0.45) < 0.01
          and abs(unweighted(slo) - 0.13) < 0.01 and abs(weighted(slo) - 0.95) < 0.01)

    # --- the failure split for Qwen2.5-3B under code -----------------------
    sel = [r for r in base if r["model"] == "qwen2.5-3b" and r["mode"] == "code"
           and r.get("hit") is not None]
    counts = Counter(r.get("failure") or "hit" for r in sel)
    check("166 of the 240 Qwen2.5-3B code runs fail before producing a number",
          "execution_error=%d, no_answer.rejected=%d"
          % (counts["execution_error"],
             counts["no_answer"] + counts.get("rejected_construct", 0)),
          counts["execution_error"] == 166 and len(sel) == 240)
    check("then 18 never reference a weight, 22 sit on the unweighted value, "
          "8 compute with weights but miss tolerance",
          "no_weight=%d, unweighted_route=%d, weighted_but_inaccurate=%d"
          % (counts["no_weight_used"], counts["unweighted_route"],
             counts["weighted_but_inaccurate"]),
          counts["no_weight_used"] == 18 and counts["unweighted_route"] == 22
          and counts["weighted_but_inaccurate"] == 8)

    # --- specification stage ----------------------------------------------
    named = total = 0
    for path in glob.glob(os.path.join(GEN_DIR, "__fw__".join(["*", "main__s120.jsonl"]))):
        if not any(m in os.path.basename(path) for m in api):
            continue                    # the sentence is about the two API runs
        for rec in rows(path):
            if rec.get("skipped") or not rec.get("spec_json"):
                continue
            want = weight_of.get(rec["task_id"], "")
            total += 1
            if str(rec.get("spec_weight", "")).strip() == want:
                named += 1
    check("the models name the correct sampling weight in 87% of "
          "specification steps",
          "%d/%d = %.1f%%" % (named, total, 100 * named / max(total, 1)),
          abs(100 * named / max(total, 1) - 87.0) < 0.5)

    # --- the gate ----------------------------------------------------------
    abstain = {}
    for model in ("deepseek-flash", "deepseek-v4-pro"):
        # gate_agree lives on the generation record; the scored file keeps the
        # derived rungs only.
        path = os.path.join(GEN_DIR, "%s__fw__main__s120.jsonl" % model)
        fw = [r for r in rows(path) if not r.get("skipped")]
        abstain[model] = 1 - float(np.mean([r.get("gate_agree") is True for r in fw]))
    check("the gate abstains on 39.2% of DeepSeek-flash runs and 57.0% of "
          "DeepSeek-V4-Pro runs",
          "abstention: flash %.1f%%, v4-pro %.1f%%"
          % (100 * abstain["deepseek-flash"], 100 * abstain["deepseek-v4-pro"]),
          abs(abstain["deepseek-flash"] - 0.392) < 0.01
          and abs(abstain["deepseek-v4-pro"] - 0.570) < 0.01)
    for model in ("qwen2.5-7b", "qwen2.5-3b", "phi-3.5-mini"):
        path = os.path.join(GEN_DIR, "%s__fw__main__s120.jsonl" % model)
        fw = [r for r in rows(path) if not r.get("skipped")]
        rate = 1 - float(np.mean([r.get("gate_agree") is True for r in fw]))
        check("the gate abstains on %.0f%% of %s runs" % (100 * rate, model),
              "%.1f%% of %d runs" % (100 * rate, len(fw)), rate > 0.8)
    for model, free, constrained, gated in (("qwen2.5-7b", 0.165, 0.101, 0.013),
                                            ("qwen2.5-3b", 0.087, 0.063, 0.038),
                                            ("phi-3.5-mini", 0.114, 0.177, 0.089)):
        got = (hit_of.get((model, "code")), hit_of.get((model, "fw_impl")),
               hit_of.get((model, "fw_gated")))
        check("ladder for %s: free code %.3f, constrained %.3f, gate %.3f"
              % (model, free, constrained, gated),
              "code %.3f, spec+code %.3f, gate %.3f" % got,
              all(abs((a or 0) - b) < 0.001 for a, b in zip(got, (free, constrained, gated))))
    spec_rates = {}
    for model in ("qwen2.5-7b", "qwen2.5-3b", "phi-3.5-mini"):
        named = total = 0
        for path in glob.glob(os.path.join(GEN_DIR, "%s__fw*__main__s120.jsonl" % model)):
            for rec in rows(path):
                if rec.get("skipped") or not rec.get("spec_json"):
                    continue
                total += 1
                if str(rec.get("spec_weight", "")).strip() == weight_of.get(
                        rec["task_id"], ""):
                    named += 1
        spec_rates[model] = 100 * named / max(total, 1)
    check("specification stage names the right weight: 73% / 56% / 27% for "
          "Qwen2.5-7B, Qwen2.5-3B, Phi-3.5-mini",
          "%.0f%% / %.0f%% / %.0f%%" % (spec_rates["qwen2.5-7b"],
                                        spec_rates["qwen2.5-3b"],
                                        spec_rates["phi-3.5-mini"]),
          abs(spec_rates["qwen2.5-7b"] - 73) < 1
          and abs(spec_rates["qwen2.5-3b"] - 56) < 1
          and abs(spec_rates["phi-3.5-mini"] - 27) < 1)

    # --- robustness: median relative gap by extract size -------------------
    def median_gap(level: str, families: tuple[str, ...]) -> float | None:
        vals, ids = [], set()
        for r in scored:
            if r.get("variant", "base") != "base" or r["level"] != level:
                continue
            if (r["family"] not in families or r.get("truth_weighted") is None
                    or r["task_id"] in ids):
                continue
            ids.add(r["task_id"])
            w, u = r["truth_weighted"], r["truth_unweighted"]
            if abs(w) > 1e-9:
                vals.append(abs(w - u) / abs(w))
        return float(np.median(vals)) if vals else None
    sizes = {level: median_gap(level, ("proportion", "mean"))
             for level in ("s120", "s250", "s500")}
    check("median relative gap over the proportion and mean families moves "
          "0.057 -> 0.041 -> 0.047",
          "%.3f -> %.3f -> %.3f" % (sizes["s120"] or 0, sizes["s250"] or 0,
                                    sizes["s500"] or 0),
          all(round(sizes[l] or 0, 3) == e
              for l, e in (("s120", 0.057), ("s250", 0.041), ("s500", 0.047))))

    width = max(len(c) for _, c, _, _ in RESULTS)
    for ok, claim, got, verdict in RESULTS:
        print("[%s] %-*s  recomputed: %s" % (verdict, width, claim, got))
    failed = [c for ok, c, _, _ in RESULTS if not ok]
    print("\n%d checks, %d failed" % (len(RESULTS), len(failed)))
    for c in failed:
        print("  FAIL:", c)


if __name__ == "__main__":
    main()
