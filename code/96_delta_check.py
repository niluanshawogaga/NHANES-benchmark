"""Exclusion test for the design-omission share (review finding K2 / DA-C2).

The manuscript reads a high delta as evidence that the model used the unweighted
route.  A competing explanation is that the answer is merely arithmetic noise
that happened to land near the unweighted value.  This script separates the two
cases with two independent signals that were never used in the original claim:

  1. how far the answer actually sits from the unweighted value (being inside
     the task tolerance means the model reproduced it; being far outside means
     the proximity was incidental);
  2. whether the generated program mentions a weight column at all -- a fact
     recorded per run and independent of the numeric outcome.

Output: results/delta_exclusion.md
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULT_DIR  # noqa: E402

ROWS = os.path.join(RESULT_DIR, "by_task_main.jsonl")
OUT = os.path.join(RESULT_DIR, "delta_exclusion.md")


def load() -> list[dict]:
    with open(ROWS, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def inside(a, b, tol) -> bool:
    return a is not None and b is not None and tol and abs(a - b) <= tol


def main() -> None:
    rows = [r for r in load() if r.get("design_share") is not None
            and r.get("answer") is not None]
    lines = ["# Design-omission share: exclusion test", "",
             "Delta is the projection of the answer onto the weighted-unweighted",
             "segment (0 = design-correct, 1 = unweighted).", ""]
    summary = {}
    for mode in ("direct", "code", "sql"):
        sub = [r for r in rows if r["mode"] == mode and r["level"] == "s120"]
        if len(sub) < 20:
            continue
        high = [r for r in sub if r["design_share"] >= 0.7]
        low = [r for r in sub if r["design_share"] <= 0.3]

        def stats(group, label):
            if not group:
                return None
            near = sum(1 for r in group
                       if inside(r["answer"], r["truth_unweighted"],
                                 r.get("tolerance") or 0.02))
            far = len(group) - near
            weighted_mentioned = sum(1 for r in group if r.get("any_weight_mentioned"))
            return {"n": len(group), "reproduces_unweighted": near, "other": far,
                    "mentions_any_weight": weighted_mentioned}

        sh, sl = stats(high, "high"), stats(low, "low")
        summary[mode] = (sh, sl)
        lines.append("## Protocol: %s" % mode)
        if sh:
            lines.append("- delta >= 0.7: n=%d, of which %d (%.0f%%) land within the task "
                         "tolerance of the *unweighted* value; %d mention any weight column"
                         % (sh["n"], sh["reproduces_unweighted"],
                            100 * sh["reproduces_unweighted"] / sh["n"],
                            sh["mentions_any_weight"]))
        if sl:
            lines.append("- delta <= 0.3: n=%d, of which %d (%.0f%%) land within tolerance "
                         "of the unweighted value; %d mention any weight column"
                         % (sl["n"], sl["reproduces_unweighted"],
                            100 * sl["reproduces_unweighted"] / sl["n"],
                            sl["mentions_any_weight"]))
        if sh and sl:
            lines.append("- weight-mention rate: %.0f%% (high delta) vs %.0f%% (low delta)"
                         % (100 * sh["mentions_any_weight"] / sh["n"],
                            100 * sl["mentions_any_weight"] / sl["n"])
                         if mode != "direct" else
                         "- weight mention is not observable in the direct protocol")
        lines.append("")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("written:", OUT)


if __name__ == "__main__":
    main()
