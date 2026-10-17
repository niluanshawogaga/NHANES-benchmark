"""Generate the manuscript figures from the scored per-task records.

    python 70_figures.py --tag main

Figures are drawn for a single-column IEEE layout (3.5 in wide, 300 dpi) and use
grayscale-safe encodings so they survive black-and-white printing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (FIG_DIR, GEN_DIR, PAPER_DIR, PROC_DIR, RESULT_DIR,  # noqa: E402
                    VECTOR_DIR, ensure_dirs)

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "figure.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4})

FAMILIES = ["proportion", "mean", "difference", "ranking", "association"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
HATCHES = ["", "//", "\\\\", "xx", "..", "++", "oo", "**"]
MODE_LABELS = {
    "direct": "Direct", "code": "Code", "sql": "SQL", "fw_spec": "Spec only",
    "fw_impl": "Spec+code", "fw_gated": "Gate (0.15 rel)",
    "fw_gate@0.5tol": "Gate 0.5$\\times$tol", "fw_gate@1tol": "Gate 1$\\times$tol",
    "fw_gate@2tol": "Gate 2$\\times$tol",
}

# The two subset-evaluated configurations carry partial main-level records from
# an interrupted batch; those must not enter a figure, or the same bar would mix
# two different question sets.  Same rule as 65_make_tables.py.
SUBSET_ONLY_MODELS = {"qwen2.5-7b", "phi-3.5-mini", "mistral-7b"}
MODEL_ORDER = ["deepseek-flash", "deepseek-v4-pro", "qwen2.5-7b",
               "phi-3.5-mini", "qwen2.5-3b"]
MODEL_SHADE = {"deepseek-flash": "0.12", "deepseek-v4-pro": "0.40",
               "qwen2.5-7b": "0.62", "phi-3.5-mini": "0.80",
               "qwen2.5-3b": "0.92"}
MODEL_MARKER = {"deepseek-flash": "o", "deepseek-v4-pro": "s",
                "qwen2.5-7b": "^", "phi-3.5-mini": "D", "qwen2.5-3b": "v"}



def save_figure(fig, path: str) -> None:
    """Write the raster figure and its vector twin.

    LaTeX gets the PDF (line art at any zoom, no dpi question), the Word build
    keeps the 300 dpi PNG because docx cannot embed vector graphics here.
    """
    fig.savefig(path, bbox_inches="tight")
    os.makedirs(VECTOR_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    fig.savefig(os.path.join(VECTOR_DIR, stem + ".pdf"), bbox_inches="tight")


def subset_ids() -> set[str]:
    with open(os.path.join(PROC_DIR, "tasks_main.json"), encoding="utf-8") as fh:
        return {t["task_id"] for t in json.load(fh)["tasks"] if t.get("in_subset")}


def in_scope(row: dict, ids: set[str]) -> bool:
    return not (row["model"] in SUBSET_ONLY_MODELS and row["task_id"] not in ids)


def scope(rows: list[dict]) -> list[dict]:
    """Baseline condition of the main extract level, restricted to the tasks a
    configuration was really evaluated on."""
    ids = subset_ids()
    return [r for r in rows if r.get("variant", "base") == "base"
            and r.get("level") == "s120" and in_scope(r, ids)]


def hit_rate(rows: list[dict]) -> tuple[int, float]:
    hits = [1.0 if r["hit"] else 0.0 for r in rows if r.get("hit") is not None]
    if not hits:
        return 0, float("nan")
    return len(hits), float(np.mean(hits))


def boot_ci(values: list[float], n: int = 2000, seed: int = 20261015) -> tuple[float, float]:
    """Percentile bootstrap interval; same seed and resampling scheme as the
    intervals printed in Table IX, so figure and table cannot drift apart."""
    values = [v for v in values if v is not None]
    if len(values) < 5:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in range(len(values)))
                   / len(values) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def fit_text(ax, fig, x, y, w, h, text, max_size=7.0, min_size=4.4):
    """Draw centred text that is guaranteed to stay inside its box.

    The wrap width is estimated from the font size, then the rendered bounding
    box is measured; the font shrinks until the block really fits.
    """
    renderer = fig.canvas.get_renderer()
    size = max_size
    artist = None
    while size >= min_size:
        # 0.62 em is the average advance width of mixed-case DejaVu Sans; a
        # smaller factor wraps too late and the block then fails the width test
        # at every size, which silently shrinks the label to the floor.
        per_line = max(6, int((w * fig.get_figwidth() * 72.0 - 7.0) / (0.62 * size)))
        lines: list[str] = []
        for para in text.split("\n"):
            lines.extend(textwrap.wrap(para, per_line) or [""])
        artist = ax.text(x + w / 2.0, y + h / 2.0, "\n".join(lines), ha="center",
                         va="center", fontsize=size, linespacing=1.2)
        box = artist.get_window_extent(renderer=renderer).transformed(
            ax.transData.inverted())
        if box.width <= w * 0.985 and box.height <= h * 0.985:
            return artist
        artist.remove()
        size -= 0.2
    return ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
                   fontsize=min_size)


def load_rows(tag: str) -> list[dict]:
    path = os.path.join(RESULT_DIR, f"by_task_{tag}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_generations() -> list[dict]:
    """Raw generation records for the baseline condition of the main extract
    level.  These, not the scored file, carry the prompt sizes and the billed
    token counts."""
    ids = subset_ids()
    out: list[dict] = []
    for name in sorted(os.listdir(GEN_DIR)):
        if not name.endswith("__main__s120.jsonl"):
            continue
        with open(os.path.join(GEN_DIR, name), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    continue
                if rec.get("variant", "base") != "base":
                    continue
                if rec.get("level") != "s120" or not in_scope(rec, ids):
                    continue
                out.append(rec)
    return out


def labelled(row: dict) -> str:
    variant = row.get("variant", "base")
    return f"{row['model']} · {row['mode']}" + ("" if variant == "base" else f" · {variant}")


def fig_framework(path: str) -> None:
    """Architecture of the study, drawn at its printed size.

    The canvas is the column width and every label is set at the size it prints
    at, so nothing is scaled on the way into the manuscript.  Solid boxes are
    the objects the study manipulates; dashed groups are the stages, as in the
    framework figure of the companion paper.
    """
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def group(x: float, y: float, w: float, h: float, label: str) -> None:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.03",
            linestyle=(0, (5, 4)), linewidth=0.7, edgecolor="0.25",
            facecolor="none"))
        ax.text(x + 0.014, y + h - 0.008, label, fontsize=6.2,
                fontweight="bold", ha="left", va="top", color="0.1")

    def box(x: float, y: float, w: float, h: float, text: str,
            size: float = 6.0, facecolor: str = "white") -> None:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=facecolor,
                               edgecolor="black", linewidth=0.6))
        fit_text(ax, fig, x, y, w, h, text, max_size=size, min_size=3.8)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", linewidth=0.6,
                                    shrinkA=0, shrinkB=0, color="black"))

    # stage 1: what the model is given.  Every label sits in clear space: the
    # boxes of a stage start below the stage label, never across it.
    group(0.005, 0.765, 0.990, 0.230, "Task and prompt")
    box(0.015, 0.876, 0.300, 0.078, "Question")
    box(0.330, 0.876, 0.340, 0.078, "Extract of 120 rows")
    box(0.685, 0.876, 0.300, 0.078, "Design notes\nand codebook")
    box(0.150, 0.775, 0.700, 0.070,
        "Prompt: column names, weight and design variables, answer format")
    for x in (0.165, 0.500, 0.835):
        arrow(x, 0.876, x, 0.845)

    # stage 2: the three protocols
    group(0.005, 0.365, 0.475, 0.345, "Baselines: three protocols")
    box(0.015, 0.588, 0.450, 0.076, "Direct answering\none number, no tools")
    box(0.015, 0.480, 0.450, 0.076,
        "Code generation\na program reads the extract")
    box(0.015, 0.372, 0.450, 0.076, "Text-to-SQL\none query, one number")

    # stage 2': the framework, chained
    group(0.520, 0.365, 0.475, 0.345, "Design-aware framework")
    box(0.535, 0.588, 0.450, 0.076,
        "1  Specification\nmachine-checkable estimand")
    box(0.535, 0.480, 0.450, 0.076,
        "2  Constrained implementation\nmust reference the weight")
    box(0.535, 0.372, 0.450, 0.076, "3  Recomputation gate")
    arrow(0.760, 0.588, 0.760, 0.556)
    arrow(0.760, 0.480, 0.760, 0.448)

    # Frame to frame, as between the branches and the scoring stage: each arrow
    # starts on the bottom edge of the stage above and ends on the top edge of
    # the stage below, at the horizontal centre of the branch it feeds.  The two
    # frame gaps are the same size, so the three stages are evenly stepped.
    arrow(0.2425, 0.765, 0.2425, 0.710)
    arrow(0.7575, 0.765, 0.7575, 0.710)

    # stage 3: reference values and scoring
    group(0.005, 0.025, 0.990, 0.285, "Scoring")
    box(0.015, 0.185, 0.470, 0.080,
        "Weighted reference estimate $w$\nthe design-correct value")
    box(0.515, 0.185, 0.470, 0.080,
        "Unweighted value $u$\nwithout the weights")
    box(0.015, 0.038, 0.970, 0.090,
        "Score: a hit when $|\\hat{y}-w|\\leq\\tau$, and the design-omission share "
        "$\\delta=(\\hat{y}-w)/(u-w)$")
    arrow(0.2425, 0.365, 0.2425, 0.310)
    arrow(0.7575, 0.365, 0.7575, 0.310)
    arrow(0.250, 0.185, 0.250, 0.128)
    arrow(0.750, 0.185, 0.750, 0.128)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    save_figure(fig, path)
    plt.close(fig)


def fig_benchmark(rows: list[dict], path: str) -> None:
    """Relative weighted-unweighted gap by task family, in one panel.

    The earlier two-panel version drew task counts per family -- 60 each by
    construction, so every bar had the same height -- beside a box plot of
    absolute gaps that mixed proportions with intake totals on one axis.  Only
    the two families whose reference value cannot approach zero are shown on a
    relative scale; for between-group differences, rankings, and correlations
    the reference can sit near zero and a relative gap carries no meaning.
    """
    gaps: dict[str, list[float]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row.get("level") != "s120" or row["task_id"] in seen:
            continue
        w, u = row.get("truth_weighted"), row.get("truth_unweighted")
        if w is None or u is None or abs(w) < 1e-9:
            continue
        seen.add(row["task_id"])
        gaps[row["family"]].append(abs(w - u) / abs(w))
    if not gaps:
        return
    families = [("proportion", "population proportion"),
                ("mean", "weighted mean or median")]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.boxplot([gaps.get(f, [0.0]) for f, _ in families], widths=0.55,
               medianprops=dict(color="black"), boxprops=dict(color="black"),
               whiskerprops=dict(color="black"), capprops=dict(color="black"),
               flierprops=dict(markerfacecolor="0.6", markersize=2))
    ax.set_xticks(range(1, len(families) + 1))
    ax.set_xticklabels([label for _, label in families], rotation=22, ha="right")
    ax.set_ylabel("relative gap  $|w-u|/|w|$")
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)
    for family, _ in families:
        values = sorted(gaps.get(family, []))
        if values:
            print("  %-11s n=%-3d median=%.3f p90=%.3f max=%.3f"
                  % (family, len(values), values[len(values) // 2],
                     values[int(0.9 * len(values)) - 1], max(values)))


def fig_heatmap(rows: list[dict], path: str) -> None:
    """Hit rate within tolerance, by configuration, protocol, and task family.

    Rows are the three protocols of each configuration and columns are the five
    families, so pooling a row reproduces the cell of Table III for that
    configuration and protocol.  The scope is Table III's scope -- the baseline
    condition of the main extract level, restricted to the tasks a configuration
    was evaluated on.  An earlier version of this figure averaged every extract
    size and every wording or serialisation variant into the same cell, which is
    why its numbers disagreed with the table beside it.
    """
    scoped = scope(rows)
    protocols = [("direct", "direct answering"), ("code", "code generation"),
                 ("sql", "text-to-SQL")]
    rows_of: list[tuple[str, list[dict]]] = []
    for model in MODEL_ORDER:
        for mode, label in protocols:
            sel = [r for r in scoped if r["model"] == model and r["mode"] == mode
                   and r.get("hit") is not None]
            if sel:
                rows_of.append((f"{model} · {label}", sel))
    if not rows_of:
        return

    matrix = np.full((len(rows_of), len(FAMILIES)), np.nan)
    for i, (_, sel) in enumerate(rows_of):
        for j, family in enumerate(FAMILIES):
            vals = [1.0 if r["hit"] else 0.0 for r in sel if r["family"] == family]
            if vals:
                matrix[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(3.5, 0.22 * len(rows_of) + 1.5))
    im = ax.imshow(matrix, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels(FAMILIES, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(rows_of)))
    ax.set_yticklabels([label for label, _ in rows_of], fontsize=5.5)
    for i in range(len(rows_of)):
        for j in range(len(FAMILIES)):
            if np.isnan(matrix[i, j]):
                ax.text(j, i, "n.s.", ha="center", va="center",
                        fontsize=4.5, color="0.45")
            else:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if matrix[i, j] > 0.55 else "black")
    # one separator between the protocols of neighbouring configurations
    for i in range(1, len(rows_of)):
        if rows_of[i][0].split(" · ")[0] != rows_of[i - 1][0].split(" · ")[0]:
            ax.axhline(i - 0.5, color="0.55", linewidth=0.7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)
    return None


def fig_gap(rows: list[dict], path: str) -> None:
    """Error against the gap, one small panel per series.

    In relative terms so that proportions, means in kcal and correlations can
    share one axis; a point on the diagonal sits exactly where an unweighted
    statistic would.  The four series get a panel each because one cloud of four
    overlapping series needs a legend, and in this figure the legend is large
    enough to cover answers.
    """
    by_mode: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        w, u = row.get("truth_weighted"), row.get("truth_unweighted")
        answer = row.get("answer")
        if answer is None or w is None or u is None or abs(w) < 1e-9:
            continue
        by_mode[row["mode"]].append((abs(w - u) / abs(w), abs(answer - w) / abs(w)))
    lo, hi = 5e-4, 20

    def points(modes: set[str]) -> tuple[list[float], list[float]]:
        xs, ys = [], []
        for mode in sorted(modes):
            for gx, gy in by_mode.get(mode, []):
                if gx <= 0 or gy <= 0 or gy > 20:
                    continue
                xs.append(gx)
                ys.append(gy)
        return xs, ys

    fig = plt.figure(figsize=(3.5, 4.8))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.05],
                            hspace=0.62, wspace=0.18)
    panels = [("direct", "(a) Direct answering", "s", "0.25", 0, 0),
              ("code", "(b) Code generation", "o", "0.45", 0, 1),
              ("sql", "(c) Text-to-SQL", "^", "0.15", 1, 0),
              (None, "(d) Framework stages", "x", "0.60", 1, 1)]
    for key, label, marker, colour, r, c in panels:
        ax = fig.add_subplot(grid[r, c])
        modes = set(by_mode) - {"direct", "code", "sql"} if key is None else {key}
        xs, ys = points(modes)
        ax.scatter(xs, ys, s=3.5, marker=marker, facecolors=colour,
                   edgecolors=None if marker == "x" else "0.2", linewidths=0.4)
        ax.plot([lo, hi], [lo, hi], color="black", linestyle=":", linewidth=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(label, fontsize=6.5)
        ax.tick_params(labelsize=5.5)
        if c:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("relative error", fontsize=6)
        if r:
            ax.set_xlabel("relative gap", fontsize=6)
        else:
            ax.tick_params(labelbottom=False)
    ax = fig.add_subplot(grid[2, :])
    shares = defaultdict(list)
    for row in rows:
        if row.get("design_share") is not None:
            shares[row["mode"]].append(row["design_share"])
    # A mode whose runs are still in flight would draw a box from a handful of
    # answers; the framework modes carry 79 tasks once complete, so anything
    # far below that is a partial run and stays out of the panel.
    keys = sorted(k for k, v in shares.items() if len(v) >= 50)
    for k, v in sorted(shares.items()):
        if len(v) < 50:
            print("  note: %s has only %d scored answers; left out of the "
                  "decomposition panel" % (k, len(v)))
    if keys:
        ax.boxplot([shares[k] for k in keys], widths=0.6,
                   orientation="horizontal",
                   medianprops=dict(color="black"), boxprops=dict(color="black"),
                   whiskerprops=dict(color="black"), capprops=dict(color="black"),
                   flierprops=dict(markerfacecolor="0.6", markersize=1.6))
        ax.set_yticks(range(1, len(keys) + 1))
        ax.set_yticklabels([MODE_LABELS.get(k, k) for k in keys], fontsize=6)
        ax.axvline(1.0, color="black", linestyle=":", linewidth=0.7)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.7)
        ax.set_xlim(-1.3, 2.2)
        ax.set_xlabel("share of the error explained\nby omitting the design")
        ax.set_title("(e) weighted-observation gap decomposition", fontsize=7)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def fig_ablation(rows: list[dict], path: str) -> None:
    """The framework ladder, one group of bars per rung.

    The protocol comparison that used to share this figure with the ladder is the
    accuracy matrix of Table III; repeating it here made the figure a second copy
    of that table, so only the ladder remains.
    """
    scoped = scope(rows)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    stages = [("code", "code\ngeneration"),
              ("fw_spec", "spec +\nfree code"),
              ("fw_impl", "spec +\nconstrained code"),
              ("fw_gated", "constrained\ncode + gate")]
    # Every configuration that was run through the ladder gets bars, in the same
    # style the other figures use for configurations.
    ladder = [m for m in MODEL_ORDER
              if any(r["model"] == m and r["mode"] in ("fw_spec", "fw_impl",
                                                       "fw_gated") for r in scoped)]
    width = 0.8 / max(len(ladder), 1)
    for i, model in enumerate(ladder):
        vals = []
        for mode, _ in stages:
            n, hit = hit_rate([r for r in scoped
                               if r["model"] == model and r["mode"] == mode])
            vals.append(hit if n else np.nan)
        ax.bar(np.arange(len(stages)) + i * width, vals, width=width,
                    color=MODEL_SHADE[model], edgecolor="black", linewidth=0.5,
                    hatch=HATCHES[MODEL_ORDER.index(model) % len(HATCHES)],
                    label=model)
    ax.set_xticks(np.arange(len(stages)) + (len(ladder) - 1) * width / 2.0)
    ax.set_xticklabels([label for _, label in stages], fontsize=6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("hit rate within\ntolerance", fontsize=7)
    ax.legend(fontsize=5, ncol=2, loc="upper left", borderpad=0.3,
                   labelspacing=0.22, handlelength=1.2, handletextpad=0.3,
                   columnspacing=0.8)

    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.4)
    save_figure(fig, path)
    plt.close(fig)


def fig_taxonomy(rows: list[dict], path: str) -> None:
    """Composition of the scored tasks, one panel per protocol.

    The mechanical labels of Table V are collapsed into four interpretable
    buckets so that the figure says what the failure profile is rather than
    which of ten labels fired; the largest bucket of every bar carries its
    share, and the pale band on top is the share answered within tolerance.
    """
    scoped = scope(rows)
    buckets = [
        ("Execution failure", ("execution_error", "no_answer", "rejected_construct"),
         "0.15", ""),
        ("Design omission", ("no_weight_used", "unweighted_route",
                             "wrong_weight_variable"), "0.42", "//"),
        ("Weighted but inaccurate", ("weighted_but_inaccurate",), "0.68", "\\\\"),
        ("Other", ("ranking_mismatch", "gated_refusal", "not_applicable"), "0.88", "xx"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.2))
    for ax, mode, title in ((axes[0], "direct", "(a) direct answering"),
                            (axes[1], "code", "(b) code generation")):
        present = [m for m in MODEL_ORDER
                   if any(r["model"] == m and r["mode"] == mode
                          and r.get("hit") is not None for r in scoped)]
        counts: dict[str, dict[str, int]] = {}
        totals: list[int] = []
        for model in present:
            sel = [r for r in scoped if r["model"] == model and r["mode"] == mode
                   and r.get("hit") is not None]
            totals.append(len(sel))
            counter: dict[str, int] = defaultdict(int)
            for r in sel:
                counter[r.get("failure") or "hit"] += 1
            counts[model] = counter

        bottom = np.zeros(len(present))
        for label, members, shade, hatch in buckets:
            vals = np.array([sum(counts[m].get(k, 0) for k in members) / totals[i]
                             if totals[i] else 0.0
                             for i, m in enumerate(present)])
            ax.bar(range(len(present)), vals, bottom=bottom, color=shade,
                   width=0.62, edgecolor="black", linewidth=0.4, hatch=hatch,
                   label=None if mode == "code" else label)
            for i, value in enumerate(vals):
                if value >= 0.17:
                    ax.text(i, bottom[i] + value / 2.0, "%.0f%%" % (100 * value),
                            ha="center", va="center", fontsize=5.5,
                            color="white" if float(shade) < 0.5 else "black")
            bottom += vals
        hits = np.array([counts[m].get("hit", 0) / totals[i] if totals[i] else 0.0
                         for i, m in enumerate(present)])
        ax.bar(range(len(present)), hits, bottom=bottom, color="0.99", width=0.62,
               edgecolor="black", linewidth=0.4,
               label=None if mode == "code" else "Hit (within tolerance)")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([m.replace("deepseek-", "deepseek-\n")
                            .replace("qwen2.5-", "qwen2.5-\n")
                            .replace("phi-3.5-mini", "phi-3.5-\nmini")
                            for m in present], fontsize=6)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("share of scored\ntasks", fontsize=7)
        ax.set_title(title)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index("Hit (within tolerance)")] + \
            [labels.index(lbl) for lbl, _, _, _ in buckets]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               fontsize=5, ncol=2, loc="lower center", frameon=False,
               columnspacing=0.7, handlelength=1.2, handletextpad=0.35)
    fig.tight_layout(rect=(0, 0.13, 1, 1), pad=0.4)
    save_figure(fig, path)
    plt.close(fig)


def fig_robustness(rows: list[dict], path: str) -> None:
    """Interval plot of the nine robustness conditions of Table IX.

    Every condition is the same 100 estimands, so the question is whether the
    interval moves away from the baseline.  Nine conditions drawn as grouped
    bars in one column are unreadable; a point estimate with an interval is not.
    """
    ids = subset_ids()
    flash = "deepseek-flash"
    repl = load_rows("repl")

    def cell(mode: str, level: str, variant: str, pool: list[dict],
             restrict: bool) -> list[dict]:
        sel = [r for r in pool if r["model"] == flash and r["mode"] == mode
               and r["level"] == level and r.get("variant", "base") == variant]
        if restrict:
            sel = [r for r in sel if r["task_id"] in ids]
        return sel

    blocks = [
        ("Direct answering", "o", "0.12", "0.15", [
            ("120-row extract", cell("direct", "s120", "base", rows, True)),
            ("Paraphrased wording", cell("direct", "s120", "paraphrase", rows, False)),
            ("JSON serialisation", cell("direct", "s120", "fmt_json", rows, False)),
            ("Markdown serialisation", cell("direct", "s120", "fmt_md", rows, False)),
            ("250-row extract", cell("direct", "s250", "base", rows, False)),
            ("500-row extract", cell("direct", "s500", "base", rows, False)),
            ("Second cycle 2015-2016", cell("direct", "s120", "base", repl, False)),
        ]),
        ("Code generation", "s", "0.55", "0.55", [
            ("120-row extract", cell("code", "s120", "base", rows, True)),
            ("250-row extract", cell("code", "s250", "base", rows, False)),
            ("500-row extract", cell("code", "s500", "base", rows, False)),
            ("Second cycle 2015-2016", cell("code", "s120", "base", repl, False)),
            ("Extract inlined in prompt", cell("code", "s120", "inline", rows, False)),
        ]),
    ]

    entries: list[tuple] = []
    for title, marker, colour, base_colour, items in blocks:
        baseline = float("nan")
        start = len(entries)
        for label, sel in items:
            n, hit = hit_rate(sel)
            lo, hi = boot_ci([1.0 if r["hit"] else 0.0 for r in sel
                              if r.get("hit") is not None])
            if label == "120-row extract":
                baseline = hit
            entries.append(("row", label, n, hit, lo, hi, marker, colour))
        entries.insert(start, ("header", title, None, None, None, None, None,
                               base_colour, baseline))

    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    positions = list(range(len(entries)))[::-1]
    for idx, (entry, y) in enumerate(zip(entries, positions)):
        if entry[0] == "header":
            _, title, _, _, _, _, _, base_colour, baseline = entry
            following = [positions[i] for i in range(idx + 1, len(entries))
                         if entries[i][0] == "row"]
            if following and baseline == baseline:      # not NaN
                ax.plot([baseline, baseline],
                        [min(following) - 0.45, max(following) + 0.45],
                        linestyle=":", linewidth=0.7, color=base_colour, zorder=1)
        else:
            _, label, n, hit, lo, hi, marker, colour = entry
            ax.errorbar([hit], [y],
                        xerr=[[max(0.0, hit - lo)], [max(0.0, hi - hit)]],
                        fmt=marker, markersize=3.4, color=colour,
                        markerfacecolor=colour, markeredgecolor="black",
                        markeredgewidth=0.4, elinewidth=0.7, capsize=1.8, zorder=3)
            # The value column replaces the table that used to print these
            # numbers: the rate first, then its interval.
            ax.text(1.05, y, "%.3f [%.3f, %.3f]" % (hit, lo, hi),
                    fontsize=4.3, va="center", ha="left", clip_on=False)

    ax.set_yticks(positions)
    ax.set_yticklabels(["%s (%d)" % (e[1], e[2]) if e[0] == "row" else e[1]
                        for e in entries], fontsize=6.2)
    for tick, entry in zip(ax.get_yticklabels(), entries):
        if entry[0] == "header":
            tick.set_fontweight("bold")
    # room for the value column on the right
    ax.set_xlim(0, 1.72)
    ax.axvline(1.02, color="0.85", linewidth=0.5, zorder=0)
    ax.text(1.05, len(entries) - 0.55, "hit [95% interval]", fontsize=4.3,
            ha="left", va="center", clip_on=False)
    ax.set_ylim(-0.7, len(entries) - 0.3)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("hit rate within tolerance")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.4)
    save_figure(fig, path)
    plt.close(fig)


def fig_cost(rows: list[dict], generations: list[dict], path: str) -> None:
    """Accuracy against the prompt volume, one point per configuration and protocol.

    The horizontal axis is categorical rather than a continuous size: the two
    programmatic protocols issue a short prompt and the in-context protocol a
    long one, and the mean size of each cluster is printed above the points, so
    the reader does not need a logarithmic ruler to see the eightfold gap.  The
    printed sizes come from the generation records, which is where the prompt
    length is stored; a small horizontal offset inside a cluster keeps
    configurations with the same hit rate apart.
    """
    scoped = scope(rows)
    sizes: dict[tuple, list[float]] = defaultdict(list)
    for rec in generations:
        if rec.get("prompt_chars") is not None:
            sizes[(rec["model"], rec["mode"])].append(rec["prompt_chars"])
    models = [m for m in MODEL_ORDER if any(k[0] == m for k in sizes)]
    if not models:
        return

    # One panel per protocol, ordered by the size of the prompt it sends.  A
    # scatter with the prompt size on the axis forces configurations that share a
    # hit rate to be displaced sideways to stay visible, which is what made the
    # earlier version hard to read; bars give every configuration its own slot.
    protocols = [("sql", "text-to-SQL"), ("code", "code generation"),
                 ("direct", "direct answering")]
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 3.3), sharey=True,
                             gridspec_kw={"hspace": 0.62})
    for ax, (mode, label) in zip(axes, protocols):
        vals, shaded, names = [], [], []
        for model in models:
            if (model, mode) not in sizes:
                continue
            n, hit = hit_rate([r for r in scoped
                               if r["model"] == model and r["mode"] == mode])
            if not n:
                continue
            vals.append(hit)
            shaded.append(MODEL_SHADE[model])
            names.append(model.replace("deepseek-", "deepseek-\n")
                         .replace("qwen2.5-", "qwen2.5-\n")
                         .replace("phi-3.5-mini", "phi-3.5-\nmini"))
        xs = np.arange(len(vals))
        ax.bar(xs, vals, width=0.66, color=shaded, edgecolor="black",
               linewidth=0.5)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.025, "%.3f" % v, ha="center", va="bottom",
                    fontsize=5)
        mean_chars = float(np.mean([v for m in models
                                    for v in sizes.get((m, mode), [])]))
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=5.5)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        # The protocol label sits inside the panel, in the empty top-right
        # corner: as a title it would print on top of the tick labels of the
        # panel above.
        ax.text(0.985, 0.95,
                "(%s) %s, %.2fk prompt characters"
                % (chr(97 + protocols.index((mode, label))), label,
                   mean_chars / 1000.0),
                transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.grid(axis="x", visible=False)
    ax.set_ylabel("hit rate")
    fig.tight_layout(pad=0.4)
    save_figure(fig, path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    ensure_dirs()
    rows = load_rows(args.tag)
    print(f"loaded {len(rows)} scored records")
    generations = load_generations() if args.tag == "main" else []
    done = []
    fig_framework(os.path.join(FIG_DIR, "fig1_framework.png"))
    done.append("fig1_framework.png")
    fig_benchmark(rows, os.path.join(FIG_DIR, "fig2_benchmark.png"))
    done.append("fig2_benchmark.png")
    fig_heatmap(rows, os.path.join(FIG_DIR, "fig3_heatmap.png"))
    done.append("fig3_heatmap.png")
    fig_gap(rows, os.path.join(FIG_DIR, "fig4_gap.png"))
    done.append("fig4_gap.png")
    fig_ablation(rows, os.path.join(FIG_DIR, "fig5_ablation.png"))
    done.append("fig5_ablation.png")
    fig_taxonomy(rows, os.path.join(FIG_DIR, "fig6_taxonomy.png"))
    done.append("fig6_taxonomy.png")
    fig_robustness(rows, os.path.join(FIG_DIR, "fig7_robustness.png"))
    done.append("fig7_robustness.png")
    fig_cost(rows, generations, os.path.join(FIG_DIR, "fig8_cost.png"))
    done.append("fig8_cost.png")
    print("figures written to", FIG_DIR)
    print("  " + ", ".join(done))


if __name__ == "__main__":
    main()
