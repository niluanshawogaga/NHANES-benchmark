"""Prompt construction, response parsing and the code / SQL sandboxes.

Every paradigm receives the same task statement and the same data extract; they
differ only in how the estimate has to be produced.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from config import WEIGHT_LABELS, ensure_dirs  # noqa: E402

SANDBOX = os.path.join(HERE, "sandbox_runner.py")
PYTHON = sys.executable

FORBIDDEN = [
    r"\bopen\s*\(", r"\bto_csv\b", r"\bto_parquet\b", r"\bto_excel\b", r"\bos\.remove\b",
    r"\bos\.system\b", r"\bsubprocess\b", r"\bsocket\b", r"\brequests\b", r"\burllib\b",
    r"\bshutil\b", r"\bhttpx\b",
]


def column_notes(spec: dict, codebook: dict) -> str:
    lines = []
    for col in spec["slice_columns"]:
        if col == "SEQN":
            lines.append("- SEQN: respondent identifier")
        elif col in WEIGHT_LABELS:
            lines.append(f"- {col}: {WEIGHT_LABELS[col]}")
        elif col == "SDMVPSU":
            lines.append("- SDMVPSU: masked variance unit (cluster) code")
        elif col == "SDMVSTRA":
            lines.append("- SDMVSTRA: masked variance unit (stratum) code")
        else:
            entry = codebook.get(col, {})
            label = entry.get("label") or col
            values = entry.get("values") or {}
            extra = ""
            if values:
                shown = [f"{k} = {v.get('label')}" for k, v in list(values.items())[:6]]
                extra = " (" + "; ".join(shown) + ")"
            lines.append(f"- {col}: {label}{extra}")
    return "\n".join(lines)


def design_note() -> str:
    return ("The extract comes from NHANES, a complex, stratified, multistage probability "
            "sample of the US non-institutionalized population. Every respondent carries a "
            "sampling weight that stands for the number of people they represent, and "
            "SDMVSTRA / SDMVPSU describe the strata and clusters of the sample design.")


def answer_rule(spec: dict) -> str:
    if spec["family"] == "ranking":
        names = " > ".join(lv["id"] for lv in spec["all_levels"])
        return ("End with a line of the form\nFINAL ANSWER: <ordering>\nwhere <ordering> "
                "lists these group ids from highest to lowest separated by ' > ': " + names)
    if spec["family"] == "association":
        return ("End with a line of the form\nFINAL ANSWER: <value>\nwhere <value> is the "
                "correlation coefficient as a number between -1 and 1.")
    return ("End with a line of the form\nFINAL ANSWER: <value>\nwhere <value> is a single "
            "number without units.")


def csv_block(path: str) -> str:
    return pd.read_csv(path).to_csv(index=False)


def serialize(path: str, fmt: str = "csv") -> str:
    """Serialise an extract for the in-context paradigms.

    The JSON form is column-oriented so that it stays close to CSV in length;
    a row-oriented dump repeats every column name and would blow the context.
    """
    df = pd.read_csv(path)
    if fmt == "csv":
        return df.to_csv(index=False)
    if fmt == "json":
        payload = {"columns": list(df.columns),
                   "rows": [[None if pd.isna(v) else v for v in row]
                            for row in df.itertuples(index=False, name=None)]}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if fmt == "md":
        header = "| " + " | ".join(str(c) for c in df.columns) + " |"
        rule = "|" + "|".join([" --- "] * len(df.columns)) + "|"
        body = "\n".join("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
                         for row in df.itertuples(index=False, name=None))
        return "\n".join([header, rule, body])
    raise ValueError(fmt)


def build_prompt(spec: dict, codebook: dict, paradigm: str, data_ref: str,
                 variant: str = "base") -> str:
    head = ["You are a survey analyst working with a public data extract.", "",
            "Column notes:", column_notes(spec, codebook), "", design_note()]
    question = spec["question"]
    fmt = variant[4:] if variant.startswith("fmt_") else "csv"
    if variant == "paraphrase":
        question = ("Using the extract provided, compute the following population quantity. "
                    + question[0].lower() + question[1:])
    if paradigm == "direct":
        body = ["", f"Data extract ({fmt.upper()}, one row per respondent):",
                serialize(data_ref, fmt), "", "Question: " + question, "", answer_rule(spec)]
    elif paradigm == "code":
        body = ["", "The extract is stored as a CSV file at: " + data_ref,
                "Write Python code that reads that file and computes the requested quantity.",
                "Rules: use only pandas and numpy; read the file from the path above; store the "
                "final number in a variable named `answer`; do not write any file.",
                "", "Question: " + question]
        if variant == "inline":
            # Control arm: same protocol, but the table is also placed in the
            # context, so the comparison against direct answering isolates
            # "where the computation happens" from "how much text precedes it".
            body += ["", "For reference, the same extract is reproduced here:",
                     serialize(data_ref, "csv")]
        body += ["",
                "Return exactly one ```python code block."]
    elif paradigm == "sql":
        body = ["", "The extract has been loaded into a SQLite database at: " + data_ref,
                "The table is named `survey` and has the columns described above.",
                "Write a single SQL query that returns the requested quantity as one number.",
                "", "Question: " + question, "",
                "Return exactly one ```sql code block."]
    elif paradigm == "fw_s1":
        body = ["", "The extract is stored as a CSV file at: " + data_ref,
                "", "Question: " + question, "",
                "Step 1. Translate the question into a machine-checkable estimation "
                "specification. Reply with one ```json block holding exactly these keys: "
                "target_variable, subgroup_filter, estimator (one of: weighted proportion, "
                "weighted mean, weighted median, weighted correlation), weight_variable, "
                "design_variables, denominator_rule. Use the exact column names of the extract."]
    elif paradigm == "fw_s2":
        body = ["", "The extract is stored as a CSV file at: " + data_ref,
                "", "Question: " + question, "",
                "Step 2. Implement the specification below as Python code that reads the CSV and "
                "stores the result in a variable named `answer`.",
                "Constraints: pandas and numpy only; the code must use the sampling weight "
                "variable and the design variables named in the specification; no file writes.",
                "", "```json", spec.get("spec_json", "{}"), "```", "",
                "Return exactly one ```python code block."]
    elif paradigm == "fw_spec":
        body = ["", "The extract is stored as a CSV file at: " + data_ref,
                "", "Question: " + question, "",
                "Estimation specification produced for this question:",
                "", "```json", spec.get("spec_json", "{}"), "```",
                "", "Write Python code that reads the CSV and implements this "
                "specification, storing the final number in a variable named `answer`.",
                "", "Return exactly one ```python code block."]
    elif paradigm == "fw_full":
        body = ["", "The extract is stored as a CSV file at: " + data_ref,
                "", "Question: " + question, "",
                "Step 3. Write a second, independent implementation of the same estimate, using a "
                "different computational route from the first one, and store the result in a "
                "variable named `answer`.",
                "Constraints: pandas and numpy only; the code must use the sampling weight "
                "variable; no file writes.",
                "", "The first implementation returned: " + str(spec.get("first_answer")), "",
                "Return exactly one ```python code block."]
    else:
        raise ValueError(paradigm)
    return "\n".join(head + body)


NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_numeric(text: str) -> float | None:
    if not text:
        return None
    marker = re.search(r"FINAL ANSWER\s*:?\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    tail = marker.group(1) if marker else text
    tail = re.sub(r"[-–]\s*(\d+(?:\.\d+)?)\s*(?:%|percent)", r"-\1", tail)
    values = NUMBER.findall(tail)
    if not values:
        return None
    return float(values[0]) if marker else float(values[-1])


def parse_ranking(text: str, level_ids: list[str]) -> list[str] | None:
    if not text:
        return None
    marker = re.search(r"FINAL ANSWER\s*:?\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    tail = marker.group(1) if marker else text
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", tail)
    seen, order = set(), []
    for token in tokens:
        if token in level_ids and token not in seen:
            seen.add(token)
            order.append(token)
    if len(order) < 2:
        return None
    for lid in level_ids:
        if lid not in seen:
            order.append(lid)
    return order


def extract_block(text: str, lang: str) -> str | None:
    match = re.search(r"```" + lang + r"\s*(.*?)```", text or "", re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)```", text or "", re.DOTALL)
    return match.group(1).strip() if match else None


def extract_json(text: str) -> dict | None:
    block = extract_block(text, "json") or (text or "")
    match = re.search(r"\{.*\}", block, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def code_violations(code: str) -> list[str]:
    return [p for p in FORBIDDEN if re.search(p, code or "")]


def run_code(code: str, csv_path: str, timeout: int = 60) -> dict:
    bad = code_violations(code)
    if bad:
        return {"ok": False, "answer": None, "error": "rejected: " + ",".join(bad),
                "rejected": True}
    with tempfile.TemporaryDirectory() as tmp:
        code_path = os.path.join(tmp, "model_code.py")
        out_path = os.path.join(tmp, "out.json")
        with open(code_path, "w", encoding="utf-8") as fh:
            fh.write(code)
        try:
            proc = subprocess.run([PYTHON, "-I", SANDBOX, code_path, csv_path, out_path],
                                  capture_output=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return {"ok": False, "answer": None, "error": f"timeout after {timeout}s"}
        if not os.path.exists(out_path):
            return {"ok": False, "answer": None,
                    "error": "sandbox produced no result: " + proc.stderr.decode()[-400:]}
        with open(out_path, encoding="utf-8") as fh:
            return json.load(fh)


def run_sql(sql: str, db_path: str, timeout: int = 30) -> dict:
    if re.search(r"\b(drop|delete|update|insert|attach|pragma|create)\b", sql or "", re.I):
        return {"ok": False, "answer": None, "error": "rejected: non-read-only SQL"}
    try:
        con = sqlite3.connect("file:" + db_path + "?mode=ro", uri=True, timeout=timeout)
        try:
            row = con.execute(sql).fetchone()
        finally:
            con.close()
    except Exception as exc:
        return {"ok": False, "answer": None, "error": f"{type(exc).__name__}: {exc}"}
    if row is None:
        return {"ok": False, "answer": None, "error": "query returned no rows"}
    for value in row:
        try:
            return {"ok": True, "answer": float(value), "error": None}
        except (TypeError, ValueError):
            continue
    return {"ok": False, "answer": None, "error": "no numeric column in the first row"}


def slice_paths(spec: dict, level: str, slice_dir: str) -> tuple[str, str]:
    base = spec["slices"][level]["file"].replace(".parquet", "")
    return os.path.join(slice_dir, base + ".csv"), os.path.join(slice_dir, base + ".db")


def materialize(spec: dict, level: str, slice_dir: str) -> tuple[str, str]:
    """Write CSV and SQLite copies of a slice, refreshing them when the parquet
    is newer.  Stale copies made every downstream number compare a model answer
    against a different data set, so freshness is checked on every call."""
    ensure_dirs()
    src = os.path.join(slice_dir, spec["slices"][level]["file"])
    src_mtime = os.path.getmtime(src)
    csv_path, db_path = slice_paths(spec, level, slice_dir)
    df = None
    if not (os.path.exists(csv_path) and os.path.getmtime(csv_path) >= src_mtime):
        df = pd.read_parquet(src)
        df.to_csv(csv_path, index=False)
    if not (os.path.exists(db_path) and os.path.getmtime(db_path) >= src_mtime):
        if df is None:
            df = pd.read_parquet(src)
        if os.path.exists(db_path):
            os.remove(db_path)
        con = sqlite3.connect(db_path)
        try:
            df.to_sql("survey", con, index=False)
        finally:
            con.close()
    return csv_path, db_path
