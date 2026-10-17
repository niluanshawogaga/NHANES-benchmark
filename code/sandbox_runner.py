"""Isolated executor for model-written estimation code.

    python -I sandbox_runner.py <code.py> <data.csv> <out.json>

Only pandas and numpy are placed in the namespace; the answer must end up in a
variable named ``answer``.  Anything that raises is reported instead of crashing
the caller.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import traceback


def main() -> None:
    code_path, data_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    buf = io.StringIO()
    try:
        import numpy as np
        import pandas as pd
    except Exception as exc:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"ok": False, "error": f"import failed: {exc}", "stdout": ""}, fh)
        return
    namespace = {"pd": pd, "pandas": pd, "np": np, "numpy": np, "DATA_PATH": data_path}
    try:
        with open(code_path, encoding="utf-8") as fh:
            source = fh.read()
        with contextlib.redirect_stdout(buf):
            exec(compile(source, "<model_code>", "exec"), namespace)
        answer = namespace.get("answer")
        if answer is None:
            for key in ("result", "estimate", "value"):
                if key in namespace:
                    answer = namespace[key]
                    break
        numeric = None
        if answer is not None:
            try:
                numeric = float(answer)
            except (TypeError, ValueError):
                try:
                    numeric = float(list(answer)[0])
                except Exception:
                    numeric = None
        if numeric is not None and not math.isfinite(numeric):
            numeric = None
        result = {"ok": numeric is not None, "answer": numeric,
                  "error": None if numeric is not None else "no finite numeric answer",
                  "stdout": buf.getvalue()[-1500:]}
    except Exception as exc:
        result = {"ok": False, "answer": None,
                  "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()[-1500:],
                  "stdout": buf.getvalue()[-1500:]}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False)


if __name__ == "__main__":
    main()
