# Reproducing the study

All commands are run from the repository root.

## 1. Environment

    python -m venv .venv
    .venv/Scripts/activate        # Windows; use source .venv/bin/activate elsewhere
    pip install -r requirements.txt

## 2. Survey data and benchmark construction

    python code/10_build_data.py --all   # downloads the NHANES XPT modules from CDC
    python code/20_build_tasks.py
    python code/oracle.py

Set NHANES_SOURCE_DIR only if you keep a local mirror of the CDC files; the
downloader copies from it instead of fetching over the network.

oracle.py recomputes every reference value with the same library the scorers use
and compares it with the stored value; it must report 240/240 on the main level,
79/79 on the replication cycle and 79 on each of the two larger extracts.

## 3. Scoring the released completions

    python code/60_evaluate.py

This writes results/by_task_main.jsonl and results/by_task_repl.jsonl, which are
the inputs of every table and figure.

## 4. Tables, figures and integrity checks

    python code/65_make_tables.py
    python code/70_figures.py --tag main
    python code/91_verify_claims.py

## 5. Re-running the models

Interface models: put the API key in the environment variable DEEPSEEK_API_KEY and
run code/40_run_api.py with the same arguments documented in its docstring.

Local models: place the Hugging Face snapshots in a directory of your choice, edit
config.LOCAL_MODELS, then run code/50_run_local.py. The framework stages use
50_run_local.py --mode fw (specification, constrained implementation, gate) and
--mode fw_spec (specification with free code).
