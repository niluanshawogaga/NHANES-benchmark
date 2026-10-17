# Design-aware population estimation benchmark (NHANES)

This repository contains the benchmark, the reference implementation, the raw
model completions and the scoring code of a study that asks whether large
language models can compute design-weighted population quantities from complex
survey microdata, and how much of the answer depends on the interface through
which the model is asked.

## Contents

    code/                 benchmark construction, runners, scoring, tables, figures
    data/processed/       tasks_main.json (300 tasks), tasks_repl.json (100-task
                          mirror on 2015-2016), variable metadata, a trimmed
                          CDC codebook
    results/              per-task scores, per-family and per-table summaries,
                          verification records
    results/generations/  raw completions, one JSONL file per configuration,
                          protocol and condition
    figures/              vector copies of the manuscript figures

The analysis extracts are not stored: prompts.materialize() recreates a 120-row
extract for a task from the two public NHANES cycles with a fixed seed, so every
extract is reproducible from the task definition.

The open-weight checkpoints (Qwen2.5-3B/7B, Phi-3.5-mini) are not redistributed.
Download them from their official Hugging Face repositories and set the paths in
config.LOCAL_MODELS (or the QWEN25_3B_PATH environment variable) to the local
copies.

## Reproducing

See REPRODUCE.md. Briefly:

    pip install -r requirements.txt
    python code/10_build_data.py --all  # download the NHANES modules from CDC
    python code/20_build_tasks.py       # rebuild the 300 tasks and the mirror set
    python code/oracle.py               # check every reference value
    python code/60_evaluate.py          # score the stored completions
    python code/65_make_tables.py       # write the manuscript tables
    python code/70_figures.py --tag main
    python code/91_verify_claims.py     # recompute every number quoted in the text

Every reference value is reproduced exactly by the reference implementation that
runs in the same sandbox as the models; results/reference_verification.md records
that check.

## Data source and ethics

The benchmark is derived from the public-use National Health and Nutrition
Examination Survey (NHANES), cycles 2015-2016 and 2017-2018, distributed by the
National Center for Health Statistics (NCHS) of the US Centers for Disease
Control and Prevention (https://wwwn.cdc.gov/nchs/nhanes/). NHANES public-use
files are de-identified and released for research; NCHS reviewed the survey
protocol and obtained consent from participants. This repository does not
redistribute any respondent-level record: it ships task definitions, aggregate
reference values, and the model outputs. Use of the data here is consistent with
the NCHS public-use terms. NCHS/CDC does not endorse this work or its
conclusions.

The repository contains no personal contact details, credentials or API keys.
Model access is read from the DEEPSEEK_API_KEY environment variable at run time.

## Path normalization in released completions

The completions are stored as the models produced them, with one mechanical
change: absolute file paths from the development environment were rewritten to
repository-relative paths (for example `data/slices/<task>.csv`), and local
Python installation paths in execution tracebacks were replaced with
`<python>`. No answer, score or token count was altered.

## License

Code: MIT (see LICENSE). Derived benchmark data and the released completions:
CC BY 4.0, consistent with the public-use NHANES source data. See NOTICE for
attribution. The model completions remain subject to the terms of the model
providers that generated them.

## Citing

If you use this benchmark, please cite the accompanying manuscript. A DOI for
this repository will be minted when the paper is accepted.
