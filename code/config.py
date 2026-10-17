"""Project configuration for the survey-estimation benchmark.

Everything downstream imports its paths, cycles, seeds and model inventory
from here, so a run is reproducible from this file plus the numbered scripts.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------- paths
PAPER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PAPER_DIR, "code")
DATA_DIR = os.path.join(PAPER_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
SLICE_DIR = os.path.join(DATA_DIR, "slices")
MODEL_DIR = os.path.join(PAPER_DIR, "models")
RESULT_DIR = os.path.join(PAPER_DIR, "results")
GEN_DIR = os.path.join(RESULT_DIR, "generations")
LOG_DIR = os.path.join(RESULT_DIR, "logs")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
VECTOR_DIR = FIG_DIR

# Optional local copy of the CDC XPT files; empty means "download from CDC".
SRC_NHANES = os.environ.get("NHANES_SOURCE_DIR", "")
SRC_CODEBOOK = os.path.join(PROC_DIR, "codebook.json")

# ---------------------------------------------------------------- cycles
# NHANES public-use files are addressed as <YEAR>/DataFiles/<STEM>_<SUFFIX>.xpt
CYCLES = {
    "2017-2018": {"year": "2017", "suffix": "J"},
    "2015-2016": {"year": "2015", "suffix": "I"},
}
PRIMARY_CYCLE = "2017-2018"
REPLICATION_CYCLE = "2015-2016"
MODULE_STEMS = ["DEMO", "BMX", "DBQ", "PAQ", "SLQ", "SMQ", "ALQ", "DR1TOT"]
DOWNLOAD_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{stem}.xpt"

# ---------------------------------------------------------------- design
WEIGHTS = {
    "interview": "WTINT2YR",     # questionnaire-only estimands
    "exam": "WTMEC2YR",          # estimands that use the mobile examination centre
    "dietary": "WTDRD1",         # 24-hour dietary recall totals
}
WEIGHT_LABELS = {
    "WTINT2YR": "2-year interview weight",
    "WTMEC2YR": "2-year MEC exam weight",
    "WTDRD1": "day-1 dietary recall weight",
}
STRATA = "SDMVSTRA"
PSU = "SDMVPSU"
ADULT_MIN_AGE = 20

# ---------------------------------------------------------------- sampling
RANDOM_SEED = 20261015
# 250-row slices keep the prompt inside every model's context window (including
# the 8k-context open weights) while still requiring a genuine weighted aggregation.
# 120 rows fits every model's context and memory budget (a 3B model already
# exceeds 16 GB of VRAM on a 250-row inline extract), while a weighted estimate
# on 120 respondents is still a well-defined survey estimand.  The larger levels
# are the extract-size axis and are built for the subset only.
SLICE_LEVELS = {"s120": 120, "s250": 250, "s500": 500}
MAIN_SLICE = "s120"
ROBUST_SLICE = "s250"
EXTRA_SLICE_LEVELS = ["s250", "s500"]

# ---------------------------------------------------------------- benchmark
TASK_FAMILIES = [
    ("proportion", 60),
    ("mean", 60),
    ("difference", 60),
    ("ranking", 60),
    ("association", 60),
]
SUBSET_SIZE = 100  # tasks used for the extended model set and robustness work

FAMILY_LABELS = {
    "proportion": "population proportion",
    "mean": "population mean or median",
    "difference": "between-group difference",
    "ranking": "subgroup ranking",
    "association": "association between two variables",
}

ABS_TOL = {
    "proportion": 0.02,
    "mean": 0.02,        # relative
    "difference": 0.03,
    "ranking": 0.0,      # exact ordering of the requested pair/triple
    "association": 0.05,
}

# ---------------------------------------------------------------- models
API_MODELS = [
    {"id": "deepseek-flash", "label": "DeepSeek-flash", "vendor": "DeepSeek",
     "backend": "api", "params": "undisclosed", "note": "non-thinking mode, temperature 0"},
    {"id": "deepseek-v4-pro", "label": "DeepSeek-V4-Pro", "vendor": "DeepSeek",
     "backend": "api", "params": "undisclosed", "note": "non-thinking mode, temperature 0"},
]

LOCAL_MODELS = [
    {"id": "qwen2.5-3b", "repo": "Qwen/Qwen2.5-3B-Instruct", "label": "Qwen2.5-3B-Instruct",
     "vendor": "Alibaba", "params": "3B",
     "path": os.environ.get("QWEN25_3B_PATH", "")},
    {"id": "qwen2.5-7b", "repo": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen2.5-7B-Instruct",
     "vendor": "Alibaba", "params": "7B"},
    {"id": "qwen2.5-14b", "repo": "Qwen/Qwen2.5-14B-Instruct", "label": "Qwen2.5-14B-Instruct",
     "vendor": "Alibaba", "params": "14B"},
    {"id": "llama-3.1-8b", "repo": "meta-llama/Llama-3.1-8B-Instruct",
     "label": "Llama-3.1-8B-Instruct", "vendor": "Meta", "params": "8B"},
    {"id": "mistral-7b", "repo": "mistralai/Mistral-7B-Instruct-v0.3",
     "label": "Mistral-7B-Instruct-v0.3", "vendor": "Mistral AI", "params": "7B",
     "scope": "subset"},
    {"id": "phi-3.5-mini", "repo": "microsoft/Phi-3.5-mini-instruct",
     "label": "Phi-3.5-mini-instruct", "vendor": "Microsoft", "params": "3.8B"},
]

ALREADY_LOCAL = {"qwen2.5-3b"}
CORE_MODELS = ["deepseek-flash", "deepseek-v4-pro", "qwen2.5-7b", "llama-3.1-8b"]

# ---------------------------------------------------------------- paradigms
PARADIGMS = [
    ("direct", "direct answering"),
    ("code", "free-form code generation"),
    ("sql", "text-to-SQL"),
]
FRAMEWORK_STAGES = [
    ("fw_s1", "estimand specification"),
    ("fw_s2", "constrained code generation"),
    ("fw_full", "specification + constrained code + recomputation gate"),
]


def ensure_dirs() -> None:
    for path in (CODE_DIR, DATA_DIR, RAW_DIR, PROC_DIR, SLICE_DIR, MODEL_DIR,
                 RESULT_DIR, GEN_DIR, LOG_DIR, FIG_DIR, VECTOR_DIR):
        os.makedirs(path, exist_ok=True)
