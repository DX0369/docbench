from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "src").is_dir() and (candidate / "README.md").is_file():
            return candidate
    raise RuntimeError(f"Unable to locate GlobalDocBench repo root from {current}")


REPO_ROOT = find_repo_root()
SRC_ROOT = REPO_ROOT / "src"
PROJECT_ROOT = REPO_ROOT.parent
DATA_ROOT = Path(
    os.path.expandvars(os.environ.get("GDB_DATA_ROOT", str(PROJECT_ROOT / "data")))
).expanduser().resolve()
PDF_ROOT = DATA_ROOT / "pdf_tree"
MODEL_ROOT = Path(
    os.path.expandvars(os.environ.get("GDB_MODEL_ROOT", str(PROJECT_ROOT / "models")))
).expanduser().resolve()
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
METHOD_ROOT = Path(
    os.path.expandvars(
        os.environ.get("GDB_METHOD_ROOT", str(THIRD_PARTY_ROOT / "methods"))
    )
).expanduser().resolve()
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
TMP_ROOT = PROJECT_ROOT / "tmp"
BATCH_RESULT_ROOT = EXPERIMENT_ROOT / "batch_result"
LOG_ROOT = EXPERIMENT_ROOT / "logs"
OUTPUTS_FORMAL_ROOT = EXPERIMENT_ROOT / "outputs_formal"
THROUGHPUT_SUBSET_ROOT = EXPERIMENT_ROOT / "throughput_subsets"
REPORT_ROOT = PROJECT_ROOT / "reports"
FINAL_TABLE_ROOT = REPORT_ROOT / "final_experiment_tables"
THROUGHPUT_REPORT_ROOT = REPORT_ROOT / "throughput_metric_summary"
CONDA_ROOT = Path(
    os.path.expandvars(
        os.environ.get("GDB_CONDA_ROOT", "/data/wangsq/software/miniconda")
    )
).expanduser().resolve()
CONDA_EXE = CONDA_ROOT / "bin" / "conda"
CONDA_ENV_ROOT = CONDA_ROOT / "envs"
