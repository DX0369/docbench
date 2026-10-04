from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from paths import CONDA_ENV_ROOT, CONDA_EXE


RunnerBuilder = Callable[[argparse.Namespace, int, int | None, Path], list[str]]
WorkerBuilder = Callable[[argparse.Namespace, int, Path, str], list[str]]
ServiceBuilder = Callable[[argparse.Namespace, int, int], list[str]]
HealthcheckBuilder = Callable[[int], str]


def conda_python(env_name: str) -> list[str]:
    return [str(CONDA_EXE), "run", "-n", env_name, "python"]


def env_vllm(env_name: str) -> str:
    return str(CONDA_ENV_ROOT / env_name / "bin" / "vllm")
