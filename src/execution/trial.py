#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Sequence




def compute_percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = max(0, math.ceil((percentile / 100.0) * len(sorted_values)) - 1)
    return sorted_values[index]


def get_pdf_page_count(pdf_path: Path) -> int:
    completed = subprocess.run(
        ["pdfinfo", str(pdf_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "pdfinfo failed for " f"{pdf_path}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in completed.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Unable to determine page count for {pdf_path}")


class GpuMonitor:
    def __init__(self, gpu_id: int, interval_seconds: float) -> None:
        self.gpu_id = gpu_id
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode == 0:
                    for line in completed.stdout.splitlines():
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) != 4:
                            continue
                        index = int(parts[0])
                        if index != self.gpu_id:
                            continue
                        self.samples.append(
                            {
                                "timestamp": time.time(),
                                "utilization_gpu": float(parts[1]),
                                "memory_used_mb": float(parts[2]),
                                "memory_total_mb": float(parts[3]),
                            }
                        )
                        break
            except Exception:
                pass
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, float | None]:
        if not self.samples:
            return {
                "avg_gpu_util_percent": None,
                "peak_gpu_util_percent": None,
                "avg_memory_used_mb": None,
                "peak_memory_used_mb": None,
            }
        return {
            "avg_gpu_util_percent": round(
                sum(item["utilization_gpu"] for item in self.samples) / len(self.samples),
                3,
            ),
            "peak_gpu_util_percent": max(item["utilization_gpu"] for item in self.samples),
            "avg_memory_used_mb": round(
                sum(item["memory_used_mb"] for item in self.samples) / len(self.samples),
                3,
            ),
            "peak_memory_used_mb": max(item["memory_used_mb"] for item in self.samples),
        }


class CpuMonitor:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))

    def _read_cpu_totals(self) -> tuple[float, float] | None:
        try:
            first_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0]
        except Exception:
            return None
        parts = first_line.split()
        if not parts or parts[0] != "cpu" or len(parts) < 5:
            return None
        try:
            counters = [float(value) for value in parts[1:]]
        except ValueError:
            return None
        idle = counters[3] + (counters[4] if len(counters) > 4 else 0.0)
        total = sum(counters)
        return total, idle

    def _run(self) -> None:
        previous_totals = self._read_cpu_totals()
        while not self._stop.is_set():
            cpu_util_percent: float | None = None
            load1_per_core: float | None = None

            current_totals = self._read_cpu_totals()
            if previous_totals is not None and current_totals is not None:
                total_delta = current_totals[0] - previous_totals[0]
                idle_delta = current_totals[1] - previous_totals[1]
                if total_delta > 0:
                    cpu_util_percent = max(
                        0.0,
                        min(100.0, 100.0 * (1.0 - (idle_delta / total_delta))),
                    )
            previous_totals = current_totals

            try:
                load1 = os.getloadavg()[0]
                cpu_count = os.cpu_count() or 1
                load1_per_core = load1 / cpu_count
            except (AttributeError, OSError):
                load1_per_core = None

            if cpu_util_percent is not None or load1_per_core is not None:
                sample: dict[str, float] = {"timestamp": time.time()}
                if cpu_util_percent is not None:
                    sample["cpu_util_percent"] = cpu_util_percent
                if load1_per_core is not None:
                    sample["load1_per_core"] = load1_per_core
                self.samples.append(sample)
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict[str, Any]:
        cpu_util_values = [float(item["cpu_util_percent"]) for item in self.samples if "cpu_util_percent" in item]
        load_values = [float(item["load1_per_core"]) for item in self.samples if "load1_per_core" in item]
        avg_cpu_util_percent = round(sum(cpu_util_values) / len(cpu_util_values), 3) if cpu_util_values else None
        p95_cpu_util_percent = compute_percentile(cpu_util_values, 95.0)
        avg_load1_per_core = round(sum(load_values) / len(load_values), 6) if load_values else None
        cpu_bottleneck_flag = bool(
            (avg_cpu_util_percent is not None and avg_cpu_util_percent >= 85.0)
            or (p95_cpu_util_percent is not None and float(p95_cpu_util_percent) >= 95.0)
            or (avg_load1_per_core is not None and float(avg_load1_per_core) >= 1.0)
        )
        return {
            "sample_count": len(self.samples),
            "avg_cpu_util_percent": avg_cpu_util_percent,
            "p95_cpu_util_percent": (
                round(float(p95_cpu_util_percent), 3) if p95_cpu_util_percent is not None else None
            ),
            "avg_load1_per_core": avg_load1_per_core,
            "cpu_bottleneck_flag": cpu_bottleneck_flag,
            "cpu_affects_throughput_flag": None,
        }
