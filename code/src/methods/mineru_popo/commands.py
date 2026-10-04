"""Commands for one GPU coordinator and P model-free document clients."""
from pathlib import Path
import os
import signal
import subprocess

from execution.processes import terminate_process

from paths import CONDA_ENV_ROOT, SRC_ROOT

DEFAULT_POPO_MAX_MODEL_LEN = 32768


def add_arguments(parser, *, include_mineru_memory=True, include_scheduling=True):
    if include_scheduling:
        parser.add_argument("--popo-concurrency", type=int, default=1)
    parser.add_argument("--popo-vllm-python", type=Path,
                        help="Explicit interpreter for a verified Popo vLLM environment; never auto-install.")
    parser.add_argument("--popo-gpu-memory-utilization", type=float,
                        help="Required shared-GPU budget; no unverified 3090 default.")
    parser.add_argument("--popo-max-model-len", type=int,
                        default=DEFAULT_POPO_MAX_MODEL_LEN,
                        help=("Popo service context length (default: "
                              f"{DEFAULT_POPO_MAX_MODEL_LEN}); override only with a verified profile."))
    if include_mineru_memory:
        parser.add_argument("--mineru-vllm-gpu-memory-utilization", type=float, default=0.55)
    if include_scheduling:
        parser.add_argument("--mineru-api-port", type=int, default=19890)
        parser.add_argument("--popo-port", type=int, default=19891)
    parser.add_argument("--popo-startup-timeout", type=float, default=600)


OPTIONS = ("popo_concurrency", "popo_vllm_python", "popo_gpu_memory_utilization",
           "popo_max_model_len", "mineru_vllm_gpu_memory_utilization",
           "mineru_api_port", "popo_port", "popo_startup_timeout")


def runtime_options(args):
    """Serializable options shared by launchers and selected throughput profiles."""
    return {name: str(value) if isinstance(value, Path) else value
            for name in OPTIONS if (value := getattr(args, name)) is not None}


def cli_options(args, *, exclude=()):
    return [token for name, value in runtime_options(args).items()
            if name not in exclude
            for token in ("--" + name.replace("_", "-"), str(value))]


def validate_options(args):
    if args.popo_concurrency < 1:
        raise ValueError("--popo-concurrency must be positive")
    if args.popo_vllm_python is None or args.popo_gpu_memory_utilization is None:
        raise ValueError("Set --popo-vllm-python and --popo-gpu-memory-utilization explicitly; dual-service capacity is not yet GPU-verified")
    if args.popo_max_model_len <= 8192:
        raise ValueError("Popo context must exceed 8192 to leave room for the input")
    for value in (args.popo_gpu_memory_utilization, args.mineru_vllm_gpu_memory_utilization):
        if not 0 < value < 1:
            raise ValueError("Each engine memory utilization must be between 0 and 1")
    if args.popo_gpu_memory_utilization + args.mineru_vllm_gpu_memory_utilization >= 1:
        raise ValueError("Two engine budgets must leave space for MinerU auxiliary components")
    if args.mineru_api_port == args.popo_port or any(not 1 <= p <= 65535 for p in (args.mineru_api_port, args.popo_port)):
        raise ValueError("Two distinct valid service ports are required")
    if args.popo_startup_timeout <= 0:
        raise ValueError("Startup timeout must be positive")


def runtime_root(queue_dir, worker_id):
    return Path(queue_dir).parent / f"{Path(queue_dir).name}_{worker_id}_services"


def wait_for_coordinator(process):
    """The native launcher owns this separate session, including on Ctrl-C."""
    try:
        return process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            # Allow both services' existing bounded cleanup to finish before
            # killing the coordinator that owns their separate process groups.
            process.wait(timeout=60)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        finally:
            terminate_process(process)
        raise


def build_worker(args, gpu_id, queue_dir, worker_id):
    validate_options(args)
    cmd = [str(CONDA_ENV_ROOT / "gdb_mineru_popo/bin/python"), "-u",
           str(SRC_ROOT / "methods/mineru_popo/worker.py"),
           "--input-root", args.input_root,
           "--output-root", str(Path(args.output_root_base) / "mineru_344_2605_popo"),
           "--queue-dir", str(queue_dir), "--worker-id", worker_id,
           "--runtime-root", str(runtime_root(queue_dir, worker_id))]
    for option in OPTIONS:
        cmd.extend(["--" + option.replace("_", "-"), str(getattr(args, option))])
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd
