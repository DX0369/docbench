"""One managed MinerU API + Popo vLLM service, P persistent CPU clients."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import collect_pdf_paths, write_json
from config.method_versions import get_method_version
from execution.processes import assert_service_port_available, run_command, terminate_process, wait_for_http_or_process_exit
from execution.queue import collect_done_statuses, collect_failed_docs, collect_unprocessed_docs, write_pending_job_list
from execution.workers import write_timing_state, start_queue_processing, queue_processing_seconds
from method_paths import method_repo_path
from methods.mineru.versions.v344_2605 import PROFILE
from methods.mineru_popo.commands import add_arguments, validate_options
from paths import CONDA_ENV_ROOT, SRC_ROOT, TMP_ROOT

METHOD = "mineru_344_2605_popo"
EXECUTION_PROFILE = METHOD + "_vllm_services"


def preflight(args):
    for repo, revision in ((PROFILE.repository_path, PROFILE.source_revision),
                           (method_repo_path("mineru_popo"), get_method_version(METHOD).source_revision)):
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise ValueError(f"Source revision mismatch: {repo}: {actual} != {revision}")
    patches = Path(__file__).resolve().parent / "patches"
    subprocess.run(["git", "-C", str(method_repo_path("mineru_popo")), "apply", "--reverse", "--check",
                    *(str(patches / name) for name in
                      ("same_page.patch", "pr12.patch", "pr15.patch", "failure_propagation.patch"))], check=True)
    for python in (CONDA_ENV_ROOT / PROFILE.conda_env / "bin/python", args.popo_vllm_python):
        if not python.is_file():
            raise FileNotFoundError(f"Missing service interpreter: {python}")
    # Refuse to attach accidentally to unrelated services using the same ports.
    for port in (args.mineru_api_port, args.popo_port):
        assert_service_port_available(port)


def service_commands(args, work):
    mineru = [str(CONDA_ENV_ROOT / PROFILE.conda_env / "bin/python"), "-u",
              str(SRC_ROOT / "methods/mineru/api_service.py"),
              "--host", "127.0.0.1", "--port", str(args.mineru_api_port),
              "--preload-models", "--method-id", PROFILE.method_id,
              "--backend", PROFILE.api_backend, "--method", "auto", "--lang", "auto",
              "--gpu-memory-utilization", str(args.mineru_vllm_gpu_memory_utilization),
              "--api-max-concurrent-requests", str(args.popo_concurrency),
              "--api-output-root", str(work / "mineru_service")]
    popo = [str(args.popo_vllm_python), "-u", "-m", "vllm.entrypoints.openai.api_server",
            "--host", "127.0.0.1", "--port", str(args.popo_port),
            "--model", str(get_method_version(METHOD).model_path),
            "--served-model-name", "Popo", "--dtype", "bfloat16",
            "--tensor-parallel-size", "1", "--pipeline-parallel-size", "1",
            "--gpu-memory-utilization", str(args.popo_gpu_memory_utilization),
            "--max-model-len", str(args.popo_max_model_len)]
    return [("mineru", mineru, f"http://127.0.0.1:{args.mineru_api_port}/ready"),
            ("popo", popo, f"http://127.0.0.1:{args.popo_port}/health")]


def client_command(args, worker_id):
    command = [sys.executable, "-u", str(Path(__file__).with_name("client.py")),
               "--input-root", str(args.input_root), "--output-root", str(args.output_root),
               "--queue-dir", str(args.queue_dir), "--worker-id", worker_id,
               "--mineru-url", f"http://127.0.0.1:{args.mineru_api_port}",
               "--popo-url", f"http://127.0.0.1:{args.popo_port}/v1"]
    if args.skip_existing:
        command.append("--skip-existing")
    return command


def run(args):
    root = args.runtime_root
    root.mkdir(parents=True, exist_ok=False)
    state = {"status": "initializing", "execution_profile": EXECUTION_PROFILE,
             "concurrency": args.popo_concurrency, "warmup": False,
             "mineru_gpu_memory_utilization": args.mineru_vllm_gpu_memory_utilization,
             "popo_gpu_memory_utilization": args.popo_gpu_memory_utilization,
             "popo_max_model_len": args.popo_max_model_len,
             "popo_vllm_python": str(args.popo_vllm_python)}
    started = None
    work = None
    begin = time.perf_counter()
    try:
        preflight(args)
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="pv_", dir=TMP_ROOT))
        state["temporary_work"] = str(work)
        write_json(work / "mineru.json", {"bucket_info": {}, "models-dir": {
            "pipeline": str(PROFILE.pipeline_model_path), "vlm": str(PROFILE.model_path)},
            "model-source": "local", "config_version": "1.3.2"})
        if args.queue_dir is None:
            args.queue_dir = root / "queue"
            pdfs = args.pdf or collect_pdf_paths(args.input_root)
            if not pdfs:
                raise ValueError("No PDF inputs")
            for index, pdf in enumerate(pdfs):
                write_pending_job_list(args.queue_dir / "pending", index, [pdf])
        timing = args.queue_dir / "timing"
        timing.mkdir()
        write_timing_state(timing / "control.json", {"parent_pid": os.getpid()})
        env = {**PROFILE.environment_overrides(),
               "MINERU_TOOLS_CONFIG_JSON": str(work / "mineru.json"),
               "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
               "PYTHONPATH": "", "PYTHONHOME": "", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "TMPDIR": str(work)}
        plan = service_commands(args, work)
        (root / "command.txt").write_text(
            "\n".join(shlex.join(command) for _, command, _ in plan) + "\n")
        with ExitStack() as stack:
            services = []
            for name, command, url in plan:
                assert_service_port_available(args.mineru_api_port if name == "mineru" else args.popo_port)
                process = run_command(command, root / f"{name}.log", env_overrides=env, wait=False)
                stack.callback(terminate_process, process)
                services.append(process)
                wait_for_http_or_process_exit(url, proc=process,
                    timeout_seconds=args.popo_startup_timeout, poll_seconds=0.5,
                    **({"identity_url": url, "expected_fields": {
                        "status": "ready", "method_id": PROFILE.method_id,
                        "release_version": PROFILE.version,
                    }} if name == "mineru" else {"expected_model": "Popo"}))
            clients = []
            for index in range(args.popo_concurrency):
                worker_id = f"{args.worker_id}.popo{index:02d}"
                log = root / f"{worker_id}.log"
                process = run_command(client_command(args, worker_id), log,
                                      env_overrides={**env, "CUDA_VISIBLE_DEVICES": ""}, wait=False)
                stack.callback(terminate_process, process)
                clients.append((worker_id, log, process))
            started = start_queue_processing(args.queue_dir, clients,
                timeout=args.popo_startup_timeout, services=services)
            state.update(status="processing", startup_seconds=time.perf_counter() - begin)
            write_json(root / "run_status.json", state)
            while True:
                if any(proc.poll() is not None for proc in services):
                    raise RuntimeError("A model service exited during processing; pending documents were not retried")
                states = []
                for worker_id, _log, proc in clients:
                    path = timing / f"{worker_id}.json"
                    item = json.loads(path.read_text())
                    if item["status"] == "failed" or (proc.poll() is not None and item["status"] not in ("done", "failed")):
                        raise RuntimeError(f"Client {worker_id} failed; see {path}")
                    states.append(item)
                if all(item["status"] == "done" for item in states):
                    break
                time.sleep(0.05)
            state["processing_seconds"] = queue_processing_seconds(
                args.queue_dir, [item[0] for item in clients], started)
            # End timestamps already exist. Client close/exit is outside timing,
            # but a post-processing client crash must not become a successful run.
            return_codes = [process.wait(timeout=20) for _, _, process in clients]
            statuses = collect_done_statuses(args.queue_dir / "done")
            failed = collect_failed_docs(statuses)
            unprocessed = collect_unprocessed_docs(statuses, args.queue_dir)
            state.update(status="failed" if failed or unprocessed or any(return_codes) or any(job["failed"] for job in statuses) else "completed",
                         failed_docs=failed, unprocessed_docs=unprocessed)
            # No timing includes client/service teardown.
            write_json(root / "run_status.json", state)
        if state["status"] == "completed":
            shutil.rmtree(work)
            state.pop("temporary_work")
        return state["status"] != "completed"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        if started is None:
            state["startup_seconds"] = time.perf_counter() - begin
        if args.queue_dir is not None:
            state["unprocessed_docs"] = collect_unprocessed_docs(
                collect_done_statuses(args.queue_dir / "done"), args.queue_dir)
        # Missing finish events are not replaced with a wall-clock throughput.
        raise
    finally:
        write_json(root / "run_status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "output-root", "runtime-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--worker-id", default="gpu")
    parser.add_argument("--pdf", type=Path, action="append")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_arguments(parser)
    args = parser.parse_args()
    try:
        validate_options(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.input_root, args.output_root = args.input_root.resolve(), args.output_root.resolve()
    args.runtime_root = args.runtime_root.resolve()
    if args.pdf and args.queue_dir:
        parser.error("--pdf and --queue-dir are mutually exclusive")
    if args.pdf:
        args.pdf = [pdf.resolve() for pdf in args.pdf]
        if len(set(args.pdf)) != len(args.pdf):
            parser.error("Duplicate PDF selections")
        for pdf in args.pdf:
            if not pdf.is_relative_to(args.input_root) or not pdf.is_file() or pdf.suffix.lower() != ".pdf":
                parser.error(f"Not an input PDF: {pdf}")
    if args.dry_run:
        if args.queue_dir is None:
            args.queue_dir = args.runtime_root / "queue"
        print(json.dumps({"services": service_commands(args, TMP_ROOT / "<runtime>"),
                          "clients": [client_command(args, f"{args.worker_id}.popo{i:02d}")
                                      for i in range(args.popo_concurrency)],
                          "client_cuda_visible_devices": ""}, indent=2))
        return
    if len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x]) != 1:
        parser.error("Set CUDA_VISIBLE_DEVICES to exactly one GPU")
    os.environ.pop("TORCHDYNAMO_DISABLE", None)
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(1 if run(args) else 0)


if __name__ == "__main__":
    main()
