#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, build_opener, urlopen

import psutil


LOCAL_NO_PROXY_HOSTS = (
    "127.0.0.1",
    "localhost",
    "::1",
    "0.0.0.0",
)


def with_local_no_proxy(env: dict[str, str] | None = None) -> dict[str, str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)

    existing = []
    for key in ("NO_PROXY", "no_proxy"):
        value = merged.get(key)
        if value:
            existing.extend(item.strip() for item in value.split(",") if item.strip())
    seen = set(existing)
    for host in LOCAL_NO_PROXY_HOSTS:
        if host not in seen:
            existing.append(host)
            seen.add(host)
    no_proxy = ",".join(existing)
    merged["NO_PROXY"] = no_proxy
    merged["no_proxy"] = no_proxy
    return merged


def is_local_http_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in LOCAL_NO_PROXY_HOSTS


def run_command(
    command: list[str],
    log_path: Path,
    *,
    env_overrides: dict[str, str] | None = None,
    wait: bool = True,
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    env = with_local_no_proxy(env_overrides)
    proc = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        preexec_fn=os.setsid,
    )
    if wait:
        proc.wait()
        log_file.close()
    else:
        # The child process keeps its stdout fd; close the parent copy.
        log_file.close()
    return proc


def run_subprocess(command: list[str], log_path: Path) -> int:
    proc = run_command(command, log_path, wait=False)
    return proc.wait()


def terminate_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    # A launcher may exit while its model children keep the owned process group
    # alive. Clean that group even when the original leader has already exited.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def http_service_ready(url: str) -> bool:
    """One bounded probe after a failed job; no retry or inference request."""
    request = build_opener(ProxyHandler({})).open if is_local_http_url(url) else urlopen
    try:
        with request(url, timeout=5) as response:
            return 200 <= response.status < 300
    except (URLError, OSError):
        return False


def wait_for_http(url: str, timeout_seconds: int, poll_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    opener = build_opener(ProxyHandler({})) if is_local_http_url(url) else None
    while time.time() < deadline:
        try:
            request = opener.open if opener is not None else urlopen
            with request(url, timeout=5) as resp:
                if 200 <= getattr(resp, "status", 200) < 500:
                    return
        except (URLError, OSError):
            time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for service readiness: {url}")


def wait_for_http_or_process_exit(
    url: str,
    *,
    proc: subprocess.Popen[str],
    timeout_seconds: int,
    poll_seconds: float,
    expected_model: str | None = None,
    identity_url: str | None = None,
    expected_fields: dict[str, str] | None = None,
) -> None:
    """Wait for our managed listener, not an unrelated HTTP service on its port."""
    if expected_model is None and not expected_fields:
        raise ValueError("Managed service readiness requires an expected identity")
    address = urlparse(url)
    if not is_local_http_url(url) or address.port is None:
        raise ValueError(f"Managed service must use an explicit local port: {url}")
    identity_url = identity_url or f"{address.scheme}://{address.netloc}/v1/models"
    if urlparse(identity_url).netloc != address.netloc:
        raise ValueError("Service identity must be checked on the managed port")
    deadline = time.monotonic() + timeout_seconds
    opener = build_opener(ProxyHandler({})) if is_local_http_url(url) else None
    while time.monotonic() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Service process exited before readiness: url={url} return_code={return_code}"
            )
        try:
            if not _owned_listener(address.port, proc):
                time.sleep(poll_seconds)
                continue
            request = opener.open if opener is not None else urlopen
            with request(url, timeout=5) as resp:
                if not 200 <= resp.status < 300:
                    time.sleep(poll_seconds)
                    continue
            with request(identity_url, timeout=5) as resp:
                if not 200 <= resp.status < 300:
                    time.sleep(poll_seconds)
                    continue
                try:
                    payload = json.load(resp)
                    if expected_model is not None:
                        matches = expected_model in [item["id"] for item in payload["data"]]
                    else:
                        matches = all(payload[key] == value for key, value in expected_fields.items())
                except (ValueError, TypeError, KeyError) as exc:
                    raise RuntimeError(f"Invalid service identity response: {identity_url}") from exc
                if not matches:
                    raise RuntimeError(
                        f"Service identity mismatch: {identity_url}; "
                        f"expected={expected_model if expected_model is not None else expected_fields}; "
                        f"received={payload}"
                    )
            # Recheck after HTTP: the launcher/listener may have exited during the probe.
            if proc.poll() is not None:
                raise RuntimeError(f"Service process exited during readiness: {url}")
            if _owned_listener(address.port, proc):
                return
        except (URLError, OSError):
            pass
        time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for service readiness: {url}")


def assert_service_port_available(port: int) -> None:
    """Fail before launching; never choose a different port or kill its occupant."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Managed service port {port} is unavailable: {exc}") from exc


def _owned_listener(port: int, proc: subprocess.Popen[str]) -> bool:
    # run_command creates a session/process group with proc.pid as its leader.
    # Conda/worker launchers may delegate the actual listening socket to a child.
    try:
        connections = psutil.net_connections(kind="tcp")
    except (psutil.AccessDenied, OSError) as exc:
        raise RuntimeError(f"Cannot inspect listener ownership on port {port}: {exc}") from exc
    listeners = [conn for conn in connections
                 if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port
                 and conn.laddr.ip in ("127.0.0.1", "0.0.0.0", "::1", "::")]
    for conn in listeners:
        if conn.pid is None:
            raise RuntimeError(f"Cannot verify owner of managed service port {port}")
        try:
            group = os.getpgid(conn.pid)
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            raise RuntimeError(f"Cannot verify owner of managed service port {port}") from exc
        if group != proc.pid:
            raise RuntimeError(
                f"Managed service port {port} belongs to another process: "
                f"listener_pid={conn.pid} pgid={group}; expected_pgid={proc.pid}"
            )
    return bool(listeners)
