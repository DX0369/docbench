#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from config.method_versions import METHOD_CATALOG_PATH, load_method_catalog


@dataclass(frozen=True)
class Repository:
    id: str
    path: Path
    url: str
    ref: str
    update_ref: str
    revision: str
    release: str | None = None
    version_file: str | None = None
    source_version: str | None = None


def run_git(args: Sequence[str], *, cwd: Path | None = None, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def load_repositories(path: Path) -> list[Repository]:
    payload = load_method_catalog(path)

    repositories: list[Repository] = []
    seen_ids: set[str] = set()
    seen_paths: set[Path] = set()
    for raw in payload["repositories"]:
        repository = Repository(
            id=raw["id"],
            path=PROJECT_ROOT / raw["path"],
            url=raw["url"],
            ref=raw["ref"],
            update_ref=raw.get("update_ref", raw["ref"]),
            revision=raw["revision"],
            release=raw.get("release"),
            version_file=raw.get("version_file"),
            source_version=raw.get("source_version"),
        )
        if repository.version_file is not None and repository.release is None:
            raise ValueError(f"{repository.id}: version_file requires release")
        if repository.id in seen_ids:
            raise ValueError(f"Duplicate repository id: {repository.id}")
        if repository.path in seen_paths:
            raise ValueError(f"Duplicate repository path: {repository.path}")
        if len(repository.revision) != 40:
            raise ValueError(f"{repository.id}: revision must be a full 40-character commit")
        seen_ids.add(repository.id)
        seen_paths.add(repository.path)
        repositories.append(repository)
    if not repositories:
        raise ValueError(f"No repositories in {path}")
    return repositories


def select(repositories: list[Repository], ids: list[str] | None) -> list[Repository]:
    if not ids:
        return repositories
    wanted = set(ids)
    selected = [repository for repository in repositories if repository.id in wanted]
    missing = wanted - {repository.id for repository in selected}
    if missing:
        raise ValueError(f"Unknown repository ids: {', '.join(sorted(missing))}")
    return selected


VERSION_PATTERN = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.MULTILINE)


def snapshot_state(repository: Repository) -> tuple[str, str]:
    """Third-party sources are plain snapshots (no .git): check presence and recorded version metadata."""
    if not repository.path.is_dir() or not any(repository.path.iterdir()):
        return "missing", "-"
    if repository.version_file is None:
        return "present", "-"
    version_path = repository.path / repository.version_file
    match = VERSION_PATTERN.search(version_path.read_text(encoding="utf-8")) if version_path.is_file() else None
    if match is None:
        return "version-unreadable", "-"
    # source_version records a known release/version-file difference; otherwise they must match.
    expected = repository.source_version or repository.release
    return ("version-ok" if match.group(1) == expected else "version-mismatch"), match.group(1)


def repository_at(path: Path) -> Repository:
    """Catalog entry whose snapshot directory is `path`."""
    matches = [item for item in load_repositories(METHOD_CATALOG_PATH) if item.path.resolve() == path.resolve()]
    if len(matches) != 1:
        raise ValueError(f"No unique catalog repository for {path}")
    return matches[0]


def print_status(repositories: list[Repository]) -> int:
    print("id\tstate\tsnapshot version\trelease\tlocked revision\tpath")
    failed = False
    for repository in repositories:
        state, version = snapshot_state(repository)
        failed |= state not in {"present", "version-ok"}
        print(
            f"{repository.id}\t{state}\t{version}\t{repository.release or '-'}\t"
            f"{repository.revision}\t{repository.path.relative_to(PROJECT_ROOT)}"
        )
    return 1 if failed else 0


def check_upstream(repositories: list[Repository]) -> int:
    print("id\tstatus\tlocked revision\tupstream revision\tref")
    updates = False
    for repository in repositories:
        output = run_git(["ls-remote", repository.url, repository.update_ref])
        lines = [line for line in output.splitlines() if line]
        if len(lines) != 1:
            raise RuntimeError(
                f"{repository.id}: unable to resolve update ref {repository.update_ref!r}"
            )
        upstream = lines[0].split()[0]
        status = "current" if upstream == repository.revision else "update-available"
        updates |= status != "current"
        print(
            f"{repository.id}\t{status}\t{repository.revision}\t"
            f"{upstream}\t{repository.update_ref}"
        )
    return 2 if updates else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check third-party source snapshots against the method catalog metadata."
    )
    parser.add_argument("--config-file", type=Path, default=METHOD_CATALOG_PATH)
    parser.add_argument(
        "--method", action="append", dest="methods", help="Limit operation to one repository id."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate the unified method catalog without touching repositories.")
    subparsers.add_parser("status", help="Check snapshot presence and recorded version metadata.")
    subparsers.add_parser(
        "check-upstream", help="Read remote refs and report updates without changing the lock."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repositories = select(load_repositories(args.config_file.resolve()), args.methods)
    if args.command == "validate":
        print(f"Validated {len(repositories)} repositories in {args.config_file}")
        return 0
    if args.command == "status":
        return print_status(repositories)
    if args.command == "check-upstream":
        return check_upstream(repositories)
    raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
