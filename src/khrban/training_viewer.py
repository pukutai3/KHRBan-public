"""Launch one KHR policy from the active velocity-training run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from typing import Iterable


VELOCITY_TASK_ID = "Mjlab-KHR-Velocity-Flat"
VIEWER_NUM_ENVS = 16
DEFAULT_LOG_ROOT = Path("logs/rsl_rl/khr_velocity")
_CHECKPOINT_PATTERN = re.compile(r"model_(\d+)\.pt")


def _process_arguments(proc_root: Path = Path("/proc")) -> Iterable[list[str]]:
    """Yield command arguments for readable numeric process directories."""

    for process_dir in sorted(proc_root.iterdir(), key=lambda path: path.name):
        if not process_dir.name.isdigit() or int(process_dir.name) == os.getpid():
            continue
        try:
            raw = (process_dir / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        arguments = [
            item.decode(errors="replace") for item in raw.split(b"\0") if item
        ]
        if arguments:
            yield arguments


def _option(arguments: list[str], name: str) -> str | None:
    try:
        return arguments[arguments.index(name) + 1]
    except (ValueError, IndexError):
        return None


def active_velocity_run_name(proc_root: Path = Path("/proc")) -> str | None:
    """Return the longest active velocity run's explicit run name."""

    candidates: list[tuple[int, str]] = []
    for arguments in _process_arguments(proc_root):
        if not any(Path(argument).name == "khrban-train" for argument in arguments):
            continue
        if _option(arguments, "--task") != "velocity":
            continue
        run_name = _option(arguments, "--run-name")
        if not run_name:
            continue
        try:
            iterations = int(_option(arguments, "--iterations") or "0")
        except ValueError:
            iterations = 0
        candidates.append((iterations, run_name))
    return max(candidates, default=(0, None))[1]


def latest_checkpoint(run_dir: Path) -> Path:
    """Select the highest numbered complete checkpoint in a run directory."""

    numbered = []
    for path in run_dir.glob("model_*.pt"):
        match = _CHECKPOINT_PATTERN.fullmatch(path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            numbered.append((int(match.group(1)), path))
    if not numbered:
        raise FileNotFoundError(f"チェックポイントがありません: {run_dir}")
    return max(numbered, key=lambda item: item[0])[1]


def resolve_training_checkpoint(
    log_root: Path = DEFAULT_LOG_ROOT,
    proc_root: Path = Path("/proc"),
) -> Path:
    """Resolve the latest checkpoint, preferring the active velocity run."""

    if not log_root.is_dir():
        raise FileNotFoundError(f"歩行学習ログがありません: {log_root}")

    run_name = active_velocity_run_name(proc_root)
    run_dirs = [path for path in log_root.iterdir() if path.is_dir()]
    if run_name:
        active_dirs = [path for path in run_dirs if path.name.endswith(f"_{run_name}")]
        if active_dirs:
            return latest_checkpoint(max(active_dirs, key=lambda path: path.stat().st_mtime))

    usable = []
    for run_dir in run_dirs:
        try:
            usable.append((run_dir.stat().st_mtime, latest_checkpoint(run_dir)))
        except FileNotFoundError:
            pass
    if not usable:
        raise FileNotFoundError(f"歩行チェックポイントがありません: {log_root}")
    return max(usable, key=lambda item: item[0])[1]


def playback_arguments(checkpoint: Path) -> list[str]:
    """Build playback arguments for a bounded 16-robot visualization."""

    return [
        VELOCITY_TASK_ID,
        "--checkpoint-file",
        str(checkpoint),
        "--num-envs",
        str(VIEWER_NUM_ENVS),
        "--viewer",
        "viser",
    ]


def viewer_is_running(proc_root: Path = Path("/proc")) -> bool:
    """Return whether another KHR Viser policy viewer already exists."""

    for arguments in _process_arguments(proc_root):
        command = " ".join(arguments)
        if "--viewer viser" not in command:
            continue
        if "khrban-play" in command or "khrban-training-viewer" in command:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="実行中のKHR歩行学習を1体のViser画面で確認します"
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--print-checkpoint", action="store_true")
    args = parser.parse_args()

    checkpoint = resolve_training_checkpoint(args.log_root)
    if args.print_checkpoint:
        print(checkpoint)
        return
    if viewer_is_running():
        print("KHR学習ビューアは既に実行中です: http://localhost:8080/")
        return

    print(
        f"KHR学習ビューア: {checkpoint.name} / {VIEWER_NUM_ENVS}体 / "
        "http://localhost:8080/"
    )
    sys.argv = ["khrban-play", *playback_arguments(checkpoint)]
    from khrban.play import main as play_main

    play_main()


if __name__ == "__main__":
    main()
