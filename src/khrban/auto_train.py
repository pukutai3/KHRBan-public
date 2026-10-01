"""Train, evaluate, resume, and prune KHR checkpoints until success."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Collection, Iterator

from khrban.evaluation_precision import (
    EVALUATION_DECIMAL_PLACES,
    evaluation_value,
    format_evaluation_values,
)


class TrainingAlreadyRunningError(RuntimeError):
    """Raised when another training process owns the shared workspace."""


ACTIVE_LINEAGE_FILENAME = "active_auto_training.json"
EVALUATION_CONTRACT_VERSION = 4
DEFAULT_AUTO_TRAIN_NUM_ENVS = 1536
DEFAULT_MAX_LINEAR_VELOCITY_RMSE = 0.10
DEFAULT_MAX_YAW_VELOCITY_RMSE = 0.28
DEFAULT_MAX_DIRECTIONAL_LINEAR_VELOCITY_RMSE = 0.15
DEFAULT_MAX_ROTATION_YAW_VELOCITY_RMSE = 0.31
DEFAULT_MAX_ROTATION_LINEAR_DRIFT_RMSE = 0.08


class EvaluationStage(str, Enum):
    """Ordered objectives used to choose the next training checkpoint."""

    SAFETY = "safety"
    TRACKING = "tracking"
    GAIT = "gait"
    COMPLETE = "complete"


def evaluation_max_satisfied(value: float, threshold: float) -> bool:
    return evaluation_value(value) <= evaluation_value(threshold)


def evaluation_min_satisfied(value: float, threshold: float) -> bool:
    return evaluation_value(value) >= evaluation_value(threshold)


@dataclass(frozen=True)
class EvaluationResult:
    fall_rate: float
    linear_velocity_rmse: float
    yaw_velocity_rmse: float
    mean_reward: float
    num_envs: int
    num_steps: int
    directional_linear_velocity_rmse: float = 1.0e9
    direction_sample_count: int = 0
    rotation_fall_rate: float = 1.0
    rotation_yaw_velocity_rmse: float = 1.0e9
    rotation_linear_drift_rmse: float = 1.0e9
    rotation_sample_count: int = 0
    foot_lift_success_rate: float = 0.0
    foot_peak_height_mean: float = 0.0
    foot_air_time_mean: float = 0.0
    foot_landing_count: int = 0
    standing_foot_airborne_rate: float = 1.0
    standing_sample_count: int = 0
    action_rms: float = 1.0e9
    action_outside_unit_rate: float = 1.0
    action_outside_unit_rate_by_joint: dict[str, float] = field(default_factory=dict)
    action_sample_count_by_mode: dict[str, int] = field(default_factory=dict)
    action_outside_unit_rate_by_mode: dict[str, float] = field(default_factory=dict)
    action_outside_unit_rate_by_mode_and_joint: dict[str, dict[str, float]] = field(
        default_factory=dict
    )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationCriteria:
    max_fall_rate: float = 0.05
    max_linear_velocity_rmse: float = DEFAULT_MAX_LINEAR_VELOCITY_RMSE
    max_yaw_velocity_rmse: float = DEFAULT_MAX_YAW_VELOCITY_RMSE
    max_directional_linear_velocity_rmse: float = (
        DEFAULT_MAX_DIRECTIONAL_LINEAR_VELOCITY_RMSE
    )
    min_direction_sample_count: int = 1
    max_rotation_fall_rate: float = 0.05
    max_rotation_yaw_velocity_rmse: float = DEFAULT_MAX_ROTATION_YAW_VELOCITY_RMSE
    max_rotation_linear_drift_rmse: float = (
        DEFAULT_MAX_ROTATION_LINEAR_DRIFT_RMSE
    )
    min_rotation_sample_count: int = 1
    min_foot_lift_success_rate: float = 0.50
    min_foot_landing_count: int = 1
    max_standing_foot_airborne_rate: float = 0.05
    min_standing_sample_count: int = 1
    max_action_outside_unit_rate: float = 0.05
    max_standing_action_outside_unit_rate: float = 0.05

    def has_required_samples(self, result: EvaluationResult) -> bool:
        return (
            result.direction_sample_count >= self.min_direction_sample_count
            and result.rotation_sample_count >= self.min_rotation_sample_count
            and result.foot_landing_count >= self.min_foot_landing_count
            and result.standing_sample_count >= self.min_standing_sample_count
        )

    def safety_is_satisfied_by(self, result: EvaluationResult) -> bool:
        standing_action_rate = result.action_outside_unit_rate_by_mode.get(
            "standing", result.action_outside_unit_rate
        )
        return (
            self.has_required_samples(result)
            and evaluation_max_satisfied(result.fall_rate, self.max_fall_rate)
            and evaluation_max_satisfied(
                result.rotation_fall_rate, self.max_rotation_fall_rate
            )
            and evaluation_max_satisfied(
                result.standing_foot_airborne_rate,
                self.max_standing_foot_airborne_rate,
            )
            and evaluation_max_satisfied(
                result.action_outside_unit_rate,
                self.max_action_outside_unit_rate,
            )
            and evaluation_max_satisfied(
                standing_action_rate,
                self.max_standing_action_outside_unit_rate,
            )
        )

    def tracking_is_satisfied_by(self, result: EvaluationResult) -> bool:
        return (
            self.safety_is_satisfied_by(result)
            and evaluation_max_satisfied(
                result.linear_velocity_rmse, self.max_linear_velocity_rmse
            )
            and evaluation_max_satisfied(
                result.yaw_velocity_rmse, self.max_yaw_velocity_rmse
            )
            and evaluation_max_satisfied(
                result.directional_linear_velocity_rmse,
                self.max_directional_linear_velocity_rmse,
            )
            and evaluation_max_satisfied(
                result.rotation_yaw_velocity_rmse,
                self.max_rotation_yaw_velocity_rmse,
            )
            and evaluation_max_satisfied(
                result.rotation_linear_drift_rmse,
                self.max_rotation_linear_drift_rmse,
            )
        )

    def stage_for(self, results: Collection[EvaluationResult]) -> EvaluationStage:
        candidates = tuple(results)
        if any(self.is_satisfied_by(result) for result in candidates):
            return EvaluationStage.COMPLETE
        if any(self.tracking_is_satisfied_by(result) for result in candidates):
            return EvaluationStage.GAIT
        if any(self.safety_is_satisfied_by(result) for result in candidates):
            return EvaluationStage.TRACKING
        return EvaluationStage.SAFETY

    def is_satisfied_by(self, result: EvaluationResult) -> bool:
        return (
            evaluation_max_satisfied(result.fall_rate, self.max_fall_rate)
            and evaluation_max_satisfied(
                result.linear_velocity_rmse, self.max_linear_velocity_rmse
            )
            and evaluation_max_satisfied(
                result.yaw_velocity_rmse, self.max_yaw_velocity_rmse
            )
            and result.direction_sample_count >= self.min_direction_sample_count
            and evaluation_max_satisfied(
                result.directional_linear_velocity_rmse,
                self.max_directional_linear_velocity_rmse,
            )
            and result.rotation_sample_count >= self.min_rotation_sample_count
            and evaluation_max_satisfied(
                result.rotation_fall_rate, self.max_rotation_fall_rate
            )
            and evaluation_max_satisfied(
                result.rotation_yaw_velocity_rmse,
                self.max_rotation_yaw_velocity_rmse,
            )
            and evaluation_max_satisfied(
                result.rotation_linear_drift_rmse,
                self.max_rotation_linear_drift_rmse,
            )
            and evaluation_min_satisfied(
                result.foot_lift_success_rate,
                self.min_foot_lift_success_rate,
            )
            and result.foot_landing_count >= self.min_foot_landing_count
            and result.standing_sample_count >= self.min_standing_sample_count
            and evaluation_max_satisfied(
                result.standing_foot_airborne_rate,
                self.max_standing_foot_airborne_rate,
            )
            and evaluation_max_satisfied(
                result.action_outside_unit_rate,
                self.max_action_outside_unit_rate,
            )
            and evaluation_max_satisfied(
                result.action_outside_unit_rate_by_mode.get(
                    "standing", result.action_outside_unit_rate
                ),
                self.max_standing_action_outside_unit_rate,
            )
        )

    def comparison_values(self, result: EvaluationResult) -> dict[str, float]:
        """Return the rounded scalar values used by the formal decision."""

        return {
            "fall_rate": evaluation_value(result.fall_rate),
            "linear_velocity_rmse": evaluation_value(result.linear_velocity_rmse),
            "yaw_velocity_rmse": evaluation_value(result.yaw_velocity_rmse),
            "directional_linear_velocity_rmse": evaluation_value(
                result.directional_linear_velocity_rmse
            ),
            "rotation_fall_rate": evaluation_value(result.rotation_fall_rate),
            "rotation_yaw_velocity_rmse": evaluation_value(
                result.rotation_yaw_velocity_rmse
            ),
            "rotation_linear_drift_rmse": evaluation_value(
                result.rotation_linear_drift_rmse
            ),
            "foot_lift_success_rate": evaluation_value(
                result.foot_lift_success_rate
            ),
            "standing_foot_airborne_rate": evaluation_value(
                result.standing_foot_airborne_rate
            ),
            "action_outside_unit_rate": evaluation_value(
                result.action_outside_unit_rate
            ),
            "standing_action_outside_unit_rate": evaluation_value(
                result.action_outside_unit_rate_by_mode.get(
                    "standing", result.action_outside_unit_rate
                )
            ),
        }

    def selection_rank(
        self,
        result: EvaluationResult,
        *,
        stage: EvaluationStage | None = None,
    ) -> tuple[float, ...]:
        """Rank failed candidates without changing the formal pass criteria.

        Valid sample coverage is a prerequisite, then more passed performance
        gates wins. Ties are broken by the total relative shortfall of failed
        gates, so metrics with different units remain comparable.
        """

        standing_action_rate = result.action_outside_unit_rate_by_mode.get(
            "standing", result.action_outside_unit_rate
        )
        safety_gates = (
            (result.fall_rate, self.max_fall_rate, False),
            (result.rotation_fall_rate, self.max_rotation_fall_rate, False),
            (
                result.standing_foot_airborne_rate,
                self.max_standing_foot_airborne_rate,
                False,
            ),
            (
                result.action_outside_unit_rate,
                self.max_action_outside_unit_rate,
                False,
            ),
            (
                standing_action_rate,
                self.max_standing_action_outside_unit_rate,
                False,
            ),
        )
        tracking_gates = (
            (result.linear_velocity_rmse, self.max_linear_velocity_rmse, False),
            (result.yaw_velocity_rmse, self.max_yaw_velocity_rmse, False),
            (
                result.directional_linear_velocity_rmse,
                self.max_directional_linear_velocity_rmse,
                False,
            ),
            (
                result.rotation_yaw_velocity_rmse,
                self.max_rotation_yaw_velocity_rmse,
                False,
            ),
            (
                result.rotation_linear_drift_rmse,
                self.max_rotation_linear_drift_rmse,
                False,
            ),
        )
        performance_gates = (
            *safety_gates,
            *tracking_gates,
            (
                result.foot_lift_success_rate,
                self.min_foot_lift_success_rate,
                True,
            ),
        )

        def gate_rank(
            gates: Collection[tuple[float, float, bool]],
        ) -> tuple[int, float]:
            passed = 0
            relative_shortfall = 0.0
            for value, threshold, higher_is_better in gates:
                if not math.isfinite(value):
                    relative_shortfall = float("inf")
                    continue
                value = evaluation_value(value)
                threshold = evaluation_value(threshold)
                satisfied = value >= threshold if higher_is_better else value <= threshold
                if satisfied:
                    passed += 1
                    continue
                shortfall = threshold - value if higher_is_better else value - threshold
                relative_shortfall += max(0.0, shortfall) / max(
                    abs(threshold), 1.0e-12
                )
            return passed, -relative_shortfall

        samples_valid = self.has_required_samples(result)
        if stage is EvaluationStage.SAFETY:
            passed, shortfall = gate_rank(safety_gates)
            return int(samples_valid), passed, shortfall
        if stage is EvaluationStage.TRACKING:
            passed, shortfall = gate_rank(tracking_gates)
            return (
                int(self.safety_is_satisfied_by(result)),
                passed,
                shortfall,
            )
        if stage in (EvaluationStage.GAIT, EvaluationStage.COMPLETE):
            foot_passed = (
                evaluation_min_satisfied(
                    result.foot_lift_success_rate,
                    self.min_foot_lift_success_rate,
                )
                and result.foot_landing_count >= self.min_foot_landing_count
            )
            return (
                int(self.tracking_is_satisfied_by(result)),
                int(foot_passed),
                evaluation_value(result.foot_lift_success_rate),
            )

        passed = 0
        relative_shortfall = 0.0
        for value, threshold, higher_is_better in performance_gates:
            if not math.isfinite(value):
                relative_shortfall = float("inf")
                continue
            value = evaluation_value(value)
            threshold = evaluation_value(threshold)
            satisfied = value >= threshold if higher_is_better else value <= threshold
            if satisfied:
                passed += 1
                continue
            shortfall = threshold - value if higher_is_better else value - threshold
            relative_shortfall += max(0.0, shortfall) / max(abs(threshold), 1.0e-12)

        return int(samples_valid), passed, -relative_shortfall


def checkpoint_iteration(path: Path) -> int | None:
    if not path.name.startswith("model_") or path.suffix != ".pt":
        return None
    try:
        return int(path.stem.removeprefix("model_"))
    except ValueError:
        return None


def _checkpoint_round(path: Path, run_prefix: str) -> int | None:
    marker = f"{run_prefix}-round-"
    marker_index = path.parent.name.rfind(marker)
    if marker_index < 0:
        return None
    suffix = path.parent.name[marker_index + len(marker) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def find_latest_checkpoint(
    log_root: Path, *, run_prefix: str | None = None
) -> Path | None:
    checkpoints = [
        path
        for path in log_root.glob("*/model_*.pt")
        if checkpoint_iteration(path) is not None
        and (run_prefix is None or _checkpoint_round(path, run_prefix) is not None)
    ]
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda path: (
            checkpoint_iteration(path) or -1,
            path.stat().st_mtime_ns,
            str(path),
        ),
    )


def begin_training_lineage(log_root: Path, requested_run_prefix: str) -> str:
    """Persist a unique series prefix before a fresh training child is launched."""

    started_at = datetime.now(timezone.utc)
    series_prefix = (
        f"{requested_run_prefix}-{started_at.strftime('%Y%m%dT%H%M%S%fZ')}"
    )
    record = {
        "version": 1,
        "requested_run_prefix": requested_run_prefix,
        "series_prefix": series_prefix,
        "started_at": started_at.isoformat(),
    }
    log_root.mkdir(parents=True, exist_ok=True)
    lineage_path = log_root / ACTIVE_LINEAGE_FILENAME
    temporary_path = lineage_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, lineage_path)
    return series_prefix


def read_active_training_lineage(log_root: Path) -> str | None:
    """Return the persisted active series, rejecting corrupt lineage records."""

    lineage_path = log_root / ACTIVE_LINEAGE_FILENAME
    if not lineage_path.exists():
        return None
    record = json.loads(lineage_path.read_text(encoding="utf-8"))
    series_prefix = record.get("series_prefix")
    if not isinstance(series_prefix, str) or not series_prefix:
        raise RuntimeError(f"Invalid active training lineage: {lineage_path}")
    return series_prefix


def next_round_number(log_root: Path, run_prefix: str) -> int:
    """Continue the recorded round sequence after a supervisor restart."""

    rounds = [
        round_number
        for path in log_root.glob("*/model_*.pt")
        if (round_number := _checkpoint_round(path, run_prefix)) is not None
    ]
    return max(rounds, default=0) + 1


def find_new_checkpoint(log_root: Path, previous: set[Path]) -> Path | None:
    """Return the most advanced checkpoint created by the latest train call."""

    created = [
        path.resolve()
        for path in log_root.glob("*/model_*.pt")
        if path.resolve() not in previous and checkpoint_iteration(path) is not None
    ]
    if not created:
        return None
    return max(
        created,
        key=lambda path: (
            checkpoint_iteration(path) or -1,
            path.stat().st_mtime_ns,
            str(path),
        ),
    )


def prune_checkpoints(
    log_root: Path,
    *,
    keep: int,
    protected: Collection[Path] = (),
) -> list[Path]:
    """Remove old model files only; retain logs and evaluation evidence."""

    if keep < 1:
        raise ValueError("keep must be at least 1")
    checkpoints = sorted(
        (
            path
            for path in log_root.glob("*/model_*.pt")
            if checkpoint_iteration(path) is not None
        ),
        key=lambda path: (
            path.stat().st_mtime_ns,
            checkpoint_iteration(path) or -1,
            str(path),
        ),
    )
    protected_paths = {path.resolve() for path in protected}
    retained_paths = {
        path.resolve() for path in checkpoints if path.resolve() in protected_paths
    }
    for path in reversed(checkpoints):
        if len(retained_paths) >= keep:
            break
        retained_paths.add(path.resolve())
    removed = [path for path in checkpoints if path.resolve() not in retained_paths]
    for path in removed:
        path.unlink()
    return removed


def append_evaluation_record(
    history_path: Path,
    checkpoint: Path,
    result: EvaluationResult,
    achieved: bool,
    *,
    evaluation_stage: EvaluationStage | None = None,
    comparison_values: dict[str, float] | None = None,
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint.resolve()),
        "achieved": achieved,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "evaluation_decimal_places": EVALUATION_DECIMAL_PLACES,
        "metrics": result.as_dict(),
    }
    if evaluation_stage is not None:
        record["evaluation_stage"] = evaluation_stage.value
    if comparison_values is not None:
        record["comparison_values"] = comparison_values
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_best_evaluated_checkpoint(
    history_path: Path,
    *,
    log_root: Path,
    run_prefix: str,
    criteria: EvaluationCriteria,
) -> Path | None:
    """Choose the best existing evaluated checkpoint in the active lineage."""

    if not history_path.exists():
        return None
    root = log_root.resolve()
    evaluated: list[tuple[EvaluationResult, int, int, Path]] = []
    for record_number, line in enumerate(
        history_path.read_text(encoding="utf-8").splitlines()
    ):
        try:
            record = json.loads(line)
            if (
                record.get("evaluation_contract_version")
                != EVALUATION_CONTRACT_VERSION
            ):
                continue
            checkpoint = Path(record["checkpoint"]).resolve()
            metrics = EvaluationResult(**record["metrics"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        if (
            not checkpoint.is_file()
            or _checkpoint_round(checkpoint, run_prefix) is None
        ):
            continue
        try:
            checkpoint.relative_to(root)
        except ValueError:
            continue
        evaluated.append(
            (metrics, record_number, checkpoint_iteration(checkpoint) or -1, checkpoint)
        )
    if not evaluated:
        return None
    stage = criteria.stage_for(metrics for metrics, _, _, _ in evaluated)
    candidates = [
        (
            criteria.selection_rank(metrics, stage=stage),
            record_number,
            iteration,
            checkpoint,
        )
        for metrics, record_number, iteration, checkpoint in evaluated
    ]
    return max(candidates, key=lambda item: item[:3])[3]


def find_other_training_pids(
    proc_root: Path = Path("/proc"), *, current_pid: int | None = None
) -> tuple[int, ...]:
    """Return other KHR training PIDs without matching shell command text."""

    own_pid = os.getpid() if current_pid is None else current_pid
    found: list[int] = []
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == own_pid:
            continue
        try:
            argv = [
                value.decode(errors="replace")
                for value in (process_dir / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not argv:
            continue
        executable_name = Path(argv[0]).name
        training_scripts = {
            "khrban-auto-train",
            "khrban-auto-train-getup",
            "khrban-train",
        }
        python_launch = executable_name.startswith("python")
        direct_launch = executable_name in training_scripts
        python_script_launch = (
            python_launch
            and len(argv) > 1
            and Path(argv[1]).name in training_scripts
        )
        module_launch = (
            python_launch
            and any(
                argv[index] == "-m"
                and argv[index + 1]
                in {
                    "khrban.auto_train",
                    "khrban.auto_train_getup",
                    "khrban.train",
                }
                for index in range(len(argv) - 1)
            )
        )
        if direct_launch or python_script_launch or module_launch:
            found.append(int(process_dir.name))
    return tuple(sorted(found))


@contextmanager
def exclusive_training_lock(lock_path: Path) -> Iterator[None]:
    """Hold a non-blocking process lock for one automatic-training supervisor."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TrainingAlreadyRunningError(
                f"Automatic training is already running (lock: {lock_path})"
            ) from error
        stream.seek(0)
        stream.truncate()
        stream.write(f"{os.getpid()}\n")
        stream.flush()
        yield
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def run_process_with_retry(
    command: list[str],
    *,
    label: str,
    initial_delay_seconds: float,
    max_delay_seconds: float,
) -> None:
    """Run a child process, retrying failures with bounded exponential backoff."""

    delay = initial_delay_seconds
    while True:
        try:
            subprocess.run(command, check=True)
            return
        except subprocess.CalledProcessError as error:
            print(
                f"[AUTO] {label} failed with exit {error.returncode}; "
                f"retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            delay = min(max_delay_seconds, max(initial_delay_seconds, delay * 2))


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def run_evaluation(
    checkpoint: Path,
    *,
    num_envs: int,
    num_steps: int,
    seed: int,
    gpu_id: int,
    retry_delay_seconds: float,
    max_retry_delay_seconds: float,
) -> EvaluationResult:
    result_path = checkpoint.parent / "evaluation.json"
    command = [
        sys.executable,
        "-m",
        "khrban.evaluation",
        str(checkpoint),
        "--num-envs",
        str(num_envs),
        "--num-steps",
        str(num_steps),
        "--seed",
        str(seed),
        "--device",
        f"cuda:{gpu_id}",
        "--output",
        str(result_path),
    ]
    run_process_with_retry(
        command,
        label="evaluation",
        initial_delay_seconds=retry_delay_seconds,
        max_delay_seconds=max_retry_delay_seconds,
    )
    return EvaluationResult(**json.loads(result_path.read_text(encoding="utf-8")))


def run_auto_training(
    args: argparse.Namespace,
    persistent_viewer: Any | None = None,
) -> None:
    """Run the evaluate/train loop while the caller owns the supervisor lock."""

    log_root = Path("logs/rsl_rl/khr_velocity").resolve()
    history_path = log_root / "auto_training_evaluations.jsonl"
    criteria = EvaluationCriteria(
        max_fall_rate=args.max_fall_rate,
        max_linear_velocity_rmse=args.max_linear_velocity_rmse,
        max_yaw_velocity_rmse=args.max_yaw_velocity_rmse,
        max_directional_linear_velocity_rmse=(
            args.max_directional_linear_velocity_rmse
        ),
        min_direction_sample_count=args.min_direction_sample_count,
        max_rotation_fall_rate=args.max_rotation_fall_rate,
        max_rotation_yaw_velocity_rmse=args.max_rotation_yaw_velocity_rmse,
        max_rotation_linear_drift_rmse=args.max_rotation_linear_drift_rmse,
        min_rotation_sample_count=args.min_rotation_sample_count,
        min_foot_lift_success_rate=args.min_foot_lift_success_rate,
        min_foot_landing_count=args.min_foot_landing_count,
        max_standing_foot_airborne_rate=args.max_standing_foot_airborne_rate,
        min_standing_sample_count=args.min_standing_sample_count,
        max_action_outside_unit_rate=args.max_action_outside_unit_rate,
        max_standing_action_outside_unit_rate=(
            args.max_standing_action_outside_unit_rate
        ),
    )
    if args.fresh:
        run_prefix = begin_training_lineage(log_root, args.run_prefix)
        resume_checkpoint = None
        round_number = 1
        print(f"[AUTO] Started fresh training lineage {run_prefix}", flush=True)
    else:
        active_run_prefix = read_active_training_lineage(log_root)
        if active_run_prefix is None:
            run_prefix = args.run_prefix
            resume_checkpoint = find_latest_checkpoint(log_root)
            round_number = 1
            print(
                "[AUTO] No active lineage record; using legacy checkpoint lookup",
                flush=True,
            )
        else:
            run_prefix = active_run_prefix
            resume_checkpoint = find_best_evaluated_checkpoint(
                history_path,
                log_root=log_root,
                run_prefix=run_prefix,
                criteria=criteria,
            ) or find_latest_checkpoint(
                log_root,
                run_prefix=run_prefix,
            )
            round_number = next_round_number(log_root, run_prefix)
            print(f"[AUTO] Resuming training lineage {run_prefix}", flush=True)

    def evaluate_and_record(checkpoint: Path) -> bool:
        if persistent_viewer is not None:
            persistent_viewer.set_phase("評価中", checkpoint.parent.name)
        print(f"[AUTO] Evaluating {checkpoint.name}", flush=True)
        result = run_evaluation(
            checkpoint,
            num_envs=args.eval_num_envs,
            num_steps=args.eval_num_steps,
            seed=args.eval_seed,
            gpu_id=args.gpu_id,
            retry_delay_seconds=args.retry_delay_seconds,
            max_retry_delay_seconds=args.max_retry_delay_seconds,
        )
        achieved = criteria.is_satisfied_by(result)
        evaluation_stage = criteria.stage_for((result,))
        comparison_values = criteria.comparison_values(result)
        append_evaluation_record(
            history_path,
            checkpoint,
            result,
            achieved,
            evaluation_stage=evaluation_stage,
            comparison_values=comparison_values,
        )
        best_checkpoint = find_best_evaluated_checkpoint(
            history_path,
            log_root=log_root,
            run_prefix=run_prefix,
            criteria=criteria,
        )
        protected_checkpoints = [checkpoint]
        if best_checkpoint is not None:
            protected_checkpoints.append(best_checkpoint)
        removed = prune_checkpoints(
            log_root,
            keep=args.keep_checkpoints,
            protected=protected_checkpoints,
        )
        print(
            f"[AUTO] stage={evaluation_stage.value} achieved={achieved} "
            "evaluation_values="
            f"{json.dumps(format_evaluation_values(comparison_values))} "
            f"best_checkpoint={best_checkpoint} removed={len(removed)}",
            flush=True,
        )
        if achieved:
            if persistent_viewer is not None:
                persistent_viewer.set_phase("目標達成", checkpoint.parent.name)
            print(f"[AUTO] Target achieved by {checkpoint}", flush=True)
        elif persistent_viewer is not None:
            persistent_viewer.set_phase(
                "評価不合格・次ラウンド準備中",
                checkpoint.parent.name,
            )
        return achieved

    if resume_checkpoint is not None and evaluate_and_record(resume_checkpoint):
        return

    while True:
        run_name = f"{run_prefix}-round-{round_number:04d}"
        train_command = [
            sys.executable,
            "-m",
            "khrban.train",
            "--task",
            "velocity",
            "--num-envs",
            str(args.num_envs),
            "--iterations",
            str(args.iterations_per_round),
            "--gpu-id",
            str(args.gpu_id),
            "--run-name",
            run_name,
            "--save-interval",
            str(args.iterations_per_round),
        ]
        if resume_checkpoint is not None:
            train_command += ["--resume-from", str(resume_checkpoint)]
        if args.live_viewer:
            train_command += [
                "--live-viewer",
                "--viewer-num-envs",
                str(args.viewer_num_envs),
                "--viewer-port",
                str(args.viewer_port),
            ]

        print(f"[AUTO] Round {round_number}: training", flush=True)
        if persistent_viewer is not None:
            persistent_viewer.set_phase("学習プロセス起動中", run_name)
        previous_checkpoints = {
            path.resolve() for path in log_root.glob("*/model_*.pt")
        }
        run_process_with_retry(
            train_command,
            label="training",
            initial_delay_seconds=args.retry_delay_seconds,
            max_delay_seconds=args.max_retry_delay_seconds,
        )
        if persistent_viewer is not None:
            persistent_viewer.set_phase("評価準備中", run_name)
        checkpoint = find_new_checkpoint(log_root, previous_checkpoints)
        if checkpoint is None:
            raise RuntimeError("Training completed without a new checkpoint")

        if evaluate_and_record(checkpoint):
            return

        resume_checkpoint = find_best_evaluated_checkpoint(
            history_path,
            log_root=log_root,
            run_prefix=run_prefix,
            criteria=criteria,
        ) or checkpoint
        print(f"[AUTO] Next round resumes from {resume_checkpoint}", flush=True)
        round_number += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat KHR velocity training until evaluation criteria pass"
    )
    parser.add_argument(
        "--num-envs", type=_positive, default=DEFAULT_AUTO_TRAIN_NUM_ENVS
    )
    parser.add_argument("--iterations-per-round", type=_positive, default=2000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-prefix", default="auto-walk")
    parser.add_argument("--keep-checkpoints", type=_positive, default=3)
    parser.add_argument("--eval-num-envs", type=_positive, default=64)
    parser.add_argument("--eval-num-steps", type=_positive, default=500)
    parser.add_argument("--eval-seed", type=int, default=314159)
    parser.add_argument("--max-fall-rate", type=float, default=0.05)
    parser.add_argument(
        "--max-linear-velocity-rmse",
        type=float,
        default=DEFAULT_MAX_LINEAR_VELOCITY_RMSE,
    )
    parser.add_argument(
        "--max-yaw-velocity-rmse",
        type=float,
        default=DEFAULT_MAX_YAW_VELOCITY_RMSE,
    )
    parser.add_argument(
        "--max-directional-linear-velocity-rmse",
        type=float,
        default=DEFAULT_MAX_DIRECTIONAL_LINEAR_VELOCITY_RMSE,
    )
    parser.add_argument("--min-direction-sample-count", type=_positive, default=1)
    parser.add_argument("--max-rotation-fall-rate", type=float, default=0.05)
    parser.add_argument(
        "--max-rotation-yaw-velocity-rmse",
        type=float,
        default=DEFAULT_MAX_ROTATION_YAW_VELOCITY_RMSE,
    )
    parser.add_argument(
        "--max-rotation-linear-drift-rmse",
        type=float,
        default=DEFAULT_MAX_ROTATION_LINEAR_DRIFT_RMSE,
    )
    parser.add_argument("--min-rotation-sample-count", type=_positive, default=1)
    parser.add_argument("--min-foot-lift-success-rate", type=float, default=0.50)
    parser.add_argument("--min-foot-landing-count", type=_positive, default=1)
    parser.add_argument(
        "--max-standing-foot-airborne-rate", type=float, default=0.05
    )
    parser.add_argument("--min-standing-sample-count", type=_positive, default=1)
    parser.add_argument(
        "--max-action-outside-unit-rate", type=float, default=0.05
    )
    parser.add_argument(
        "--max-standing-action-outside-unit-rate", type=float, default=0.05
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--live-viewer", action="store_true")
    parser.add_argument("--viewer-num-envs", type=_positive, default=16)
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    parser.add_argument("--max-retry-delay-seconds", type=float, default=300.0)
    args = parser.parse_args()

    if args.live_viewer and args.viewer_num_envs < 16:
        parser.error("--viewer-num-envs must be at least 16")
    if not 0.0 <= args.min_foot_lift_success_rate <= 1.0:
        parser.error("--min-foot-lift-success-rate must be between 0 and 1")
    if not 0.0 <= args.max_action_outside_unit_rate <= 1.0:
        parser.error("--max-action-outside-unit-rate must be between 0 and 1")
    if not 0.0 <= args.max_standing_action_outside_unit_rate <= 1.0:
        parser.error(
            "--max-standing-action-outside-unit-rate must be between 0 and 1"
        )
    for name in (
        "max_fall_rate",
        "max_linear_velocity_rmse",
        "max_yaw_velocity_rmse",
        "max_directional_linear_velocity_rmse",
        "max_rotation_fall_rate",
        "max_rotation_yaw_velocity_rmse",
        "max_rotation_linear_drift_rmse",
        "max_standing_foot_airborne_rate",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.retry_delay_seconds < 0 or args.max_retry_delay_seconds < 0:
        parser.error("retry delays must be non-negative")
    if args.max_retry_delay_seconds < args.retry_delay_seconds:
        parser.error("--max-retry-delay-seconds must be at least the initial delay")

    other_pids = find_other_training_pids()
    if other_pids:
        raise SystemExit(
            "KHR training is already running (PID: "
            + ", ".join(str(pid) for pid in other_pids)
            + ")"
        )

    lock_path = Path("logs/rsl_rl/khr_velocity/.auto_train.lock").resolve()
    try:
        with exclusive_training_lock(lock_path):
            persistent_viewer = None
            try:
                if args.live_viewer:
                    from khrban.live_training import PersistentLiveTrainingViewer

                    persistent_viewer = PersistentLiveTrainingViewer(
                        num_envs=args.viewer_num_envs,
                        port=args.viewer_port,
                    )
                run_auto_training(args, persistent_viewer=persistent_viewer)
            finally:
                if persistent_viewer is not None:
                    persistent_viewer.close()
    except TrainingAlreadyRunningError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
