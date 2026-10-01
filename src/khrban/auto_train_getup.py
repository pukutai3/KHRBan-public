"""Gate, train, evaluate, and prune KHR get-up rounds until every start passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

from khrban.auto_train import (
    TrainingAlreadyRunningError,
    append_evaluation_record,
    begin_training_lineage,
    exclusive_training_lock,
    find_latest_checkpoint,
    find_new_checkpoint,
    find_other_training_pids,
    next_round_number,
    prune_checkpoints,
    read_active_training_lineage,
    run_process_with_retry,
)
from khrban.getup_evaluation import (
    GetUpEvaluationCriteria,
    GetUpEvaluationResult,
)


class VelocityTrainingGateError(RuntimeError):
    """Raised until an evaluated velocity checkpoint passes and is retained."""


def require_velocity_training_pass(
    *,
    velocity_log_root: Path = Path("logs/rsl_rl/khr_velocity"),
) -> Path:
    """Require a successful result from the active velocity lineage and its file."""

    velocity_log_root = velocity_log_root.resolve()
    series_prefix = read_active_training_lineage(velocity_log_root)
    if series_prefix is None:
        raise VelocityTrainingGateError(
            "速度追従の自動評価履歴がありません。速度追従を合格させてから "
            "起き上がり学習へ進んでください。"
        )

    history_path = velocity_log_root / "auto_training_evaluations.jsonl"
    if not history_path.is_file():
        raise VelocityTrainingGateError(
            f"速度追従の評価履歴がありません: {history_path}"
        )
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        history_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise VelocityTrainingGateError(
                f"速度追従評価履歴の{line_number}行目を読めません: {history_path}"
            ) from error
        if isinstance(record, dict):
            records.append(record)

    matching = [
        record
        for record in reversed(records)
        if series_prefix in Path(str(record.get("checkpoint", ""))).parent.name
    ]
    if not matching:
        raise VelocityTrainingGateError(
            "現在の速度追従学習系列に対する正式評価がまだありません。"
        )
    latest = matching[0]
    if latest.get("achieved") is not True:
        raise VelocityTrainingGateError(
            "現在の速度追従学習系列の最新正式評価は未達です。"
        )
    checkpoint = Path(str(latest.get("checkpoint", ""))).resolve()
    if not checkpoint.is_file():
        raise VelocityTrainingGateError(
            "速度追従は合格記録がありますが、そのチェックポイントが見つかりません: "
            f"{checkpoint}"
        )
    return checkpoint


def checkpoint_sha256(checkpoint: Path) -> str:
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_getup_evaluation(
    checkpoint: Path,
    *,
    num_envs: int,
    num_steps: int,
    hold_seconds: float,
    seed: int,
    gpu_id: int,
    retry_delay_seconds: float,
    max_retry_delay_seconds: float,
) -> GetUpEvaluationResult:
    result_path = checkpoint.parent / "evaluation.json"
    command = [
        sys.executable,
        "-m",
        "khrban.getup_evaluation",
        str(checkpoint),
        "--num-envs",
        str(num_envs),
        "--num-steps",
        str(num_steps),
        "--hold-seconds",
        str(hold_seconds),
        "--seed",
        str(seed),
        "--device",
        f"cuda:{gpu_id}",
        "--output",
        str(result_path),
    ]
    run_process_with_retry(
        command,
        label="get-up evaluation",
        initial_delay_seconds=retry_delay_seconds,
        max_delay_seconds=max_retry_delay_seconds,
    )
    return GetUpEvaluationResult(**json.loads(result_path.read_text(encoding="utf-8")))


def run_getup_auto_training(args: Namespace) -> None:
    log_root = Path("logs/rsl_rl/khr_getup").resolve()
    history_path = log_root / "auto_training_evaluations.jsonl"
    criteria = GetUpEvaluationCriteria(
        min_success_rate_per_pose=args.min_success_rate_per_pose,
        min_samples_per_pose=args.min_samples_per_pose,
    )

    if args.fresh:
        run_prefix = begin_training_lineage(log_root, args.run_prefix)
        resume_checkpoint = None
        round_number = 1
        print(f"[GETUP] Started fresh training lineage {run_prefix}", flush=True)
    else:
        active_run_prefix = read_active_training_lineage(log_root)
        if active_run_prefix is None:
            run_prefix = args.run_prefix
            resume_checkpoint = find_latest_checkpoint(log_root)
            round_number = 1
            print(
                "[GETUP] No active lineage record; using legacy checkpoint lookup",
                flush=True,
            )
        else:
            run_prefix = active_run_prefix
            resume_checkpoint = find_latest_checkpoint(
                log_root, run_prefix=run_prefix
            )
            round_number = next_round_number(log_root, run_prefix)
            print(f"[GETUP] Resuming training lineage {run_prefix}", flush=True)

    def evaluate_and_record(checkpoint: Path) -> bool:
        print(f"[GETUP] Evaluating {checkpoint.name}", flush=True)
        result = run_getup_evaluation(
            checkpoint,
            num_envs=args.eval_num_envs,
            num_steps=args.eval_num_steps,
            hold_seconds=args.eval_hold_seconds,
            seed=args.eval_seed,
            gpu_id=args.gpu_id,
            retry_delay_seconds=args.retry_delay_seconds,
            max_retry_delay_seconds=args.max_retry_delay_seconds,
        )
        achieved = criteria.is_satisfied_by(result)
        append_evaluation_record(history_path, checkpoint, result, achieved)
        removed = prune_checkpoints(
            log_root,
            keep=args.keep_checkpoints,
            protected=(checkpoint,),
        )
        print(
            f"[GETUP] achieved={achieved} "
            f"success_by_pose={json.dumps(result.success_rate_by_pose)} "
            f"samples={json.dumps(result.sample_count_by_pose)} "
            f"removed={len(removed)}",
            flush=True,
        )
        if achieved:
            print(f"[GETUP] Target achieved by {checkpoint}", flush=True)
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
            "getup",
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

        print(f"[GETUP] Round {round_number}: training", flush=True)
        previous_checkpoints = {
            path.resolve() for path in log_root.glob("*/model_*.pt")
        }
        run_process_with_retry(
            train_command,
            label="get-up training",
            initial_delay_seconds=args.retry_delay_seconds,
            max_delay_seconds=args.max_retry_delay_seconds,
        )
        checkpoint = find_new_checkpoint(log_root, previous_checkpoints)
        if checkpoint is None:
            raise RuntimeError("Get-up training completed without a new checkpoint")
        if evaluate_and_record(checkpoint):
            return
        resume_checkpoint = checkpoint
        round_number += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate KHR get-up policies until every pose passes"
    )
    parser.add_argument("--check-velocity-gate", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations-per-round", type=int, default=2000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-prefix", default="auto-getup")
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--eval-num-envs", type=int, default=256)
    parser.add_argument("--eval-num-steps", type=int, default=500)
    parser.add_argument("--eval-hold-seconds", type=float, default=1.0)
    parser.add_argument("--eval-seed", type=int, default=314159)
    parser.add_argument("--min-success-rate-per-pose", type=float, default=0.80)
    parser.add_argument("--min-samples-per-pose", type=int, default=16)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--live-viewer", action="store_true")
    parser.add_argument("--viewer-num-envs", type=int, default=16)
    parser.add_argument("--viewer-port", type=int, default=8080)
    parser.add_argument("--retry-delay-seconds", type=float, default=30.0)
    parser.add_argument("--max-retry-delay-seconds", type=float, default=300.0)
    args = parser.parse_args()

    if args.check_velocity_gate:
        try:
            checkpoint = require_velocity_training_pass()
        except (VelocityTrainingGateError, ValueError, OSError) as error:
            print(f"[GETUP] BLOCKED: {error}", file=sys.stderr, flush=True)
            raise SystemExit(2) from error
        print(
            json.dumps(
                {
                    "velocity_checkpoint": str(checkpoint),
                    "sha256": checkpoint_sha256(checkpoint),
                },
                ensure_ascii=False,
            )
        )
        return

    if min(args.num_envs, args.iterations_per_round, args.keep_checkpoints) < 1:
        parser.error("training counts must be positive")
    if args.eval_num_envs < 4 or args.eval_num_envs % 4:
        parser.error("--eval-num-envs must be a positive multiple of four")
    if args.min_samples_per_pose < 1 or args.eval_num_envs // 4 < args.min_samples_per_pose:
        parser.error("evaluation needs at least --min-samples-per-pose per fall pose")
    if not 0.0 <= args.min_success_rate_per_pose <= 1.0:
        parser.error("--min-success-rate-per-pose must be between 0 and 1")
    if args.eval_num_steps < 1 or args.eval_hold_seconds <= 0.0:
        parser.error("evaluation duration and hold duration must be positive")
    if args.live_viewer and args.viewer_num_envs < 16:
        parser.error("--viewer-num-envs must be at least 16")
    if args.retry_delay_seconds < 0 or args.max_retry_delay_seconds < 0:
        parser.error("retry delays must be non-negative")
    if args.max_retry_delay_seconds < args.retry_delay_seconds:
        parser.error("maximum retry delay must be at least the initial delay")

    try:
        velocity_checkpoint = require_velocity_training_pass()
    except (VelocityTrainingGateError, ValueError, OSError) as error:
        raise SystemExit(f"Get-up training is gated: {error}") from error
    print(
        "[GETUP] Velocity gate passed with "
        f"{velocity_checkpoint} sha256={checkpoint_sha256(velocity_checkpoint)}",
        flush=True,
    )

    other_pids = find_other_training_pids()
    if other_pids:
        raise SystemExit(
            "KHR training is already running (PID: "
            + ", ".join(str(pid) for pid in other_pids)
            + ")"
        )

    # Deliberately share the walk supervisor lock: only one KHR training GPU job
    # (including its evaluation child) may own the workspace at a time.
    lock_path = Path("logs/rsl_rl/khr_velocity/.auto_train.lock").resolve()
    try:
        with exclusive_training_lock(lock_path):
            run_getup_auto_training(args)
    except TrainingAlreadyRunningError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
