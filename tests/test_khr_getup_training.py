from __future__ import annotations

import json
from pathlib import Path

import pytest

from khrban.auto_train import begin_training_lineage, find_other_training_pids
from khrban.auto_train_getup import (
    VelocityTrainingGateError,
    checkpoint_sha256,
    require_velocity_training_pass,
)
from khrban.getup_evaluation import (
    GetUpEvaluationCriteria,
    GetUpEvaluationResult,
)
from khrban.tasks.getup import GETUP_POSE_CLASSES


def _walking_checkpoint(log_root: Path, run_prefix: str, *, achieved: bool) -> Path:
    run_dir = log_root / f"2026-09-23_{run_prefix}-round-0001"
    checkpoint = run_dir / "model_2000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"retained walking checkpoint")
    history_path = log_root / "auto_training_evaluations.jsonl"
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "checkpoint": str(checkpoint.resolve()),
                    "achieved": achieved,
                    "metrics": {},
                }
            )
            + "\n"
        )
    return checkpoint


def _passing_result() -> GetUpEvaluationResult:
    return GetUpEvaluationResult(
        success_rate_by_pose={pose: 0.80 for pose in GETUP_POSE_CLASSES},
        sample_count_by_pose={pose: 16 for pose in GETUP_POSE_CLASSES},
        final_torso_height_mean_by_pose_m={pose: 0.28 for pose in GETUP_POSE_CLASSES},
        final_upright_cos_mean_by_pose={pose: 0.95 for pose in GETUP_POSE_CLASSES},
        final_both_feet_contact_rate_by_pose={pose: 0.90 for pose in GETUP_POSE_CLASSES},
        success_rate=0.80,
        num_envs=64,
        num_steps=500,
        hold_steps=50,
    )


def test_getup_evaluation_requires_every_fall_class_to_pass() -> None:
    criteria = GetUpEvaluationCriteria(
        min_success_rate_per_pose=0.80,
        min_samples_per_pose=16,
    )
    result = _passing_result()
    assert criteria.is_satisfied_by(result)

    rates = dict(result.success_rate_by_pose)
    rates["right_side"] = 0.79
    assert not criteria.is_satisfied_by(
        GetUpEvaluationResult(**{**result.as_dict(), "success_rate_by_pose": rates})
    )

    counts = dict(result.sample_count_by_pose)
    counts["left_side"] = 15
    assert not criteria.is_satisfied_by(
        GetUpEvaluationResult(**{**result.as_dict(), "sample_count_by_pose": counts})
    )


def test_getup_training_gate_requires_a_retained_pass_in_active_walk_lineage(
    tmp_path: Path,
) -> None:
    walk_root = tmp_path / "logs" / "rsl_rl" / "khr_velocity"
    run_prefix = begin_training_lineage(walk_root, "walk-stage")
    checkpoint = _walking_checkpoint(walk_root, run_prefix, achieved=True)

    assert require_velocity_training_pass(velocity_log_root=walk_root) == checkpoint.resolve()
    assert len(checkpoint_sha256(checkpoint)) == 64

    failed_prefix = begin_training_lineage(walk_root, "new-walk-stage")
    _walking_checkpoint(walk_root, failed_prefix, achieved=False)
    with pytest.raises(VelocityTrainingGateError, match="未達"):
        require_velocity_training_pass(velocity_log_root=walk_root)


def test_getup_training_gate_fails_if_the_passing_checkpoint_was_pruned(
    tmp_path: Path,
) -> None:
    walk_root = tmp_path / "logs" / "rsl_rl" / "khr_velocity"
    run_prefix = begin_training_lineage(walk_root, "walk-stage")
    checkpoint = _walking_checkpoint(walk_root, run_prefix, achieved=True)
    checkpoint.unlink()

    with pytest.raises(VelocityTrainingGateError, match="見つかりません"):
        require_velocity_training_pass(velocity_log_root=walk_root)


def test_training_process_discovery_includes_getup_supervisor(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    process = proc_root / "4321"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(
        b"/home/user/.venv/bin/khrban-auto-train-getup\0--live-viewer\0"
    )

    assert find_other_training_pids(proc_root, current_pid=1234) == (4321,)
