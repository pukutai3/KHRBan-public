from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from khrban.auto_train import (
    DEFAULT_AUTO_TRAIN_NUM_ENVS,
    EVALUATION_CONTRACT_VERSION,
    EvaluationCriteria,
    EvaluationStage,
    EvaluationResult,
    TrainingAlreadyRunningError,
    append_evaluation_record,
    begin_training_lineage,
    checkpoint_iteration,
    exclusive_training_lock,
    find_other_training_pids,
    format_evaluation_values,
    find_latest_checkpoint,
    find_new_checkpoint,
    next_round_number,
    prune_checkpoints,
    read_active_training_lineage,
    run_auto_training,
    run_process_with_retry,
)


def test_formal_evaluation_display_uses_contract_precision() -> None:
    formatted = format_evaluation_values(
        {
            "rounds_down": 0.1044,
            "rounds_up": 0.1073,
            "sample_count": 123,
            "nested": {"rate": 0.5},
        }
    )

    assert formatted == {
        "rounds_down": "0.10",
        "rounds_up": "0.11",
        "sample_count": 123,
        "nested": {"rate": "0.50"},
    }


def test_default_tracking_thresholds_are_calibrated_to_microban_baseline() -> None:
    criteria = EvaluationCriteria()

    assert criteria.max_linear_velocity_rmse == 0.10
    assert criteria.max_yaw_velocity_rmse == 0.28
    assert criteria.max_directional_linear_velocity_rmse == 0.15
    assert criteria.max_rotation_yaw_velocity_rmse == 0.31
    assert criteria.max_rotation_linear_drift_rmse == 0.08

    assert 0.09172603142815448 <= criteria.max_linear_velocity_rmse
    assert 0.26377313978628464 <= criteria.max_yaw_velocity_rmse
    assert 0.13626331828358929 <= criteria.max_directional_linear_velocity_rmse
    assert 0.2886325184513805 <= criteria.max_rotation_yaw_velocity_rmse
    assert 0.07086006761819197 <= criteria.max_rotation_linear_drift_rmse

    # KHR-specific safety and user-requested gait criteria remain strict.
    assert criteria.min_foot_lift_success_rate == 0.50
    assert criteria.max_standing_foot_airborne_rate == 0.05
    assert criteria.max_action_outside_unit_rate == 0.05
    assert criteria.max_standing_action_outside_unit_rate == 0.05


def test_training_guide_documents_the_active_microban_calibrated_thresholds() -> None:
    criteria = EvaluationCriteria()
    guide = Path("docs/training-and-evaluation.md").read_text(encoding="utf-8")
    expected_lines = (
        f"- 前後・左右速度のRMSE: {criteria.max_linear_velocity_rmse:.2f} m/s以下",
        "- 前・後・左・右を各1sample以上含み、4方向で最も悪い軸速度RMSE: "
        f"{criteria.max_directional_linear_velocity_rmse:.2f} m/s以下",
        f"- ヨー角速度のRMSE: {criteria.max_yaw_velocity_rmse:.2f} rad/s以下",
        f"- 純旋回のヨー角速度RMSE: {criteria.max_rotation_yaw_velocity_rmse:.2f} rad/s以下",
        f"- 純旋回中の直線ドリフトRMSE: {criteria.max_rotation_linear_drift_rmse:.2f} m/s以下",
    )

    for line in expected_lines:
        assert line in guide


def _checkpoint(path: Path, iteration: int, mtime: int) -> Path:
    checkpoint = path / f"model_{iteration}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.touch()
    checkpoint.chmod(0o600)
    import os

    os.utime(checkpoint, (mtime, mtime))
    return checkpoint


def _evaluation_result_for_selection(
    *, linear_rmse: float = 0.09, foot_lift_rate: float = 0.60
) -> EvaluationResult:
    return EvaluationResult(
        fall_rate=0.0,
        linear_velocity_rmse=linear_rmse,
        yaw_velocity_rmse=0.20,
        mean_reward=0.0,
        num_envs=64,
        num_steps=500,
        directional_linear_velocity_rmse=0.14,
        direction_sample_count=100,
        rotation_fall_rate=0.0,
        rotation_yaw_velocity_rmse=0.25,
        rotation_linear_drift_rmse=0.06,
        rotation_sample_count=100,
        foot_lift_success_rate=foot_lift_rate,
        foot_peak_height_mean=0.025,
        foot_air_time_mean=0.20,
        foot_landing_count=100,
        standing_foot_airborne_rate=0.01,
        standing_sample_count=100,
        action_rms=0.50,
        action_outside_unit_rate=0.04,
        action_outside_unit_rate_by_mode={"standing": 0.01},
    )


def test_best_evaluated_checkpoint_is_preferred_over_latest(tmp_path: Path) -> None:
    from khrban.auto_train import find_best_evaluated_checkpoint

    run_prefix = "same-lineage"
    older_best = _checkpoint(
        tmp_path / f"round-{run_prefix}-round-0077", 153_923, 100
    )
    latest_but_worse = _checkpoint(
        tmp_path / f"round-{run_prefix}-round-0079", 157_921, 200
    )
    history = tmp_path / "auto_training_evaluations.jsonl"
    append_evaluation_record(
        history,
        older_best,
        _evaluation_result_for_selection(
            linear_rmse=0.1149, foot_lift_rate=0.53622
        ),
        False,
    )
    append_evaluation_record(
        history,
        latest_but_worse,
        _evaluation_result_for_selection(
            linear_rmse=0.1151, foot_lift_rate=0.44362
        ),
        False,
    )

    assert find_latest_checkpoint(tmp_path, run_prefix=run_prefix) == latest_but_worse
    assert (
        find_best_evaluated_checkpoint(
            history,
            log_root=tmp_path,
            run_prefix=run_prefix,
            criteria=EvaluationCriteria(),
        )
        == older_best
    )


def test_append_evaluation_record_persists_metric_contract_version(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "current-round-0001", 100, 100)
    history = tmp_path / "auto_training_evaluations.jsonl"

    append_evaluation_record(
        history,
        checkpoint,
        _evaluation_result_for_selection(),
        False,
    )

    record = json.loads(history.read_text(encoding="utf-8").splitlines()[-1])
    assert record["evaluation_contract_version"] == EVALUATION_CONTRACT_VERSION


def test_append_evaluation_record_persists_evaluation_stage(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "current-round-0001", 100, 100)
    history = tmp_path / "auto_training_evaluations.jsonl"

    append_evaluation_record(
        history,
        checkpoint,
        _evaluation_result_for_selection(linear_rmse=0.099, foot_lift_rate=0.30),
        False,
        evaluation_stage=EvaluationStage.GAIT,
    )

    record = json.loads(history.read_text(encoding="utf-8").splitlines()[-1])
    assert record["evaluation_stage"] == "gait"


def test_append_evaluation_record_persists_rounded_comparison_values(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "current-round-0001", 100, 100)
    history = tmp_path / "auto_training_evaluations.jsonl"
    criteria = EvaluationCriteria()
    result = _evaluation_result_for_selection(
        linear_rmse=0.1044,
        foot_lift_rate=0.4949,
    )

    append_evaluation_record(
        history,
        checkpoint,
        result,
        False,
        evaluation_stage=EvaluationStage.GAIT,
        comparison_values=criteria.comparison_values(result),
    )

    record = json.loads(history.read_text(encoding="utf-8").splitlines()[-1])
    assert record["evaluation_decimal_places"] == 2
    assert record["comparison_values"]["linear_velocity_rmse"] == 0.10
    assert record["comparison_values"]["foot_lift_success_rate"] == 0.49


def test_best_checkpoint_ignores_legacy_unversioned_evaluation_metrics(
    tmp_path: Path,
) -> None:
    from khrban.auto_train import find_best_evaluated_checkpoint

    run_prefix = "obsolete-foot-metric"
    legacy_checkpoint = _checkpoint(
        tmp_path / f"round-{run_prefix}-round-0001", 100, 100
    )
    history = tmp_path / "auto_training_evaluations.jsonl"
    history.write_text(
        json.dumps(
            {
                "evaluated_at": "2026-09-27T00:00:00+00:00",
                "checkpoint": str(legacy_checkpoint.resolve()),
                "achieved": False,
                "metrics": _evaluation_result_for_selection(
                    linear_rmse=0.101, foot_lift_rate=0.99
                ).as_dict(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert (
        find_best_evaluated_checkpoint(
            history,
            log_root=tmp_path,
            run_prefix=run_prefix,
            criteria=EvaluationCriteria(),
        )
        is None
    )


def test_best_checkpoint_ignores_prior_command_timing_contract(
    tmp_path: Path,
) -> None:
    from khrban.auto_train import find_best_evaluated_checkpoint

    run_prefix = "old-command-timing"
    old_checkpoint = _checkpoint(
        tmp_path / f"round-{run_prefix}-round-0001", 100, 100
    )
    history = tmp_path / "auto_training_evaluations.jsonl"
    history.write_text(
        json.dumps(
            {
                "evaluated_at": "2026-09-28T00:00:00+00:00",
                "checkpoint": str(old_checkpoint.resolve()),
                "achieved": False,
                "evaluation_contract_version": 2,
                "metrics": _evaluation_result_for_selection().as_dict(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert EVALUATION_CONTRACT_VERSION == 4
    assert (
        find_best_evaluated_checkpoint(
            history,
            log_root=tmp_path,
            run_prefix=run_prefix,
            criteria=EvaluationCriteria(),
        )
        is None
    )


def test_selection_rank_prioritizes_valid_samples_then_relative_shortfall() -> None:
    criteria = EvaluationCriteria()
    marginal_linear_miss = _evaluation_result_for_selection(
        linear_rmse=0.101, foot_lift_rate=0.60
    )
    larger_foot_miss = _evaluation_result_for_selection(
        linear_rmse=0.09, foot_lift_rate=0.49
    )
    invalid_samples_but_good_metrics = _evaluation_result_for_selection()
    invalid_samples_but_good_metrics = EvaluationResult(
        **{
            **invalid_samples_but_good_metrics.as_dict(),
            "direction_sample_count": 0,
        }
    )

    assert criteria.selection_rank(marginal_linear_miss) > criteria.selection_rank(
        larger_foot_miss
    )
    assert criteria.selection_rank(marginal_linear_miss) > criteria.selection_rank(
        invalid_samples_but_good_metrics
    )


def test_tracking_stage_prefers_tracking_progress_over_already_passing_gait() -> None:
    criteria = EvaluationCriteria()
    integrated_but_slower = _evaluation_result_for_selection(
        linear_rmse=0.1151, foot_lift_rate=0.506237
    )
    tracking_progress = _evaluation_result_for_selection(
        linear_rmse=0.1051, foot_lift_rate=0.274533
    )

    assert criteria.stage_for((integrated_but_slower, tracking_progress)) is EvaluationStage.TRACKING
    assert criteria.selection_rank(
        tracking_progress, stage=EvaluationStage.TRACKING
    ) > criteria.selection_rank(
        integrated_but_slower, stage=EvaluationStage.TRACKING
    )


def test_gait_stage_preserves_a_checkpoint_that_passes_tracking() -> None:
    criteria = EvaluationCriteria()
    tracking_pass = _evaluation_result_for_selection(
        linear_rmse=0.099, foot_lift_rate=0.30
    )
    gait_better_but_tracking_regressed = _evaluation_result_for_selection(
        linear_rmse=0.1051, foot_lift_rate=0.60
    )
    integrated_progress = _evaluation_result_for_selection(
        linear_rmse=0.098, foot_lift_rate=0.45
    )

    assert criteria.stage_for(
        (tracking_pass, gait_better_but_tracking_regressed, integrated_progress)
    ) is EvaluationStage.GAIT
    assert criteria.selection_rank(
        integrated_progress, stage=EvaluationStage.GAIT
    ) > criteria.selection_rank(
        gait_better_but_tracking_regressed, stage=EvaluationStage.GAIT
    )


def test_final_stage_still_requires_the_original_integrated_contract() -> None:
    criteria = EvaluationCriteria()
    tracking_only = _evaluation_result_for_selection(
        linear_rmse=0.099, foot_lift_rate=0.49
    )
    complete = _evaluation_result_for_selection(
        linear_rmse=0.099, foot_lift_rate=0.50
    )

    assert not criteria.is_satisfied_by(tracking_only)
    assert criteria.is_satisfied_by(complete)
    assert criteria.stage_for((tracking_only,)) is EvaluationStage.GAIT
    assert criteria.stage_for((complete,)) is EvaluationStage.COMPLETE


def test_checkpoint_pruning_keeps_best_and_latest_within_limit(
    tmp_path: Path,
) -> None:
    best = _checkpoint(tmp_path / "best-round", 100, 100)
    stale = _checkpoint(tmp_path / "stale-round", 200, 200)
    newest = _checkpoint(tmp_path / "newest-round", 300, 300)
    current = _checkpoint(tmp_path / "current-round", 400, 400)

    removed = prune_checkpoints(tmp_path, keep=3, protected=(best, current))

    assert removed == [stale]
    assert best.exists()
    assert newest.exists()
    assert current.exists()
    assert len(list(tmp_path.glob("*/model_*.pt"))) == 3


def test_auto_training_restarts_from_best_evaluation_not_recent_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from khrban import auto_train as auto_train_module

    class StopAfterTwoTrainingStarts(Exception):
        pass

    monkeypatch.chdir(tmp_path)
    log_root = tmp_path / "logs/rsl_rl/khr_velocity"
    run_prefix = begin_training_lineage(log_root, "same-lineage")
    best = _checkpoint(
        log_root / f"2026-09-27_best_{run_prefix}-round-0001", 100, 100
    )
    recent = _checkpoint(
        log_root / f"2026-09-27_recent_{run_prefix}-round-0002", 200, 200
    )
    history = log_root / "auto_training_evaluations.jsonl"
    append_evaluation_record(
        history,
        best,
        _evaluation_result_for_selection(
            linear_rmse=0.1149, foot_lift_rate=0.53622
        ),
        False,
    )
    append_evaluation_record(
        history,
        recent,
        _evaluation_result_for_selection(
            linear_rmse=0.1151, foot_lift_rate=0.44362
        ),
        False,
    )

    training_commands: list[list[str]] = []

    def fake_run_process(command: list[str], **kwargs: object) -> None:
        assert kwargs["label"] == "training"
        training_commands.append(command)
        if len(training_commands) == 1:
            run_name = command[command.index("--run-name") + 1]
            _checkpoint(log_root / f"2026-09-27_{run_name}", 300, 300)
            return
        raise StopAfterTwoTrainingStarts

    def fake_evaluation(checkpoint: Path, **kwargs: object) -> EvaluationResult:
        del kwargs
        if checkpoint == best:
            return _evaluation_result_for_selection(
                linear_rmse=0.1149, foot_lift_rate=0.53622
            )
        return _evaluation_result_for_selection(
            linear_rmse=0.1151, foot_lift_rate=0.44362
        )

    monkeypatch.setattr(
        auto_train_module, "run_process_with_retry", fake_run_process
    )
    monkeypatch.setattr(auto_train_module, "run_evaluation", fake_evaluation)
    args = Namespace(
        fresh=False,
        run_prefix="same-lineage",
        num_envs=1,
        iterations_per_round=1,
        gpu_id=0,
        keep_checkpoints=3,
        eval_num_envs=64,
        eval_num_steps=500,
        eval_seed=314159,
        max_fall_rate=0.05,
        max_linear_velocity_rmse=0.10,
        max_yaw_velocity_rmse=0.28,
        max_directional_linear_velocity_rmse=0.15,
        min_direction_sample_count=1,
        max_rotation_fall_rate=0.05,
        max_rotation_yaw_velocity_rmse=0.31,
        max_rotation_linear_drift_rmse=0.08,
        min_rotation_sample_count=1,
        min_foot_lift_success_rate=0.50,
        min_foot_landing_count=1,
        max_standing_foot_airborne_rate=0.05,
        min_standing_sample_count=1,
        max_action_outside_unit_rate=0.05,
        max_standing_action_outside_unit_rate=0.05,
        retry_delay_seconds=0.0,
        max_retry_delay_seconds=0.0,
        live_viewer=False,
    )

    with pytest.raises(StopAfterTwoTrainingStarts):
        auto_train_module.run_auto_training(args)

    assert len(training_commands) == 2
    for command in training_commands:
        assert command[command.index("--resume-from") + 1] == str(best)
    assert best.exists()
    assert len(list(log_root.glob("*/model_*.pt"))) <= 3


def test_auto_training_uses_benchmarked_parallel_count() -> None:
    assert DEFAULT_AUTO_TRAIN_NUM_ENVS == 1536


def test_evaluation_criteria_require_every_metric() -> None:
    criteria = EvaluationCriteria(
        max_fall_rate=0.05,
        max_linear_velocity_rmse=0.08,
        max_yaw_velocity_rmse=0.20,
        max_directional_linear_velocity_rmse=0.08,
        min_direction_sample_count=1,
        max_rotation_fall_rate=0.05,
        max_rotation_yaw_velocity_rmse=0.20,
        max_rotation_linear_drift_rmse=0.08,
        min_rotation_sample_count=1,
        min_foot_lift_success_rate=0.50,
        min_foot_landing_count=1,
        max_standing_foot_airborne_rate=0.05,
        min_standing_sample_count=1,
        max_action_outside_unit_rate=0.05,
        max_standing_action_outside_unit_rate=0.05,
    )
    passing = EvaluationResult(
        fall_rate=0.05,
        linear_velocity_rmse=0.08,
        yaw_velocity_rmse=0.20,
        mean_reward=1.0,
        num_envs=64,
        num_steps=500,
        directional_linear_velocity_rmse=0.08,
        direction_sample_count=1,
        rotation_fall_rate=0.05,
        rotation_yaw_velocity_rmse=0.20,
        rotation_linear_drift_rmse=0.08,
        rotation_sample_count=1,
        foot_lift_success_rate=0.50,
        foot_peak_height_mean=0.021,
        foot_air_time_mean=0.10,
        foot_landing_count=1,
        standing_foot_airborne_rate=0.05,
        standing_sample_count=1,
        action_rms=0.50,
        action_outside_unit_rate=0.05,
    )
    assert criteria.is_satisfied_by(passing)
    assert not criteria.is_satisfied_by(
        EvaluationResult(**{**passing.as_dict(), "fall_rate": 0.055})
    )
    assert not criteria.is_satisfied_by(
        EvaluationResult(**{**passing.as_dict(), "rotation_sample_count": 0})
    )
    assert not criteria.is_satisfied_by(
        EvaluationResult(**{**passing.as_dict(), "rotation_yaw_velocity_rmse": 0.205})
    )
    assert not criteria.is_satisfied_by(
        EvaluationResult(**{**passing.as_dict(), "foot_lift_success_rate": 0.4949})
    )
    assert not criteria.is_satisfied_by(
        EvaluationResult(**{**passing.as_dict(), "foot_landing_count": 0})
    )
    assert not criteria.is_satisfied_by(
        EvaluationResult(
            **{**passing.as_dict(), "directional_linear_velocity_rmse": 0.085}
        )
    )
    assert not criteria.is_satisfied_by(
        EvaluationResult(**{**passing.as_dict(), "direction_sample_count": 0})
    )
    assert not criteria.is_satisfied_by(
        EvaluationResult(
            **{**passing.as_dict(), "standing_foot_airborne_rate": 0.055}
        )
    )
    assert not criteria.is_satisfied_by(
        EvaluationResult(**{**passing.as_dict(), "standing_sample_count": 0})
    )
    assert not criteria.is_satisfied_by(
        EvaluationResult(**{**passing.as_dict(), "action_outside_unit_rate": 0.055})
    )
    assert not criteria.is_satisfied_by(
        EvaluationResult(
            **{
                **passing.as_dict(),
                "action_outside_unit_rate_by_mode": {"standing": 0.055},
            }
        )
    )


def test_evaluation_uses_the_same_two_decimal_places_as_thresholds() -> None:
    criteria = EvaluationCriteria()
    displayed_as_limit = _evaluation_result_for_selection(
        linear_rmse=0.1044,
        foot_lift_rate=0.5044,
    )
    displayed_above_limit = _evaluation_result_for_selection(
        linear_rmse=0.1051,
        foot_lift_rate=0.5044,
    )
    displayed_below_foot_limit = _evaluation_result_for_selection(
        linear_rmse=0.1044,
        foot_lift_rate=0.4949,
    )

    assert criteria.is_satisfied_by(displayed_as_limit)
    assert not criteria.is_satisfied_by(displayed_above_limit)
    assert not criteria.is_satisfied_by(displayed_below_foot_limit)


def test_tracking_rank_does_not_use_hidden_decimal_places() -> None:
    criteria = EvaluationCriteria()
    first = _evaluation_result_for_selection(
        linear_rmse=0.1044,
        foot_lift_rate=0.30,
    )
    second = _evaluation_result_for_selection(
        linear_rmse=0.1001,
        foot_lift_rate=0.30,
    )

    assert criteria.selection_rank(
        first, stage=EvaluationStage.TRACKING
    ) == criteria.selection_rank(second, stage=EvaluationStage.TRACKING)


def test_checkpoint_iteration_rejects_unrelated_files(tmp_path: Path) -> None:
    assert checkpoint_iteration(tmp_path / "model_3900.pt") == 3900
    assert checkpoint_iteration(tmp_path / "optimizer.pt") is None
    assert checkpoint_iteration(tmp_path / "model_latest.pt") is None


def test_prune_checkpoints_keeps_only_newest_across_runs(tmp_path: Path) -> None:
    oldest = _checkpoint(tmp_path / "run-a", 100, 100)
    middle = _checkpoint(tmp_path / "run-a", 200, 200)
    newest = _checkpoint(tmp_path / "run-b", 300, 300)
    latest = _checkpoint(tmp_path / "run-b", 400, 400)

    removed = prune_checkpoints(tmp_path, keep=3)

    assert removed == [oldest]
    assert not oldest.exists()
    assert middle.exists() and newest.exists() and latest.exists()


def test_prune_checkpoints_prefers_recent_run_over_larger_iteration_number(
    tmp_path: Path,
) -> None:
    legacy_oldest = _checkpoint(tmp_path / "legacy-a", 48_000, 100)
    legacy_middle = _checkpoint(tmp_path / "legacy-b", 49_998, 200)
    legacy_latest = _checkpoint(tmp_path / "legacy-c", 50_000, 300)
    current_resume = _checkpoint(tmp_path / "current", 1_999, 400)

    removed = prune_checkpoints(tmp_path, keep=3)

    assert removed == [legacy_oldest]
    assert current_resume.exists()
    assert legacy_middle.exists() and legacy_latest.exists()


def test_prune_checkpoints_never_removes_protected_resume(tmp_path: Path) -> None:
    protected = _checkpoint(tmp_path / "current", 1_999, 100)
    old = _checkpoint(tmp_path / "other", 2_000, 200)
    middle = _checkpoint(tmp_path / "other", 4_000, 300)
    latest = _checkpoint(tmp_path / "other", 6_000, 400)

    removed = prune_checkpoints(tmp_path, keep=3, protected=(protected,))

    assert removed == [old]
    assert protected.exists()
    assert middle.exists() and latest.exists()


def test_latest_checkpoint_prefers_training_progress_over_newer_smoke(
    tmp_path: Path,
) -> None:
    trained = _checkpoint(tmp_path / "trained", 1999, 100)
    _checkpoint(tmp_path / "newer-smoke", 0, 200)

    assert find_latest_checkpoint(tmp_path) == trained


def test_latest_checkpoint_does_not_cross_active_training_series(
    tmp_path: Path,
) -> None:
    _checkpoint(
        tmp_path / "2026-09-14_03-41-23_failed-series-round-0007",
        11_994,
        100,
    )
    current = _checkpoint(
        tmp_path / "2026-09-14_09-04-35_current-series-round-0001",
        1_999,
        200,
    )

    assert find_latest_checkpoint(tmp_path, run_prefix="current-series") == current


def test_active_training_lineage_is_unique_and_restores_round_sequence(
    tmp_path: Path,
) -> None:
    series_prefix = begin_training_lineage(tmp_path, "current-series")
    assert series_prefix.startswith("current-series-")
    assert read_active_training_lineage(tmp_path) == series_prefix

    _checkpoint(
        tmp_path / f"2026-09-14_09-04-35_{series_prefix}-round-0001",
        1_999,
        100,
    )
    latest = _checkpoint(
        tmp_path / f"2026-09-14_10-04-35_{series_prefix}-round-0002",
        3_998,
        200,
    )
    _checkpoint(tmp_path / "2026-09-14_03-41-23_failed-round-0009", 17_991, 300)

    assert find_latest_checkpoint(tmp_path, run_prefix=series_prefix) == latest
    assert next_round_number(tmp_path, series_prefix) == 3


def test_new_checkpoint_is_selected_even_when_fresh_iteration_is_lower(
    tmp_path: Path,
) -> None:
    trained = _checkpoint(tmp_path / "trained", 1999, 100)
    previous = {trained.resolve()}
    fresh = _checkpoint(tmp_path / "fresh", 0, 200)

    assert find_new_checkpoint(tmp_path, previous) == fresh


def test_evaluation_history_is_preserved_when_models_are_pruned(
    tmp_path: Path,
) -> None:
    history = tmp_path / "auto_training_evaluations.jsonl"
    result = EvaluationResult(
        fall_rate=0.10,
        linear_velocity_rmse=0.12,
        yaw_velocity_rmse=0.25,
        mean_reward=0.5,
        num_envs=16,
        num_steps=50,
        rotation_fall_rate=0.20,
        rotation_yaw_velocity_rmse=0.30,
        rotation_linear_drift_rmse=0.10,
        rotation_sample_count=4,
    )
    append_evaluation_record(history, tmp_path / "run/model_100.pt", result, False)

    record = json.loads(history.read_text(encoding="utf-8"))
    assert record["checkpoint"].endswith("model_100.pt")
    assert record["achieved"] is False
    assert record["metrics"]["fall_rate"] == 0.10
    assert record["metrics"]["rotation_sample_count"] == 4


def test_auto_training_limits_intermediate_checkpoint_frequency() -> None:
    source = (Path(__file__).parents[1] / "src/khrban/auto_train.py").read_text()
    assert '"--save-interval"' in source
    assert 'str(args.iterations_per_round)' in source


def test_auto_training_lock_rejects_a_second_supervisor(tmp_path: Path) -> None:
    lock_path = tmp_path / ".auto_train.lock"

    with exclusive_training_lock(lock_path):
        with pytest.raises(TrainingAlreadyRunningError):
            with exclusive_training_lock(lock_path):
                pass

    with exclusive_training_lock(lock_path):
        assert lock_path.read_text(encoding="utf-8").strip().isdigit()


def test_existing_console_script_and_module_training_are_detected(
    tmp_path: Path,
) -> None:
    for pid, command in {
        5: b".venv/bin/khrban-train\0--task\0velocity\0",
        10: b"python3\0.venv/bin/khrban-auto-train\0--num-envs\01024\0",
        20: b"python3\0-m\0khrban.train\0--task\0velocity\0",
        30: b"bash\0-c\0echo khrban-auto-train\0",
        40: b"timeout\0--kill-after=5s\025s\0.venv/bin/khrban-auto-train\0--fresh\0",
    }.items():
        process = tmp_path / str(pid)
        process.mkdir()
        (process / "cmdline").write_bytes(command)

    assert find_other_training_pids(tmp_path, current_pid=999) == (5, 10, 20)


def test_failed_child_process_is_retried_until_success(monkeypatch) -> None:
    attempts = 0
    delays: list[float] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("khrban.auto_train.time.sleep", delays.append)

    run_process_with_retry(
        ["training-child"],
        label="training",
        initial_delay_seconds=1.0,
        max_delay_seconds=2.0,
    )

    assert attempts == 3
    assert delays == [1.0, 2.0]


def test_auto_training_repeats_without_round_limit_until_all_criteria_pass(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    passing = EvaluationResult(
        fall_rate=0.0,
        linear_velocity_rmse=0.08,
        yaw_velocity_rmse=0.20,
        mean_reward=1.0,
        num_envs=64,
        num_steps=500,
        directional_linear_velocity_rmse=0.08,
        direction_sample_count=1,
        rotation_fall_rate=0.0,
        rotation_yaw_velocity_rmse=0.20,
        rotation_linear_drift_rmse=0.08,
        rotation_sample_count=1,
        foot_lift_success_rate=0.50,
        foot_peak_height_mean=0.021,
        foot_air_time_mean=0.10,
        foot_landing_count=1,
        standing_foot_airborne_rate=0.05,
        standing_sample_count=1,
        action_rms=0.50,
        action_outside_unit_rate=0.05,
    )
    evaluations = [
        EvaluationResult(
            **{**passing.as_dict(), "directional_linear_velocity_rmse": 0.085}
        ),
        passing,
    ]
    training_commands: list[list[str]] = []

    def fake_process(command: list[str], **_: object) -> None:
        training_commands.append(command)
        run_name = command[command.index("--run-name") + 1]
        checkpoint = (
            tmp_path
            / "logs/rsl_rl/khr_velocity"
            / run_name
            / "model_1999.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(run_name.encode())

    def fake_evaluation(*_: object, **__: object) -> EvaluationResult:
        return evaluations.pop(0)

    monkeypatch.setattr("khrban.auto_train.run_process_with_retry", fake_process)
    monkeypatch.setattr("khrban.auto_train.run_evaluation", fake_evaluation)
    args = Namespace(
        max_fall_rate=0.05,
        max_linear_velocity_rmse=0.08,
        max_yaw_velocity_rmse=0.20,
        max_directional_linear_velocity_rmse=0.08,
        min_direction_sample_count=1,
        max_rotation_fall_rate=0.05,
        max_rotation_yaw_velocity_rmse=0.20,
        max_rotation_linear_drift_rmse=0.08,
        min_rotation_sample_count=1,
        min_foot_lift_success_rate=0.50,
        min_foot_landing_count=1,
        max_standing_foot_airborne_rate=0.05,
        min_standing_sample_count=1,
        max_action_outside_unit_rate=0.05,
        max_standing_action_outside_unit_rate=0.05,
        fresh=True,
        eval_num_envs=64,
        eval_num_steps=500,
        eval_seed=314159,
        gpu_id=0,
        retry_delay_seconds=0.0,
        max_retry_delay_seconds=0.0,
        keep_checkpoints=3,
        run_prefix="until-pass",
        num_envs=1024,
        iterations_per_round=2000,
        live_viewer=False,
        viewer_num_envs=16,
        viewer_port=8080,
    )

    run_auto_training(args)

    assert len(training_commands) == 2
    assert "--resume-from" not in training_commands[0]
    assert "--resume-from" in training_commands[1]
    history_path = (
        tmp_path
        / "logs/rsl_rl/khr_velocity"
        / "auto_training_evaluations.jsonl"
    )
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    assert [record["achieved"] for record in history] == [False, True]
