import json
import math
from pathlib import Path
import sys

import torch

import khrban.evaluation as evaluation
from khrban.evaluation import (
    VelocityEvaluationAccumulator,
    make_velocity_evaluation_env_cfg,
)


def test_evaluation_cli_displays_two_decimals_but_saves_raw_metrics(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "evaluation.json"
    raw_result = {
        "linear_velocity_rmse": 0.1073,
        "foot_lift_success_rate": 0.5049,
        "num_envs": 64,
    }
    monkeypatch.setattr(
        evaluation,
        "evaluate_velocity_checkpoint",
        lambda *args, **kwargs: raw_result,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["khrban-evaluation", "model.pt", "--output", str(output)],
    )

    evaluation.main()

    displayed = json.loads(capsys.readouterr().out)
    assert displayed["linear_velocity_rmse"] == "0.11"
    assert displayed["foot_lift_success_rate"] == "0.50"
    assert json.loads(output.read_text())["linear_velocity_rmse"] == 0.1073


def test_velocity_evaluation_uses_pinned_microban_air_time_minimum() -> None:
    accumulator = VelocityEvaluationAccumulator(num_envs=1, device="cpu")

    assert accumulator.foot_min_air_time == 0.125


def test_velocity_evaluation_reports_rotation_specific_metrics() -> None:
    accumulator = VelocityEvaluationAccumulator(
        num_envs=3,
        device="cpu",
        action_names=("left_joint", "right_joint"),
    )
    command = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    root_lin_vel_b = torch.tensor(
        [
            [1.2, 0.0, 0.0],
            [0.3, 0.4, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    root_ang_vel_b = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -1.5],
        ]
    )

    accumulator.update(
        root_lin_vel_b=root_lin_vel_b,
        root_ang_vel_b=root_ang_vel_b,
        command=command,
        actions=torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        rewards=torch.tensor([1.0, 2.0, 3.0]),
        dones=torch.tensor([False, True, False]),
        rotation_mask=torch.tensor([False, True, True]),
    )

    result = accumulator.as_dict(num_steps=1)

    assert result["rotation_sample_count"] == 2
    assert result["rotation_fall_rate"] == 0.5
    assert math.isclose(result["rotation_yaw_velocity_rmse"], 0.5)
    assert math.isclose(result["rotation_linear_drift_rmse"], 0.25)
    assert math.isclose(result["action_rms"], math.sqrt(5.0 / 6.0))
    assert math.isclose(result["action_outside_unit_rate"], 1.0 / 6.0)
    assert result["action_outside_unit_rate_by_joint"] == {
        "left_joint": 1.0 / 3.0,
        "right_joint": 0.0,
    }
    assert result["action_sample_count_by_mode"] == {
        "standing": 0,
        "translation": 1,
        "rotation": 2,
    }
    assert result["action_outside_unit_rate_by_mode"] == {
        "standing": 0.0,
        "translation": 0.0,
        "rotation": 0.25,
    }
    assert result["action_outside_unit_rate_by_mode_and_joint"] == {
        "standing": {"left_joint": 0.0, "right_joint": 0.0},
        "translation": {"left_joint": 0.0, "right_joint": 0.0},
        "rotation": {"left_joint": 0.5, "right_joint": 0.0},
    }


def test_velocity_evaluation_reports_standing_action_saturation() -> None:
    accumulator = VelocityEvaluationAccumulator(
        num_envs=2,
        device="cpu",
        action_names=("hip_pitch", "knee_pitch"),
    )
    command = torch.zeros((2, 3))

    accumulator.update(
        root_lin_vel_b=torch.zeros((2, 3)),
        root_ang_vel_b=torch.zeros((2, 3)),
        command=command,
        actions=torch.tensor([[1.1, 0.0], [-1.2, 1.0]]),
        rewards=torch.ones(2),
        dones=torch.zeros(2, dtype=torch.bool),
        rotation_mask=torch.zeros(2, dtype=torch.bool),
    )

    result = accumulator.as_dict(num_steps=1)

    assert result["action_sample_count_by_mode"]["standing"] == 2
    assert result["action_outside_unit_rate_by_mode"]["standing"] == 0.5
    assert result["action_outside_unit_rate_by_mode_and_joint"]["standing"] == {
        "hip_pitch": 1.0,
        "knee_pitch": 0.0,
    }


def test_velocity_evaluation_requires_bilateral_landed_foot_lifts() -> None:
    accumulator = VelocityEvaluationAccumulator(num_envs=3, device="cpu")
    command = torch.tensor(
        [
            [0.3, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.6],
        ]
    )
    no_done = torch.zeros(3, dtype=torch.bool)

    accumulator.update_foot_lifts(
        command=command,
        foot_heights=torch.tensor(
            [[0.023, 0.0], [0.0, 0.024], [0.010, 0.0]]
        ),
        foot_contacts=torch.tensor(
            [[False, True], [True, False], [False, True]]
        ),
        first_contacts=torch.zeros((3, 2), dtype=torch.bool),
        last_air_time=torch.zeros((3, 2)),
        dones=no_done,
    )
    accumulator.update_foot_lifts(
        command=command,
        foot_heights=torch.zeros((3, 2)),
        foot_contacts=torch.ones((3, 2), dtype=torch.bool),
        first_contacts=torch.tensor(
            [[True, False], [False, True], [True, False]]
        ),
        last_air_time=torch.tensor([[0.125, 0.0], [0.0, 0.125], [0.12, 0.0]]),
        dones=no_done,
    )

    result = accumulator.as_dict(num_steps=2)

    assert result["foot_landing_count"] == 1
    assert math.isclose(result["foot_lift_success_rate"], 0.5)
    assert math.isclose(result["foot_peak_height_mean"], 0.019, abs_tol=1.0e-8)
    assert math.isclose(result["foot_air_time_mean"], 0.37 / 3.0, abs_tol=1.0e-8)


def test_velocity_evaluation_rejects_air_time_below_microban_minimum() -> None:
    accumulator = VelocityEvaluationAccumulator(num_envs=2, device="cpu")
    command = torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.2, 0.0]])
    no_done = torch.zeros(2, dtype=torch.bool)

    accumulator.update_foot_lifts(
        command=command,
        foot_heights=torch.tensor([[0.023, 0.0], [0.0, 0.023]]),
        foot_contacts=torch.tensor([[False, True], [True, False]]),
        first_contacts=torch.zeros((2, 2), dtype=torch.bool),
        last_air_time=torch.zeros((2, 2)),
        dones=no_done,
    )
    accumulator.update_foot_lifts(
        command=command,
        foot_heights=torch.zeros((2, 2)),
        foot_contacts=torch.ones((2, 2), dtype=torch.bool),
        first_contacts=torch.tensor([[True, False], [False, True]]),
        last_air_time=torch.tensor([[0.125, 0.0], [0.0, 0.124]]),
        dones=no_done,
    )

    result = accumulator.as_dict(num_steps=2)

    assert result["foot_landing_count"] == 1
    assert result["foot_lift_success_rate"] == 0.0


def test_velocity_evaluation_covers_four_directions_and_standing() -> None:
    accumulator = VelocityEvaluationAccumulator(num_envs=5, device="cpu")
    command = torch.tensor(
        [
            [0.3, 0.0, 0.0],
            [-0.3, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, -0.2, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    accumulator.update(
        root_lin_vel_b=torch.cat((command[:, :2], torch.zeros((5, 1))), dim=1),
        root_ang_vel_b=torch.zeros((5, 3)),
        command=command,
        actions=torch.zeros((5, 18)),
        rewards=torch.ones(5),
        dones=torch.zeros(5, dtype=torch.bool),
        rotation_mask=torch.zeros(5, dtype=torch.bool),
    )
    accumulator.update_standing(
        command=command,
        foot_contacts=torch.tensor(
            [
                [True, True],
                [True, True],
                [True, True],
                [True, True],
                [True, False],
            ]
        ),
    )

    result = accumulator.as_dict(num_steps=1)

    assert result["direction_sample_count"] == 1
    assert result["directional_linear_velocity_rmse"] == 0.0
    assert result["standing_sample_count"] == 1
    assert math.isclose(result["standing_foot_airborne_rate"], 0.5)


def test_velocity_evaluation_uses_final_training_command_distribution() -> None:
    cfg = make_velocity_evaluation_env_cfg(num_envs=4)
    command = cfg.commands["twist"]

    assert command.ranges.lin_vel_x == (-0.7, 0.7)
    assert command.ranges.lin_vel_y == (-0.3, 0.3)
    assert command.ranges.ang_vel_z == (-1.5, 1.5)
    assert command.rotation_env_ang_vel_range == (-3.0, 3.0)
    assert command.rel_standing_envs == 0.1
    assert command.rel_rotation_envs == 0.1
    assert not cfg.curriculum
