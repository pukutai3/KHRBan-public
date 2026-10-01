import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from khrban.model import (
    build_khr_spec,
    sample_motion_home_joint_positions,
    sample_motion_joint_limits,
)
from khrban.rewards import action_target_excess_l2, excessive_air_time_penalty
from khrban.tasks import KHR_VELOCITY_TASK_ID
from khrban.tasks.velocity import (
    FROZEN_POLICY_JOINT_NAMES,
    KHR_FOOT_CLEARANCE_TARGET_M,
    KHR_FOOT_TRAINING_TARGET_M,
    KHR_MIN_FOOT_DISTANCE_M,
    KHR_ROTATION_ENV_ANG_VEL_RANGE,
    KHR_ROTATION_ENV_FRACTION,
    KHR_ROTATION_MIN_ANG_VEL,
    LOCOMOTION_ACTION_SCALE,
    make_velocity_env_cfg,
    make_velocity_ppo_cfg,
)


def _set_home_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[:7] = (0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0)
    for joint_name, value in sample_motion_home_joint_positions().items():
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def test_khr_model_exposes_locomotion_foot_references() -> None:
    model = build_khr_spec().compile()
    data = mujoco.MjData(model)
    _set_home_pose(model, data)

    site_ids = []
    for site_name in ("left_foot", "right_foot"):
        site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        assert site_id >= 0
        site_ids.append(site_id)
    for geom_name in ("left_foot_collision", "right_foot_collision"):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
        ) >= 0

    foot_positions = data.site_xpos[site_ids]
    assert np.allclose(foot_positions[:, 2], 0.00819802, atol=1.0e-7)
    home_foot_distance = np.linalg.norm(
        foot_positions[0, :2] - foot_positions[1, :2]
    )
    assert math.isclose(home_foot_distance, 0.07176859, abs_tol=1.0e-7)
    assert KHR_MIN_FOOT_DISTANCE_M < home_foot_distance


def test_velocity_action_scale_stays_inside_confirmed_sample_envelope() -> None:
    home = sample_motion_home_joint_positions()
    limits = sample_motion_joint_limits()

    assert set(LOCOMOTION_ACTION_SCALE) == set(home) - set(
        FROZEN_POLICY_JOINT_NAMES
    )
    for joint_name, scale in LOCOMOTION_ACTION_SCALE.items():
        lower, upper = limits[joint_name]
        assert home[joint_name] - scale >= lower
        assert home[joint_name] + scale <= upper
    for left_index, right_index in (
        (3, 7), (4, 8), (6, 10), (11, 17), (12, 18),
        (13, 19), (14, 20), (15, 21), (16, 22),
    ):
        left_name = f"continuous_joint_{left_index:02d}"
        right_name = f"continuous_joint_{right_index:02d}"
        assert LOCOMOTION_ACTION_SCALE[left_name] == LOCOMOTION_ACTION_SCALE[right_name]


def test_velocity_policy_excludes_frozen_upper_body_yaw_joints() -> None:
    cfg = make_velocity_env_cfg(num_envs=32)
    action = cfg.actions["joint_pos"]
    frozen_joint_names = {
        "continuous_joint_01",  # waist yaw
        "continuous_joint_02",  # head yaw
        "continuous_joint_05",  # left elbow yaw
        "continuous_joint_09",  # right elbow yaw
    }

    assert set(FROZEN_POLICY_JOINT_NAMES) == frozen_joint_names
    controlled_joint_names = tuple(action.actuator_names)
    assert len(controlled_joint_names) == 18
    assert frozen_joint_names.isdisjoint(controlled_joint_names)
    assert set(LOCOMOTION_ACTION_SCALE) == set(controlled_joint_names)
    assert all(
        sample_motion_home_joint_positions()[name] == 0.0
        for name in frozen_joint_names
    )


def test_velocity_action_safety_preserves_target_envelope() -> None:
    home = sample_motion_home_joint_positions()
    cfg = make_velocity_env_cfg(num_envs=32)
    ppo = make_velocity_ppo_cfg()

    assert ppo.clip_actions is None
    target_clip = cfg.actions["joint_pos"].clip
    assert target_clip is not None
    for joint_name, scale in LOCOMOTION_ACTION_SCALE.items():
        assert target_clip[joint_name] == (
            home[joint_name] - scale,
            home[joint_name] + scale,
        )


def test_velocity_action_safety_penalizes_constant_mean_saturation() -> None:
    cfg = make_velocity_env_cfg(num_envs=32)

    assert "action_magnitude_l2" not in cfg.rewards
    excess_penalty = cfg.rewards["action_target_excess_l2"]
    assert excess_penalty.func.__name__ == "action_target_excess_l2"
    assert excess_penalty.weight == -1.0

    env = SimpleNamespace(
        action_manager=SimpleNamespace(
            action=torch.tensor(
                [
                    [-1.0, -0.5, 0.5, 1.0],
                    [-2.0, -1.5, 1.5, 2.0],
                ]
            )
        )
    )
    assert torch.equal(
        action_target_excess_l2(env),
        torch.tensor([0.0, 2.5]),
    )


def test_training_foot_height_reward_targets_margin_above_pass_threshold() -> None:
    cfg = make_velocity_env_cfg(num_envs=32)
    evaluation_threshold = KHR_FOOT_CLEARANCE_TARGET_M
    training_targets = (
        cfg.rewards["foot_clearance"].params["target_height"],
        cfg.rewards["foot_swing_height"].params["target_height"],
    )

    assert math.isclose(
        KHR_FOOT_TRAINING_TARGET_M - evaluation_threshold,
        0.004,
    )
    assert all(target == KHR_FOOT_TRAINING_TARGET_M for target in training_targets)


def test_velocity_task_uses_khr_specific_geometry_and_rewards() -> None:
    cfg = make_velocity_env_cfg(num_envs=32)
    ppo = make_velocity_ppo_cfg()
    command = cfg.commands["twist"]

    assert KHR_VELOCITY_TASK_ID == "Mjlab-KHR-Velocity-Flat"
    assert cfg.scene.num_envs == 32
    assert set(cfg.scene.entities) == {"robot"}
    assert math.isclose(KHR_FOOT_CLEARANCE_TARGET_M, 0.021)
    assert math.isclose(KHR_MIN_FOOT_DISTANCE_M, 0.062)
    assert cfg.rewards["foot_clearance"].params["target_height"] == 0.025
    assert cfg.rewards["foot_swing_height"].params["target_height"] == 0.025
    assert cfg.rewards["feet_distance"].params["min_dist"] == 0.062
    assert command.ranges.lin_vel_x == (-0.50, 0.50)
    assert command.ranges.lin_vel_y == (-0.30, 0.30)
    assert command.ranges.ang_vel_z == (-0.75, 0.75)
    assert command.rel_standing_envs == 0.10
    assert command.rel_forward_envs == 0.0
    assert command.rel_heading_envs == 0.0
    assert command.rel_rotation_envs == KHR_ROTATION_ENV_FRACTION
    assert command.rotation_min_ang_vel == KHR_ROTATION_MIN_ANG_VEL
    assert command.rotation_env_ang_vel_range == KHR_ROTATION_ENV_ANG_VEL_RANGE
    assert cfg.rewards["air_time"].weight == 3.0
    assert cfg.rewards["no_stepping"].weight == 0.0
    assert "microban_style_velocity" in cfg.curriculum
    assert cfg.rewards["angular_momentum"].weight == -0.02
    assert "root_height" not in cfg.terminations
    assert set(cfg.terminations) == {"time_out", "fell_over"}
    assert ppo.experiment_name == "khr_velocity"


def test_velocity_reward_step_has_root_angular_momentum_sensor_on_cpu() -> None:
    env = ManagerBasedRlEnv(
        cfg=make_velocity_env_cfg(num_envs=1, play=True),
        device="cpu",
    )
    try:
        env.reset(seed=314159)
        assert env.scene["robot/root_angmom"] is not None
        _, rewards, _, _, _ = env.step(torch.zeros((1, 18), device="cpu"))
        assert torch.isfinite(rewards).all()
    finally:
        env.close()


def test_velocity_training_penalizes_air_time_beyond_evaluation_window() -> None:
    cfg = make_velocity_env_cfg(num_envs=32)

    penalty = cfg.rewards["excessive_air_time"]
    assert penalty.func.__name__ == "excessive_air_time_penalty"
    assert penalty.weight == -3.0
    assert penalty.params["threshold_max"] == 0.30
    assert penalty.params["command_threshold"] == 0.01

    env = SimpleNamespace(
        scene=SimpleNamespace(
            sensors={
                "feet": SimpleNamespace(
                    data=SimpleNamespace(
                        current_air_time=torch.tensor(
                            [
                                [0.20, 0.31],
                                [0.40, 0.50],
                            ]
                        )
                    )
                )
            }
        ),
        command_manager=SimpleNamespace(
            get_command=lambda _: torch.tensor(
                [
                    [0.20, 0.00, 0.00],
                    [0.00, 0.00, 0.00],
                ]
            )
        ),
    )
    assert torch.equal(
        excessive_air_time_penalty(
            env,
            sensor_name="feet",
            command_name="twist",
            threshold_max=0.30,
            command_threshold=0.01,
        ),
        torch.tensor([1.0, 0.0]),
    )


def test_velocity_task_curriculum_expands_to_microban_style_targets() -> None:
    cfg = make_velocity_env_cfg(num_envs=32)
    stage = cfg.curriculum["microban_style_velocity"].params["stages"][0]

    assert stage["step"] == 3000 * 24
    assert stage["lin_vel_x"] == (-0.70, 0.70)
    assert stage["lin_vel_y"] == (-0.30, 0.30)
    assert stage["ang_vel_z"] == (-1.50, 1.50)
    assert stage["rotation_env_ang_vel_range"] == (-3.00, 3.00)
    assert stage["air_time_weight"] == 3.0
    assert stage["no_stepping_weight"] == -1.0
    assert stage["rel_standing_envs"] == 0.10
    assert stage["rel_rotation_envs"] == KHR_ROTATION_ENV_FRACTION


def test_velocity_actor_observations_follow_microban_privilege_boundary() -> None:
    cfg = make_velocity_env_cfg(num_envs=32)

    assert "base_lin_vel" not in cfg.observations["actor"].terms
    assert "base_lin_vel" in cfg.observations["critic"].terms
    assert "height_scan" not in cfg.observations["actor"].terms
    assert "height_scan" not in cfg.observations["critic"].terms


def test_velocity_non_model_settings_match_pinned_microban_contract() -> None:
    """Reject learning-style drift outside KHR morphology and actuator settings."""

    cfg = make_velocity_env_cfg(num_envs=32)
    ppo = make_velocity_ppo_cfg()
    actor = cfg.observations["actor"].terms

    def noise_contract(term_name: str) -> tuple[float | None, float | None, int, int, int]:
        term = actor[term_name]
        noise = term.noise
        return (
            None if noise is None else noise.n_min,
            None if noise is None else noise.n_max,
            term.delay_min_lag,
            term.delay_max_lag,
            term.delay_update_period,
        )

    actual = {
        "joint_pos_observation": noise_contract("joint_pos"),
        "joint_vel_observation": noise_contract("joint_vel"),
        "base_ang_vel_observation": noise_contract("base_ang_vel"),
        "projected_gravity_observation": noise_contract("projected_gravity"),
        "linear_velocity_reward_std": cfg.rewards["track_linear_velocity"].params[
            "std"
        ],
        "angular_momentum_reward_weight": (
            cfg.rewards["angular_momentum"].weight
            if "angular_momentum" in cfg.rewards
            else None
        ),
        "upright_reward_std": cfg.rewards["upright"].params["std"],
        "hip_pitch_standing_pose_std": cfg.rewards["pose"].params[
            "std_standing"
        ].get(r"continuous_joint_(13|19)"),
        "knee_standing_pose_std": cfg.rewards["pose"].params["std_standing"].get(
            r"continuous_joint_(14|20)"
        ),
        "push_interval": cfg.events["push_robot"].interval_range_s,
        "push_velocity": cfg.events["push_robot"].params["velocity_range"],
        "armature_randomization": (
            cfg.events.get("dof_armature_randomization").params["ranges"]
            if "dof_armature_randomization" in cfg.events
            else None
        ),
        "joint_friction_randomization": (
            cfg.events.get("dof_friction_randomization").params["ranges"]
            if "dof_friction_randomization" in cfg.events
            else None
        ),
        "episode_length_s": cfg.episode_length_s,
        "minimum_air_time": cfg.rewards["air_time"].params["threshold_min"],
        "policy_init_std": ppo.actor.distribution_cfg["init_std"],
    }
    expected = {
        "joint_pos_observation": (-0.001, 0.001, 0, 0, 0),
        "joint_vel_observation": (-0.25, 0.25, 0, 1, 0),
        "base_ang_vel_observation": (-0.03, 0.03, 0, 3, 64),
        "projected_gravity_observation": (-0.01, 0.01, 0, 3, 64),
        "linear_velocity_reward_std": math.sqrt(0.1),
        "angular_momentum_reward_weight": -0.02,
        "upright_reward_std": math.sqrt(0.1),
        "hip_pitch_standing_pose_std": 0.15,
        "knee_standing_pose_std": 0.15,
        "push_interval": (1.0, 3.0),
        "push_velocity": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        "armature_randomization": (0.9, 1.1),
        "joint_friction_randomization": (0.9, 1.1),
        "episode_length_s": 20.0,
        "minimum_air_time": 0.125,
        "policy_init_std": 1.0,
    }

    differences = {
        key: {"actual": actual[key], "expected": expected_value}
        for key, expected_value in expected.items()
        if actual[key] != expected_value
    }
    assert not differences, differences


@pytest.mark.skipif(not torch.cuda.is_available(), reason="velocity task requires CUDA")
def test_velocity_environment_steps_on_gpu() -> None:
    env = ManagerBasedRlEnv(
        cfg=make_velocity_env_cfg(num_envs=2, play=True),
        device="cuda",
    )
    try:
        observations, _ = env.reset(seed=42)
        assert observations["actor"].shape == (2, 65)
        assert observations["critic"].shape == (2, 80)

        robot = env.scene["robot"]
        frozen_joint_ids, frozen_joint_names = robot.find_joints(
            FROZEN_POLICY_JOINT_NAMES, preserve_order=True
        )
        assert tuple(frozen_joint_names) == FROZEN_POLICY_JOINT_NAMES
        frozen_home = torch.tensor(
            [
                sample_motion_home_joint_positions()[name]
                for name in FROZEN_POLICY_JOINT_NAMES
            ],
            device="cuda",
        )
        max_frozen_deviation = torch.tensor(0.0, device="cuda")

        actions = torch.zeros((2, 18), device="cuda")
        ever_terminated = torch.zeros(2, dtype=torch.bool, device="cuda")
        ever_truncated = torch.zeros(2, dtype=torch.bool, device="cuda")
        for _ in range(100):
            observations, rewards, terminated, truncated, _ = env.step(actions)
            ever_terminated |= terminated
            ever_truncated |= truncated
            frozen_positions = robot.data.joint_pos[:, frozen_joint_ids]
            max_frozen_deviation = torch.maximum(
                max_frozen_deviation,
                torch.max(torch.abs(frozen_positions - frozen_home)),
            )

        assert torch.isfinite(observations["actor"]).all()
        assert torch.isfinite(observations["critic"]).all()
        assert torch.isfinite(rewards).all()
        assert torch.allclose(
            robot.data.joint_pos_target[:, frozen_joint_ids],
            frozen_home.expand(2, -1),
        )
        assert max_frozen_deviation.item() <= math.radians(1.0)
        assert not ever_terminated.any()
        assert not ever_truncated.any()
    finally:
        env.close()
