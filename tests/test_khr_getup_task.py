from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

from khrban.model import sample_motion_joint_limits
from khrban.getup_evaluation import evaluate_getup_checkpoint
from khrban.tasks import GETUP_TASK_ID
from khrban.tasks.getup import (
    GETUP_POSE_CLASSES,
    GETUP_ROOT_HEIGHT_BY_POSE_M,
    GETUP_SUCCESS_TORSO_HEIGHT_M,
    GETUP_UPRIGHT_COS_THRESHOLD,
    KHR_HOME_TORSO_HEIGHT_M,
    getup_action_scale,
    getup_stable_mask,
    make_getup_env_cfg,
)
from khrban.tasks.velocity import FROZEN_POLICY_JOINT_NAMES, LOCOMOTION_JOINT_NAMES


def test_getup_task_is_registered_and_keeps_the_khr_actor_boundary() -> None:
    cfg = make_getup_env_cfg(num_envs=32)

    assert GETUP_TASK_ID == "Mjlab-KHR-GetUp-Flat"
    assert cfg.commands == {}
    assert "fell_over" not in cfg.terminations
    assert tuple(cfg.terminations) == ("time_out",)
    assert set(cfg.actions["joint_pos"].actuator_names) == set(LOCOMOTION_JOINT_NAMES)
    assert set(cfg.actions["joint_pos"].actuator_names).isdisjoint(
        FROZEN_POLICY_JOINT_NAMES
    )
    assert len(cfg.actions["joint_pos"].actuator_names) == 18

    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms
    assert "command" not in actor_terms
    assert "base_lin_vel" not in actor_terms
    assert "torso_height" not in actor_terms
    assert "foot_contact" not in actor_terms
    assert "torso_height" in critic_terms
    assert "base_lin_vel" in critic_terms
    assert "foot_contact" in critic_terms


def test_getup_action_targets_use_only_the_confirmed_sample_envelope() -> None:
    cfg = make_getup_env_cfg(num_envs=16)
    limits = sample_motion_joint_limits()
    scales = getup_action_scale()
    action_cfg = cfg.actions["joint_pos"]

    assert set(scales) == set(LOCOMOTION_JOINT_NAMES)
    assert set(action_cfg.clip) == set(LOCOMOTION_JOINT_NAMES)
    for joint_name in LOCOMOTION_JOINT_NAMES:
        lower, upper = limits[joint_name]
        assert action_cfg.clip[joint_name] == (lower, upper)
        assert math.isfinite(scales[joint_name]) and scales[joint_name] > 0.0


def test_getup_targets_use_khr_measured_geometry_and_four_fall_classes() -> None:
    assert GETUP_POSE_CLASSES == ("face_down", "face_up", "left_side", "right_side")
    assert set(GETUP_ROOT_HEIGHT_BY_POSE_M) == set(GETUP_POSE_CLASSES)
    assert all(0.0 < GETUP_ROOT_HEIGHT_BY_POSE_M[name] < 0.2 for name in GETUP_POSE_CLASSES)
    assert math.isclose(KHR_HOME_TORSO_HEIGHT_M, 0.30475, abs_tol=1.0e-7)
    assert GETUP_SUCCESS_TORSO_HEIGHT_M < KHR_HOME_TORSO_HEIGHT_M
    assert math.isclose(
        GETUP_UPRIGHT_COS_THRESHOLD,
        math.cos(math.radians(20.0)),
        abs_tol=1.0e-12,
    )


def test_cpu_environment_resets_balanced_poses_and_computes_all_terms() -> None:
    cfg = make_getup_env_cfg(num_envs=4, play=True, balanced_reset=True)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        env.reset()
        robot = env.scene["robot"]
        assert env.getup_pose_class_ids.tolist() == [0, 1, 2, 3]
        expected_root_z = torch.tensor(
            [GETUP_ROOT_HEIGHT_BY_POSE_M[name] for name in GETUP_POSE_CLASSES]
        )
        assert torch.allclose(robot.data.root_link_pos_w[:, 2], expected_root_z, atol=1.0e-6)

        observations = env.observation_manager.compute()
        assert observations["actor"].shape == (4, 62)
        assert observations["critic"].shape == (4, 68)

        actions = torch.zeros((4, 18), device="cpu")
        for _ in range(8):
            observations, rewards, terminated, truncated, _ = env.step(actions)
            assert torch.isfinite(rewards).all()
            assert torch.isfinite(observations["actor"]).all()
            assert torch.isfinite(observations["critic"]).all()
            assert not terminated.any()
            assert not truncated.any()

        torso_cfg = SceneEntityCfg(
            "robot", body_names=("c_chest_c_1",), preserve_order=True
        )
        stable = getup_stable_mask(
            env,
            asset_cfg=torso_cfg,
            sensor_name="feet_ground_contact",
        )
        assert stable.shape == (4,)
        assert stable.dtype == torch.bool
    finally:
        env.close()


def test_cpu_evaluator_loads_a_checkpoint_and_reports_each_pose(tmp_path: Path) -> None:
    cfg = make_getup_env_cfg(num_envs=4, play=True, balanced_reset=True)
    raw_env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    agent_cfg = load_rl_cfg(GETUP_TASK_ID)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    checkpoint = tmp_path / "model_0.pt"
    try:
        runner_cls = load_runner_cls(GETUP_TASK_ID)
        assert runner_cls is not None
        runner = runner_cls(env, asdict(agent_cfg), device="cpu")
        runner.save(str(checkpoint))
    finally:
        env.close()

    result = evaluate_getup_checkpoint(
        checkpoint,
        num_envs=4,
        num_steps=2,
        hold_seconds=0.02,
        seed=7,
        device="cpu",
    )
    assert result.num_envs == 4
    assert result.sample_count_by_pose == {pose: 1 for pose in GETUP_POSE_CLASSES}
    assert set(result.success_rate_by_pose) == set(GETUP_POSE_CLASSES)
    assert all(0.0 <= rate <= 1.0 for rate in result.success_rate_by_pose.values())
