import math

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from khrban.model import sample_motion_home_joint_positions
from khrban.tasks.standing import make_standing_env_cfg, make_standing_ppo_cfg


def test_standing_task_contract() -> None:
    home = sample_motion_home_joint_positions()
    env_cfg = make_standing_env_cfg(num_envs=32)
    ppo_cfg = make_standing_ppo_cfg()

    assert len(home) == 22
    assert env_cfg.scene.num_envs == 32
    assert math.isclose(env_cfg.sim.mujoco.timestep, 0.005)
    assert env_cfg.decimation == 4
    assert math.isclose(env_cfg.sim.mujoco.timestep * env_cfg.decimation, 0.02)
    assert set(env_cfg.observations) == {"actor", "critic"}
    for group in env_cfg.observations.values():
        assert "leg_flexion_ratio" in group.terms
    assert set(env_cfg.actions) == {"joint_pos"}
    assert set(env_cfg.metrics) == {
        "left_leg_flexion_deg",
        "right_leg_flexion_deg",
    }
    assert {"alive", "upright", "posture", "joint_velocity", "action_rate"} <= set(
        env_cfg.rewards
    )
    assert {"time_out", "fell_over"} == set(env_cfg.terminations)
    assert ppo_cfg.experiment_name == "khr_standing"
    assert ppo_cfg.logger == "tensorboard"
    assert ppo_cfg.clip_actions == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="standing task requires CUDA")
def test_standing_environment_steps_on_gpu() -> None:
    env = ManagerBasedRlEnv(
        cfg=make_standing_env_cfg(num_envs=2, play=True),
        device="cuda",
    )
    try:
        observations, _ = env.reset(seed=42)
        assert observations["actor"].shape == (2, 77)

        actions = torch.zeros((2, 22), device="cuda")
        ever_terminated = torch.zeros(2, dtype=torch.bool, device="cuda")
        ever_truncated = torch.zeros(2, dtype=torch.bool, device="cuda")
        for _ in range(100):
            observations, rewards, terminated, truncated, _ = env.step(actions)
            ever_terminated |= terminated
            ever_truncated |= truncated

        assert torch.isfinite(observations["actor"]).all()
        assert torch.isfinite(rewards).all()
        assert not ever_terminated.any()
        assert not ever_truncated.any()
        metric_values = dict(env.metrics_manager.get_active_iterable_terms(0))
        assert set(metric_values) == {
            "left_leg_flexion_deg",
            "right_leg_flexion_deg",
        }
        assert all(0.0 <= values[0] <= 180.0 for values in metric_values.values())
    finally:
        env.close()
