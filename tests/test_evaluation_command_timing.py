"""Regression tests for the command used by a saved-policy evaluation step."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

import khrban.evaluation as evaluation


def test_velocity_evaluation_uses_the_command_seen_by_the_action(
    monkeypatch, tmp_path
) -> None:
    checkpoint = tmp_path / "model_1.pt"
    checkpoint.write_bytes(b"fake checkpoint")
    captured = {}
    command_term = SimpleNamespace(
        command=torch.tensor([[0.5, 0.0, 0.0]]),
        is_rotation_env=torch.tensor([False]),
    )
    foot_contact = SimpleNamespace(
        data=SimpleNamespace(
            last_air_time=torch.zeros((1, 2)),
            found=torch.ones((1, 2), dtype=torch.bool),
        ),
        compute_first_contact=lambda dt: torch.zeros((1, 2), dtype=torch.bool),
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_lin_vel_b=torch.tensor([[0.5, 0.0, 0.0]]),
            root_link_ang_vel_b=torch.zeros((1, 3)),
        )
    )

    class FakeRawEnv:
        def __init__(self, *, cfg, device):
            self.scene = {
                "robot": robot,
                "feet_ground_contact": foot_contact,
                "foot_height_scan": SimpleNamespace(
                    data=SimpleNamespace(heights=torch.zeros((1, 2)))
                ),
            }
            self.command_manager = SimpleNamespace(get_term=lambda name: command_term)
            self.step_dt = 0.02

        def close(self):
            pass

    class FakeWrapper:
        def __init__(self, raw_env, clip_actions):
            self.raw_env = raw_env

        def close(self):
            self.raw_env.close()

        def get_observations(self):
            return None

        def step(self, actions):
            # MuJoCo advances under the old command, then resamples the next one.
            command_term.command[:] = torch.tensor([[0.0, 0.0, 1.0]])
            command_term.is_rotation_env[:] = True
            return None, torch.ones(1), torch.zeros(1, dtype=torch.bool), {}

    class FakeRunner:
        def __init__(self, env, cfg, device):
            pass

        def load(self, *args, **kwargs):
            pass

        def get_inference_policy(self, *, device):
            return lambda observations: torch.zeros((1, 18))

    class FakeAccumulator:
        def __init__(self, **kwargs):
            pass

        def update(self, **kwargs):
            captured["velocity_command"] = kwargs["command"].clone()
            captured["rotation_mask"] = kwargs["rotation_mask"].clone()

        def update_foot_lifts(self, **kwargs):
            captured["foot_command"] = kwargs["command"].clone()

        def update_standing(self, **kwargs):
            captured["standing_command"] = kwargs["command"].clone()

        def as_dict(self, *, num_steps):
            return {}

    @dataclass
    class FakeAgentCfg:
        clip_actions: float | None = None

    monkeypatch.setattr(evaluation, "configure_torch_backends", lambda: None)
    monkeypatch.setattr(
        evaluation,
        "make_velocity_evaluation_env_cfg",
        lambda num_envs: SimpleNamespace(seed=0),
    )
    monkeypatch.setattr(evaluation, "load_rl_cfg", lambda task: FakeAgentCfg())
    monkeypatch.setattr(evaluation, "load_runner_cls", lambda task: FakeRunner)
    monkeypatch.setattr(evaluation, "ManagerBasedRlEnv", FakeRawEnv)
    monkeypatch.setattr(evaluation, "RslRlVecEnvWrapper", FakeWrapper)
    monkeypatch.setattr(evaluation, "VelocityEvaluationAccumulator", FakeAccumulator)

    evaluation.evaluate_velocity_checkpoint(
        checkpoint, num_envs=1, num_steps=1, device="cpu"
    )

    old_command = torch.tensor([[0.5, 0.0, 0.0]])
    assert torch.equal(captured["velocity_command"], old_command)
    assert torch.equal(captured["foot_command"], old_command)
    assert torch.equal(captured["standing_command"], old_command)
    assert torch.equal(captured["rotation_mask"], torch.tensor([False]))
