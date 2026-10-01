from pathlib import Path
import inspect

import pytest
from mjlab.viewer import NativeMujocoViewer

from khrban.keyboard_policy import (
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
    KEY_V,
    KEY_X,
    KeyboardVelocityState,
    MicrobanKeyboardViewer,
    apply_keyboard_command,
    scaled_velocity_command,
)


def test_keyboard_playback_is_locked_to_microban_native_mujoco_path() -> None:
    assert issubclass(MicrobanKeyboardViewer, NativeMujocoViewer)
    assert "mujoco.viewer.launch_passive" in inspect.getsource(
        NativeMujocoViewer.setup
    )

    script = (
        Path(__file__).resolve().parents[1] / "tools/windows/KHRBan-GUI.ps1"
    ).read_text()
    launch_line = next(
        line
        for line in script.splitlines()
        if ".venv/bin/python -m khrban.keyboard_policy" in line
    )
    assert "--viewer" not in launch_line
    assert "http://" not in launch_line


def test_microban_velocity_scaling_is_preserved() -> None:
    assert scaled_velocity_command({"vx": 1.0, "vy": 0.0, "vtheta": 0.0}) == (
        0.7,
        0.0,
        0.0,
    )
    assert scaled_velocity_command({"vx": -1.0, "vy": 0.0, "vtheta": 0.0}) == (
        -0.5,
        0.0,
        0.0,
    )
    assert scaled_velocity_command({"vx": 0.0, "vy": 1.0, "vtheta": 1.0}) == (
        0.0,
        0.3,
        1.5,
    )
    assert scaled_velocity_command({"vx": 0.0, "vy": 0.0, "vtheta": 1.0}) == (
        0.0,
        0.0,
        3.0,
    )


def test_microban_keyboard_meanings_and_clamping_are_preserved() -> None:
    state = KeyboardVelocityState()
    assert not state.policy_enabled

    assert state.handle_key(KEY_V) == "policy_enabled"
    assert state.policy_enabled

    for _ in range(20):
        state.handle_key(KEY_UP)
        state.handle_key(KEY_RIGHT)
    assert state.normalized_velocity == {"vx": 1.0, "vy": 0.0, "vtheta": 1.0}
    assert state.command == pytest.approx((0.7, 0.0, 1.5))

    state.handle_key(KEY_DOWN)
    state.handle_key(KEY_LEFT)
    assert state.normalized_velocity == pytest.approx(
        {"vx": 0.9, "vy": 0.0, "vtheta": 0.9}
    )

    assert state.handle_key(KEY_X) == "velocity_zero"
    assert state.command == (0.0, 0.0, 0.0)


def test_control_desk_exposes_independent_keyboard_policy_playback() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "tools/windows/KHRBan-GUI.ps1"
    ).read_text()

    assert "キーボード操作" in script
    assert "python -m khrban.keyboard_policy" in script
    assert "学習と独立" in script


def test_keyboard_command_overrides_resampling_masks() -> None:
    import torch

    class Term:
        vel_command_b = torch.ones((2, 3))
        vel_command_w = torch.ones((2, 3))
        is_standing_env = torch.ones(2, dtype=torch.bool)
        is_heading_env = torch.ones(2, dtype=torch.bool)
        is_world_env = torch.ones(2, dtype=torch.bool)
        is_rotation_env = torch.ones(2, dtype=torch.bool)
        time_left = torch.zeros(2)

    class CommandManager:
        @staticmethod
        def get_term(name: str):
            assert name == "twist"
            return Term

    class Env:
        device = "cpu"
        command_manager = CommandManager()

    state = KeyboardVelocityState()
    state.handle_key(KEY_V)
    state.handle_key(KEY_UP)
    state.handle_key(KEY_RIGHT)
    apply_keyboard_command(Env(), state)

    torch.testing.assert_close(
        Term.vel_command_b,
        torch.tensor([[0.07, 0.0, 0.15], [0.07, 0.0, 0.15]]),
    )
    assert not Term.vel_command_w.any()
    assert not Term.is_standing_env.any()
    assert not Term.is_heading_env.any()
    assert not Term.is_world_env.any()
    assert not Term.is_rotation_env.any()
    assert torch.all(Term.time_left == 1.0e9)
