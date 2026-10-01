"""Replay a trained KHR velocity policy with Microban-style keyboard input."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Mapping

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.base import VerbosityLevel
from mjlab.viewer.native.keys import (
    KEY_DOWN,
    KEY_LEFT,
    KEY_Q,
    KEY_R,
    KEY_RIGHT,
    KEY_UP,
    KEY_V,
    KEY_X,
)

from khrban.training_viewer import DEFAULT_LOG_ROOT, resolve_training_checkpoint


BUNDLED_WALKING_POLICY = (
    Path(__file__).resolve().parents[2] / "policies/velocity/model_179910.pt"
)


def resolve_keyboard_checkpoint(
    checkpoint: Path | None,
    log_root: Path = DEFAULT_LOG_ROOT,
    bundled_policy: Path = BUNDLED_WALKING_POLICY,
) -> Path:
    """Prefer an explicit or local-training checkpoint, then the bundled policy."""

    if checkpoint is not None:
        return checkpoint.resolve()
    try:
        return resolve_training_checkpoint(log_root)
    except FileNotFoundError:
        if bundled_policy.is_file():
            return bundled_policy
        raise


VELOCITY_TASK_ID = "Mjlab-KHR-Velocity-Flat"
VELOCITY_STEP = 0.1
VELOCITY_MAX = 1.0

# These are the physical limits used by Rhoban/microban's scale_velocity().
# They are also the final KHR velocity curriculum ranges.
VX_MAX = 0.7
VX_MAX_BACKWARD = 0.5
VY_MAX = 0.3
VTHETA_MAX_STATIONARY = 3.0
VTHETA_MAX_MOVING = 1.5


def scaled_velocity_command(
    velocity: Mapping[str, float],
) -> tuple[float, float, float]:
    """Map Microban's normalized command to physical velocity units."""

    vx = max(-1.0, min(1.0, float(velocity.get("vx", 0.0))))
    vy = max(-1.0, min(1.0, float(velocity.get("vy", 0.0))))
    vtheta = max(-1.0, min(1.0, float(velocity.get("vtheta", 0.0))))

    moving = abs(vx) > 1.0e-6 or abs(vy) > 1.0e-6
    vx_max = VX_MAX if vx >= 0.0 else VX_MAX_BACKWARD
    vtheta_max = VTHETA_MAX_MOVING if moving else VTHETA_MAX_STATIONARY
    return vx * vx_max, vy * VY_MAX, vtheta * vtheta_max


class KeyboardVelocityState:
    """Thread-safe Microban-compatible keyboard command state."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._policy_enabled = False
        self._velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}

    @property
    def policy_enabled(self) -> bool:
        with self._lock:
            return self._policy_enabled

    @property
    def normalized_velocity(self) -> dict[str, float]:
        with self._lock:
            return dict(self._velocity)

    @property
    def command(self) -> tuple[float, float, float]:
        with self._lock:
            if not self._policy_enabled:
                return 0.0, 0.0, 0.0
            return scaled_velocity_command(self._velocity)

    def handle_key(self, keycode: int) -> str | None:
        """Update state from a MuJoCo/GLFW keycode and return the event name."""

        with self._lock:
            if keycode == KEY_V:
                self._policy_enabled = not self._policy_enabled
                return "policy_enabled" if self._policy_enabled else "policy_disabled"
            if keycode == KEY_X:
                self._velocity = {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}
                return "velocity_zero"
            if keycode == KEY_R:
                return "reset"
            if keycode == KEY_Q:
                return "quit"
            if keycode == KEY_UP:
                self._adjust_velocity("vx", VELOCITY_STEP)
                return "velocity_changed"
            if keycode == KEY_DOWN:
                self._adjust_velocity("vx", -VELOCITY_STEP)
                return "velocity_changed"
            if keycode == KEY_RIGHT:
                self._adjust_velocity("vtheta", VELOCITY_STEP)
                return "velocity_changed"
            if keycode == KEY_LEFT:
                self._adjust_velocity("vtheta", -VELOCITY_STEP)
                return "velocity_changed"
        return None

    def _adjust_velocity(self, axis: str, delta: float) -> None:
        value = self._velocity[axis] + delta
        self._velocity[axis] = max(-VELOCITY_MAX, min(VELOCITY_MAX, value))


def apply_keyboard_command(env: ManagerBasedRlEnv, state: KeyboardVelocityState) -> None:
    """Write the current command into every playback environment."""

    term = env.command_manager.get_term("twist")
    command = torch.tensor(state.command, device=env.device)
    term.vel_command_b[:, :] = command
    if hasattr(term, "vel_command_w"):
        term.vel_command_w[:, :] = 0.0
    for mask_name in (
        "is_standing_env",
        "is_heading_env",
        "is_world_env",
        "is_rotation_env",
    ):
        mask = getattr(term, mask_name, None)
        if mask is not None:
            mask[:] = False
    term.time_left[:] = 1.0e9


class _GatedPolicy:
    def __init__(self, policy, state: KeyboardVelocityState, env) -> None:
        self._policy = policy
        self._state = state
        self._action_shape = env.unwrapped.action_space.shape
        self._device = env.unwrapped.device

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        if self._state.policy_enabled:
            return self._policy(observation)
        return torch.zeros(self._action_shape, device=self._device)


class MicrobanKeyboardViewer(NativeMujocoViewer):
    """Native viewer that applies Microban's keyboard meanings to KHR."""

    def __init__(self, env, policy, state: KeyboardVelocityState) -> None:
        super().__init__(
            env,
            _GatedPolicy(policy, state, env),
            frame_rate=60.0,
            key_callback=None,
            verbosity=VerbosityLevel.INFO,
        )
        self._keyboard_state = state
        self._quit_requested = False

    def is_running(self) -> bool:
        return not self._quit_requested and super().is_running()

    def _execute_step(self) -> bool:
        apply_keyboard_command(self.env.unwrapped, self._keyboard_state)
        return super()._execute_step()

    def _safe_key_callback(self, key: int) -> None:
        """Queue simulation actions; never mutate MuJoCo from its viewer thread."""

        event = self._keyboard_state.handle_key(key)
        if event is None:
            super()._safe_key_callback(key)
            return
        if event == "reset":
            self.request_reset()
            print("KHRを初期姿勢へリセットします")
        elif event == "quit":
            self._quit_requested = True
            print("KHRポリシー再生を終了します")
        elif event == "policy_enabled":
            print("学習ポリシー: 有効")
        elif event == "policy_disabled":
            print("学習ポリシー: 無効（ホーム姿勢指令）")
        elif event == "velocity_zero":
            print("速度指令: vx=0.00 m/s, yaw=0.00 rad/s")
        else:
            vx, _, yaw = self._keyboard_state.command
            print(f"速度指令: vx={vx:+.2f} m/s, yaw={yaw:+.2f} rad/s")


def run_keyboard_policy(checkpoint: Path, device: str = "cpu") -> None:
    """Load one trained KHR policy and run it in the native MuJoCo viewer."""

    configure_torch_backends()

    # Importing the task package registers the KHR task IDs with MjLab.
    import khrban.tasks  # noqa: F401

    env_cfg = load_env_cfg(VELOCITY_TASK_ID, play=True)
    agent_cfg = load_rl_cfg(VELOCITY_TASK_ID)
    env_cfg.scene.num_envs = 1

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(VELOCITY_TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    state = KeyboardVelocityState()
    viewer = MicrobanKeyboardViewer(wrapped_env, policy, state)

    print("Microban互換 KHRキーボード操作:")
    print("  [V]          学習ポリシー 有効/無効")
    print("  [↑]/[↓]      前進/後進速度を0.1段階で変更")
    print("  [←]/[→]      左/右旋回速度を0.1段階で変更")
    print("  [X]          速度指令をゼロ")
    print("  [R]          KHRを初期姿勢へリセット")
    print("  [Q]          終了")
    print("MuJoCoウィンドウへフォーカスし、最初に[V]を押してください。")

    try:
        viewer.run()
    finally:
        wrapped_env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="microbanと同じキー操作でKHR学習ポリシーを再生します"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--print-checkpoint", action="store_true")
    args = parser.parse_args()

    checkpoint = resolve_keyboard_checkpoint(args.checkpoint, args.log_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"チェックポイントがありません: {checkpoint}")
    if args.print_checkpoint:
        print(checkpoint)
        return

    print(f"KHR学習ポリシー: {checkpoint}")
    run_keyboard_policy(checkpoint, device=args.device)


if __name__ == "__main__":
    main()
