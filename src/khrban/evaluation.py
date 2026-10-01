"""Bounded, headless evaluation for a trained KHR velocity policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from khrban.evaluation_precision import format_evaluation_values
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from khrban.tasks import KHR_VELOCITY_TASK_ID
from khrban.tasks.velocity import (
    KHR_FOOT_AIR_TIME_MAX_S,
    KHR_FOOT_AIR_TIME_MIN_S,
    KHR_FOOT_CLEARANCE_TARGET_M,
    LOCOMOTION_JOINT_NAMES,
    make_velocity_env_cfg,
)


NO_SAMPLED_METRIC = 1.0e9


def make_velocity_evaluation_env_cfg(
    num_envs: int = 64,
) -> ManagerBasedRlEnvCfg:
    """Build the evaluator at the final command ranges used during training."""

    env_cfg = make_velocity_env_cfg(num_envs=num_envs, play=True)
    curriculum = env_cfg.curriculum.get("microban_style_velocity")
    stages = curriculum.params.get("stages") if curriculum is not None else None
    if not stages:
        raise ValueError("velocity evaluation requires a final training curriculum stage")

    final_stage = stages[-1]
    command = env_cfg.commands["twist"]
    command.ranges.lin_vel_x = final_stage["lin_vel_x"]
    command.ranges.lin_vel_y = final_stage["lin_vel_y"]
    command.ranges.ang_vel_z = final_stage["ang_vel_z"]
    command.rotation_env_ang_vel_range = final_stage[
        "rotation_env_ang_vel_range"
    ]
    command.rel_standing_envs = final_stage["rel_standing_envs"]
    command.rel_rotation_envs = final_stage["rel_rotation_envs"]
    env_cfg.curriculum = {}
    return env_cfg


def rotation_mask_for_command(command_term: object, command: torch.Tensor) -> torch.Tensor:
    """Return the pure-rotation mask used for mode-specific evaluation."""

    explicit_mask = getattr(command_term, "is_rotation_env", None)
    if isinstance(explicit_mask, torch.Tensor) and explicit_mask.shape[:1] == command.shape[:1]:
        return explicit_mask.to(device=command.device, dtype=torch.bool)
    linear_is_zero = torch.linalg.vector_norm(command[:, :2], dim=-1) <= 1.0e-6
    yaw_is_nonzero = command[:, 2].abs() > 1.0e-6
    return linear_is_zero & yaw_is_nonzero


def command_snapshot_for_step(command_term: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Capture the command and mode that produced the next policy action."""

    command = command_term.command.clone()
    rotation_mask = rotation_mask_for_command(command_term, command).clone()
    return command, rotation_mask


@dataclass
class VelocityEvaluationAccumulator:
    """Accumulate aggregate, directional, rotation, and standing metrics."""

    num_envs: int
    device: str | torch.device
    foot_target_height: float = KHR_FOOT_CLEARANCE_TARGET_M
    foot_min_air_time: float = KHR_FOOT_AIR_TIME_MIN_S
    foot_max_air_time: float = KHR_FOOT_AIR_TIME_MAX_S
    foot_command_threshold: float = 0.01
    direction_command_threshold: float = 0.01
    action_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.ever_fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rotation_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rotation_fell = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.linear_squared_error = 0.0
        self.yaw_squared_error = 0.0
        self.direction_squared_error = [0.0, 0.0, 0.0, 0.0]
        self.direction_sample_counts = [0, 0, 0, 0]
        self.rotation_yaw_squared_error = 0.0
        self.rotation_linear_drift_squared_error = 0.0
        self.rotation_sample_count = 0
        self.action_squared_sum = 0.0
        self.action_outside_unit_count = 0
        self.action_element_count = 0
        self.action_sample_count = 0
        self.action_outside_unit_count_by_joint: torch.Tensor | None = None
        self.action_sample_count_by_mode = {
            "standing": 0,
            "translation": 0,
            "rotation": 0,
        }
        self.action_outside_unit_count_by_mode = {
            "standing": 0,
            "translation": 0,
            "rotation": 0,
        }
        self.action_outside_unit_count_by_mode_and_joint: dict[
            str, torch.Tensor | None
        ] = {
            "standing": None,
            "translation": None,
            "rotation": None,
        }
        self.reward_sum = 0.0
        self.foot_peak_heights = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self.foot_landing_count_by_side = torch.zeros(
            2, dtype=torch.int64, device=self.device
        )
        self.foot_lift_success_count_by_side = torch.zeros(
            2, dtype=torch.int64, device=self.device
        )
        self.foot_peak_height_sum = 0.0
        self.foot_air_time_sum = 0.0
        self.standing_sample_count = 0
        self.standing_airborne_foot_count = 0

    def update(
        self,
        *,
        root_lin_vel_b: torch.Tensor,
        root_ang_vel_b: torch.Tensor,
        command: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        rotation_mask: torch.Tensor,
    ) -> None:
        linear_error = root_lin_vel_b[:, :2] - command[:, :2]
        yaw_error = root_ang_vel_b[:, 2] - command[:, 2]
        if actions.ndim != 2 or actions.shape[0] != self.num_envs:
            raise ValueError(
                f"actions must have shape ({self.num_envs}, action_dim), "
                f"got {actions.shape}"
            )
        actions = actions.to(device=self.device)
        action_dim = actions.shape[1]
        if self.action_outside_unit_count_by_joint is None:
            if self.action_names is None:
                self.action_names = tuple(f"action_{index}" for index in range(action_dim))
            elif len(self.action_names) != action_dim:
                raise ValueError(
                    f"action_names has {len(self.action_names)} entries for "
                    f"{action_dim} actions"
                )
            self.action_outside_unit_count_by_joint = torch.zeros(
                action_dim, dtype=torch.int64, device=self.device
            )
            self.action_outside_unit_count_by_mode_and_joint = {
                mode: torch.zeros(action_dim, dtype=torch.int64, device=self.device)
                for mode in self.action_sample_count_by_mode
            }
        elif self.action_outside_unit_count_by_joint.shape != (action_dim,):
            raise ValueError(
                "action dimension changed during evaluation: "
                f"expected {self.action_outside_unit_count_by_joint.numel()}, "
                f"got {action_dim}"
            )

        action_outside_unit = actions.abs() > 1.0
        self.action_squared_sum += float(actions.square().sum().item())
        self.action_outside_unit_count += int(action_outside_unit.sum().item())
        self.action_element_count += actions.numel()
        self.action_sample_count += actions.shape[0]
        self.action_outside_unit_count_by_joint += action_outside_unit.sum(dim=0)
        self.linear_squared_error += float(linear_error.square().sum().item())
        self.yaw_squared_error += float(yaw_error.square().sum().item())
        self.reward_sum += float(rewards.sum().item())
        self.ever_fell |= dones.bool()

        direction_masks = (
            command[:, 0] > self.direction_command_threshold,
            command[:, 0] < -self.direction_command_threshold,
            command[:, 1] > self.direction_command_threshold,
            command[:, 1] < -self.direction_command_threshold,
        )
        direction_axes = (0, 0, 1, 1)
        for index, (mask, axis) in enumerate(zip(direction_masks, direction_axes)):
            count = int(mask.sum().item())
            self.direction_sample_counts[index] += count
            if count:
                self.direction_squared_error[index] += float(
                    linear_error[mask, axis].square().sum().item()
                )

        rotation_mask = rotation_mask.to(device=self.device, dtype=torch.bool)
        command_speed = (
            torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()
        )
        standing_mask = command_speed < self.foot_command_threshold
        action_mode_masks = {
            "standing": standing_mask,
            "translation": ~(standing_mask | rotation_mask),
            "rotation": rotation_mask,
        }
        for mode, mask in action_mode_masks.items():
            count = int(mask.sum().item())
            self.action_sample_count_by_mode[mode] += count
            if count:
                outside_for_mode = action_outside_unit[mask]
                self.action_outside_unit_count_by_mode[mode] += int(
                    outside_for_mode.sum().item()
                )
                by_joint = self.action_outside_unit_count_by_mode_and_joint[mode]
                assert by_joint is not None
                by_joint += outside_for_mode.sum(dim=0)
        if not rotation_mask.any():
            return
        rotation_count = int(rotation_mask.sum().item())
        self.rotation_sample_count += rotation_count
        self.rotation_seen |= rotation_mask
        self.rotation_fell |= rotation_mask & dones.bool()
        self.rotation_yaw_squared_error += float(
            yaw_error[rotation_mask].square().sum().item()
        )
        self.rotation_linear_drift_squared_error += float(
            root_lin_vel_b[rotation_mask, :2].square().sum().item()
        )

    def update_standing(
        self,
        *,
        command: torch.Tensor,
        foot_contacts: torch.Tensor,
    ) -> None:
        """Measure airborne feet while the command requests a stationary stance."""

        foot_contacts = foot_contacts.to(device=self.device, dtype=torch.bool)
        if foot_contacts.ndim == 3:
            foot_contacts = foot_contacts.any(dim=-1)
        expected_shape = (self.num_envs, 2)
        if foot_contacts.shape != expected_shape:
            raise ValueError(
                f"foot_contacts must have shape {expected_shape}, got {foot_contacts.shape}"
            )
        command_speed = (
            torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()
        )
        standing = command_speed < self.foot_command_threshold
        count = int(standing.sum().item())
        self.standing_sample_count += count
        if count:
            self.standing_airborne_foot_count += int(
                (~foot_contacts[standing]).sum().item()
            )

    def update_foot_lifts(
        self,
        *,
        command: torch.Tensor,
        foot_heights: torch.Tensor,
        foot_contacts: torch.Tensor,
        first_contacts: torch.Tensor,
        last_air_time: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Count bilateral landed swings that meet KHR height and air-time targets."""

        foot_contacts = foot_contacts.to(device=self.device, dtype=torch.bool)
        first_contacts = first_contacts.to(device=self.device, dtype=torch.bool)
        if foot_contacts.ndim == 3:
            foot_contacts = foot_contacts.any(dim=-1)
        if first_contacts.ndim == 3:
            first_contacts = first_contacts.any(dim=-1)
        expected_shape = (self.num_envs, 2)
        for name, value in (
            ("foot_heights", foot_heights),
            ("foot_contacts", foot_contacts),
            ("first_contacts", first_contacts),
            ("last_air_time", last_air_time),
        ):
            if value.shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")

        foot_heights = foot_heights.to(device=self.device)
        last_air_time = last_air_time.to(device=self.device)
        dones = dones.to(device=self.device, dtype=torch.bool)
        command_speed = (
            torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()
        )
        active = command_speed > self.foot_command_threshold
        active_feet = active.unsqueeze(1)
        in_air = ~foot_contacts
        self.foot_peak_heights = torch.where(
            active_feet & in_air,
            torch.maximum(self.foot_peak_heights, foot_heights),
            self.foot_peak_heights,
        )

        landing = (
            active_feet
            & first_contacts
            & ~dones.unsqueeze(1)
            & (last_air_time > 0.0)
        )
        self.foot_landing_count_by_side += landing.sum(dim=0)
        self.foot_peak_height_sum += float(self.foot_peak_heights[landing].sum().item())
        self.foot_air_time_sum += float(last_air_time[landing].sum().item())
        successful = (
            landing
            & (self.foot_peak_heights >= self.foot_target_height)
            & (last_air_time >= self.foot_min_air_time)
            & (last_air_time <= self.foot_max_air_time)
        )
        self.foot_lift_success_count_by_side += successful.sum(dim=0)

        reset = first_contacts | dones.unsqueeze(1) | ~active_feet
        self.foot_peak_heights = torch.where(
            reset,
            torch.zeros_like(self.foot_peak_heights),
            self.foot_peak_heights,
        )

    def as_dict(self, *, num_steps: int) -> dict[str, float | int]:
        samples = self.num_envs * num_steps
        landing_count_total = int(self.foot_landing_count_by_side.sum().item())
        foot_landing_count = int(self.foot_landing_count_by_side.min().item())
        if foot_landing_count:
            success_by_side = (
                self.foot_lift_success_count_by_side.float()
                / self.foot_landing_count_by_side.float()
            )
            foot_lift_success_rate = float(success_by_side.min().item())
        else:
            foot_lift_success_rate = 0.0
        if landing_count_total:
            foot_peak_height_mean = self.foot_peak_height_sum / landing_count_total
            foot_air_time_mean = self.foot_air_time_sum / landing_count_total
        else:
            foot_peak_height_mean = 0.0
            foot_air_time_mean = 0.0
        direction_sample_count = min(self.direction_sample_counts)
        if direction_sample_count:
            directional_linear_velocity_rmse = max(
                (error_sum / count) ** 0.5
                for error_sum, count in zip(
                    self.direction_squared_error,
                    self.direction_sample_counts,
                )
            )
        else:
            directional_linear_velocity_rmse = NO_SAMPLED_METRIC
        if self.standing_sample_count:
            standing_foot_airborne_rate = (
                self.standing_airborne_foot_count / (self.standing_sample_count * 2)
            )
        else:
            standing_foot_airborne_rate = 1.0
        if self.rotation_sample_count:
            rotation_fall_denominator = max(int(self.rotation_seen.sum().item()), 1)
            rotation_fall_rate = (
                float(self.rotation_fell.float().sum().item())
                / rotation_fall_denominator
            )
            rotation_yaw_velocity_rmse = (
                self.rotation_yaw_squared_error / self.rotation_sample_count
            ) ** 0.5
            rotation_linear_drift_rmse = (
                self.rotation_linear_drift_squared_error
                / (self.rotation_sample_count * 2)
            ) ** 0.5
        else:
            rotation_fall_rate = 1.0
            rotation_yaw_velocity_rmse = NO_SAMPLED_METRIC
            rotation_linear_drift_rmse = NO_SAMPLED_METRIC
        if self.action_element_count:
            action_rms = (
                self.action_squared_sum / self.action_element_count
            ) ** 0.5
            action_outside_unit_rate = (
                self.action_outside_unit_count / self.action_element_count
            )
        else:
            action_rms = NO_SAMPLED_METRIC
            action_outside_unit_rate = 1.0

        if self.action_outside_unit_count_by_joint is not None:
            assert self.action_names is not None
            joint_denominator = max(self.action_sample_count, 1)
            action_outside_unit_rate_by_joint = {
                name: float(count) / joint_denominator
                for name, count in zip(
                    self.action_names,
                    self.action_outside_unit_count_by_joint.tolist(),
                )
            }
            action_outside_unit_rate_by_mode = {}
            action_outside_unit_rate_by_mode_and_joint = {}
            for mode, sample_count in self.action_sample_count_by_mode.items():
                aggregate_denominator = sample_count * len(self.action_names)
                action_outside_unit_rate_by_mode[mode] = (
                    self.action_outside_unit_count_by_mode[mode]
                    / aggregate_denominator
                    if aggregate_denominator
                    else 0.0
                )
                by_joint = self.action_outside_unit_count_by_mode_and_joint[mode]
                assert by_joint is not None
                action_outside_unit_rate_by_mode_and_joint[mode] = {
                    name: float(count) / sample_count if sample_count else 0.0
                    for name, count in zip(self.action_names, by_joint.tolist())
                }
        else:
            action_outside_unit_rate_by_joint = {}
            action_outside_unit_rate_by_mode = {
                mode: 0.0 for mode in self.action_sample_count_by_mode
            }
            action_outside_unit_rate_by_mode_and_joint = {
                mode: {} for mode in self.action_sample_count_by_mode
            }

        return {
            "fall_rate": float(self.ever_fell.float().mean().item()),
            "linear_velocity_rmse": (
                self.linear_squared_error / (samples * 2)
            )
            ** 0.5,
            "yaw_velocity_rmse": (self.yaw_squared_error / samples) ** 0.5,
            "directional_linear_velocity_rmse": directional_linear_velocity_rmse,
            "direction_sample_count": direction_sample_count,
            "rotation_fall_rate": rotation_fall_rate,
            "rotation_yaw_velocity_rmse": rotation_yaw_velocity_rmse,
            "rotation_linear_drift_rmse": rotation_linear_drift_rmse,
            "rotation_sample_count": self.rotation_sample_count,
            "foot_lift_success_rate": foot_lift_success_rate,
            "foot_peak_height_mean": foot_peak_height_mean,
            "foot_air_time_mean": foot_air_time_mean,
            "foot_landing_count": foot_landing_count,
            "standing_foot_airborne_rate": standing_foot_airborne_rate,
            "standing_sample_count": self.standing_sample_count,
            "action_rms": action_rms,
            "action_outside_unit_rate": action_outside_unit_rate,
            "action_outside_unit_rate_by_joint": action_outside_unit_rate_by_joint,
            "action_sample_count_by_mode": self.action_sample_count_by_mode,
            "action_outside_unit_rate_by_mode": action_outside_unit_rate_by_mode,
            "action_outside_unit_rate_by_mode_and_joint": (
                action_outside_unit_rate_by_mode_and_joint
            ),
            "mean_reward": self.reward_sum / samples,
            "num_envs": self.num_envs,
            "num_steps": num_steps,
        }


def evaluate_velocity_checkpoint(
    checkpoint: Path,
    *,
    num_envs: int = 64,
    num_steps: int = 500,
    seed: int = 42,
    device: str | None = None,
) -> dict[str, float | int]:
    """Evaluate command tracking and falls without training the policy."""

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if num_envs < 1 or num_steps < 1:
        raise ValueError("num_envs and num_steps must be positive")

    configure_torch_backends()
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = make_velocity_evaluation_env_cfg(num_envs=num_envs)
    env_cfg.seed = seed
    agent_cfg = load_rl_cfg(KHR_VELOCITY_TASK_ID)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=resolved_device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(KHR_VELOCITY_TASK_ID) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=resolved_device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=resolved_device,
        )
        policy = runner.get_inference_policy(device=resolved_device)
        observations = env.get_observations()
        accumulator = VelocityEvaluationAccumulator(
            num_envs=num_envs,
            device=resolved_device,
            action_names=LOCOMOTION_JOINT_NAMES,
        )
        foot_contact_sensor = raw_env.scene["feet_ground_contact"]
        foot_height_sensor = raw_env.scene["foot_height_scan"]

        with torch.inference_mode():
            for _ in range(num_steps):
                command_term = raw_env.command_manager.get_term("twist")
                command, rotation_mask = command_snapshot_for_step(command_term)
                actions = policy(observations)
                observations, rewards, dones, _ = env.step(actions)
                robot = raw_env.scene["robot"]
                accumulator.update(
                    root_lin_vel_b=robot.data.root_link_lin_vel_b,
                    root_ang_vel_b=robot.data.root_link_ang_vel_b,
                    command=command,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    rotation_mask=rotation_mask,
                )
                last_air_time = foot_contact_sensor.data.last_air_time
                if last_air_time is None:
                    raise RuntimeError("feet_ground_contact does not track air time")
                accumulator.update_foot_lifts(
                    command=command,
                    foot_heights=foot_height_sensor.data.heights,
                    foot_contacts=foot_contact_sensor.data.found,
                    first_contacts=foot_contact_sensor.compute_first_contact(
                        dt=raw_env.step_dt
                    ),
                    last_air_time=last_air_time,
                    dones=dones,
                )
                accumulator.update_standing(
                    command=command,
                    foot_contacts=foot_contact_sensor.data.found,
                )

        return accumulator.as_dict(num_steps=num_steps)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a KHR velocity checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_velocity_checkpoint(
        args.checkpoint,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        seed=args.seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            format_evaluation_values(result),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
