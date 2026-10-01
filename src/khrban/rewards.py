"""KHR-specific locomotion rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


def feet_distance_penalty(
    env: "ManagerBasedRlEnv",
    min_dist: float,
    asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Return the horizontal foot-distance shortfall in meters."""

    asset = env.scene[asset_cfg.name]
    positions = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
    if positions.shape[1:] != (2, 2):
        raise ValueError("feet_distance_penalty requires exactly two foot sites")
    distance = torch.linalg.vector_norm(positions[:, 0] - positions[:, 1], dim=-1)
    return torch.clamp(min_dist - distance, min=0.0)


def no_stepping_penalty(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    command_name: str,
    command_threshold: float,
) -> torch.Tensor:
    """Count airborne feet only when the requested planar motion is near zero."""

    command = env.command_manager.get_command(command_name)
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()
    found = env.scene.sensors[sensor_name].data.found
    if found.ndim == 3:
        found = found.any(dim=-1)
    return (~found.bool()).float().sum(dim=-1) * (speed < command_threshold)


def excessive_air_time_penalty(
    env: "ManagerBasedRlEnv",
    sensor_name: str,
    command_name: str,
    threshold_max: float,
    command_threshold: float,
) -> torch.Tensor:
    """Count feet kept airborne beyond the accepted swing-time window."""

    current_air_time = env.scene.sensors[sensor_name].data.current_air_time
    if current_air_time is None:
        raise RuntimeError(f"{sensor_name} does not track air time")
    command = env.command_manager.get_command(command_name)
    speed = torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs()
    overlong = (current_air_time > threshold_max).float().sum(dim=-1)
    return overlong * (speed > command_threshold)


def action_target_excess_l2(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Penalize only the raw policy amount hidden beyond the target clip."""

    excess = torch.relu(torch.abs(env.action_manager.action) - 1.0)
    return torch.sum(torch.square(excess), dim=1)
