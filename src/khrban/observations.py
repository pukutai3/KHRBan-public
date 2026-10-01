"""KHR-specific geometric observations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


LEG_PITCH_AXIS_BODY_NAMES = (
    "l_upperreg_1",
    "l_lowerreg_1",
    "l_ankle_1",
    "r_upperreg_1",
    "r_lowerreg_1",
    "r_ankle_1",
)


def leg_flexion_angle_from_axis_positions(
    axis_positions: torch.Tensor,
    *,
    validate_link_lengths: bool = True,
) -> torch.Tensor:
    """Return the A-B angle in radians for each leg.

    axis_positions has shape (..., 2, 3, 3). The last dimensions are leg,
    pitch axis (hip, knee, ankle), and xyz. Link A points from hip to knee and
    link B points from knee to ankle, so aligned extended links yield zero and
    links folded back on themselves yield pi.
    """

    if axis_positions.shape[-3:] != (2, 3, 3):
        raise ValueError(
            "axis_positions must end with (2 legs, 3 pitch axes, xyz)"
        )

    link_a = axis_positions[..., 1, :] - axis_positions[..., 0, :]
    link_b = axis_positions[..., 2, :] - axis_positions[..., 1, :]
    length_a = torch.linalg.vector_norm(link_a, dim=-1)
    length_b = torch.linalg.vector_norm(link_b, dim=-1)
    epsilon = torch.finfo(axis_positions.dtype).eps

    if validate_link_lengths and torch.any(
        (length_a <= epsilon) | (length_b <= epsilon)
    ).item():
        raise ValueError("A and B links must have non-zero length")

    denominator = (length_a * length_b).clamp_min(epsilon)
    cosine = torch.sum(link_a * link_b, dim=-1) / denominator
    return torch.acos(cosine.clamp(-1.0, 1.0))


def leg_flexion_ratio_from_axis_positions(
    axis_positions: torch.Tensor,
    *,
    validate_link_lengths: bool = True,
) -> torch.Tensor:
    """Return left/right A-B angles normalized from 0 to 1."""

    angles = leg_flexion_angle_from_axis_positions(
        axis_positions,
        validate_link_lengths=validate_link_lengths,
    )
    return (angles / math.pi).clamp(0.0, 1.0)


def _leg_axis_positions(
    env: "ManagerBasedRlEnv",
    asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Read the six pitch-axis origins and group them as two three-axis legs."""

    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    if positions.shape[1:] != (6, 3):
        raise ValueError("leg pitch-axis selection must resolve to six xyz positions")
    return positions.reshape(positions.shape[0], 2, 3, 3)


def leg_flexion_ratio(
    env: "ManagerBasedRlEnv",
    asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Return measured left/right flexion ratios for policy observation."""

    return leg_flexion_ratio_from_axis_positions(
        _leg_axis_positions(env, asset_cfg),
        validate_link_lengths=False,
    )


def leg_flexion_angle_deg(
    env: "ManagerBasedRlEnv",
    asset_cfg: "SceneEntityCfg",
    leg_index: int,
) -> torch.Tensor:
    """Return one measured A-B angle in degrees for human-readable metrics."""

    if leg_index not in (0, 1):
        raise ValueError("leg_index must be 0 for left or 1 for right")
    ratio = leg_flexion_ratio(env, asset_cfg)
    return ratio[:, leg_index] * 180.0
