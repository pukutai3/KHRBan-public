"""KHR-specific stand-up task based on the KHR velocity-training recipe."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul

from khrban.model import sample_motion_home_joint_positions, sample_motion_joint_limits
from khrban.rewards import action_target_excess_l2
from khrban.tasks.velocity import (
    FROZEN_POLICY_JOINT_NAMES,
    KHR_FOOT_GEOM_NAMES,
    KHR_TORSO_BODY_NAME,
    LOCOMOTION_JOINT_NAMES,
    make_velocity_env_cfg,
    make_velocity_ppo_cfg,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


GETUP_TASK_ID = "Mjlab-KHR-GetUp-Flat"
GETUP_POSE_CLASSES = ("face_down", "face_up", "left_side", "right_side")

# Root-link heights were measured with the actual KHR collision meshes, HTH4 home
# joint angles, 0 yaw, and a MuJoCo ground plane. A 2 mm release clearance lets
# the first 20 ms simulation step settle onto the floor without starting interpenetrating.
GETUP_ROOT_HEIGHT_BY_POSE_M = {
    "face_down": 0.083,
    "face_up": 0.070,
    "left_side": 0.122,
    "right_side": 0.103,
}
KHR_HOME_TORSO_HEIGHT_M = 0.30475
GETUP_SUCCESS_TORSO_HEIGHT_M = 0.270
GETUP_UPRIGHT_COS_THRESHOLD = math.cos(math.radians(20.0))
GETUP_HOLD_SECONDS = 1.0
GETUP_EPISODE_SECONDS = 12.0


def getup_action_scale() -> dict[str, float]:
    """Scale active KHR joints to the confirmed sample-motion envelope."""

    home = sample_motion_home_joint_positions()
    limits = sample_motion_joint_limits()
    return {
        name: max(home[name] - limits[name][0], limits[name][1] - home[name])
        for name in LOCOMOTION_JOINT_NAMES
    }


def reset_khr_fall_pose(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    pose_class: str | None = None,
    balanced: bool = False,
) -> None:
    """Place the KHR in one of four measured floor-contacting fall orientations."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
    asset = env.scene["robot"]

    if pose_class is not None:
        if pose_class not in GETUP_POSE_CLASSES:
            raise ValueError(f"Unknown KHR get-up pose class: {pose_class}")
        class_ids = torch.full(
            (len(env_ids),),
            GETUP_POSE_CLASSES.index(pose_class),
            dtype=torch.long,
            device=env.device,
        )
    elif balanced:
        class_ids = torch.arange(len(env_ids), device=env.device) % len(
            GETUP_POSE_CLASSES
        )
    else:
        class_ids = torch.randint(
            len(GETUP_POSE_CLASSES), (len(env_ids),), device=env.device
        )

    # Initialize lazily because reset events run during environment construction.
    if not hasattr(env, "getup_pose_class_ids"):
        env.getup_pose_class_ids = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=env.device
        )
    env.getup_pose_class_ids[env_ids] = class_ids

    default_state = asset.data.default_root_state[env_ids].clone()
    positions = default_state[:, :3] + env.scene.env_origins[env_ids]
    height_by_class = torch.tensor(
        [GETUP_ROOT_HEIGHT_BY_POSE_M[name] for name in GETUP_POSE_CLASSES],
        device=env.device,
        dtype=positions.dtype,
    )
    positions[:, 2] = env.scene.env_origins[env_ids, 2] + height_by_class[class_ids]

    roll_values = torch.tensor(
        [0.0, 0.0, -math.pi / 2.0, math.pi / 2.0],
        device=env.device,
        dtype=positions.dtype,
    )
    pitch_values = torch.tensor(
        [math.pi / 2.0, -math.pi / 2.0, 0.0, 0.0],
        device=env.device,
        dtype=positions.dtype,
    )
    zeros = torch.zeros(len(env_ids), device=env.device, dtype=positions.dtype)
    orientations_delta = quat_from_euler_xyz(
        roll_values[class_ids], pitch_values[class_ids], zeros
    )
    orientations = quat_mul(default_state[:, 3:7], orientations_delta)
    root_state = torch.cat(
        (
            positions,
            orientations,
            torch.zeros((len(env_ids), 6), device=env.device, dtype=positions.dtype),
        ),
        dim=-1,
    )
    asset.write_root_state_to_sim(root_state, env_ids=env_ids)


def torso_height_observation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return torso height for the privileged critic, not the deployable actor."""

    if asset_cfg.body_names is not None and not isinstance(asset_cfg.body_ids, list):
        asset_cfg.resolve(env.scene)
    asset = env.scene[asset_cfg.name]
    torso_height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(-1)
    return (torso_height - env.scene.env_origins[:, 2]).unsqueeze(-1)


def _foot_contact_mask(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    found = env.scene[sensor_name].data.found
    if found is None:
        raise RuntimeError(f"{sensor_name} does not report contact state")
    if found.ndim == 3:
        found = found.any(dim=-1)
    if found.ndim != 2 or found.shape[1] != 2:
        raise ValueError(
            "KHR get-up evaluation requires one contact state for each foot; "
            f"got shape {tuple(found.shape)}"
        )
    return found.bool()


def getup_stable_mask(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    sensor_name: str,
) -> torch.Tensor:
    """Require measured KHR torso height, uprightness, and both planted feet."""

    asset = env.scene[asset_cfg.name]
    torso_height = torso_height_observation(env, asset_cfg).squeeze(-1)
    upright_cos = -asset.data.projected_gravity_b[:, 2]
    feet_planted = _foot_contact_mask(env, sensor_name).all(dim=-1)
    return (
        (torso_height >= GETUP_SUCCESS_TORSO_HEIGHT_M)
        & (upright_cos >= GETUP_UPRIGHT_COS_THRESHOLD)
        & feet_planted
    )


def getup_upright_reward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Shape a continuous reward toward the world-upright orientation."""

    asset = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0, max=1.0)


def getup_torso_height_reward(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    minimum_height_m: float = 0.05,
) -> torch.Tensor:
    """Reward progress toward the measured KHR home-pose torso height."""

    height = torso_height_observation(env, asset_cfg).squeeze(-1)
    return torch.clamp(
        (height - minimum_height_m) / (KHR_HOME_TORSO_HEIGHT_M - minimum_height_m),
        min=0.0,
        max=1.0,
    )


def getup_foot_support_reward(
    env: ManagerBasedRlEnv,
    *,
    asset_cfg: SceneEntityCfg,
    sensor_name: str,
) -> torch.Tensor:
    """Reward planted feet only as the torso approaches upright."""

    upright = getup_upright_reward(env, asset_cfg)
    planted_count = _foot_contact_mask(env, sensor_name).float().sum(dim=-1)
    return planted_count * upright.square()


def joint_home_error_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize unnecessary departure from the confirmed KHR home pose."""

    asset = env.scene[asset_cfg.name]
    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(error), dim=-1)


def make_getup_env_cfg(
    num_envs: int = 1024,
    play: bool = False,
    initial_pose_class: str | None = None,
    balanced_reset: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create a KHR stand-up task while retaining the velocity task's sim2real base."""

    if initial_pose_class is not None and initial_pose_class not in GETUP_POSE_CLASSES:
        raise ValueError(f"Unknown KHR get-up pose class: {initial_pose_class}")
    if balanced_reset and num_envs % len(GETUP_POSE_CLASSES) != 0:
        raise ValueError(
            f"balanced_reset requires num_envs to be divisible by "
            f"{len(GETUP_POSE_CLASSES)}"
        )

    cfg = make_velocity_env_cfg(num_envs=num_envs, play=play)
    cfg.scene.sensors = tuple(
        sensor
        for sensor in (cfg.scene.sensors or ())
        if sensor.name == "feet_ground_contact"
    )
    active_joints = SceneEntityCfg(
        "robot", joint_names=LOCOMOTION_JOINT_NAMES, preserve_order=True
    )
    torso = SceneEntityCfg(
        "robot", body_names=(KHR_TORSO_BODY_NAME,), preserve_order=True
    )

    cfg.commands = {}
    cfg.observations["actor"].terms.pop("command", None)
    critic_terms = cfg.observations["critic"].terms
    critic_terms.pop("command", None)
    for name in ("foot_height", "foot_air_time", "foot_contact_forces"):
        critic_terms.pop(name, None)
    critic_terms["torso_height"] = ObservationTermCfg(
        func=torso_height_observation,
        params={"asset_cfg": torso},
    )
    cfg.observations["actor"].enable_corruption = not play

    cfg.actions["joint_pos"] = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=LOCOMOTION_JOINT_NAMES,
        preserve_order=True,
        scale=getup_action_scale(),
        clip={
            name: limits
            for name, limits in sample_motion_joint_limits().items()
            if name not in FROZEN_POLICY_JOINT_NAMES
        },
        use_default_offset=True,
    )

    cfg.events["reset_base"] = EventTermCfg(
        func=reset_khr_fall_pose,
        mode="reset",
        params={"pose_class": initial_pose_class, "balanced": balanced_reset},
    )
    cfg.events["reset_robot_joints"].params["asset_cfg"] = active_joints
    cfg.events.pop("push_robot", None)
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = KHR_FOOT_GEOM_NAMES

    cfg.rewards = {
        "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.1),
        "upright_progress": RewardTermCfg(
            func=getup_upright_reward,
            weight=2.0,
            params={"asset_cfg": torso},
        ),
        "torso_height_progress": RewardTermCfg(
            func=getup_torso_height_reward,
            weight=2.0,
            params={"asset_cfg": torso},
        ),
        "foot_support": RewardTermCfg(
            func=getup_foot_support_reward,
            weight=0.25,
            params={"asset_cfg": torso, "sensor_name": "feet_ground_contact"},
        ),
        "stable_stand": RewardTermCfg(
            func=getup_stable_mask,
            weight=5.0,
            params={"asset_cfg": torso, "sensor_name": "feet_ground_contact"},
        ),
        "home_pose": RewardTermCfg(
            func=joint_home_error_l2,
            weight=-0.05,
            params={"asset_cfg": active_joints},
        ),
        "joint_velocity": RewardTermCfg(
            func=envs_mdp.joint_vel_l2,
            weight=-0.002,
            params={"asset_cfg": active_joints},
        ),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.005),
        "torque": RewardTermCfg(
            func=envs_mdp.joint_torques_l2,
            weight=-0.00005,
            params={"asset_cfg": active_joints},
        ),
        "action_target_excess_l2": RewardTermCfg(
            func=action_target_excess_l2,
            weight=-0.04,
        ),
    }
    cfg.terminations.pop("fell_over", None)
    cfg.curriculum = {}
    cfg.metrics = {}
    cfg.episode_length_s = GETUP_EPISODE_SECONDS
    cfg.viewer.body_name = KHR_TORSO_BODY_NAME
    cfg.viewer.distance = 0.9
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 1024

    if play:
        cfg.episode_length_s = 1.0e9

    return cfg


def make_getup_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """Reuse the KHR velocity PPO recipe with a separate get-up log namespace."""

    return replace(make_velocity_ppo_cfg(), experiment_name="khr_getup")
