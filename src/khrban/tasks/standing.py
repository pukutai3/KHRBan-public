"""Flat-ground standing task for the KHR-3HV."""

from __future__ import annotations

import math

from bam.mjlab import bam_init
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from khrban.model import make_flat_terrain_cfg, make_khr_entity_cfg
from khrban.observations import (
    LEG_PITCH_AXIS_BODY_NAMES,
    leg_flexion_angle_deg,
    leg_flexion_ratio,
)


def make_standing_env_cfg(
    num_envs: int = 1024,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the first-stage task: retain the confirmed HTH4 home pose."""

    khr_all_joints = SceneEntityCfg(
        "khr", joint_names=(r"continuous_joint_.*",)
    )
    leg_pitch_axes = SceneEntityCfg(
        "khr",
        body_names=LEG_PITCH_AXIS_BODY_NAMES,
        preserve_order=True,
    )
    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.base_lin_vel,
            params={"asset_cfg": khr_all_joints},
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            params={"asset_cfg": khr_all_joints},
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            params={"asset_cfg": khr_all_joints},
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": khr_all_joints},
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": khr_all_joints},
        ),
        "last_action": ObservationTermCfg(func=mdp.last_action),
        "leg_flexion_ratio": ObservationTermCfg(
            func=leg_flexion_ratio,
            params={"asset_cfg": leg_pitch_axes},
            clip=(0.0, 1.0),
        ),
    }
    observations = {
        "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
        "critic": ObservationGroupCfg({**actor_terms}, enable_corruption=False),
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="khr",
            actuator_names=(r"continuous_joint_.*",),
            scale=0.25,
            use_default_offset=True,
        )
    }

    events = {
        "bam_init": EventTermCfg(func=bam_init, mode="startup"),
        "reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
    }

    metrics = {
        "left_leg_flexion_deg": MetricsTermCfg(
            func=leg_flexion_angle_deg,
            params={"asset_cfg": leg_pitch_axes, "leg_index": 0},
        ),
        "right_leg_flexion_deg": MetricsTermCfg(
            func=leg_flexion_angle_deg,
            params={"asset_cfg": leg_pitch_axes, "leg_index": 1},
        ),
    }

    rewards = {
        "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
        "upright": RewardTermCfg(
            func=mdp.flat_orientation_l2,
            weight=-2.0,
            params={"asset_cfg": khr_all_joints},
        ),
        "posture": RewardTermCfg(
            func=mdp.posture,
            weight=2.0,
            params={"std": {r".*": 0.20}, "asset_cfg": khr_all_joints},
        ),
        "joint_velocity": RewardTermCfg(
            func=mdp.joint_vel_l2,
            weight=-0.005,
            params={"asset_cfg": khr_all_joints},
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "torque": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-0.0001,
            params={"asset_cfg": khr_all_joints},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={
                "limit_angle": math.radians(45.0),
                "asset_cfg": khr_all_joints,
            },
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=make_flat_terrain_cfg(),
            entities={"khr": make_khr_entity_cfg()},
            num_envs=num_envs,
            env_spacing=1.0,
        ),
        observations=observations,
        actions=actions,
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="khr",
            body_name="base_link",
            distance=0.8,
            elevation=-15.0,
            azimuth=180.0,
        ),
        sim=SimulationCfg(
            nconmax=64,
            njmax=512,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=10.0 if not play else 1.0e9,
    )


def make_standing_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """Return a conservative PPO baseline for the standing task."""

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="khr_standing",
        logger="tensorboard",
        upload_model=False,
        clip_actions=1.0,
        save_interval=50,
        num_steps_per_env=32,
        max_iterations=2000,
    )
