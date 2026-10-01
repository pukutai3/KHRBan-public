"""Flat-ground velocity tracking task for the KHR-3HV."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

from bam.mjlab import bam_init
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg as make_base_cfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
import torch

from khrban.model import (
    KHR_TORSO_BODY_NAME,
    make_khr_entity_cfg,
    sample_motion_home_joint_positions,
)
from khrban.observations import LEG_PITCH_AXIS_BODY_NAMES, leg_flexion_ratio
from khrban.rewards import (
    action_target_excess_l2,
    excessive_air_time_penalty,
    feet_distance_penalty,
    no_stepping_penalty,
)


KHR_FOOT_CLEARANCE_TARGET_M = 0.021
KHR_FOOT_TRAINING_MARGIN_M = 0.004
KHR_FOOT_TRAINING_TARGET_M = (
    KHR_FOOT_CLEARANCE_TARGET_M + KHR_FOOT_TRAINING_MARGIN_M
)
KHR_FOOT_AIR_TIME_MIN_S = 0.125
KHR_FOOT_AIR_TIME_MAX_S = 0.30
KHR_FOOT_SCAN_RADIUS_M = 0.042
KHR_MIN_FOOT_DISTANCE_M = 0.062
KHR_FOOT_SITE_NAMES = ("left_foot", "right_foot")
KHR_FOOT_GEOM_NAMES = ("left_foot_collision", "right_foot_collision")
KHR_ROTATION_ENV_FRACTION = 0.10
KHR_ROTATION_MIN_ANG_VEL = 0.50
KHR_ROTATION_ENV_ANG_VEL_RANGE = (-1.50, 1.50)
KHR_ACTION_TARGET_EXCESS_WEIGHT = -1.0
KHR_EXCESSIVE_AIR_TIME_WEIGHT = -3.0
MICROBAN_TRAINING_REFERENCE_COMMIT = "d594a6088bb7b6600fe8098321169031fbca680c"

# These upper-body yaw joints stay at their confirmed 0 rad home positions.
# They remain physically actuated but are excluded from policy actions and
# joint observations.
FROZEN_POLICY_JOINT_NAMES = (
    "continuous_joint_01",  # waist yaw
    "continuous_joint_02",  # head yaw
    "continuous_joint_05",  # left elbow yaw
    "continuous_joint_09",  # right elbow yaw
)

# The walking policy controls the remaining 18 joints in stable numeric order.
LOCOMOTION_JOINT_NAMES = tuple(
    joint_name
    for index in range(1, 23)
    if (joint_name := f"continuous_joint_{index:02d}")
    not in FROZEN_POLICY_JOINT_NAMES
)

# Symmetric, paired action ranges stay inside the confirmed sample-motion
# envelope.  Knee and ankle pitch allow the clearance needed for KHR steps.
LOCOMOTION_ACTION_SCALE = {
    r"continuous_joint_03": 0.50,
    r"continuous_joint_04": 0.42,
    r"continuous_joint_06": 0.50,
    r"continuous_joint_07": 0.50,
    r"continuous_joint_08": 0.42,
    r"continuous_joint_10": 0.50,
    r"continuous_joint_11": 0.33,
    r"continuous_joint_12": 0.20,
    r"continuous_joint_13": 0.50,
    r"continuous_joint_14": 0.55,
    r"continuous_joint_15": 0.55,
    r"continuous_joint_16": 0.14,
    r"continuous_joint_17": 0.33,
    r"continuous_joint_18": 0.20,
    r"continuous_joint_19": 0.50,
    r"continuous_joint_20": 0.55,
    r"continuous_joint_21": 0.55,
    r"continuous_joint_22": 0.14,
}


def _pose_std_walking() -> dict[str, float]:
    """Return KHR joint-role tolerances in radians."""

    return {
        r"continuous_joint_(03|07)": 0.40,  # shoulder pitch
        r"continuous_joint_(04|08)": 0.20,  # shoulder roll
        r"continuous_joint_(06|10)": 0.20,  # elbow pitch
        r"continuous_joint_(11|17)": 0.20,  # hip yaw
        r"continuous_joint_(12|18)": 0.20,  # hip roll
        r"continuous_joint_(13|19)": 0.40,  # hip pitch
        r"continuous_joint_(14|20)": 0.40,  # knee
        r"continuous_joint_(15|21)": 0.30,  # ankle pitch
        r"continuous_joint_(16|22)": 0.20,  # ankle roll
    }


def _pose_std_standing() -> dict[str, float]:
    """Map Microban's joint-role tolerances onto the KHR joint names."""

    return {
        r"continuous_joint_(03|07)": 0.10,  # shoulder pitch
        r"continuous_joint_(04|08)": 0.10,  # shoulder roll
        r"continuous_joint_(06|10)": 0.10,  # elbow pitch
        r"continuous_joint_(11|17)": 0.10,  # hip yaw
        r"continuous_joint_(12|18)": 0.10,  # hip roll
        r"continuous_joint_(13|19)": 0.15,  # hip pitch
        r"continuous_joint_(14|20)": 0.15,  # knee
        r"continuous_joint_(15|21)": 0.10,  # ankle pitch
        r"continuous_joint_(16|22)": 0.10,  # ankle roll
    }


def _action_target_clip() -> dict[str, tuple[float, float]]:
    """Keep KHR targets inside the confirmed sample-motion envelope."""

    home = sample_motion_home_joint_positions()
    return {
        joint_name: (home[joint_name] - scale, home[joint_name] + scale)
        for joint_name, scale in LOCOMOTION_ACTION_SCALE.items()
    }


class KhrUniformVelocityCommandWithRotation(UniformVelocityCommand):
    """Add Microban-style pure-rotation command samples for KHR training."""

    cfg: "KhrUniformVelocityCommandWithRotationCfg"

    def __init__(
        self,
        cfg: "KhrUniformVelocityCommandWithRotationCfg",
        env,
    ):
        super().__init__(cfg, env)
        self.is_rotation_env = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)

        rel_rotation_envs = self.cfg.rel_rotation_envs
        sample = torch.empty(len(env_ids), device=self.device)
        rotation_mask = sample.uniform_(0.0, 1.0) <= rel_rotation_envs
        self.is_rotation_env[env_ids] = rotation_mask
        rotation_ids = env_ids[rotation_mask]
        if len(rotation_ids) == 0:
            return

        self.vel_command_b[rotation_ids, 0] = 0.0
        self.vel_command_b[rotation_ids, 1] = 0.0
        ang = torch.empty(len(rotation_ids), device=self.device).uniform_(
            *self.cfg.rotation_env_ang_vel_range
        )
        too_small = ang.abs() < self.cfg.rotation_min_ang_vel
        if too_small.any():
            signs = torch.where(
                torch.rand(too_small.sum(), device=self.device) > 0.5,
                torch.ones(too_small.sum(), device=self.device),
                -torch.ones(too_small.sum(), device=self.device),
            )
            ang[too_small] = signs * self.cfg.rotation_min_ang_vel
        self.vel_command_b[rotation_ids, 2] = ang


@dataclass(kw_only=True)
class KhrUniformVelocityCommandWithRotationCfg(UniformVelocityCommandCfg):
    """Velocity command config with dedicated pure-rotation environments."""

    rel_rotation_envs: float = 0.0
    rotation_min_ang_vel: float = KHR_ROTATION_MIN_ANG_VEL
    rotation_env_ang_vel_range: tuple[float, float] = KHR_ROTATION_ENV_ANG_VEL_RANGE

    def build(self, env) -> KhrUniformVelocityCommandWithRotation:
        return KhrUniformVelocityCommandWithRotation(self, env)


class khr_microban_style_velocity_curriculum:
    """Apply the Microban velocity-stage idea with KHR-specific terms intact."""

    def __init__(self, cfg: CurriculumTermCfg, env):
        del cfg, env
        self.current_stage = 0

    def __call__(self, env, env_ids: torch.Tensor, stages: list[dict]) -> dict:
        del env_ids
        if self.current_stage >= len(stages):
            return {"stage": torch.tensor(self.current_stage, device=env.device)}

        stage = stages[self.current_stage]
        if env.common_step_counter < stage["step"]:
            return {"stage": torch.tensor(self.current_stage, device=env.device)}

        command = env.command_manager.get_term_cfg("twist")
        command.ranges.lin_vel_x = stage["lin_vel_x"]
        command.ranges.lin_vel_y = stage["lin_vel_y"]
        command.ranges.ang_vel_z = stage["ang_vel_z"]
        command.rotation_env_ang_vel_range = stage["rotation_env_ang_vel_range"]
        command.rel_standing_envs = stage["rel_standing_envs"]
        command.rel_rotation_envs = stage["rel_rotation_envs"]
        env.reward_manager.get_term_cfg("air_time").weight = stage["air_time_weight"]
        env.reward_manager.get_term_cfg("no_stepping").weight = stage[
            "no_stepping_weight"
        ]
        self.current_stage += 1
        return {"stage": torch.tensor(self.current_stage, device=env.device)}


def make_velocity_env_cfg(
    num_envs: int = 1024,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the first KHR flat-ground velocity-tracking candidate."""

    cfg = make_base_cfg()
    cfg.scene.entities = {"robot": make_khr_entity_cfg()}
    cfg.scene.num_envs = num_envs
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    foot_height_scan = next(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name == "foot_height_scan"
    )
    assert isinstance(foot_height_scan, TerrainHeightSensorCfg)
    foot_height_scan.frame = tuple(
        ObjRef(type="site", name=name, entity="robot")
        for name in KHR_FOOT_SITE_NAMES
    )
    foot_height_scan.pattern = RingPatternCfg.single_ring(
        radius=KHR_FOOT_SCAN_RADIUS_M,
        num_samples=6,
    )
    foot_height_scan.debug_vis = False

    feet_ground_sensor = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(l_foot_1|r_foot_1)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_sensor = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (foot_height_scan, feet_ground_sensor, self_collision_sensor)

    locomotion_joints = SceneEntityCfg(
        "robot", joint_names=LOCOMOTION_JOINT_NAMES, preserve_order=True
    )
    leg_pitch_axes = SceneEntityCfg(
        "robot", body_names=LEG_PITCH_AXIS_BODY_NAMES, preserve_order=True
    )
    actor = cfg.observations["actor"].terms
    critic = cfg.observations["critic"].terms
    for terms in (actor, critic):
        terms.pop("height_scan", None)
        terms["leg_flexion_ratio"] = ObservationTermCfg(
            func=leg_flexion_ratio,
            params={"asset_cfg": leg_pitch_axes},
            clip=(0.0, 1.0),
        )
    actor["base_ang_vel"] = ObservationTermCfg(
        func=envs_mdp.base_ang_vel,
        params={"asset_cfg": locomotion_joints},
        noise=Unoise(n_min=-0.03, n_max=0.03),
        delay_min_lag=0,
        delay_max_lag=3,
        delay_update_period=64,
    )
    actor["joint_pos"] = ObservationTermCfg(
        func=envs_mdp.joint_pos_rel,
        params={"asset_cfg": locomotion_joints},
        noise=Unoise(n_min=-0.001, n_max=0.001),
        delay_min_lag=0,
        delay_max_lag=0,
    )
    actor["joint_vel"] = ObservationTermCfg(
        func=envs_mdp.joint_vel_rel,
        params={"asset_cfg": locomotion_joints},
        noise=Unoise(n_min=-0.25, n_max=0.25),
        delay_min_lag=0,
        delay_max_lag=1,
    )
    actor["projected_gravity"] = deepcopy(actor["projected_gravity"])
    actor["projected_gravity"].noise = Unoise(n_min=-0.01, n_max=0.01)
    actor["projected_gravity"].delay_min_lag = 0
    actor["projected_gravity"].delay_max_lag = 3
    actor["projected_gravity"].delay_update_period = 64
    critic["base_ang_vel"] = ObservationTermCfg(
        func=envs_mdp.base_ang_vel,
        params={"asset_cfg": locomotion_joints},
    )
    critic["joint_pos"] = ObservationTermCfg(
        func=envs_mdp.joint_pos_rel,
        params={"asset_cfg": locomotion_joints},
    )
    critic["joint_vel"] = ObservationTermCfg(
        func=envs_mdp.joint_vel_rel,
        params={"asset_cfg": locomotion_joints},
    )
    actor.pop("base_lin_vel", None)
    critic["base_lin_vel"] = ObservationTermCfg(
        func=envs_mdp.base_lin_vel,
        params={"asset_cfg": locomotion_joints},
    )
    cfg.observations["actor"].enable_corruption = not play

    cfg.actions["joint_pos"] = JointPositionActionCfg(
        entity_name="robot",
        actuator_names=LOCOMOTION_JOINT_NAMES,
        preserve_order=True,
        scale=LOCOMOTION_ACTION_SCALE,
        clip=_action_target_clip(),
        use_default_offset=True,
    )

    old_command = cfg.commands["twist"]
    assert isinstance(old_command, UniformVelocityCommandCfg)
    command = KhrUniformVelocityCommandWithRotationCfg(
        resampling_time_range=old_command.resampling_time_range,
        debug_vis=old_command.debug_vis,
        entity_name=old_command.entity_name,
        heading_command=False,
        heading_control_stiffness=old_command.heading_control_stiffness,
        rel_standing_envs=0.10,
        rel_heading_envs=0.0,
        rel_world_envs=old_command.rel_world_envs,
        rel_forward_envs=0.0,
        init_velocity_prob=old_command.init_velocity_prob,
        ranges=old_command.ranges,
        viz=old_command.viz,
        rel_rotation_envs=KHR_ROTATION_ENV_FRACTION,
        rotation_min_ang_vel=KHR_ROTATION_MIN_ANG_VEL,
        rotation_env_ang_vel_range=KHR_ROTATION_ENV_ANG_VEL_RANGE,
    )
    cfg.commands["twist"] = command
    command.heading_command = False
    command.ranges.heading = None
    command.rel_heading_envs = 0.0
    command.rel_forward_envs = 0.0
    command.rel_standing_envs = 0.10
    command.ranges.lin_vel_x = (-0.50, 0.50)
    command.ranges.lin_vel_y = (-0.30, 0.30)
    command.ranges.ang_vel_z = (-0.75, 0.75)
    command.viz.z_offset = 0.40

    cfg.events["bam_init"] = EventTermCfg(func=bam_init, mode="startup")
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.005)
    cfg.events["reset_robot_joints"].params["asset_cfg"] = locomotion_joints
    cfg.events["push_robot"].interval_range_s = (1.0, 3.0)
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.50, 0.50),
        "y": (-0.50, 0.50),
    }
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = KHR_FOOT_GEOM_NAMES
    cfg.events["base_com"].params = {
        "asset_cfg": SceneEntityCfg("robot", body_names=(KHR_TORSO_BODY_NAME,)),
        "operation": "add",
        "ranges": {
            0: (-0.005, 0.005),
            1: (-0.005, 0.005),
            2: (-0.005, 0.005),
        },
    }
    all_khr_joints = SceneEntityCfg(
        "robot", joint_names=(r"continuous_joint_.*",)
    )
    cfg.events["dof_armature_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": all_khr_joints,
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )
    cfg.events["dof_friction_randomization"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": all_khr_joints,
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )

    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    pose = cfg.rewards["pose"]
    pose.params["asset_cfg"] = locomotion_joints
    pose.params["std_standing"] = _pose_std_standing()
    pose.params["std_walking"] = _pose_std_walking()
    pose.params["std_running"] = _pose_std_walking()
    pose.params["walking_threshold"] = 0.01
    cfg.rewards["upright"].params["asset_cfg"].body_names = (KHR_TORSO_BODY_NAME,)
    cfg.rewards["upright"].params["std"] = math.sqrt(0.1)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (
        KHR_TORSO_BODY_NAME,
    )
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards.pop("soft_landing", None)
    cfg.rewards["air_time"].weight = 3.0
    cfg.rewards["air_time"].params.update(
        threshold_min=KHR_FOOT_AIR_TIME_MIN_S,
        threshold_max=KHR_FOOT_AIR_TIME_MAX_S,
        command_threshold=0.01,
    )
    cfg.rewards["excessive_air_time"] = RewardTermCfg(
        func=excessive_air_time_penalty,
        weight=KHR_EXCESSIVE_AIR_TIME_WEIGHT,
        params={
            "sensor_name": feet_ground_sensor.name,
            "command_name": "twist",
            "threshold_max": KHR_FOOT_AIR_TIME_MAX_S,
            "command_threshold": 0.01,
        },
    )
    for reward_name in ("foot_clearance", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = KHR_FOOT_SITE_NAMES
    cfg.rewards["foot_clearance"].params.update(
        target_height=KHR_FOOT_TRAINING_TARGET_M,
        command_threshold=0.01,
    )
    cfg.rewards["foot_swing_height"].params.update(
        target_height=KHR_FOOT_TRAINING_TARGET_M,
        command_threshold=0.01,
    )
    cfg.rewards["foot_slip"].weight = -1.0
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01
    cfg.rewards["action_target_excess_l2"] = RewardTermCfg(
        func=action_target_excess_l2,
        weight=KHR_ACTION_TARGET_EXCESS_WEIGHT,
    )
    cfg.rewards["feet_distance"] = RewardTermCfg(
        func=feet_distance_penalty,
        weight=-1000.0,
        params={
            "min_dist": KHR_MIN_FOOT_DISTANCE_M,
            "asset_cfg": SceneEntityCfg(
                "robot", site_names=KHR_FOOT_SITE_NAMES, preserve_order=True
            ),
        },
    )
    cfg.rewards["no_stepping"] = RewardTermCfg(
        func=no_stepping_penalty,
        weight=0.0,
        params={
            "sensor_name": feet_ground_sensor.name,
            "command_name": "twist",
            "command_threshold": 0.01,
        },
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_sensor.name},
    )

    cfg.terminations["fell_over"].params["limit_angle"] = math.radians(45.0)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {
        "microban_style_velocity": CurriculumTermCfg(
            func=khr_microban_style_velocity_curriculum,
            params={
                "stages": [
                    {
                        "name": "increase KHR velocity and standing discipline",
                        "step": 3000 * 24,
                        "lin_vel_x": (-0.70, 0.70),
                        "lin_vel_y": (-0.30, 0.30),
                        "ang_vel_z": (-1.50, 1.50),
                        "rotation_env_ang_vel_range": (-3.00, 3.00),
                        "air_time_weight": 3.0,
                        "no_stepping_weight": -1.0,
                        "rel_standing_envs": 0.10,
                        "rel_rotation_envs": KHR_ROTATION_ENV_FRACTION,
                    },
                ],
            },
        )
    }

    cfg.viewer.body_name = KHR_TORSO_BODY_NAME
    cfg.viewer.distance = 0.8
    cfg.sim.nconmax = 96
    cfg.sim.njmax = 768
    cfg.sim.mujoco.ccd_iterations = 100
    cfg.episode_length_s = 20.0

    if play:
        cfg.episode_length_s = 1.0e9
        cfg.events.pop("push_robot", None)

    return cfg


def make_velocity_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    """Return the PPO configuration for KHR flat-ground velocity tracking."""

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="khr_velocity",
        logger="tensorboard",
        upload_model=False,
        clip_actions=None,
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=15_000,
    )
