"""KHRBan MjLab task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from khrban.live_training import KhrLiveViewerOnPolicyRunner

from .getup import GETUP_TASK_ID, make_getup_env_cfg, make_getup_ppo_cfg
from .standing import make_standing_env_cfg, make_standing_ppo_cfg
from .velocity import make_velocity_env_cfg, make_velocity_ppo_cfg


KHR_STANDING_TASK_ID = "Mjlab-KHR-Standing"
KHR_VELOCITY_TASK_ID = "Mjlab-KHR-Velocity-Flat"

register_mjlab_task(
    task_id=KHR_STANDING_TASK_ID,
    env_cfg=make_standing_env_cfg(),
    play_env_cfg=make_standing_env_cfg(num_envs=1, play=True),
    rl_cfg=make_standing_ppo_cfg(),
)

register_mjlab_task(
    task_id=KHR_VELOCITY_TASK_ID,
    env_cfg=make_velocity_env_cfg(),
    play_env_cfg=make_velocity_env_cfg(num_envs=1, play=True),
    rl_cfg=make_velocity_ppo_cfg(),
    runner_cls=KhrLiveViewerOnPolicyRunner,
)

register_mjlab_task(
    task_id=GETUP_TASK_ID,
    env_cfg=make_getup_env_cfg(),
    play_env_cfg=make_getup_env_cfg(num_envs=1, play=True),
    rl_cfg=make_getup_ppo_cfg(),
    runner_cls=KhrLiveViewerOnPolicyRunner,
)

__all__ = [
    "GETUP_TASK_ID",
    "KHR_STANDING_TASK_ID",
    "KHR_VELOCITY_TASK_ID",
    "make_getup_env_cfg",
    "make_getup_ppo_cfg",
    "make_standing_env_cfg",
    "make_standing_ppo_cfg",
    "make_velocity_env_cfg",
    "make_velocity_ppo_cfg",
]
