"""Evaluate Microban's pinned pretrained policy with KHRBan's metric definitions."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import subprocess
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

MICROBAN_TRAINING_REFERENCE_COMMIT = "d594a6088bb7b6600fe8098321169031fbca680c"
MICROBAN_TASK_ID = "Mjlab-Velocity-Microban"
MICROBAN_FOOT_AIR_TIME_MIN_S = 0.125
MICROBAN_FOOT_AIR_TIME_MAX_S = 0.300
MICROBAN_FOOT_CLEARANCE_TARGET_M = 0.020


def _load_shared_evaluation_metrics() -> tuple[type[Any], Any]:
    """Load KHRBan's exact accumulator, including from Microban's isolated venv."""

    if __package__:
        from khrban.evaluation import (
            VelocityEvaluationAccumulator,
            command_snapshot_for_step,
        )

        return VelocityEvaluationAccumulator, command_snapshot_for_step

    # Standalone execution uses Microban's dependency environment.  Load the
    # shared metric source directly without importing KHR's model/BAM modules.
    khrban_stub = types.ModuleType("khrban")
    khrban_stub.__path__ = []
    tasks_stub = types.ModuleType("khrban.tasks")
    tasks_stub.__path__ = []
    tasks_stub.KHR_VELOCITY_TASK_ID = "KhrBan-Velocity"
    velocity_stub = types.ModuleType("khrban.tasks.velocity")
    velocity_stub.KHR_FOOT_AIR_TIME_MAX_S = 0.30
    velocity_stub.KHR_FOOT_AIR_TIME_MIN_S = MICROBAN_FOOT_AIR_TIME_MIN_S
    velocity_stub.KHR_FOOT_CLEARANCE_TARGET_M = 0.02
    velocity_stub.LOCOMOTION_JOINT_NAMES = ()
    velocity_stub.make_velocity_env_cfg = None
    sys.modules.setdefault("khrban", khrban_stub)
    sys.modules.setdefault("khrban.tasks", tasks_stub)
    sys.modules.setdefault("khrban.tasks.velocity", velocity_stub)

    evaluation_path = Path(__file__).with_name("evaluation.py")
    spec = importlib.util.spec_from_file_location(
        "_khrban_shared_evaluation", evaluation_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared evaluation metrics: {evaluation_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.VelocityEvaluationAccumulator, module.command_snapshot_for_step


VelocityEvaluationAccumulator, command_snapshot_for_step = (
    _load_shared_evaluation_metrics()
)


def require_reference_commit(actual: str, expected: str) -> None:
    """Reject an unpinned Microban checkout before loading executable source."""

    if actual != expected:
        raise RuntimeError(
            "Microban reference commit mismatch: "
            f"expected {expected}, got {actual}"
        )


def reference_commit(reference_root: Path) -> str:
    """Return the exact Git commit checked out by the Microban reference tree."""

    return subprocess.run(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def prepare_reference_eval_cfg(cfg: Any, *, num_envs: int, seed: int) -> Any:
    """Keep training commands while disabling state-changing training behavior."""

    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.episode_length_s = 1.0e9
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    return cfg


def _load_reference_tasks(reference_root: Path) -> None:
    source_root = (reference_root / "src").resolve()
    sys.path.insert(0, str(source_root))
    importlib.invalidate_caches()
    tasks_module = importlib.import_module("mjlab_microban.tasks")
    module_path = Path(tasks_module.__file__).resolve()
    if not module_path.is_relative_to(source_root):
        raise RuntimeError(
            "Loaded mjlab_microban from an unexpected location: "
            f"{module_path}, expected below {source_root}"
        )


def evaluate_microban_reference_checkpoint(
    reference_root: Path,
    checkpoint: Path,
    *,
    num_envs: int = 64,
    num_steps: int = 500,
    seed: int = 314159,
    device: str | None = None,
) -> dict[str, object]:
    """Evaluate the pinned Microban model without modifying its repository."""

    reference_root = reference_root.resolve()
    checkpoint = checkpoint.resolve()
    if not reference_root.is_dir():
        raise FileNotFoundError(f"Microban reference not found: {reference_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Microban checkpoint not found: {checkpoint}")
    if num_envs < 1 or num_steps < 1:
        raise ValueError("num_envs and num_steps must be positive")

    actual_commit = reference_commit(reference_root)
    require_reference_commit(actual_commit, MICROBAN_TRAINING_REFERENCE_COMMIT)
    _load_reference_tasks(reference_root)

    from mjlab_microban.tasks.microban_velocity_env_cfg import (
        make_microban_velocity_env_cfg,
    )

    configure_torch_backends()
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = prepare_reference_eval_cfg(
        make_microban_velocity_env_cfg(play=False),
        num_envs=num_envs,
        seed=seed,
    )
    agent_cfg = load_rl_cfg(MICROBAN_TASK_ID)

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=resolved_device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(MICROBAN_TASK_ID) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=resolved_device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=resolved_device,
        )
        policy = runner.get_inference_policy(device=resolved_device)
        observations = env.get_observations()
        action_term = raw_env.action_manager.get_term("joint_pos")
        accumulator = VelocityEvaluationAccumulator(
            num_envs=num_envs,
            device=resolved_device,
            foot_target_height=MICROBAN_FOOT_CLEARANCE_TARGET_M,
            foot_min_air_time=MICROBAN_FOOT_AIR_TIME_MIN_S,
            foot_max_air_time=MICROBAN_FOOT_AIR_TIME_MAX_S,
            action_names=tuple(action_term.target_names),
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

        result = accumulator.as_dict(num_steps=num_steps)
        result.update(
            reference_commit=actual_commit,
            reference_checkpoint=str(checkpoint),
        )
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the pinned Microban policy with KHRBan metrics"
    )
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint or (
        args.reference_root / "src/mjlab_microban/agents/velocity.pt"
    )
    result = evaluate_microban_reference_checkpoint(
        args.reference_root,
        checkpoint,
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
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
