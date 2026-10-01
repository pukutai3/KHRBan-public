"""Per-fall-orientation evaluation for trained KHR stand-up policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from khrban.tasks.getup import (
    GETUP_HOLD_SECONDS,
    GETUP_POSE_CLASSES,
    GETUP_TASK_ID,
    GETUP_SUCCESS_TORSO_HEIGHT_M,
    GETUP_UPRIGHT_COS_THRESHOLD,
    getup_stable_mask,
    make_getup_env_cfg,
)
from khrban.tasks.velocity import KHR_TORSO_BODY_NAME


@dataclass(frozen=True)
class GetUpEvaluationResult:
    """Metrics retain per-start-pose outcomes so one easy pose cannot hide another."""

    success_rate_by_pose: dict[str, float]
    sample_count_by_pose: dict[str, int]
    final_torso_height_mean_by_pose_m: dict[str, float]
    final_upright_cos_mean_by_pose: dict[str, float]
    final_both_feet_contact_rate_by_pose: dict[str, float]
    success_rate: float
    num_envs: int
    num_steps: int
    hold_steps: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GetUpEvaluationCriteria:
    """Every fall orientation must stand and hold for the full tail window."""

    min_success_rate_per_pose: float = 0.80
    min_samples_per_pose: int = 16

    def is_satisfied_by(self, result: GetUpEvaluationResult) -> bool:
        return all(
            result.sample_count_by_pose.get(pose, 0) >= self.min_samples_per_pose
            and result.success_rate_by_pose.get(pose, 0.0)
            >= self.min_success_rate_per_pose
            for pose in GETUP_POSE_CLASSES
        )


def evaluate_getup_checkpoint(
    checkpoint: Path,
    *,
    num_envs: int = 256,
    num_steps: int = 500,
    hold_seconds: float = GETUP_HOLD_SECONDS,
    seed: int = 314159,
    device: str | None = None,
) -> GetUpEvaluationResult:
    """Test balanced KHR fall starts and require a stable final hold interval."""

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if num_envs < len(GETUP_POSE_CLASSES) or num_envs % len(GETUP_POSE_CLASSES):
        raise ValueError("num_envs must be a positive multiple of four")
    if num_steps < 1 or hold_seconds <= 0.0:
        raise ValueError("num_steps and hold_seconds must be positive")

    configure_torch_backends()
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = make_getup_env_cfg(
        num_envs=num_envs,
        play=True,
        balanced_reset=True,
    )
    env_cfg.seed = seed
    agent_cfg = load_rl_cfg(GETUP_TASK_ID)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=resolved_device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_cls = load_runner_cls(GETUP_TASK_ID) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=resolved_device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=resolved_device,
        )
        policy = runner.get_inference_policy(device=resolved_device)
        observations = env.get_observations()
        pose_ids = raw_env.getup_pose_class_ids.clone()
        if torch.any(pose_ids < 0):
            raise RuntimeError("Get-up reset did not record its fall-pose classes")

        hold_steps = max(1, round(hold_seconds / raw_env.step_dt))
        if hold_steps > num_steps:
            raise ValueError("hold_seconds must not exceed the evaluation duration")
        stable_history = torch.zeros(
            (num_steps, num_envs), dtype=torch.bool, device=resolved_device
        )
        torso_cfg = SceneEntityCfg(
            "robot", body_names=(KHR_TORSO_BODY_NAME,), preserve_order=True
        )
        torso_cfg.resolve(raw_env.scene)

        with torch.inference_mode():
            for step in range(num_steps):
                actions = policy(observations)
                observations, _, _, _ = env.step(actions)
                stable_history[step] = getup_stable_mask(
                    raw_env,
                    asset_cfg=torso_cfg,
                    sensor_name="feet_ground_contact",
                )

        stable_at_end = stable_history[-hold_steps:].all(dim=0)
        asset = raw_env.scene["robot"]
        torso_height = (
            asset.data.body_link_pos_w[:, torso_cfg.body_ids, 2].squeeze(-1)
            - raw_env.scene.env_origins[:, 2]
        )
        upright_cos = -asset.data.projected_gravity_b[:, 2]
        feet_found = raw_env.scene["feet_ground_contact"].data.found
        if feet_found is None:
            raise RuntimeError("feet_ground_contact sensor has no contact data")
        if feet_found.ndim == 3:
            feet_found = feet_found.any(dim=-1)
        if feet_found.ndim != 2 or feet_found.shape[1] != 2:
            raise ValueError(
                "Expected separate final contact states for both feet, got "
                f"{tuple(feet_found.shape)}"
            )
        both_feet_found = feet_found.bool().all(dim=-1)

        success_rates: dict[str, float] = {}
        sample_counts: dict[str, int] = {}
        torso_heights: dict[str, float] = {}
        upright_values: dict[str, float] = {}
        contact_rates: dict[str, float] = {}
        for class_id, pose in enumerate(GETUP_POSE_CLASSES):
            mask = pose_ids == class_id
            count = int(mask.sum().item())
            sample_counts[pose] = count
            if count == 0:
                success_rates[pose] = 0.0
                torso_heights[pose] = 0.0
                upright_values[pose] = -1.0
                contact_rates[pose] = 0.0
                continue
            success_rates[pose] = float(stable_at_end[mask].float().mean().item())
            torso_heights[pose] = float(torso_height[mask].mean().item())
            upright_values[pose] = float(upright_cos[mask].mean().item())
            contact_rates[pose] = float(both_feet_found[mask].float().mean().item())

        macro_success_rate = sum(success_rates.values()) / len(GETUP_POSE_CLASSES)
        return GetUpEvaluationResult(
            success_rate_by_pose=success_rates,
            sample_count_by_pose=sample_counts,
            final_torso_height_mean_by_pose_m=torso_heights,
            final_upright_cos_mean_by_pose=upright_values,
            final_both_feet_contact_rate_by_pose=contact_rates,
            success_rate=macro_success_rate,
            num_envs=num_envs,
            num_steps=num_steps,
            hold_steps=hold_steps,
        )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a KHR get-up checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--hold-seconds", type=float, default=GETUP_HOLD_SECONDS)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_getup_checkpoint(
        args.checkpoint,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        hold_seconds=args.hold_seconds,
        seed=args.seed,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.as_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
