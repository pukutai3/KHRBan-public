"""Compare one saved KHR policy under two knee and ankle target widths.

This is a diagnostic replay. It does not change training or the formal evaluator.
Run it from the KHRBan repository with its installed Python environment.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import khrban.evaluation as evaluation
from khrban.evaluation import VelocityEvaluationAccumulator
from khrban.model import (
    sample_motion_home_joint_positions,
    sample_motion_joint_limits,
)


KNEE_ANKLE_PITCH_JOINTS = (
    "continuous_joint_14",
    "continuous_joint_15",
    "continuous_joint_20",
    "continuous_joint_21",
)


class SideMetricsAccumulator(VelocityEvaluationAccumulator):
    """Expose the left/right counts behind the evaluator's minimum rate."""

    def as_dict(self, *, num_steps: int) -> dict[str, object]:
        result = super().as_dict(num_steps=num_steps)
        landings = self.foot_landing_count_by_side.cpu().tolist()
        successes = self.foot_lift_success_count_by_side.cpu().tolist()
        result["diagnostic_landings_by_side"] = landings
        result["diagnostic_foot_success_by_side"] = [
            float(good) / count if count else 0.0
            for good, count in zip(successes, landings)
        ]
        return result


def make_widened_factory(original_factory, width: float):
    home = sample_motion_home_joint_positions()
    limits = sample_motion_joint_limits()
    for name in KNEE_ANKLE_PITCH_JOINTS:
        lower, upper = limits[name]
        if home[name] - width < lower or home[name] + width > upper:
            raise ValueError(f"{name}: width {width} exceeds sample-motion range")

    def factory(*args, **kwargs):
        cfg = original_factory(*args, **kwargs)
        action = cfg.actions["joint_pos"]
        scales = dict(action.scale)
        for name in KNEE_ANKLE_PITCH_JOINTS:
            scales[name] = width
        action.scale = scales
        action.clip = {
            name: (home[name] - scale, home[name] + scale)
            for name, scale in scales.items()
        }
        return cfg

    return factory


def run_replay(
    checkpoint: Path,
    *,
    factory,
    num_envs: int,
    num_steps: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    evaluation.make_velocity_env_cfg = factory
    started = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        result = evaluation.evaluate_velocity_checkpoint(
            checkpoint,
            num_envs=num_envs,
            num_steps=num_steps,
            seed=seed,
            device=device,
        )
    result["diagnostic_elapsed_s"] = round(time.perf_counter() - started, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--width", type=float, default=0.55)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.width <= 0:
        parser.error("--width must be positive")

    checkpoint = args.checkpoint.resolve(strict=True)
    original_factory = evaluation.make_velocity_env_cfg
    original_accumulator = evaluation.VelocityEvaluationAccumulator
    evaluation.VelocityEvaluationAccumulator = SideMetricsAccumulator
    try:
        baseline = run_replay(
            checkpoint,
            factory=original_factory,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            seed=args.seed,
            device=args.device,
        )
        alternate = run_replay(
            checkpoint,
            factory=make_widened_factory(original_factory, args.width),
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            seed=args.seed,
            device=args.device,
        )
    finally:
        evaluation.make_velocity_env_cfg = original_factory
        evaluation.VelocityEvaluationAccumulator = original_accumulator

    with checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    artifact = {
        "kind": "diagnostic_policy_replay_not_formal_evaluation",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "device": args.device,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "alternate_change": {
            name: args.width for name in KNEE_ANKLE_PITCH_JOINTS
        },
        "baseline": baseline,
        "alternate": alternate,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(artifact, ensure_ascii=False))


if __name__ == "__main__":
    main()
