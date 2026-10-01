"""Command-line launcher for KHRBan training tasks."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import replace
from pathlib import Path

from mjlab.scripts.train import TrainConfig, launch_training

from khrban.tasks import GETUP_TASK_ID, KHR_STANDING_TASK_ID, KHR_VELOCITY_TASK_ID
from khrban.live_training import LIVE_VIEWER_ENV, LIVE_VIEWER_PORT_ENV


TASK_IDS = {
    "standing": KHR_STANDING_TASK_ID,
    "velocity": KHR_VELOCITY_TASK_ID,
    "getup": GETUP_TASK_ID,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a KHR policy")
    parser.add_argument("--task", choices=tuple(TASK_IDS), default="standing")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--live-viewer", action="store_true")
    parser.add_argument("--viewer-num-envs", type=int, default=16)
    parser.add_argument("--viewer-port", type=int, default=8080)
    args = parser.parse_args()

    if args.task == "getup":
        from khrban.auto_train_getup import (
            VelocityTrainingGateError,
            require_velocity_training_pass,
        )

        try:
            require_velocity_training_pass()
        except (VelocityTrainingGateError, ValueError, OSError) as error:
            parser.error(f"get-up training is gated until velocity passes: {error}")

    if args.live_viewer:
        if args.viewer_num_envs < 16:
            parser.error("--viewer-num-envs must be at least 16")
        os.environ[LIVE_VIEWER_ENV] = str(args.viewer_num_envs)
        os.environ[LIVE_VIEWER_PORT_ENV] = str(args.viewer_port)
    else:
        os.environ.pop(LIVE_VIEWER_ENV, None)
        os.environ.pop(LIVE_VIEWER_PORT_ENV, None)

    task_id = TASK_IDS[args.task]
    cfg = TrainConfig.from_task(task_id)
    cfg.env.scene.num_envs = args.num_envs
    cfg.agent.max_iterations = args.iterations
    cfg.agent.run_name = args.run_name
    if args.save_interval is not None:
        if args.save_interval < 1:
            parser.error("--save-interval must be at least 1")
        cfg.agent.save_interval = args.save_interval
    if args.resume_from is not None:
        checkpoint = args.resume_from.resolve()
        if not checkpoint.is_file():
            parser.error(f"--resume-from does not exist: {checkpoint}")
        cfg.agent.resume = True
        cfg.agent.load_run = f"^{re.escape(checkpoint.parent.name)}$"
        cfg.agent.load_checkpoint = f"^{re.escape(checkpoint.name)}$"
    cfg = replace(cfg, gpu_ids=[args.gpu_id])
    launch_training(task_id, cfg)


if __name__ == "__main__":
    main()
