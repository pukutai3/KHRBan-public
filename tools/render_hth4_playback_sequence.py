"""Render every HTH4 Pos pose in actual motion-graph playback order."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

from khrban.hth4 import (
    frame_to_joint_positions,
    parse_trim,
    playback_position_sequence,
)
from khrban.model import build_khr_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("motion", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument(
        "--view", choices=("front", "left", "perspective"), default="front"
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    trim = parse_trim(next(project_dir.glob("*.h4p")))
    sequence = playback_position_sequence(args.motion.resolve())
    model = build_khr_spec().compile()
    data = mujoco.MjData(model)
    width = height = 320
    images = []

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.15)
    camera.distance = 0.82
    camera.azimuth = {
        "front": 180.0,
        "left": 90.0,
        "perspective": 135.0,
    }[args.view]
    camera.elevation = -20.0

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        accumulated = {device: 7500 for device in trim}
        for index, (activity_name, frame) in enumerate(sequence, start=1):
            accumulated.update(frame)
            mujoco.mj_resetData(model, data)
            for joint_name, angle in frame_to_joint_positions(
                accumulated, trim
            ).items():
                joint_id = model.joint(joint_name).id
                data.qpos[model.jnt_qposadr[joint_id]] = angle
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            image = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, width, 28), fill=(255, 255, 255))
            draw.text(
                (8, 7),
                f"Playback {index:02d} - {activity_name}",
                fill=(0, 0, 0),
            )
            images.append(image)

    rows = math.ceil(len(images) / args.columns)
    sheet = Image.new("RGB", (width * args.columns, height * rows), "white")
    for index, image in enumerate(images):
        x = (index % args.columns) * width
        y = (index // args.columns) * height
        sheet.paste(image, (x, y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Rendered {len(images)} poses in playback order")


if __name__ == "__main__":
    main()
