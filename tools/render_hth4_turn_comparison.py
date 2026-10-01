"""Render corresponding Pos frames from the HTH4 left/right turn motions."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

from khrban.hth4 import frame_to_joint_positions, parse_trim, position_frames
from khrban.model import build_khr_spec


def apply_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: dict[int, int],
    trim: dict[int, dict],
) -> None:
    for joint_name, angle in frame_to_joint_positions(frame, trim).items():
        joint_id = model.joint(joint_name).id
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    mujoco.mj_forward(model, data)


def render_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    title: str,
    view: str,
) -> Image.Image:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.15)
    camera.distance = 0.75 if view == "top" else 0.68
    camera.azimuth = 180.0
    camera.elevation = -72.0 if view == "top" else -28.0
    renderer.update_scene(data, camera=camera)
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 28), fill=(255, 255, 255))
    draw.text((8, 7), title, fill=(0, 0, 0))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--view", choices=("front", "top"), default="front")
    parser.add_argument(
        "--turn", choices=("both", "left", "right"), default="both"
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    trim = parse_trim(next(project_dir.glob("*.h4p")))
    all_motions = (
        ("left turn", next(project_dir.glob("22D020*.xml"))),
        ("right turn", next(project_dir.glob("22D021*.xml"))),
    )
    motions = tuple(
        motion
        for motion in all_motions
        if args.turn == "both" or motion[0].startswith(args.turn)
    )
    model = build_khr_spec().compile()
    data = mujoco.MjData(model)
    width = height = 360
    rows: list[list[Image.Image]] = []

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for motion_name, motion_path in motions:
            frames = position_frames(motion_path)
            row = []
            accumulated = {device: 7500 for device in trim}
            for index, frame in enumerate(frames):
                accumulated.update(frame)
                mujoco.mj_resetData(model, data)
                apply_frame(model, data, accumulated, trim)
                left_yaw = accumulated[10] - 7500
                right_yaw = accumulated[11] - 7500
                title = (
                    f"{motion_name} F{index + 1} "
                    f"hip yaw L{left_yaw:+d} R{right_yaw:+d}"
                )
                row.append(render_frame(renderer, data, title, args.view))
            rows.append(row)

    columns = max(len(row) for row in rows)
    sheet = Image.new("RGB", (width * columns, height * len(rows)), "white")
    for row_index, row in enumerate(rows):
        for column_index, image in enumerate(row):
            sheet.paste(image, (column_index * width, row_index * height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
