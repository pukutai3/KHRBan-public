"""Render the KHR URDF zero pose from multiple viewpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

from khrban.hth4 import frame_to_joint_positions, parse_trim, position_frames
from khrban.model import build_khr_spec


VIEWS = (
    ("front", 180.0, -5.0),
    ("left", 90.0, -5.0),
    ("perspective", 135.0, -12.0),
)


def render_view(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    label: str,
    azimuth: float,
    elevation: float,
) -> Image.Image:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.16)
    camera.distance = 0.72
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer.update_scene(data, camera=camera)
    image = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 210, 28), fill=(255, 255, 255))
    draw.text((10, 7), label, fill=(0, 0, 0))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--hth4-project",
        type=Path,
        help="Render a pose using the supplied HTH4 project directory",
    )
    parser.add_argument(
        "--hth4-pose",
        choices=("trim", "home"),
        default="trim",
        help="HTH4 pose to render when --hth4-project is supplied",
    )
    args = parser.parse_args()

    model = build_khr_spec().compile()
    data = mujoco.MjData(model)
    label_prefix = "servo origin"
    if args.hth4_project:
        project_dir = args.hth4_project.resolve()
        trim = parse_trim(next(project_dir.glob("*.h4p")))
        if args.hth4_pose == "trim":
            frame = {device: 7500 for device in trim}
        else:
            home_file = next(
                path
                for path in project_dir.rglob("*.xml")
                if "ホームポジション" in path.name
            )
            frame = next(
                frame
                for frame in reversed(position_frames(home_file))
                if len(frame) == 22
            )
        for joint_name, angle in frame_to_joint_positions(frame, trim).items():
            joint_id = model.joint(joint_name).id
            data.qpos[model.jnt_qposadr[joint_id]] = angle
        label_prefix = f"HTH4 {args.hth4_pose}"
    mujoco.mj_forward(model, data)

    width, height = 480, 480
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        images = [
            render_view(
                renderer, data, f"{label_prefix} {label}", azimuth, elevation
            )
            for label, azimuth, elevation in VIEWS
        ]

    sheet = Image.new("RGB", (width * len(images), height), "white")
    for index, image in enumerate(images):
        sheet.paste(image, (index * width, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
