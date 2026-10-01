"""Interactive HTH4 sample-motion playback in the physical KHR world."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

from khrban.hth4 import (
    frame_to_joint_positions,
    parse_trim,
    playback_position_sequence,
    position_frames,
)
from khrban.tasks.standing import make_standing_env_cfg


FRAME_PERIOD_S = 0.015
HOME_MOTION_PREFIX = "22D002_"
MUJOCO_FLOOR_COLORS = ((51, 76, 102), (26, 51, 76))


def mujoco_checker_meshes(
    half_extent: float = 6.0,
    cell_size: float = 0.25,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]]:
    """Build two meshes matching MjLab's default MuJoCo checker colors."""

    if half_extent <= 0.0 or cell_size <= 0.0:
        raise ValueError("Checker dimensions must be positive")
    cells = max(2, int(np.ceil(2.0 * half_extent / cell_size)))
    start = -0.5 * cells * cell_size
    meshes = []
    for parity, color in enumerate(MUJOCO_FLOOR_COLORS):
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int]] = []
        for row in range(cells):
            for column in range(cells):
                if (row + column) % 2 != parity:
                    continue
                x0 = start + column * cell_size
                y0 = start + row * cell_size
                base = len(vertices)
                vertices.extend(
                    (
                        (x0, y0, 0.0),
                        (x0 + cell_size, y0, 0.0),
                        (x0 + cell_size, y0 + cell_size, 0.0),
                        (x0, y0 + cell_size, 0.0),
                    )
                )
                faces.extend(
                    (
                        (base, base + 1, base + 2),
                        (base, base + 2, base + 3),
                    )
                )
        meshes.append(
            (
                np.asarray(vertices, dtype=np.float32),
                np.asarray(faces, dtype=np.uint32),
                color,
            )
        )
    return meshes


def mujoco_sky_gradient(height: int = 256, width: int = 512) -> np.ndarray:
    """Return the common MuJoCo blue-to-black skybox gradient."""

    if height < 2 or width < 2:
        raise ValueError("Sky image dimensions must be at least 2")
    top = np.asarray((0.3, 0.5, 0.7), dtype=np.float32)
    bottom = np.zeros(3, dtype=np.float32)
    blend = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    column = top[None, :] * (1.0 - blend) + bottom[None, :] * blend
    image = np.repeat(column[:, None, :], width, axis=1)
    return np.rint(image * 255.0).astype(np.uint8)


def install_mujoco_standard_visuals(server) -> None:
    """Overlay MuJoCo-style visuals without changing the physical terrain."""

    server.scene.set_background_image(
        mujoco_sky_gradient(),
        format="jpeg",
        jpeg_quality=90,
    )
    for index, (vertices, faces, color) in enumerate(mujoco_checker_meshes()):
        server.scene.add_mesh_simple(
            f"/mujoco_standard/floor_{index}",
            vertices,
            faces,
            color=color,
            side="double",
            position=(0.0, 0.0, 0.0005),
            cast_shadow=False,
            receive_shadow=0.2,
        )


def list_motion_files(project_dir: Path) -> list[Path]:
    """Return selectable HTH4 motions, reserving 22D002 for reset."""

    return sorted(
        path
        for path in project_dir.glob("*.xml")
        if not path.name.startswith(HOME_MOTION_PREFIX)
    )


def motion_frame_count(command: bytes) -> int:
    """Read the 1-255 Pos interpolation-frame count from a command."""

    if len(command) < 9 or command[1] != 0x10:
        raise ValueError("Not an HTH4 Pos command")
    return max(1, command[7])


def _frame_counts(xml_path: Path) -> dict[str, int]:
    root = ET.parse(xml_path).getroot()
    result = {}
    for activity in root.findall("./Activities/anyType"):
        if activity.findtext("BaseName") != "Pos":
            continue
        code = activity.find("./ProgramCode/anyType")
        if code is None or not code.text:
            continue
        command = bytes(int(token, 16) for token in code.text.split())
        result[activity.findtext("Name", "Pos")] = motion_frame_count(command)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HTH4サンプルを選択して物理環境で再生します"
    )
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    motion_files = list_motion_files(project_dir)
    if not motion_files:
        parser.error(f"再生可能なXMLがありません: {project_dir}")
    home_path = next(project_dir.glob(f"{HOME_MOTION_PREFIX}*.xml"), None)
    trim_path = next(project_dir.glob("*.h4p"), None)
    if home_path is None or trim_path is None:
        parser.error("22D002ホームXMLまたはHTH4プロジェクト(.h4p)がありません")

    import viser
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.viser.scene import MjlabViserScene

    from khrban.viser_ja import install_japanese_viser_labels

    install_japanese_viser_labels()
    trim = parse_trim(trim_path)
    home_frames = position_frames(home_path)
    if not home_frames:
        parser.error(f"ホームPosがありません: {home_path}")
    home_raw = {device: 7500 for device in trim}
    home_raw.update(home_frames[-1])
    home_joints = frame_to_joint_positions(home_raw, trim)

    cfg = make_standing_env_cfg(num_envs=1, play=True)
    cfg.decimation = 3
    cfg.terminations = {}
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
    sim = env.unwrapped.sim
    action_term = env.action_manager.get_term("joint_pos")
    joint_names = list(action_term._target_names)
    offset = action_term._offset[0]
    scale = float(action_term._scale)

    server = viser.ViserServer(
        port=args.port,
        label="KHRBan HTH4 physical playback",
    )
    scene = MjlabViserScene(
        server=server,
        mj_model=sim.mj_model,
        num_envs=1,
    )
    scene.camera_tracking_enabled = False
    scene.show_only_selected = False
    scene.create_scene_gui(
        camera_distance=0.85,
        camera_azimuth=180.0,
        camera_elevation=-10.0,
        show_debug_viz_control=False,
    )
    install_mujoco_standard_visuals(server)

    labels = [path.stem for path in motion_files]
    paths_by_label = dict(zip(labels, motion_files, strict=True))
    initial = next(
        (label for label in labels if label.startswith("22D003_")), labels[0]
    )
    with server.gui.add_folder("モーション再生"):
        selector = server.gui.add_dropdown(
            "再生するモーション",
            options=labels,
            initial_value=initial,
        )
        play_button = server.gui.add_button("再生")
        reset_button = server.gui.add_button("ホームへ戻す")
        status = server.gui.add_html("ホームポジションへ移動中")

    request_lock = threading.Lock()
    requested_motion: list[Path | None] = [None]
    reset_requested = threading.Event()

    @play_button.on_click
    def _request_play(_event) -> None:
        with request_lock:
            if requested_motion[0] is None:
                requested_motion[0] = paths_by_label[selector.value]

    @reset_button.on_click
    def _request_reset(_event) -> None:
        reset_requested.set()

    def action_for(targets: dict[str, float]) -> torch.Tensor:
        absolute = offset.clone()
        for index, name in enumerate(joint_names):
            absolute[index] = targets.get(name, float(offset[index].cpu()))
        return ((absolute - offset) / scale).unsqueeze(0)

    def publish(state: str, pose: str = "22D002 ホームポジション") -> None:
        data = sim.data
        scene.update_from_arrays(
            data.xpos[:1].cpu().numpy(),
            data.xmat[:1].cpu().numpy(),
            None,
            None,
        )
        base_id = sim.mj_model.body("khr/base_link").id
        base_z = float(data.xpos[0, base_id, 2].cpu())
        status.content = (
            f"<div><strong>状態:</strong> {state}<br/>"
            f"<strong>姿勢:</strong> {pose}<br/>"
            "<strong>周期:</strong> 15 ms<br/>"
            "<strong>物理:</strong> 重力・床接触・KRS-2552 有効<br/>"
            f"<strong>基部Z:</strong> {base_z * 1000:.1f} mm</div>"
        )
        server.flush()

    def step_realtime(action: torch.Tensor, next_tick: float) -> float:
        env.step(action)
        next_tick += FRAME_PERIOD_S
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        return next_tick

    home_action = action_for(home_joints)

    def reset_home() -> float:
        env.reset()
        next_tick = time.monotonic()
        for index in range(100):
            next_tick = step_realtime(home_action, next_tick)
            if index % 2 == 0:
                publish("ホームへリセット中")
        reset_requested.clear()
        publish("待機中・モーションを選択して再生してください")
        return next_tick

    next_tick = reset_home()
    print(
        f"READY http://localhost:{args.port}/ motions={len(motion_files)}",
        flush=True,
    )
    try:
        while True:
            selected = None
            with request_lock:
                if requested_motion[0] is not None:
                    selected = requested_motion[0]
                    requested_motion[0] = None

            if selected is None:
                next_tick = step_realtime(home_action, next_tick)
                if reset_requested.is_set():
                    next_tick = reset_home()
                else:
                    publish("待機中・モーションを選択して再生してください")
                continue

            selector.disabled = True
            play_button.disabled = True
            reset_requested.clear()
            sequence = playback_position_sequence(selected)
            frame_counts = _frame_counts(selected)
            accumulated = home_raw.copy()
            current = home_joints.copy()
            for pose_index, (pose_name, frame) in enumerate(sequence, start=1):
                if reset_requested.is_set():
                    break
                accumulated.update(frame)
                destination = current.copy()
                destination.update(frame_to_joint_positions(accumulated, trim))
                count = frame_counts[pose_name]
                start = current.copy()
                for frame_index in range(1, count + 1):
                    if reset_requested.is_set():
                        break
                    alpha = frame_index / count
                    target = {
                        name: start.get(name, float(offset[index].cpu()))
                        + (
                            destination.get(name, float(offset[index].cpu()))
                            - start.get(name, float(offset[index].cpu()))
                        )
                        * alpha
                        for index, name in enumerate(joint_names)
                    }
                    next_tick = step_realtime(action_for(target), next_tick)
                    if frame_index % 2 == 0 or frame_index == count:
                        publish(
                            f"再生中: {selected.stem}",
                            f"{pose_index:02d}/{len(sequence):02d} {pose_name} "
                            f"({frame_index}/{count})",
                        )
                current = destination

            next_tick = reset_home()
            selector.disabled = False
            play_button.disabled = False
    finally:
        env.close()
        server.stop()


if __name__ == "__main__":
    main()
