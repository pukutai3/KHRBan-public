"""Viser bridge for displaying the environments stepped by PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import socket
import sys
import threading
from typing import Any

import numpy as np
from mjlab.rl import MjlabOnPolicyRunner


LIVE_VIEWER_ENV = "KHRBAN_LIVE_VIEWER_NUM_ENVS"
LIVE_VIEWER_PORT_ENV = "KHRBAN_LIVE_VIEWER_PORT"
LIVE_VIEWER_RELAY_SOCKET_ENV = "KHRBAN_LIVE_VIEWER_RELAY_SOCKET"
LIVE_VIEWER_SPACING = 0.65
LIVE_VIEWER_MAX_FRAME_BYTES = 256 * 1024


def requested_live_viewer_num_envs() -> int:
    """Return the requested display count, or zero when live display is off."""

    value = os.environ.get(LIVE_VIEWER_ENV, "0")
    try:
        return max(0, int(value))
    except ValueError as exc:
        raise ValueError(f"{LIVE_VIEWER_ENV} must be an integer: {value!r}") from exc


def first_environments(tensor: Any, count: int) -> np.ndarray:
    """Copy only the displayed environments from a Warp or Torch tensor."""

    return tensor[:count].cpu().numpy()


def preview_grid_shape(count: int) -> tuple[int, int]:
    """Return a compact row/column layout, square for square counts."""

    if count <= 0:
        return 0, 0
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    return rows, columns


def square_preview_grid(count: int, spacing: float = LIVE_VIEWER_SPACING) -> np.ndarray:
    """Create centered preview-only origins in row-major order."""

    rows, columns = preview_grid_shape(count)
    if count == 0:
        return np.empty((0, 3), dtype=float)
    row_index, column_index = np.unravel_index(np.arange(count), (rows, columns))
    origins = np.zeros((count, 3), dtype=float)
    origins[:, 0] = -row_index * spacing
    origins[:, 1] = column_index * spacing
    origins[:, :2] -= origins[:, :2].mean(axis=0)
    return origins


def arrange_preview_positions(
    world_positions: np.ndarray,
    source_origins: np.ndarray,
    spacing: float = LIVE_VIEWER_SPACING,
) -> np.ndarray:
    """Rebase batched world positions onto the display-only preview grid."""

    if world_positions.shape[0] != source_origins.shape[0]:
        raise ValueError("world positions and source origins must have equal counts")
    target_origins = square_preview_grid(world_positions.shape[0], spacing)
    return world_positions - source_origins[:, None, :] + target_origins[:, None, :]


@dataclass(frozen=True)
class TrainingFrame:
    """One actual post-step training frame transferred to the supervisor."""

    body_positions: np.ndarray
    body_rotations: np.ndarray
    run_name: str
    step_count: int
    total_envs: int


def encode_training_frame(
    *,
    body_positions: np.ndarray,
    body_rotations: np.ndarray,
    run_name: str,
    step_count: int,
    total_envs: int,
) -> bytes:
    """Encode a local IPC frame without allowing object deserialization."""

    stream = BytesIO()
    np.savez(
        stream,
        body_positions=np.asarray(body_positions, dtype=np.float32),
        body_rotations=np.asarray(body_rotations, dtype=np.float32),
        run_name=np.asarray(run_name),
        step_count=np.asarray(step_count, dtype=np.int64),
        total_envs=np.asarray(total_envs, dtype=np.int64),
    )
    payload = stream.getvalue()
    if len(payload) > LIVE_VIEWER_MAX_FRAME_BYTES:
        raise ValueError(f"Live training frame is too large: {len(payload)} bytes")
    return payload


def decode_training_frame(payload: bytes) -> TrainingFrame:
    """Decode a frame produced by :func:`encode_training_frame`."""

    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        return TrainingFrame(
            body_positions=archive["body_positions"],
            body_rotations=archive["body_rotations"],
            run_name=str(archive["run_name"].item()),
            step_count=int(archive["step_count"].item()),
            total_envs=int(archive["total_envs"].item()),
        )


def _actual_training_arrays(
    env: Any,
    num_envs: int,
    source_origins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = env.unwrapped.sim.data
    return (
        arrange_preview_positions(
            first_environments(data.xpos, num_envs),
            source_origins,
        ),
        first_environments(data.xmat, num_envs),
    )


class LiveTrainingRelayPublisher:
    """Send actual training states to the supervisor-owned viewer."""

    def __init__(
        self,
        env: Any,
        num_envs: int,
        socket_path: str,
        run_name: str,
    ) -> None:
        self.env = env
        self.num_envs = min(num_envs, env.num_envs)
        self.socket_path = socket_path
        self.run_name = run_name
        self.step_count = 0
        self.update_interval = 2
        self.source_origins = first_environments(
            env.unwrapped.scene.env_origins,
            self.num_envs,
        )
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.setblocking(False)
        self._failure_reported = False
        self.update(force=True)

    def update(self, force: bool = False) -> None:
        self.step_count += 1
        if not force and self.step_count % self.update_interval:
            return
        try:
            positions, rotations = _actual_training_arrays(
                self.env,
                self.num_envs,
                self.source_origins,
            )
            payload = encode_training_frame(
                body_positions=positions,
                body_rotations=rotations,
                run_name=self.run_name,
                step_count=self.step_count,
                total_envs=self.env.num_envs,
            )
            self.socket.sendto(payload, self.socket_path)
        except Exception as error:
            # Visualization is optional telemetry and must never interrupt PPO.
            if not self._failure_reported:
                print(
                    f"[LIVE VIEWER] relay unavailable; training continues: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                self._failure_reported = True
            return
        self._failure_reported = False

    def close(self) -> None:
        self.socket.close()


class PersistentLiveTrainingViewer:
    """Own one Viser server for the complete automatic-training lifetime."""

    def __init__(self, num_envs: int, port: int = 8080) -> None:
        import viser
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.viewer.viser.scene import MjlabViserScene

        from khrban.tasks.velocity import make_velocity_env_cfg
        from khrban.viser_ja import install_japanese_viser_labels

        install_japanese_viser_labels()
        self.num_envs = num_envs
        self.phase = "準備中"
        self.run_name = "未指定"
        self.step_count = 0
        self.total_envs = 0
        self._closed = False
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.socket_path = f"/tmp/khrban-live-viewer-{os.getpid()}.sock"
        Path(self.socket_path).unlink(missing_ok=True)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(self.socket_path)
        self.socket.settimeout(0.25)
        self._previous_socket_env = os.environ.get(LIVE_VIEWER_RELAY_SOCKET_ENV)
        os.environ[LIVE_VIEWER_RELAY_SOCKET_ENV] = self.socket_path

        cfg = make_velocity_env_cfg(num_envs=1, play=True)
        self.env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        self.server = viser.ViserServer(port=port, label="KHRBan live training")
        self.scene = MjlabViserScene(
            server=self.server,
            mj_model=self.env.unwrapped.sim.mj_model,
            num_envs=self.num_envs,
        )
        self.scene.camera_tracking_enabled = False
        self.scene.show_only_selected = False
        self.scene.create_scene_gui(
            camera_distance=3.2,
            camera_azimuth=45.0,
            camera_elevation=25.0,
            show_debug_viz_control=False,
        )
        with self.server.gui.add_folder("実学習"):
            self.status = self.server.gui.add_html("")
        self._show_initial_pose()
        self._update_status()
        self.server.flush()
        self.thread = threading.Thread(
            target=self._receive_frames,
            name="khrban-live-viewer-relay",
            daemon=True,
        )
        self.thread.start()

    def _show_initial_pose(self) -> None:
        data = self.env.unwrapped.sim.data
        source_origin = first_environments(
            self.env.unwrapped.scene.env_origins,
            1,
        )[0]
        relative_positions = first_environments(data.xpos, 1)[0] - source_origin
        target_origins = square_preview_grid(self.num_envs)
        positions = (
            np.repeat(relative_positions[None, :, :], self.num_envs, axis=0)
            + target_origins[:, None, :]
        )
        rotations = np.repeat(
            first_environments(data.xmat, 1),
            self.num_envs,
            axis=0,
        )
        self.scene.update_from_arrays(positions, rotations)

    def _update_status(self) -> None:
        rows, columns = preview_grid_shape(self.num_envs)
        self.status.content = (
            '<div style="font-size:0.9em;line-height:1.35;padding:0 0.5em">'
            "<strong>表示元:</strong> PPO学習プロセスの実環境<br/>"
            f"<strong>状態:</strong> {self.phase}<br/>"
            f"<strong>run:</strong> {self.run_name}<br/>"
            f"<strong>表示:</strong> {self.num_envs} / {self.total_envs or '-'} 環境<br/>"
            f"<strong>配置:</strong> {rows}行 × {columns}列<br/>"
            f"<strong>学習ステップ:</strong> {self.step_count}<br/>"
            "<strong>ラウンド間:</strong> 最後の実学習フレームを保持"
            "</div>"
        )

    def set_phase(self, phase: str, run_name: str | None = None) -> None:
        with self._lock:
            self.phase = phase
            if run_name:
                self.run_name = run_name
            self._update_status()
            self.server.flush()

    def _receive_frames(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self.socket.recv(LIVE_VIEWER_MAX_FRAME_BYTES)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            try:
                frame = decode_training_frame(payload)
                if frame.body_positions.shape[0] != self.num_envs:
                    continue
                with self._lock:
                    self.scene.update_from_arrays(
                        frame.body_positions,
                        frame.body_rotations,
                    )
                    self.phase = "学習中"
                    self.run_name = frame.run_name
                    self.step_count = frame.step_count
                    self.total_envs = frame.total_envs
                    self._update_status()
                    self.server.flush()
            except Exception as error:
                # Keep the server and receiver alive after a malformed/dropped frame.
                print(
                    f"[LIVE VIEWER] ignored one relay frame: {error}",
                    file=sys.stderr,
                    flush=True,
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self.socket.close()
        self.thread.join(timeout=2.0)
        Path(self.socket_path).unlink(missing_ok=True)
        if self._previous_socket_env is None:
            os.environ.pop(LIVE_VIEWER_RELAY_SOCKET_ENV, None)
        else:
            os.environ[LIVE_VIEWER_RELAY_SOCKET_ENV] = self._previous_socket_env
        self.server.stop()
        self.env.close()


class LiveTrainingViewer:
    """Publish actual simulator state from the training process to Viser."""

    def __init__(
        self,
        env: Any,
        num_envs: int,
        port: int = 8080,
        run_name: str = "",
    ) -> None:
        import viser
        from mjlab.viewer.viser.scene import MjlabViserScene

        from khrban.viser_ja import install_japanese_viser_labels

        install_japanese_viser_labels()
        self.env = env
        self.num_envs = min(num_envs, env.num_envs)
        self.step_count = 0
        self.run_name = run_name
        self.update_interval = 2
        self.server = viser.ViserServer(port=port, label="KHRBan live training")
        sim = env.unwrapped.sim
        self.scene = MjlabViserScene(
            server=self.server,
            mj_model=sim.mj_model,
            num_envs=self.num_envs,
        )
        self.scene.camera_tracking_enabled = False
        self.scene.show_only_selected = False
        self.scene.create_scene_gui(
            camera_distance=3.2,
            camera_azimuth=45.0,
            camera_elevation=25.0,
            show_debug_viz_control=False,
        )
        with self.server.gui.add_folder("実学習"):
            self.status = self.server.gui.add_html("")
        self.source_origins = first_environments(
            env.unwrapped.scene.env_origins,
            self.num_envs,
        )
        self._update_status()
        self.update(force=True)

    def _update_status(self) -> None:
        rows, columns = preview_grid_shape(self.num_envs)
        self.status.content = (
            '<div style="font-size:0.9em;line-height:1.35;padding:0 0.5em">'
            "<strong>表示元:</strong> PPO学習プロセスの実環境<br/>"
            f"<strong>run:</strong> {self.run_name or '未指定'}<br/>"
            f"<strong>表示:</strong> {self.num_envs} / {self.env.num_envs} 環境<br/>"
            f"<strong>配置:</strong> {rows}行 × {columns}列<br/>"
            f"<strong>学習ステップ:</strong> {self.step_count}<br/>"
            "<strong>転送時点:</strong> env.step() 完了直後"
            "</div>"
        )

    def update(self, force: bool = False) -> None:
        """Push one post-step training snapshot without advancing simulation."""

        self.step_count += 1
        if not force and self.step_count % self.update_interval:
            return
        body_positions, body_rotations = _actual_training_arrays(
            self.env,
            self.num_envs,
            self.source_origins,
        )
        data = self.env.unwrapped.sim.data
        self.scene.update_from_arrays(
            body_positions,
            body_rotations,
            arrange_preview_positions(
                first_environments(data.mocap_pos, self.num_envs),
                self.source_origins,
            )
            if self.scene.mj_model.nmocap > 0
            else None,
            first_environments(data.mocap_quat, self.num_envs)
            if self.scene.mj_model.nmocap > 0
            else None,
        )
        self._update_status()
        self.server.flush()

    def close(self) -> None:
        self.server.stop()


class KhrLiveViewerOnPolicyRunner(MjlabOnPolicyRunner):
    """MjLab runner that mirrors post-step training state to a web viewer."""

    def __init__(self, env: Any, train_cfg: dict, log_dir=None, device="cpu") -> None:
        self.live_viewer: LiveTrainingViewer | LiveTrainingRelayPublisher | None = None
        display_count = requested_live_viewer_num_envs()
        if display_count:
            run_name = Path(log_dir).name if log_dir else ""
            relay_socket = os.environ.get(LIVE_VIEWER_RELAY_SOCKET_ENV)
            if relay_socket:
                self.live_viewer = LiveTrainingRelayPublisher(
                    env,
                    display_count,
                    socket_path=relay_socket,
                    run_name=run_name,
                )
            else:
                port = int(os.environ.get(LIVE_VIEWER_PORT_ENV, "8080"))
                self.live_viewer = LiveTrainingViewer(
                    env,
                    display_count,
                    port=port,
                    run_name=run_name,
                )
            original_step = env.step

            def step_and_publish(actions):
                result = original_step(actions)
                assert self.live_viewer is not None
                self.live_viewer.update()
                return result

            env.step = step_and_publish
        super().__init__(env, train_cfg, log_dir, device)

    def learn(self, *args, **kwargs) -> None:
        try:
            super().learn(*args, **kwargs)
        finally:
            if self.live_viewer is not None:
                self.live_viewer.close()
