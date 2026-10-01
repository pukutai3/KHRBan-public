from __future__ import annotations

from pathlib import Path

import numpy as np

from khrban.live_training import (
    LIVE_VIEWER_RELAY_SOCKET_ENV,
    decode_training_frame,
    encode_training_frame,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_frame_ipc_round_trip_preserves_actual_arrays() -> None:
    positions = np.arange(16 * 27 * 3, dtype=np.float32).reshape(16, 27, 3)
    rotations = np.arange(16 * 27 * 9, dtype=np.float32).reshape(16, 27, 9)

    frame = decode_training_frame(
        encode_training_frame(
            body_positions=positions,
            body_rotations=rotations,
            run_name="round-0001",
            step_count=42,
            total_envs=1536,
        )
    )

    np.testing.assert_array_equal(frame.body_positions, positions)
    np.testing.assert_array_equal(frame.body_rotations, rotations)
    assert frame.run_name == "round-0001"
    assert frame.step_count == 42
    assert frame.total_envs == 1536


def test_auto_training_supervisor_owns_one_persistent_viewer() -> None:
    auto_train = (REPO_ROOT / "src/khrban/auto_train.py").read_text()
    live_training = (REPO_ROOT / "src/khrban/live_training.py").read_text()

    assert "PersistentLiveTrainingViewer" in auto_train
    assert "persistent_viewer.set_phase" in auto_train
    assert LIVE_VIEWER_RELAY_SOCKET_ENV in live_training
    assert "LiveTrainingRelayPublisher" in live_training
