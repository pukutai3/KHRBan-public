import numpy as np
import pytest
import torch

from khrban.live_training import (
    LIVE_VIEWER_ENV,
    arrange_preview_positions,
    first_environments,
    square_preview_grid,
    requested_live_viewer_num_envs,
)


def test_live_viewer_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LIVE_VIEWER_ENV, raising=False)
    assert requested_live_viewer_num_envs() == 0


def test_live_viewer_requires_numeric_environment_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LIVE_VIEWER_ENV, "sixteen")
    with pytest.raises(ValueError, match=LIVE_VIEWER_ENV):
        requested_live_viewer_num_envs()


def test_only_actual_first_sixteen_training_environments_are_copied() -> None:
    training_state = torch.arange(32 * 3).reshape(32, 3)
    displayed = first_environments(training_state, 16)

    assert isinstance(displayed, np.ndarray)
    assert displayed.shape == (16, 3)
    np.testing.assert_array_equal(displayed, training_state[:16].numpy())


def test_sixteen_preview_environments_form_centered_four_by_four_grid() -> None:
    grid = square_preview_grid(16, spacing=0.65)

    assert grid.shape == (16, 3)
    assert len(np.unique(grid[:, 0])) == 4
    assert len(np.unique(grid[:, 1])) == 4
    np.testing.assert_allclose(grid.mean(axis=0), np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(np.ptp(grid[:, :2], axis=0), [1.95, 1.95])


def test_preview_rebases_world_positions_without_changing_robot_pose() -> None:
    source_origins = np.array([[31.0, -31.0, 0.0], [31.0, -29.0, 0.0]])
    relative_body_positions = np.array(
        [
            [[0.0, 0.0, 0.2], [0.1, 0.0, 0.4]],
            [[0.0, 0.0, 0.2], [-0.1, 0.0, 0.4]],
        ]
    )
    world_positions = relative_body_positions + source_origins[:, None, :]

    arranged = arrange_preview_positions(
        world_positions,
        source_origins,
        spacing=0.65,
    )
    target_origins = square_preview_grid(2, spacing=0.65)

    np.testing.assert_allclose(
        arranged - target_origins[:, None, :],
        relative_body_positions,
    )
