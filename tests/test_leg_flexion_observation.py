import pytest
import torch

from khrban.observations import (
    leg_flexion_angle_from_axis_positions,
    leg_flexion_ratio_from_axis_positions,
)


def test_leg_flexion_ratio_uses_angle_between_axis_links() -> None:
    axis_positions = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, -2.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, -1.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[1.0, 2.0, 3.0], [1.0, 2.0, 2.0], [1.0, 2.0, 1.0]],
            ],
        ],
        dtype=torch.float32,
    )

    ratios = leg_flexion_ratio_from_axis_positions(axis_positions)
    angles_deg = torch.rad2deg(
        leg_flexion_angle_from_axis_positions(axis_positions)
    )

    assert ratios.shape == (2, 2)
    torch.testing.assert_close(
        angles_deg,
        torch.tensor([[0.0, 90.0], [180.0, 0.0]]),
        atol=1.0e-4,
        rtol=0.0,
    )
    assert ratios[0, 0].item() == pytest.approx(0.0, abs=1.0e-6)
    assert ratios[0, 1].item() == pytest.approx(0.5, abs=1.0e-6)
    assert ratios[1, 0].item() == pytest.approx(1.0, abs=1.0e-6)
    assert ratios[1, 1].item() == pytest.approx(0.0, abs=1.0e-6)


def test_leg_flexion_ratio_rejects_zero_length_links() -> None:
    axis_positions = torch.zeros((1, 2, 3, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="non-zero"):
        leg_flexion_ratio_from_axis_positions(axis_positions)
