import math

import pytest
import torch
from bam.mjlab import Simulator

from khrban.actuators import KRS_2552_MODEL_PATH


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MjLab GPU smoke test requires CUDA"
)


def test_krs_2552_model_steps_in_mjlab_on_gpu() -> None:
    log = {
        "dt": 0.005,
        "kp": 32.0,
        "vin": 7.5,
        "mass": 0.1,
        "arm_mass": 0.0,
        "length": 0.1,
        "entries": [
            {
                "goal_position": 0.0,
                "torque_enable": True,
                "position": 0.0,
                "speed": 0.0,
            },
            {
                "goal_position": 0.1,
                "torque_enable": True,
                "position": 0.0,
                "speed": 0.0,
            },
            {
                "goal_position": 0.1,
                "torque_enable": True,
                "position": 0.0,
                "speed": 0.0,
            },
        ],
    }

    positions, velocities, controls = Simulator(
        json_path=str(KRS_2552_MODEL_PATH), device="cuda"
    ).rollout_log(log)

    assert len(positions) == len(log["entries"])
    assert len(velocities) == len(log["entries"])
    assert len(controls) == len(log["entries"])
    assert all(math.isfinite(float(value)) for value in positions)
    assert all(math.isfinite(float(value)) for value in velocities)
    assert all(math.isfinite(float(value)) for value in controls)
