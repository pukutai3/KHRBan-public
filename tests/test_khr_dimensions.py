"""Regression checks for KHR dimensions that affect locomotion task tuning."""

import math

import mujoco
import numpy as np

from khrban.model import build_khr_spec


def _joint_anchor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return data.xanchor[joint_id]


def _distance(model: mujoco.MjModel, data: mujoco.MjData, a: str, b: str) -> float:
    return float(np.linalg.norm(_joint_anchor(model, data, a) - _joint_anchor(model, data, b)))


def test_khr_joint_axis_dimensions() -> None:
    model = build_khr_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    expected_distances = {
        # Bilateral spacing at the servo mounting-zero configuration.
        ("continuous_joint_03", "continuous_joint_07"): 0.0978,
        ("continuous_joint_11", "continuous_joint_17"): 0.0434,
        # Left and right hip-pitch -> knee links.
        ("continuous_joint_13", "continuous_joint_14"): 0.065,
        ("continuous_joint_19", "continuous_joint_20"): 0.065,
        # Left and right knee -> ankle-pitch links.
        ("continuous_joint_14", "continuous_joint_15"): math.sqrt(
            0.003363**2 + 0.0005**2 + 0.064913**2
        ),
        ("continuous_joint_20", "continuous_joint_21"): math.sqrt(
            0.003363**2 + 0.0005**2 + 0.064913**2
        ),
    }

    for pair, expected in expected_distances.items():
        assert math.isclose(
            _distance(model, data, *pair), expected, abs_tol=1.0e-9
        ), pair
