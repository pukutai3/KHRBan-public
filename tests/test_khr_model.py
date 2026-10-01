import math
import xml.etree.ElementTree as ET

import mujoco

from khrban.model import (
    KHR_TORSO_BODY_NAME,
    build_khr_spec,
    expanded_khr_urdf,
    sample_motion_joint_limits,
)


def test_expanded_urdf_has_resolved_meshes_and_no_xacro_include() -> None:
    urdf = expanded_khr_urdf()

    assert "xacro:include" not in urdf
    assert "package://KHR3_001_description" not in urdf


def test_khr_spec_compiles_with_22_actuators() -> None:
    spec = build_khr_spec()
    model = spec.compile()

    assert model.nu == 22
    assert model.nq == 29
    assert model.nv == 28

    limits = sample_motion_joint_limits()
    for joint_id in range(1, model.njnt):
        joint_name = model.joint(joint_id).name
        assert model.jnt_limited[joint_id]
        assert math.isclose(
            model.jnt_range[joint_id, 0], limits[joint_name][0], abs_tol=1e-7
        )
        assert math.isclose(
            model.jnt_range[joint_id, 1], limits[joint_name][1], abs_tol=1e-7
        )


def test_khr_spec_defines_root_angular_momentum_sensor() -> None:
    model = build_khr_spec().compile()

    sensor_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "root_angmom"
    )
    assert sensor_id >= 0, (
        "KHR model lacks root_angmom required by the inherited angular_momentum reward"
    )
    assert model.sensor_type[sensor_id] == mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
    assert model.sensor_objid[sensor_id] == mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, KHR_TORSO_BODY_NAME
    )


def test_zero_pose_has_requested_servo_mounting_offsets() -> None:
    root = ET.fromstring(expanded_khr_urdf())
    actual = {
        joint.get("name"): tuple(
            float(value) for value in joint.find("origin").get("rpy").split()
        )
        for joint in root.findall("joint")
    }
    expected = {
        "continuous_joint_03": (0.0, math.radians(-45.0), 0.0),
        "continuous_joint_04": (math.radians(90.0), 0.0, 0.0),
        "continuous_joint_06": (0.0, math.radians(-90.0), 0.0),
        "continuous_joint_07": (0.0, math.radians(-45.0), 0.0),
        "continuous_joint_08": (math.radians(-90.0), 0.0, 0.0),
        "continuous_joint_10": (0.0, math.radians(-90.0), 0.0),
        "continuous_joint_14": (0.0, math.radians(55.3), 0.0),
        "continuous_joint_15": (0.0, math.radians(23.9), 0.0),
        "continuous_joint_20": (0.0, math.radians(55.3), 0.0),
        "continuous_joint_21": (0.0, math.radians(23.9), 0.0),
    }

    for joint_name, expected_rpy in expected.items():
        assert all(
            math.isclose(value, target, abs_tol=1e-12)
            for value, target in zip(actual[joint_name], expected_rpy, strict=True)
        )
