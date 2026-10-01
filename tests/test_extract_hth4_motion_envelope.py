import json
import math

from khrban.hth4 import decode_position_code, position_to_joint_angle_rad
from khrban.model import SAMPLE_ENVELOPE_PATH


def test_decode_full_22_axis_home_position() -> None:
    code = (
        "35 10 FF FF 3F 00 00 14 "
        "4C 1D 4C 1D 2C 1A 6C 20 14 1E 84 1C 4C 1D 4C 1D "
        "7C 15 1C 25 4C 1D 4C 1D 4C 1D 4C 1D 40 1F 58 1B "
        "34 21 64 19 F4 1A A4 1F 4C 1D 4C 1D 9B"
    )

    positions = decode_position_code(code)

    assert positions is not None
    assert len(positions) == 22
    assert positions[0] == 7500
    assert positions[2] == 6700
    assert positions[3] == 8300
    assert positions[16] == 8500
    assert positions[17] == 6500


def test_decode_position_mask_supports_extension_channels() -> None:
    # Devices 0 and 24 are present; KHR uses 0-21 but the stored offset must
    # account for all 40 mask bits.
    code = "0D 10 01 00 00 01 00 08 4C 1D 34 21 00"

    positions = decode_position_code(code)

    assert positions == {0: 7500, 24: 8500}


def test_generated_envelope_uses_servo_origin_coordinates() -> None:
    envelope = json.loads(SAMPLE_ENVELOPE_PATH.read_text(encoding="utf-8"))

    assert envelope["not_mechanical_limits"] is True
    assert envelope["motion_file_count"] == 49
    assert envelope["position_frame_count"] == 440
    assert len(envelope["joints"]) == 22

    for values in envelope["joints"].values():
        assert 3500 <= values["observed_min_servo_target_raw"]
        assert values["observed_max_servo_target_raw"] <= 11500
        assert values["training_lower_rad"] <= 0.0
        assert values["training_upper_rad"] >= 0.0
        assert values["training_lower_rad"] <= values["observed_lower_rad"]
        assert values["training_upper_rad"] >= values["observed_upper_rad"]


def test_position_conversion_applies_trim_and_mounting_direction() -> None:
    # Head yaw is reversed relative to increasing HTH4 position values.
    assert math.isclose(
        math.degrees(position_to_joint_angle_rad(0, 8500, 0)),
        -33.75,
        abs_tol=1e-12,
    )

    # Waist yaw is also reversed relative to increasing HTH4 position values.
    assert math.isclose(
        math.degrees(position_to_joint_angle_rad(1, 8500, 0)),
        -33.75,
        abs_tol=1e-12,
    )

    # Left shoulder pitch: position 7500 + trim -1350, then mounting sign -1.
    angle = position_to_joint_angle_rad(2, 7500, -1350)

    assert math.isclose(math.degrees(angle), 45.5625, abs_tol=1e-12)

    # Both hip-yaw servos rotate opposite to the URDF joint coordinate.
    assert math.isclose(
        math.degrees(position_to_joint_angle_rad(10, 8100, 0)),
        -20.25,
        abs_tol=1e-12,
    )
    assert math.isclose(
        math.degrees(position_to_joint_angle_rad(11, 6900, 0)),
        20.25,
        abs_tol=1e-12,
    )

    # Both ankle-roll servos rotate opposite to the URDF joint coordinate.
    assert math.isclose(
        math.degrees(position_to_joint_angle_rad(20, 7500, 20)),
        -0.675,
        abs_tol=1e-12,
    )
    assert math.isclose(
        math.degrees(position_to_joint_angle_rad(21, 7500, -20)),
        0.675,
        abs_tol=1e-12,
    )


def test_home_pose_has_expected_mirrored_pitch_angles() -> None:
    envelope = json.loads(SAMPLE_ENVELOPE_PATH.read_text(encoding="utf-8"))
    joints = envelope["joints"]

    assert math.isclose(
        joints["continuous_joint_03"]["home_joint_angle_deg"], 72.5625
    )
    assert math.isclose(
        joints["continuous_joint_07"]["home_joint_angle_deg"], -72.5625
    )
    assert math.isclose(
        joints["continuous_joint_13"]["home_joint_angle_deg"], -17.8875
    )
    assert math.isclose(
        joints["continuous_joint_19"]["home_joint_angle_deg"], 17.8875
    )
    for left, right, expected in (
        ("continuous_joint_14", "continuous_joint_20", -26.325),
        ("continuous_joint_15", "continuous_joint_21", -40.5),
    ):
        assert math.isclose(joints[left]["home_joint_angle_deg"], expected)
        assert math.isclose(joints[right]["home_joint_angle_deg"], expected)
