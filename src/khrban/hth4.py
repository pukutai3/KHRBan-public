"""HeartToHeart4 position conversion for the KHR-3HV 22-axis layout."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET


DEVICE_TO_JOINT = {
    0: "continuous_joint_02",   # head yaw
    1: "continuous_joint_01",   # waist yaw
    2: "continuous_joint_03",   # left shoulder pitch
    3: "continuous_joint_07",   # right shoulder pitch
    4: "continuous_joint_04",   # left shoulder roll
    5: "continuous_joint_08",   # right shoulder roll
    6: "continuous_joint_05",   # left shoulder yaw
    7: "continuous_joint_09",   # right shoulder yaw
    8: "continuous_joint_06",   # left elbow pitch
    9: "continuous_joint_10",   # right elbow pitch
    10: "continuous_joint_11",  # left hip yaw
    11: "continuous_joint_17",  # right hip yaw
    12: "continuous_joint_12",  # left hip roll
    13: "continuous_joint_18",  # right hip roll
    14: "continuous_joint_13",  # left hip pitch
    15: "continuous_joint_19",  # right hip pitch
    16: "continuous_joint_14",  # left knee pitch
    17: "continuous_joint_20",  # right knee pitch
    18: "continuous_joint_15",  # left ankle pitch
    19: "continuous_joint_21",  # right ankle pitch
    20: "continuous_joint_16",  # left ankle roll
    21: "continuous_joint_22",  # right ankle roll
}

# Converts increasing KRS position values to the URDF joint coordinate.  The
# negative entries are the servos whose confirmed mounting orientation is
# opposite to the corresponding URDF coordinate.
DEVICE_TO_URDF_SIGN = {
    device: -1.0
    if device in {0, 1, 2, 3, 9, 10, 11, 14, 15, 17, 19, 20, 21}
    else 1.0
    for device in DEVICE_TO_JOINT
}

NEUTRAL_RAW = 7500
RAW_SPAN = 8000
ANGLE_SPAN_DEG = 270.0
RAD_PER_RAW_UNIT = math.radians(ANGLE_SPAN_DEG / RAW_SPAN)


def parse_trim(trim_path: Path) -> dict[int, dict]:
    """Read servo names, IDs, and trim values from an HTH4 project or trim file."""

    root = ET.parse(trim_path).getroot()
    servo_config = root.find("ServoConfigParams")
    entries = (
        servo_config.findall("DictionaryEntry")
        if servo_config is not None
        else root.findall("DictionaryEntry")
    )
    result = {}
    for entry in entries:
        value = entry.find("Value")
        if value is None:
            continue
        device = int(value.findtext("DeviceNumber", "-1"))
        if device in DEVICE_TO_JOINT:
            result[device] = {
                "servo_name": value.findtext("Name", ""),
                "ics_id": int(value.findtext("ID", "-1")),
                "trim_raw": int(value.findtext("Trim", "0")),
            }
    return result


def decode_position_code(text: str) -> dict[int, int] | None:
    """Decode one binary HTH4 Pos activity stored as hexadecimal text."""

    data = bytes(int(token, 16) for token in text.split())
    if len(data) < 9 or data[1] != 0x10 or data[0] != len(data):
        return None
    mask = int.from_bytes(data[2:7], "little")
    active = [device for device in range(40) if mask & (1 << device)]
    if len(data) != 8 + 2 * len(active) + 1:
        return None
    return {
        device: int.from_bytes(data[8 + 2 * i : 10 + 2 * i], "little")
        for i, device in enumerate(active)
    }


def position_frames(xml_path: Path) -> list[dict[int, int]]:
    """Return all servo-position frames from an HTH4 motion XML file."""

    root = ET.parse(xml_path).getroot()
    frames = []
    for activity in root.findall("./Activities/anyType"):
        if activity.findtext("BaseName") != "Pos":
            continue
        code = activity.find("./ProgramCode/anyType")
        if code is not None and code.text:
            decoded = decode_position_code(code.text)
            if decoded:
                frames.append(decoded)
    return frames


def playback_position_sequence(
    xml_path: Path,
) -> list[tuple[str, dict[int, int]]]:
    """Follow an HTH4 motion graph and return Pos frames in playback order."""

    root = ET.parse(xml_path).getroot()
    line_modes = {
        line.findtext("GUID", ""): line.findtext("ConnectMode", "Normal")
        for line in root.findall("./Lines/anyType")
    }
    activities = {}
    line_endpoints: dict[str, dict[str, str]] = {}
    start_guid = None
    loop_repetitions = None

    for activity in root.findall("./Activities/anyType"):
        guid = activity.findtext("GUID", "")
        activities[guid] = activity
        if activity.findtext("MotionFlag") == "Start":
            start_guid = guid
        line_guids = activity.findtext("ConnectedGuids", "").split(",")
        connect_types = activity.findtext("ConnectTypes", "").split(",")
        for line_guid, connect_type in zip(line_guids, connect_types, strict=True):
            endpoint = "source" if connect_type == "BeginConnect" else "target"
            line_endpoints.setdefault(line_guid, {})[endpoint] = guid
        if activity.findtext("BaseName") == "SetCounter":
            code = activity.find("./ProgramCode/anyType")
            if code is not None and code.text:
                data = bytes(int(token, 16) for token in code.text.split())
                loop_repetitions = data[-2]

    if start_guid is None:
        raise ValueError(f"No start activity in {xml_path}")
    if loop_repetitions is None:
        loop_repetitions = 1

    outgoing: dict[str, list[tuple[str, str]]] = {}
    for line_guid, endpoints in line_endpoints.items():
        if "source" in endpoints and "target" in endpoints:
            outgoing.setdefault(endpoints["source"], []).append(
                (line_modes[line_guid], endpoints["target"])
            )

    sequence = []
    loop_visits: dict[str, int] = {}
    current = start_guid
    for _ in range(10000):
        activity = activities[current]
        base_name = activity.findtext("BaseName")
        if base_name == "Pos":
            code = activity.find("./ProgramCode/anyType")
            if code is not None and code.text:
                frame = decode_position_code(code.text)
                if frame:
                    sequence.append((activity.findtext("Name", "Pos"), frame))

        choices = outgoing.get(current, [])
        if not choices:
            return sequence
        if base_name == "Loop":
            visits = loop_visits.get(current, 0) + 1
            loop_visits[current] = visits
            mode = "True" if visits >= loop_repetitions else "False"
            matches = [target for edge_mode, target in choices if edge_mode == mode]
        elif base_name and base_name.startswith("Cmp"):
            # For a finite pose list, leave an input/sensor wait loop through
            # its False branch. The loop itself contributes no Pos frame.
            matches = [
                target for edge_mode, target in choices if edge_mode == "False"
            ]
        else:
            matches = [
                target for edge_mode, target in choices if edge_mode == "Normal"
            ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one outgoing edge from {activity.findtext('Name')}, "
                f"found {len(matches)}"
            )
        current = matches[0]

    raise ValueError(f"Motion graph exceeded traversal limit in {xml_path}")


def position_to_joint_angle_rad(
    device: int, position_raw: int, trim_raw: int
) -> float:
    """Convert an HTH4 position plus trim to the URDF joint coordinate."""

    servo_target_raw = position_raw + trim_raw
    return (
        DEVICE_TO_URDF_SIGN[device]
        * (servo_target_raw - NEUTRAL_RAW)
        * RAD_PER_RAW_UNIT
    )


def frame_to_joint_positions(
    frame: dict[int, int], trim: dict[int, dict]
) -> dict[str, float]:
    """Convert an HTH4 frame to a URDF joint-name/angle mapping."""

    return {
        DEVICE_TO_JOINT[device]: position_to_joint_angle_rad(
            device, position_raw, trim[device]["trim_raw"]
        )
        for device, position_raw in frame.items()
        if device in DEVICE_TO_JOINT
    }
