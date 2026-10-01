"""Extract the observed 22-axis position envelope from an HTH4 project."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from khrban.hth4 import (
    ANGLE_SPAN_DEG,
    DEVICE_TO_JOINT,
    DEVICE_TO_URDF_SIGN,
    NEUTRAL_RAW,
    parse_trim,
    position_frames,
    position_to_joint_angle_rad,
)

def project_hash(project_dir: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(project_dir)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def extract(project_dir: Path) -> dict:
    project_path = next(project_dir.glob("*.h4p"))
    trim_path = next(project_dir.glob("*.h4t"))
    trim = parse_trim(project_path)
    standalone_trim = parse_trim(trim_path)
    trim_differences = {
        str(device): {
            "project_trim_raw": trim[device]["trim_raw"],
            "standalone_trim_raw": standalone_trim[device]["trim_raw"],
        }
        for device in DEVICE_TO_JOINT
        if trim[device]["trim_raw"] != standalone_trim[device]["trim_raw"]
    }
    motion_files = sorted(project_dir.rglob("*.xml"))
    home_file = next(path for path in motion_files if "ホームポジション" in path.name)
    home_frames = [frame for frame in position_frames(home_file) if len(frame) == 22]
    if not home_frames:
        raise ValueError(f"No complete 22-axis home frame found in {home_file}")
    home = home_frames[-1]
    samples = {device: [] for device in DEVICE_TO_JOINT}
    frame_count = 0
    for motion_file in motion_files:
        for frame in position_frames(motion_file):
            frame_count += 1
            for device, raw in frame.items():
                if device in samples and 3500 <= raw <= 11500:
                    samples[device].append(raw)

    joints = {}
    for device, joint_name in DEVICE_TO_JOINT.items():
        values = samples[device]
        if not values:
            raise ValueError(f"No position samples found for device {device}")
        home_raw = home[device]
        trim_raw = trim[device]["trim_raw"]
        angles = [
            position_to_joint_angle_rad(device, value, trim_raw)
            for value in values
        ]
        observed_lower = min(angles)
        observed_upper = max(angles)
        training_lower = min(0.0, observed_lower)
        training_upper = max(0.0, observed_upper)
        joints[joint_name] = {
            "device_number": device,
            **trim[device],
            "position_to_urdf_sign": DEVICE_TO_URDF_SIGN[device],
            "sample_count": len(values),
            "home_raw": home_raw,
            "home_servo_target_raw": home_raw + trim_raw,
            "home_joint_angle_rad": position_to_joint_angle_rad(
                device, home_raw, trim_raw
            ),
            "home_joint_angle_deg": math.degrees(
                position_to_joint_angle_rad(device, home_raw, trim_raw)
            ),
            "observed_min_raw": min(values),
            "observed_max_raw": max(values),
            "observed_min_servo_target_raw": min(values) + trim_raw,
            "observed_max_servo_target_raw": max(values) + trim_raw,
            "observed_lower_rad": observed_lower,
            "observed_upper_rad": observed_upper,
            "observed_lower_deg": math.degrees(observed_lower),
            "observed_upper_deg": math.degrees(observed_upper),
            "training_lower_rad": training_lower,
            "training_upper_rad": training_upper,
            "training_lower_deg": math.degrees(training_lower),
            "training_upper_deg": math.degrees(training_upper),
        }

    return {
        "kind": "observed_sample_motion_envelope",
        "not_mechanical_limits": True,
        "source_project": project_dir.name,
        "source_sha256": project_hash(
            project_dir, [project_path, trim_path, *motion_files]
        ),
        "trim_source": project_path.name,
        "standalone_trim_differences": trim_differences,
        "position_encoding": {
            "min_raw": 3500,
            "neutral_raw": NEUTRAL_RAW,
            "max_raw": 11500,
            "full_span_deg": ANGLE_SPAN_DEG,
            "servo_target_formula": "position_raw + trim_raw",
            "joint_angle_formula": (
                "position_to_urdf_sign * (servo_target_raw - 7500)"
            ),
        },
        "motion_file_count": len(motion_files),
        "position_frame_count": frame_count,
        "joints": joints,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = extract(args.project_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
