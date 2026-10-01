"""MuJoCo/MjLab model construction for KHR-3HV."""

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from bam.mjlab import (
    EntityArticulationInfoCfg,
    EntityCfg,
    create_motor_actuator,
)
from mjlab.terrains import TerrainEntityCfg

from .actuators import KHRBAN_ROOT, make_krs_2552_actuator_cfg


DESCRIPTION_ROOT = KHRBAN_ROOT / "KHR3_001_description"
XACRO_PATH = DESCRIPTION_ROOT / "urdf" / "KHR3_001.xacro"
MATERIALS_PATH = DESCRIPTION_ROOT / "urdf" / "materials.xacro"
JOINT_NAME_PREFIX = "continuous_joint_"
KHR_TORSO_BODY_NAME = "c_chest_c_1"
SAMPLE_ENVELOPE_PATH = (
    DESCRIPTION_ROOT
    / "config"
    / "joint_limits"
    / "khr_22dof_sample_envelope.json"
)

# Sole reference points measured from the KHR foot collision meshes in the
# confirmed HTH4 home pose.  They are body-local coordinates in meters.
KHR_FOOT_SITE_POSITIONS = {
    "l_foot_1": (-0.07185719, 0.01434324, -0.02630640),
    "r_foot_1": (-0.07185719, -0.01434332, -0.02630640),
}
KHR_FOOT_GEOM_NAMES = {
    "l_foot_1": "left_foot_collision",
    "r_foot_1": "right_foot_collision",
}
KHR_FOOT_SITE_NAMES = {
    "l_foot_1": "left_foot",
    "r_foot_1": "right_foot",
}


def sample_motion_joint_limits() -> dict[str, tuple[float, float]]:
    """Load conservative limits derived from the official sample motions."""

    data = json.loads(SAMPLE_ENVELOPE_PATH.read_text(encoding="utf-8"))
    if not data.get("not_mechanical_limits"):
        raise ValueError("Sample envelope must not be labeled as mechanical limits")
    return {
        joint_name: (
            float(values["training_lower_rad"]),
            float(values["training_upper_rad"]),
        )
        for joint_name, values in data["joints"].items()
    }


def sample_motion_home_joint_positions() -> dict[str, float]:
    """Load the confirmed HTH4 home pose in URDF joint coordinates."""

    data = json.loads(SAMPLE_ENVELOPE_PATH.read_text(encoding="utf-8"))
    return {
        joint_name: float(values["home_joint_angle_rad"])
        for joint_name, values in data["joints"].items()
    }


def expanded_khr_urdf() -> str:
    """Expand the simple Fusion xacro into a MuJoCo-readable URDF string."""

    root = ET.parse(XACRO_PATH).getroot()
    material_root = ET.parse(MATERIALS_PATH).getroot()
    xacro_include = "{http://www.ros.org/wiki/xacro}include"

    for element in list(root):
        if element.tag == xacro_include:
            root.remove(element)

    for material in reversed(list(material_root)):
        root.insert(0, deepcopy(material))

    package_prefix = "package://KHR3_001_description/"
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(package_prefix):
            relative_path = filename.removeprefix(package_prefix)
            mesh.set("filename", str(DESCRIPTION_ROOT / relative_path))

    return ET.tostring(root, encoding="unicode")


def build_khr_spec() -> mujoco.MjSpec:
    """Build a floating KHR spec with one motor per active joint."""

    spec = mujoco.MjSpec.from_string(expanded_khr_urdf())
    spec.body("base_link").add_freejoint(name="floating_base")
    spec.add_sensor(
        name="root_angmom",
        type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        objname=KHR_TORSO_BODY_NAME,
    )
    joint_names = [
        joint.name for joint in spec.joints if joint.name.startswith(JOINT_NAME_PREFIX)
    ]
    if len(joint_names) != 22:
        raise ValueError(f"Expected 22 KHR actuator joints, found {len(joint_names)}")

    joint_limits = sample_motion_joint_limits()
    for joint_name in joint_names:
        joint = spec.joint(joint_name)
        joint.limited = True
        joint.range = joint_limits[joint_name]
        create_motor_actuator(spec, joint_name, effort_limit=100.0)

    for body_name, site_position in KHR_FOOT_SITE_POSITIONS.items():
        foot_body = spec.body(body_name)
        foot_geoms = list(foot_body.geoms)
        if len(foot_geoms) != 1:
            raise ValueError(
                f"Expected one collision geom on {body_name}, found {len(foot_geoms)}"
            )
        foot_geoms[0].name = KHR_FOOT_GEOM_NAMES[body_name]
        foot_body.add_site(
            name=KHR_FOOT_SITE_NAMES[body_name],
            pos=site_position,
            size=(0.003,),
        )

    return spec


def make_khr_entity_cfg() -> EntityCfg:
    """Create an MjLab entity using the tuned KRS-2552 BAM actuator."""

    return EntityCfg(
        spec_fn=build_khr_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(make_krs_2552_actuator_cfg(),)
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.01),
            joint_pos=sample_motion_home_joint_positions(),
            joint_vel={r".*": 0.0},
        ),
    )


def make_flat_terrain_cfg() -> TerrainEntityCfg:
    """Create the flat ground used for initial standing and walking tasks."""

    return TerrainEntityCfg(terrain_type="plane", env_spacing=2.0)
