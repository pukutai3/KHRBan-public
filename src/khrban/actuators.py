"""MjLab actuator configurations for KHR robots."""

from pathlib import Path

from bam.mjlab import BamActuatorCfg


KHRBAN_ROOT = Path(__file__).resolve().parents[2]
KRS_2552_MODEL_PATH = (
    KHRBAN_ROOT
    / "KHR3_001_description"
    / "config"
    / "actuators"
    / "krs_2552_xl320_m6.json"
)


def make_krs_2552_actuator_cfg(
    target_names_expr: tuple[str, ...] = (r"continuous_joint_.*",),
) -> BamActuatorCfg:
    """Build the provisional KRS-2552 actuator config for all KHR joints.

    The model retains the XL320 M6 controller and friction parameters. Only its
    torque constant and resistance are tuned to reproduce the KRS-2552 peak
    torque and no-load speed at the XL320 controller's fixed 7.5 V supply.
    """

    return BamActuatorCfg(
        json_path=str(KRS_2552_MODEL_PATH),
        target_names_expr=target_names_expr,
        vin=7.5,
    )
