import math

from bam.model import load_model

from khrban.actuators import KRS_2552_MODEL_PATH, make_krs_2552_actuator_cfg


KRS_2552_PEAK_TORQUE_NM = 14.0 * 0.0980665
KRS_2552_NO_LOAD_SPEED_RAD_S = (math.pi / 3.0) / 0.14


def test_krs_2552_model_matches_requested_endpoints() -> None:
    model = load_model(str(KRS_2552_MODEL_PATH))
    actuator = model.actuator

    torque_nm = actuator.vin * model.kt.value / model.R.value
    speed_rad_s = actuator.vin / model.kt.value

    assert type(actuator).__name__ == "XL320Actuator"
    assert math.isclose(torque_nm, KRS_2552_PEAK_TORQUE_NM, abs_tol=1e-12)
    assert math.isclose(speed_rad_s, KRS_2552_NO_LOAD_SPEED_RAD_S, abs_tol=1e-12)


def test_mjlab_config_uses_custom_model_and_all_continuous_joints() -> None:
    cfg = make_krs_2552_actuator_cfg()

    assert cfg.json_path == str(KRS_2552_MODEL_PATH)
    assert cfg.target_names_expr == (r"continuous_joint_.*",)
    assert cfg.vin == 7.5
    assert cfg._resolved_json_path == str(KRS_2552_MODEL_PATH)
