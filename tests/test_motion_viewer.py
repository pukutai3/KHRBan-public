from pathlib import Path

from khrban.motion_viewer import (
    list_motion_files,
    motion_frame_count,
    mujoco_checker_meshes,
    mujoco_sky_gradient,
)


def test_motion_viewer_exposes_independent_device_selection() -> None:
    source = (Path(__file__).parents[1] / "src/khrban/motion_viewer.py").read_text()
    assert 'parser.add_argument("--device", default="cuda:0")' in source
    assert "device=args.device" in source


def test_motion_selector_lists_xml_files_but_reserves_home_for_reset(
    tmp_path: Path,
) -> None:
    (tmp_path / "22D003_wave.xml").touch()
    (tmp_path / "22D002_home.xml").touch()
    (tmp_path / "notes.txt").touch()

    assert [path.name for path in list_motion_files(tmp_path)] == [
        "22D003_wave.xml"
    ]


def test_pos_frame_count_is_read_from_hth4_command_byte() -> None:
    command = bytes((11, 0x10, 1, 0, 0, 0, 0, 45, 0, 0, 0))

    assert motion_frame_count(command) == 45


def test_mujoco_visuals_provide_two_floor_colors_and_sky_gradient() -> None:
    meshes = mujoco_checker_meshes(half_extent=1.0, cell_size=0.5)
    sky = mujoco_sky_gradient(height=32, width=64)

    assert len(meshes) == 2
    assert {mesh[2] for mesh in meshes} == {(51, 76, 102), (26, 51, 76)}
    assert all(mesh[0].shape[1] == 3 and mesh[1].shape[1] == 3 for mesh in meshes)
    assert sky.shape == (32, 64, 3)
    assert sky.dtype.name == "uint8"
