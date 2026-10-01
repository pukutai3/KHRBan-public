import pytest
import torch
from bam.mjlab import MujocoCfg, Scene, SceneCfg, Simulation, SimulationCfg

from khrban.model import make_flat_terrain_cfg, make_khr_entity_cfg


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="KHR MjLab smoke test requires CUDA"
)


def test_full_khr_entity_steps_on_gpu() -> None:
    scene = Scene(
        SceneCfg(
            num_envs=1,
            terrain=make_flat_terrain_cfg(),
            entities={"khr": make_khr_entity_cfg()},
        ),
        "cuda",
    )
    mj_model = scene.compile()
    sim = Simulation(
        num_envs=1,
        cfg=SimulationCfg(mujoco=MujocoCfg(timestep=0.005)),
        model=mj_model,
        device="cuda",
    )
    scene.initialize(sim.mj_model, sim.model, sim.data)
    sim.expand_model_fields(("dof_frictionloss", "dof_damping"))
    sim.reset()
    scene.reset()
    scene.write_data_to_sim()
    for _ in range(10):
        sim.step()
        scene.update(dt=0.005)

    entity = scene["khr"]
    assert sim.mj_model.nu == 22
    assert sim.mj_model.nq == 29
    assert sim.mj_model.nv == 28
    assert not entity.data.is_fixed_base
    assert sim.mj_model.geom("terrain").id >= 0
    assert len(entity.actuators) == 1
    assert torch.isfinite(entity.data.joint_pos).all()
    assert torch.isfinite(entity.data.joint_vel).all()
