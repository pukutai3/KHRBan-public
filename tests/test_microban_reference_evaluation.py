from types import SimpleNamespace

import pytest

from khrban.microban_reference_evaluation import (
    MICROBAN_FOOT_AIR_TIME_MIN_S,
    prepare_reference_eval_cfg,
    require_reference_commit,
)


def test_reference_evaluation_uses_pinned_microban_air_time_minimum() -> None:
    assert MICROBAN_FOOT_AIR_TIME_MIN_S == 0.125


def test_reference_evaluation_requires_the_pinned_microban_commit() -> None:
    expected = "d594a6088bb7b6600fe8098321169031fbca680c"

    require_reference_commit(expected, expected)

    with pytest.raises(RuntimeError, match="Microban reference commit mismatch"):
        require_reference_commit("deadbeef", expected)


def test_reference_evaluation_uses_training_commands_without_training_noise() -> None:
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=4096),
        observations={
            "actor": SimpleNamespace(enable_corruption=True),
        },
        events={"push_robot": object(), "keep_me": object()},
        curriculum={"staged_curriculum": object()},
        episode_length_s=20.0,
        seed=0,
    )

    prepared = prepare_reference_eval_cfg(cfg, num_envs=64, seed=314159)

    assert prepared is cfg
    assert cfg.scene.num_envs == 64
    assert cfg.seed == 314159
    assert cfg.episode_length_s == 1.0e9
    assert cfg.observations["actor"].enable_corruption is False
    assert cfg.events == {"keep_me": cfg.events["keep_me"]}
    assert cfg.curriculum == {}
