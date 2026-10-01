"""The public checkout must include the evaluated walking policy."""

import hashlib
import json
from pathlib import Path

import torch

from khrban.auto_train import EvaluationCriteria, EvaluationResult


POLICY = Path(__file__).resolve().parents[1] / "policies/velocity/model_179910.pt"
EVALUATION = POLICY.with_name("evaluation.json")
EXPECTED_SHA256 = "719f01fa07eb53dab2d0fb03af5e78869e10c6845dc1f7f9fb993baf2b133185"


def test_evaluated_walking_policy_is_distributed() -> None:
    assert POLICY.is_file(), "the public repository omits the trained walking policy"
    with POLICY.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == EXPECTED_SHA256

    checkpoint = torch.load(POLICY, map_location="cpu", weights_only=True)
    assert checkpoint["iter"] == 179910
    assert "actor_state_dict" in checkpoint

    result = EvaluationResult(**json.loads(EVALUATION.read_text(encoding="utf-8")))
    assert EvaluationCriteria().is_satisfied_by(result)
