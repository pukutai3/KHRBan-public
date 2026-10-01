"""The public checkout must include the evaluated walking policy."""

from pathlib import Path


POLICY = Path(__file__).resolve().parents[1] / "policies/velocity/model_179910.pt"


def test_evaluated_walking_policy_is_distributed() -> None:
    assert POLICY.is_file(), "the public repository omits the trained walking policy"
