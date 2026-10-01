"""Shared precision rules for formal evaluation decisions and display."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Any, Mapping


EVALUATION_DECIMAL_PLACES = 2


def evaluation_value(value: float) -> float:
    """Round one formal metric to the precision shown in the contract."""

    if not math.isfinite(value):
        return value
    quantum = Decimal(1).scaleb(-EVALUATION_DECIMAL_PLACES)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def format_evaluation_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Format metric floats exactly as the formal contract compares them."""

    def format_value(value: Any) -> Any:
        if isinstance(value, float):
            rounded = evaluation_value(value)
            return f"{rounded:.{EVALUATION_DECIMAL_PLACES}f}"
        if isinstance(value, Mapping):
            return {str(key): format_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [format_value(item) for item in value]
        return value

    return {str(key): format_value(value) for key, value in values.items()}
