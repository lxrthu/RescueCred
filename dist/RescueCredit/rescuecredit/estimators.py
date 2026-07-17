from __future__ import annotations

from dataclasses import dataclass, field


def residual_estimate(mu: float, draw: int, probability: float, shadow_return: float | None) -> float:
    """Horvitz-Thompson residual estimator.

    `mu` and `probability` must be fixed before `draw`. When draw is zero,
    `shadow_return` must be absent and the prediction is returned unchanged.
    """
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    if draw not in (0, 1):
        raise ValueError("draw must be Bernoulli 0/1")
    if draw == 0:
        if shadow_return is not None:
            raise ValueError("non-audited event cannot consume a shadow outcome")
        return float(mu)
    if shadow_return is None:
        raise ValueError("audited event requires shadow_return")
    return float(mu + (shadow_return - mu) / probability)


@dataclass
class PatchEMA:
    beta: float = 0.95
    cold_start: float = 0.0
    _values: dict[str, float] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must be in [0, 1)")

    def predict(self, patch_id: str) -> float:
        return self._values.get(patch_id, self.cold_start)

    def update(self, patch_id: str, observed_shadow_return: float) -> None:
        previous = self.predict(patch_id)
        self._values[patch_id] = self.beta * previous + (1.0 - self.beta) * float(observed_shadow_return)
        self._counts[patch_id] = self._counts.get(patch_id, 0) + 1

    def state_dict(self) -> dict[str, object]:
        return {"beta": self.beta, "cold_start": self.cold_start, "values": self._values, "counts": self._counts}

