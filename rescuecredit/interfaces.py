from __future__ import annotations

from typing import Any, Protocol

from .types import HarnessDecision, ShadowResult, VerificationResult


class SnapshotableAgentEnv(Protocol):
    def reset(self, task: dict[str, Any], seed: int) -> dict[str, Any]: ...
    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]: ...
    def snapshot(self) -> str: ...
    def restore(self, state_ref: str) -> None: ...
    def set_rng_state(self, rng_state: object) -> None: ...
    def get_rng_state(self) -> object: ...
    def task_success(self) -> bool: ...


class CorrectiveHarness(Protocol):
    def inspect(self, obs: dict[str, Any], proposal: dict[str, Any]) -> HarnessDecision: ...


class ActionVerifier(Protocol):
    def verify(self, state: dict[str, Any], action: dict[str, Any]) -> VerificationResult: ...


class ShadowRunner(Protocol):
    def run(
        self,
        state_ref: str,
        original_action: dict[str, Any],
        disabled_patch_id: str,
        rng_state: object,
        max_steps: int,
    ) -> ShadowResult: ...

