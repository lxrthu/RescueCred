from __future__ import annotations

from collections.abc import Callable

from rescuecredit.types import ShadowResult

from .adapter import APIBankControlledEnv


class APIBankShadowRunner:
    """Replay an original action and continue with a supplied policy callback.

    The triggering teachable patch is disabled by executing `original_action`
    directly. Permanent safety remains part of the environment; later teachable
    interventions are decided by the continuation callback.
    """

    def __init__(self, env: APIBankControlledEnv) -> None:
        self.env = env

    def run(
        self,
        state_ref: str,
        original_action: dict,
        disabled_patch_id: str,
        rng_state: object,
        max_steps: int,
        expected_state_hash: str | None = None,
        continuation: Callable[[dict, int, str, dict | None], dict] | None = None,
    ) -> ShadowResult:
        self.env.restore(state_ref)
        self.env.set_rng_state(rng_state)
        restored_hash = self.env.observation()["state_hash"]
        expected_hash = expected_state_hash or restored_hash
        if restored_hash != expected_hash:
            return ShadowResult(0.0, False, 0, "state_hash_mismatch", expected_hash, restored_hash, False)

        _, reward, done, info = self.env.step(original_action)
        steps = 1
        while not done and steps < max_steps:
            if continuation is None:
                return ShadowResult(0.0, False, steps, "continuation_unavailable", expected_hash, restored_hash, False)
            action = continuation(self.env.observation(), steps, disabled_patch_id, info.get("tool_result"))
            _, reward, done, info = self.env.step(action)
            steps += 1
        terminal_reason = info["terminal_reason"] if done else "shadow_horizon"
        replay_valid = bool(done)
        return ShadowResult(float(reward), bool(reward), steps, terminal_reason, expected_hash, restored_hash, replay_valid)
