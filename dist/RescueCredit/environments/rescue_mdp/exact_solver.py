from __future__ import annotations

from collections.abc import Iterable

from .env import RescueMDP
from .harness import RescueMDPHarness


CANONICAL_ACTION = {
    "choose_tool": {"type": "correct_tool"},
    "fill_args": {"type": "valid_call"},
    "tool_response": {"type": "finish"},
}


def rollout_value(env: RescueMDP, first_action: dict, harness_condition: str) -> float:
    harness = RescueMDPHarness(harness_condition)
    branch = env.clone()
    proposal = first_action
    while not branch.state.done:
        executed, decision = harness.execute(branch.observation(), proposal)
        if harness_condition in {"H1", "H2"} and decision.triggered and decision.corrected_action is not None:
            # Feedback/rejection leaves the state unchanged; the policy retry is
            # modeled by the canonical action rather than direct harness execution.
            executed = decision.corrected_action
        _, reward, done, _ = branch.step(executed)
        if done:
            return reward
        proposal = CANONICAL_ACTION[branch.state.stage]
    return float(branch.state.success)


def enumerate_q_values(conditions: Iterable[str] = ("H0", "H1", "H2", "H3", "Hplacebo")) -> list[dict]:
    records: list[dict] = []
    for stage in RescueMDP.ACTIONS:
        base = RescueMDP()
        base.reset(seed=0)
        while base.state.stage != stage and not base.state.done:
            base.step(CANONICAL_ACTION[base.state.stage])
        for action_name in RescueMDP.ACTIONS[stage]:
            action = {"type": action_name}
            q0 = rollout_value(base, action, "H0")
            for condition in conditions:
                qh = rollout_value(base, action, condition)
                records.append(
                    {
                        "state": stage,
                        "action": action_name,
                        "condition": condition,
                        "q0": q0,
                        "qh": qh,
                        "rescue_gain": qh - q0,
                    }
                )
    return records
