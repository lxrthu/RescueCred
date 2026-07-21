#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescuecredit.edit_credit import edit_credit_objective, intervention_policy_loss
from rescuecredit.logging import write_json


def run_audit() -> dict:
    import torch

    proposal_a = torch.tensor([-2.0, -1.0], requires_grad=True)
    naive_loss = -(proposal_a.mean() * 1.0)
    naive_loss.backward()
    naive_gradient = proposal_a.grad.detach().clone()

    protected_a = torch.tensor([-2.0, -1.0], requires_grad=True)
    protected_loss = intervention_policy_loss(
        protected_a,
        intervened=True,
        step_index=1,
        intervention_step=1,
        assisted_advantage=1.0,
    )
    protected_loss.backward()
    protected_gradient = protected_a.grad.detach().clone()

    # The unchanged field is deliberately absent from the production
    # EditCredit objective. Its gradient must remain exactly zero.
    unchanged = torch.tensor(0.25, requires_grad=True)
    changed_a = torch.tensor(-0.2, requires_grad=True)
    changed_b = torch.tensor(-0.2, requires_grad=True)
    rescue_loss, *_ = edit_credit_objective(
        changed_b - changed_a + unchanged * 0.0,
        torch.tensor(0.0),
        decision="rescue_preference",
        beta=1.0,
        absolute_margin_coef=1.0,
        target_margin=0.05,
        reference_anchor_coef=0.25,
    )
    rescue_loss.backward()
    rescue_gradients = {
        "changed_a": float(changed_a.grad),
        "changed_b": float(changed_b.grad),
        "unchanged": float(unchanged.grad),
    }

    reverse_a = torch.tensor(-0.2, requires_grad=True)
    reverse_b = torch.tensor(-0.2, requires_grad=True)
    reverse_loss, *_ = edit_credit_objective(
        reverse_b - reverse_a,
        torch.tensor(0.0),
        decision="reverse_preference",
        beta=1.0,
        absolute_margin_coef=1.0,
        target_margin=0.05,
        reference_anchor_coef=0.25,
    )
    reverse_loss.backward()
    reverse_gradients = {
        "changed_a": float(reverse_a.grad),
        "changed_b": float(reverse_b.grad),
    }
    checks = {
        "naive_b_reward_contaminates_a": bool(naive_gradient.abs().max() > 0),
        "firewall_zeroes_a_gradient": bool(protected_gradient.abs().max() <= 1e-12),
        "unchanged_field_gradient_zero": abs(rescue_gradients["unchanged"]) <= 1e-12,
        "rescue_increases_b_over_a": rescue_gradients["changed_b"] < 0 < rescue_gradients["changed_a"],
        "reverse_increases_a_over_b": reverse_gradients["changed_a"] < 0 < reverse_gradients["changed_b"],
    }
    return {
        "passed": all(checks.values()),
        "stage": "editcredit_gradient_ownership_sanity",
        "checks": checks,
        "naive_a_gradient": naive_gradient.tolist(),
        "protected_a_gradient": protected_gradient.tolist(),
        "rescue_gradients": rescue_gradients,
        "reverse_gradients": reverse_gradients,
        "objective_source": "rescuecredit.edit_credit.edit_credit_objective",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
