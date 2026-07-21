import math
import json

import pytest

from rescuecredit.edit_credit import (
    ABSENT,
    canonical_action_edits,
    credit_firewall_advantage,
    edit_credit_loss,
    edit_credit_objective,
    empirical_binary_auc,
    intervention_policy_loss,
    select_rescue_constrained_threshold,
    fold_role,
    stratified_group_folds,
    summarize_selection,
    symmetrized_edit_margin,
)
from rescuecredit.frozen_bank import file_sha256
from rescuecredit.toolsandbox_preference import public_preference_prompt


def _actions():
    return (
        {"tool": "send", "arguments": {"contact": "Ada", "body": "old"}},
        {"tool": "send", "arguments": {"contact": "Ada", "body": "new", "urgent": True}},
    )


def test_edit_credit_extracts_only_changed_canonical_fields():
    action_a, action_b = _actions()
    edits = canonical_action_edits(action_a, action_b)
    assert [edit.path for edit in edits] == ["/arguments/body", "/arguments/urgent"]
    assert edits[0].value_a == "old" and edits[0].value_b == "new"
    assert edits[1].value_a == ABSENT and edits[1].value_b is True


def test_edit_credit_rejects_identical_actions():
    action_a, _ = _actions()
    with pytest.raises(ValueError, match="distinct"):
        canonical_action_edits(action_a, action_a)


def test_credit_firewall_masks_assisted_reward_at_and_before_intervention():
    assert credit_firewall_advantage(
        intervened=True,
        step_index=0,
        intervention_step=1,
        assisted_advantage=2.0,
    ) == 0.0
    assert credit_firewall_advantage(
        intervened=True,
        step_index=1,
        intervention_step=1,
        assisted_advantage=2.0,
    ) == 0.0
    assert credit_firewall_advantage(
        intervened=True,
        step_index=2,
        intervention_step=1,
        assisted_advantage=2.0,
    ) == 2.0


def test_edit_credit_gradient_direction_changes_with_counterfactual_label():
    torch = pytest.importorskip("torch")
    rescue_margin = torch.tensor(0.0, requires_grad=True)
    edit_credit_loss(rescue_margin, decision="rescue_preference").backward()
    assert rescue_margin.grad.item() < 0.0

    reverse_margin = torch.tensor(0.0, requires_grad=True)
    edit_credit_loss(reverse_margin, decision="reverse_preference").backward()
    assert reverse_margin.grad.item() > 0.0


def test_intervention_credit_firewall_prevents_b_reward_from_updating_a_logprob():
    torch = pytest.importorskip("torch")
    logp_a = torch.tensor(-2.0, requires_grad=True)
    contaminated = -(1.0 * logp_a)
    contaminated.backward()
    assert logp_a.grad.item() == pytest.approx(-1.0)

    protected_logp_a = torch.tensor(-2.0, requires_grad=True)
    protected = intervention_policy_loss(
        protected_logp_a.unsqueeze(0),
        intervened=True,
        step_index=1,
        intervention_step=1,
        assisted_advantage=1.0,
    )
    protected.backward()
    assert protected_logp_a.grad.item() == pytest.approx(0.0)

    changed_a = torch.tensor(-0.2, requires_grad=True)
    changed_b = torch.tensor(-0.2, requires_grad=True)
    unchanged = torch.tensor(0.5, requires_grad=True)
    production_loss, *_ = edit_credit_objective(
        changed_b - changed_a + unchanged * 0.0,
        torch.tensor(0.0),
        decision="rescue_preference",
        beta=1.0,
        absolute_margin_coef=1.0,
        target_margin=0.05,
        reference_anchor_coef=0.25,
    )
    production_loss.backward()
    assert changed_b.grad.item() < 0 < changed_a.grad.item()
    assert unchanged.grad.item() == pytest.approx(0.0)


def test_symmetrized_margin_is_invariant_to_candidate_presentation_order():
    action_a, action_b = _actions()
    prompt = public_preference_prompt(
        visible_history=[{"role": "user", "content": "send new urgently"}],
        public_tool_schemas=[{"name": "send"}],
        action_a=action_a,
        action_b=action_b,
    )

    def scorer(edit_prompt: str, completion: str) -> float:
        # An intentionally left-candidate-biased scorer. Averaging both prompt
        # orders must cancel this bias and leave content preference only.
        left_pref = 0.0
        if '"candidate_left": {"arguments": {"body": "new"' in edit_prompt:
            left_pref = 5.0 if completion == '"new"' else 0.0
        elif '"candidate_left": {"arguments": {"body": "old"' in edit_prompt:
            left_pref = 5.0 if completion == '"old"' else 0.0
        return left_pref + (1.0 if completion in {'"new"', "true"} else 0.0)

    margin = symmetrized_edit_margin(
        prompt=prompt,
        action_a=action_a,
        action_b=action_b,
        scorer=scorer,
    )
    assert margin == pytest.approx(1.0)


def test_rescue_constrained_threshold_maximizes_reverse_with_zero_harm_budget():
    calibration = [
        {"decision": "rescue_preference", "margin_b_over_a": 0.6},
        {"decision": "rescue_preference", "margin_b_over_a": 0.8},
        {"decision": "reverse_preference", "margin_b_over_a": -0.5},
        {"decision": "reverse_preference", "margin_b_over_a": 0.7},
    ]
    selected = select_rescue_constrained_threshold(calibration, rescue_delta=0.0)
    assert selected.feasible
    assert selected.rescue_drop == 0.0
    assert selected.reverse_recall == 0.5
    summary = summarize_selection(calibration, threshold=selected.threshold)
    assert summary["rescue_accuracy"] == 1.0
    assert summary["reverse_recall"] == 0.5


def test_tie_aware_auc():
    assert empirical_binary_auc([1, 1, 0, 0], [1.0, 0.0, 0.0, -1.0]) == pytest.approx(0.875)
    with pytest.raises(ValueError, match="both classes"):
        empirical_binary_auc([1, 1], [0.0, 1.0])


def test_no_nan_threshold_when_no_routes_are_feasible_except_default_b():
    rows = [
        {"decision": "rescue_preference", "margin_b_over_a": -1.0},
        {"decision": "reverse_preference", "margin_b_over_a": 1.0},
    ]
    selected = select_rescue_constrained_threshold(rows, rescue_delta=0.0)
    assert selected.threshold == -math.inf
    assert selected.route_to_a == 0


def test_stratified_group_folds_keep_tasks_isolated_across_roles():
    rows = []
    for task in range(15):
        rows.extend(
            [
                {
                    "event_id": f"{task}-r",
                    "task_id_hash": f"task-{task}",
                    "decision": "rescue_preference",
                },
                {
                    "event_id": f"{task}-v",
                    "task_id_hash": f"task-{task}",
                    "decision": "reverse_preference",
                },
            ]
        )
    assignment = stratified_group_folds(rows, folds=5, seed=42)
    assert set(assignment.values()) == set(range(5))
    for test_fold in range(5):
        roles = {
            role: {
                row["task_id_hash"]
                for row in rows
                if fold_role(
                    row,
                    assignment=assignment,
                    test_fold=test_fold,
                    folds=5,
                )
                == role
            }
            for role in ("train", "calibration", "test")
        }
        assert not (roles["train"] & roles["calibration"])
        assert not (roles["train"] & roles["test"])
        assert not (roles["calibration"] & roles["test"])


def test_frozen_126_pair_protocol_has_complete_crossfit_roles(tmp_path):
    from scripts.freeze_toolsandbox_editcredit_protocol import build_protocol

    rows = []
    for index in range(126):
        rows.append(
            {
                "event_id": f"event-{index}",
                "task_id_hash": f"task-{index % 38}",
                "decision": "rescue_preference" if index < 41 else "reverse_preference",
                "replay_valid": True,
            }
        )
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    data_manifest = tmp_path / "manifest.json"
    data_manifest.write_text(
        json.dumps(
            {
                "status": "frozen",
                "passed": True,
                "events": 126,
                "train_sha256": file_sha256(train_file),
                "official_branch_metrics_in_training_file": False,
                "protected_outcomes_in_prompt": False,
                "source_event_sha256": "events",
                "source_summary_sha256": "summary",
                "source_protocol_sha256": "protocol",
            }
        ),
        encoding="utf-8",
    )
    data_gate = tmp_path / "data_gate.json"
    data_gate.write_text(json.dumps({"passed": True, "events": 126}), encoding="utf-8")
    gradient_sanity = tmp_path / "gradient_sanity.json"
    gradient_sanity.write_text(
        json.dumps({"passed": True, "checks": {"all": True}}), encoding="utf-8"
    )
    protocol = build_protocol(
        train_file, model, data_manifest, data_gate, gradient_sanity
    )
    assert protocol["events"] == 126
    assert protocol["task_groups"] == 38
    assert len(protocol["split_audit"]) == 5
    for split in protocol["split_audit"]:
        assert set(split["roles"]) == {"train", "calibration", "test"}
        for role in split["roles"].values():
            assert set(role["decisions"]) == {
                "rescue_preference",
                "reverse_preference",
            }
