from environments.appworld.deployable import AppWorldCandidateHarness


def test_unique_receipt_key_repairs_without_model():
    harness = AppWorldCandidateHarness()
    proposal = {"tool": "calendar:create", "arguments": {"title": "demo"}}
    repaired, decision = harness.repair(
        "Create the requested event.",
        {"events": [{"status": "ok", "calendar_id": 17}]},
        proposal,
        ["title", "calendar_id"],
    )
    assert decision.changed
    assert decision.selected_by == "unique_receipt_key"
    assert repaired["arguments"]["calendar_id"] == 17


def test_model_can_only_select_existing_visible_candidate():
    harness = AppWorldCandidateHarness(lambda payload: len(payload["candidates"]) - 1)
    proposal = {"tool": "mail:send", "arguments": {"subject": "demo"}}
    repaired, decision = harness.repair(
        'Recipient: "person@example.com".',
        None,
        proposal,
        ["subject", "recipient"],
    )
    assert decision.changed
    assert decision.selected_by == "frozen_model_candidate_index"
    assert repaired["arguments"]["recipient"] == "person@example.com"


def test_model_selection_abstains_without_strong_candidate_evidence():
    harness = AppWorldCandidateHarness(lambda _: 0)
    proposal = {"tool": "calendar:create", "arguments": {"title": "demo"}}
    repaired, decision = harness.repair(
        'Create "demo" on 2026-07-16.',
        None,
        proposal,
        ["title", "calendar_id"],
    )
    assert repaired == proposal
    assert not decision.changed
    assert decision.patch_id == "weak_candidate_evidence"


def test_public_schema_is_passed_to_selector():
    seen = {}

    def selector(payload):
        seen.update(payload)
        return 0

    harness = AppWorldCandidateHarness(selector)
    proposal = {"tool": "mail:send", "arguments": {"subject": "demo"}}
    schema = {"parameter_descriptions": {"recipient": "Email destination"}}
    repaired, decision = harness.repair(
        'Recipient: "person@example.com".',
        None,
        proposal,
        ["subject", "recipient"],
        public_schema=schema,
    )
    assert decision.changed
    assert repaired["arguments"]["recipient"] == "person@example.com"
    assert seen["public_schema"] == schema
    assert seen["candidate_sources"]
    assert seen["candidate_origins"]
    assert "instruction_span" in seen["candidate_origins"][0][0]


def test_selector_abstains_below_visible_context_floor():
    harness = AppWorldCandidateHarness(
        lambda _: 0,
        min_selector_candidates=6,
    )
    proposal = {"tool": "mail:send", "arguments": {"subject": "demo"}}
    repaired, decision = harness.repair(
        'Recipient: "person@example.com".',
        None,
        proposal,
        ["subject", "recipient"],
    )
    assert repaired == proposal
    assert not decision.changed
    assert decision.patch_id == "insufficient_visible_context"


def test_no_missing_field_never_changes_clean_action():
    harness = AppWorldCandidateHarness(lambda _: 0)
    proposal = {"tool": "x:y", "arguments": {"value": "kept"}}
    repaired, decision = harness.repair("value is wrong", None, proposal, ["value"])
    assert repaired == proposal
    assert not decision.changed
