import json
from pathlib import Path

from scripts.audit_route_a_v31_both_valid_bounded import (
    _audit_equivalent,
    build_audit,
)
from scripts.build_route_a_both_valid_dev_events import _schema_valid
from scripts.freeze_route_a_v31_both_valid_protocol import (
    GATE_THRESHOLDS,
    _common_pretreatment_context,
    _schema_complete,
    _single_parameter_difference,
)


ROOT = Path(__file__).resolve().parents[1]


def _event():
    return {
        "variant_kind": "visible_candidate_value_pair",
        "parameter": "password",
        "required_fields": ["username", "password"],
        "parameter_schemas": {
            "username": {"type": "string", "minLength": 1},
            "password": {"type": "string", "minLength": 3},
        },
        "action_a": {
            "tool": "post:/phone/auth/token",
            "arguments": {"username": "u", "password": "wrong"},
        },
        "action_b": {
            "tool": "post:/phone/auth/token",
            "arguments": {"username": "u", "password": "right"},
        },
        "action_a_schema_valid": True,
        "action_b_schema_valid": True,
        "continuation_context": "",
    }


def test_both_valid_fixture_requires_complete_actions_and_one_value_difference():
    event = _event()
    public = {
        "required_fields": event["required_fields"],
        "parameter_schemas": event["parameter_schemas"],
    }
    assert _schema_valid(event["action_a"], public)
    assert _schema_valid(event["action_b"], public)
    assert _schema_complete(event)
    assert _single_parameter_difference(event)
    event["action_a"]["arguments"].pop("username")
    assert not _schema_complete(event)


def test_public_schema_rejects_type_enum_pattern_and_range_violations():
    action = {"arguments": {"kind": "alpha", "count": 3}}
    schema = {
        "required_fields": ["kind", "count"],
        "parameter_schemas": {
            "kind": {"type": "string", "enum": ["alpha"], "pattern": "^a"},
            "count": {"type": "integer", "minimum": 1, "maximum": 5},
        },
    }
    assert _schema_valid(action, schema)
    action["arguments"]["count"] = 9
    assert not _schema_valid(action, schema)
    action["arguments"]["count"] = "3"
    assert not _schema_valid(action, schema)
    action["arguments"]["count"] = 3
    action["arguments"]["kind"] = "beta"
    assert not _schema_valid(action, schema)
    assert not _schema_valid({"arguments": {"kind": "alpha"}}, schema)


def test_public_schema_rejects_unresolved_ref_and_combinators():
    action = {"arguments": {"value": "x"}}
    for unsupported in (
        {"$ref": "#/components/schemas/Value"},
        {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        {"oneOf": [{"type": "string"}]},
        {"allOf": [{"type": "string"}]},
    ):
        schema = {
            "required_fields": ["value"],
            "parameter_schemas": {"value": unsupported},
        }
        assert not _schema_valid(action, schema)


def test_both_valid_fixture_rejects_tool_or_multi_field_changes():
    event = _event()
    event["action_b"]["arguments"]["username"] = "other"
    assert not _single_parameter_difference(event)


def test_original_proposal_is_identical_common_pretreatment_context():
    event = _event()
    event["continuation_context"] = json.dumps(
        {
            "task_instruction": "login",
            "public_openapi_schema": {},
            "original_visible_proposal": event["action_a"],
        }
    )
    assert _common_pretreatment_context(event)
    context = json.loads(event["continuation_context"])
    context["action_b"] = event["action_b"]
    event["continuation_context"] = json.dumps(context)
    assert not _common_pretreatment_context(event)
    event = _event()
    event["action_b"]["tool"] = "post:/other"
    assert not _single_parameter_difference(event)


def _audit_inputs():
    primary = {
        "valid_paired_events": 40,
        "nonzero_causal_events": 8,
        "mask_mean_official_score": 0.4,
        "v2_mean_official_score": 0.45,
        "score_improvement": 0.05,
        "mask_causal_selection_accuracy": 0.5,
        "v2_causal_selection_accuracy": 0.625,
        "causal_accuracy_improvement": 0.125,
        "v2_better_events": 3,
        "v2_worse_events": 1,
        "ties": 36,
    }
    raw = {
        "primary": primary,
        "primary_horizon": 8,
        "requested_horizons": [4, 8],
        "selection_disagreements": 4,
        "event_set_hash": "events",
        "event_file_sha256": "event-file",
        "mask_results_sha256": "mask-results",
        "v2_results_sha256": "v31-results",
        "protocol_lock_validated": True,
        "development_protocol": True,
        "continuation_input_excludes_evaluator_and_reference": True,
        "reference_suffix_used": False,
        "test_split_access": False,
        "cache_conflicts": 0,
        "horizon_prefix_mismatches": 0,
        "horizon_prefix_unverifiable": 0,
        "sanity_limit": None,
    }
    protocol = {
        "status": "frozen_before_both_valid_dev_outcomes",
        "method_a": "mask",
        "method_b": "v31",
        "event_set_hash": "events",
        "event_file_sha256": "event-file",
        "mask_results_sha256": "mask-results",
        "v31_results_sha256": "v31-results",
        "gate_thresholds": GATE_THRESHOLDS,
        "scope": "controlled",
    }
    raw["protocol_lock"] = protocol
    raw["protocol_lock_sha256"] = "lock-sha"
    manifest = {
        "event_set_hash": "events",
        "event_file_sha256": "event-file",
        "both_actions_schema_complete": True,
        "variant_kinds": {"visible_candidate_value_pair": 40},
        "events": 40,
        "tasks_with_events": 20,
        "max_task_event_share": 0.05,
    }
    return {
        "raw_summary": raw,
        "protocol": protocol,
        "manifest": manifest,
        "mask_selection": {
            "method": "mask",
            "results_sha256": "mask-results",
            "scoring_failures": 0,
        },
        "v31_selection": {
            "method": "v31",
            "results_sha256": "v31-results",
            "scoring_failures": 0,
        },
        "preference_gate": {
            "passed": True,
            "stage": "route_a_seed42_v31_validity_first_gate",
        },
        "source_identity_matches": True,
        "protocol_lock_sha256": "lock-sha",
        "artifact_identity_matches": True,
        "horizon_binding_matches": True,
        "bounded_row_identity_matches": True,
        "recomputed": {
            "primary": primary,
            "primary_horizon": 8,
            "selection_disagreements": 4,
            "event_set_hash": "events",
            "horizon_prefix_mismatches": 0,
            "horizon_prefix_unverifiable": 0,
        },
    }


def test_both_valid_audit_passes_bound_positive_result():
    summary, gate = build_audit(**_audit_inputs())
    assert gate["passed"] is True
    assert gate["v31_mean_official_score"] == 0.45
    assert summary["method_b"] == "v31"


def test_both_valid_audit_rejects_no_task_level_improvement():
    inputs = _audit_inputs()
    inputs["raw_summary"]["primary"]["score_improvement"] = 0.0
    inputs["raw_summary"]["primary"]["v2_better_events"] = 1
    inputs["raw_summary"]["primary"]["v2_worse_events"] = 1
    _, gate = build_audit(**inputs)
    assert gate["passed"] is False
    assert gate["outcome_checks"]["v31_improves_bounded_official_score"] is False


def test_both_valid_audit_rejects_lock_or_raw_row_tampering():
    inputs = _audit_inputs()
    inputs["protocol_lock_sha256"] = "different"
    _, gate = build_audit(**inputs)
    assert gate["passed"] is False


def test_audit_recompute_allows_only_machine_precision_roundoff():
    assert _audit_equivalent(0.4485714285714286, 0.44857142857142857)
    assert _audit_equivalent(
        {"primary": {"score": 0.4485714285714286, "wins": 1}},
        {"primary": {"score": 0.44857142857142857, "wins": 1}},
    )
    assert not _audit_equivalent(0.4485714285714286, 0.4486)
    assert not _audit_equivalent({"wins": 1}, {"wins": 2})
    inputs = _audit_inputs()
    inputs["recomputed"]["primary"] = {**inputs["recomputed"]["primary"], "score_improvement": -1.0}
    _, gate = build_audit(**inputs)
    assert gate["identity_checks"]["raw_rows_independently_recomputed"] is False
    inputs = _audit_inputs()
    inputs["horizon_binding_matches"] = False
    _, gate = build_audit(**inputs)
    assert gate["passed"] is False
    inputs = _audit_inputs()
    inputs["bounded_row_identity_matches"] = False
    _, gate = build_audit(**inputs)
    assert gate["passed"] is False


def test_bounded_evaluator_has_separate_dynamic_development_protocol():
    source = (ROOT / "scripts/evaluate_route_a_bounded.py").read_text(
        encoding="utf-8"
    )
    assert '"--development-protocol"' in source
    assert '"frozen_before_both_valid_dev_outcomes"' in source
    assert "confirmatory and development protocols are mutually exclusive" in source
    assert "str(event['event_id'])[:12]" in source
    assert '"source_identity"' in source


def test_runner_freezes_before_any_bounded_outcomes():
    source = (
        ROOT / "scripts/cloud/run_route_a_v31_both_valid_dev_seed42.sh"
    ).read_text(encoding="utf-8")
    freeze = source.index(
        '"$MODEL_PY" scripts/freeze_route_a_v31_both_valid_protocol.py'
    )
    evaluate = source.index('"$APP_PY" scripts/evaluate_route_a_bounded.py')
    assert freeze < evaluate
    assert source.count("--development-protocol") == 2
    assert "--limit 3" in source
    assert "--bounded-results" in source
    assert "cleanup_scorers" in source
