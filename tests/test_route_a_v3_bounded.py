from scripts.audit_route_a_v3_bounded import build_audit


def _inputs() -> dict:
    identity = {
        "event_set_hash": "events",
        "event_file_sha256": "event-file",
    }
    raw_summary = {
        **identity,
        "mask_results_sha256": "mask-results",
        "v2_results_sha256": "v3-results",
        "continuation_input_excludes_evaluator_and_reference": True,
        "reference_suffix_used": False,
        "test_split_access": False,
        "primary": {"v2_mean_official_score": 0.6},
    }
    return {
        "raw_summary": raw_summary,
        "raw_gate": {"passed": True, "v2_better_events": 3},
        "erratum_gate": {"passed": True, "status": "audited_arithmetic_erratum"},
        "mask_selection": {
            **identity,
            "method": "mask",
            "results_sha256": "mask-results",
            "scoring_failures": 0,
        },
        "v3_selection": {
            **identity,
            "method": "v3",
            "results_sha256": "v3-results",
            "scoring_failures": 0,
        },
        "mask_results_sha256": "mask-results",
        "v3_results_sha256": "v3-results",
    }


def test_audit_relabels_legacy_method_b_and_passes() -> None:
    summary, gate = build_audit(**_inputs())
    assert gate["passed"] is True
    assert gate["v3_better_events"] == 3
    assert summary["primary"]["v3_mean_official_score"] == 0.6
    assert summary["method_b"] == "v3"


def test_audit_rejects_unbound_v3_results() -> None:
    inputs = _inputs()
    inputs["raw_summary"]["v2_results_sha256"] = "wrong"
    _, gate = build_audit(**inputs)
    assert gate["passed"] is False
    assert gate["identity_checks"]["raw_evaluator_binds_method_b_to_v3"] is False
