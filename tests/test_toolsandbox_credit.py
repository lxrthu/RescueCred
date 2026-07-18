import pytest

from rescuecredit.toolsandbox_credit import (
    OFFICIAL_SCORE_SOURCE,
    lexicographic_counterfactual_regret,
    validate_branch_credit_evidence,
)


def _branch(
    similarity=1.0,
    progress_auc=1.0,
    errors=0,
    turns=8,
    steps=8,
    valid=True,
):
    return {
        "valid": valid,
        "score": {"similarity": similarity, "turn_count": turns},
        "progress_auc": progress_auc,
        "tool_errors": errors,
        "steps": steps,
    }


def test_final_outcome_dominates_all_efficiency_components():
    a = _branch(similarity=0.5, progress_auc=0.9, errors=0, turns=1, steps=1)
    b = _branch(similarity=0.6, progress_auc=0.1, errors=5, turns=8, steps=8)
    result = lexicographic_counterfactual_regret(a, b, horizon=8)
    assert result["decision"] == "rescue_preference"
    assert result["decision_basis"] == "final_official_similarity"


def test_progress_auc_breaks_terminal_tie_before_error_cost():
    a = _branch(progress_auc=0.4, errors=0)
    b = _branch(progress_auc=0.6, errors=3)
    result = lexicographic_counterfactual_regret(a, b, horizon=8)
    assert result["decision"] == "rescue_preference"
    assert result["decision_basis"] == "bounded_progress_auc"


def test_visible_error_breaks_outcome_and_progress_tie():
    a = _branch(errors=1)
    b = _branch(errors=0)
    result = lexicographic_counterfactual_regret(a, b, horizon=8)
    assert result["decision"] == "rescue_preference"
    assert result["decision_basis"] == "visible_tool_error_advantage"
    assert result["causal_weight"] == pytest.approx(1 / 8)


def test_reverse_preference_and_zero_tie_are_supported():
    reverse = lexicographic_counterfactual_regret(
        _branch(errors=0), _branch(errors=1), horizon=8
    )
    tied = lexicographic_counterfactual_regret(_branch(), _branch(), horizon=8)
    assert reverse["decision"] == "reverse_preference"
    assert tied["decision"] == "zero_delta"
    assert tied["decision_basis"] == "all_components_tied"


def test_invalid_replay_never_produces_preference():
    result = lexicographic_counterfactual_regret(
        _branch(valid=False), _branch(), horizon=8
    )
    assert result["decision"] == "invalid"
    assert result["causal_weight"] == 0.0


def test_official_trace_and_auc_are_independently_recomputed():
    branch = _branch(similarity=0.5, progress_auc=0.375, steps=2)
    branch["score"]["source"] = OFFICIAL_SCORE_SOURCE
    branch["score_trace"] = [
        {"source": OFFICIAL_SCORE_SOURCE, "similarity": 0.25},
        {"source": OFFICIAL_SCORE_SOURCE, "similarity": 0.5},
    ]
    branch["padded_similarity_trace"] = [0.25, 0.5]
    result = validate_branch_credit_evidence(branch, horizon=2)
    assert result == {"final_similarity": 0.5, "progress_auc": 0.375}


def test_tampered_auc_or_provenance_is_rejected():
    branch = _branch(similarity=0.5, progress_auc=0.5, steps=1)
    branch["score"]["source"] = OFFICIAL_SCORE_SOURCE
    branch["score_trace"] = [{"source": "fake", "similarity": 0.5}]
    branch["padded_similarity_trace"] = [0.5]
    with pytest.raises(ValueError, match="provenance"):
        validate_branch_credit_evidence(branch, horizon=1)
