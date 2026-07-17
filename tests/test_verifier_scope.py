from environments.api_bank.verifier import APIBankVerifier


STATE = {
    "available_tools": [{"name": "SendEmail", "required": ["to", "body"]}],
    "success_predicate_satisfied": False,
}


def test_missing_required_argument_is_deterministic_failure():
    result = APIBankVerifier().verify(STATE, {"tool": "SendEmail", "arguments": {"to": "a@b.com"}})
    assert not result.valid and result.deterministic_outcome


def test_schema_valid_does_not_claim_semantic_success():
    result = APIBankVerifier().verify(STATE, {"tool": "SendEmail", "arguments": {"to": "a@b.com", "body": "wrong"}})
    assert result.valid
    assert not result.deterministic_outcome
    assert result.score != 1.0


def test_premature_finish_is_deterministic_failure():
    result = APIBankVerifier().verify(STATE, {"type": "finish"})
    assert not result.valid and result.deterministic_outcome

