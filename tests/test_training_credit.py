from rescuecredit.training import CreditAssigner, classify_patch, group_normalize, parse_action


EXPECTED = {"tool": "SendEmail", "arguments": {"to": "a@b.com", "body": "hello"}}


def test_action_parser_and_patch_classification():
    proposal = parse_action('prefix {"tool":"SendEmail","arguments":{"to":"a@b.com"}} suffix')
    assert classify_patch(proposal, EXPECTED) == "missing_required_argument"
    assert classify_patch({"tool": "Other", "arguments": {}}, EXPECTED) == "wrong_tool_replace"
    assert classify_patch(
        {"tool": "SendEmail", "arguments": {"to": "wrong@b.com", "body": "hello"}}, EXPECTED
    ) == "semantic_argument_mismatch"
    assert classify_patch({"type": "finish"}, EXPECTED) == "premature_finish"
    assert classify_patch(EXPECTED, EXPECTED) is None


def test_mask_correction_masks_intervened_prefix():
    record = CreditAssigner("mask_correction").assign({"type": "finish"}, EXPECTED)
    assert record.intervened and record.assigned_prefix_score == 0.0


def test_rescuecredit_uses_exact_label_for_missing_argument():
    record = CreditAssigner("rescuecredit").assign({"tool": "SendEmail", "arguments": {"to": "a@b.com"}}, EXPECTED)
    assert record.g0_truth == 0.0
    assert record.assigned_prefix_score == 0.0
    assert record.audit_draw is None


def test_rescuecredit_audits_semantic_argument_mismatch():
    record = CreditAssigner("rescuecredit", audit_probability=1.0).assign(
        {"tool": "SendEmail", "arguments": {"to": "wrong@b.com", "body": "hello"}}, EXPECTED
    )
    assert record.intervened and record.teachable
    assert record.audit_draw == 1
    assert record.shadow_steps == 1


def test_group_normalize_has_zero_mean():
    values = group_normalize([0.0, 1.0, 2.0])
    assert abs(sum(values)) < 1e-9
