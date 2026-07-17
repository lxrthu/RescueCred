from rescuecredit.route_a_preference import (
    completion,
    preference_kind,
    stratified_split,
    training_preference,
    validity_relation,
)
from scripts.train_route_a_preference import validity_first_epoch_order


def row(event_id: str, decision: str, delta: float):
    return {
        "event_id": event_id,
        "decision": decision,
        "delta": delta,
        "action_a": {"tool": "mail__send", "arguments": {"body": "A"}},
        "action_b": {"tool": "mail__send", "arguments": {"body": "B"}},
    }


def test_stratified_split_is_deterministic_disjoint_and_complete():
    rows = [
        row(f"r{i}", "rescue_preference", 1.0) for i in range(6)
    ] + [row(f"v{i}", "reverse_preference", -1.0) for i in range(6)]
    train_a, validation_a = stratified_split(rows, seed=42, validation_fraction=0.25)
    train_b, validation_b = stratified_split(rows, seed=42, validation_fraction=0.25)
    assert train_a == train_b
    assert validation_a == validation_b
    train_ids = {item["event_id"] for item in train_a}
    validation_ids = {item["event_id"] for item in validation_a}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == {item["event_id"] for item in rows}
    assert {item["decision"] for item in validation_a} == {
        "rescue_preference",
        "reverse_preference",
    }


def test_mask_always_teaches_b_over_a():
    item = row("x", "reverse_preference", -0.5)
    chosen, rejected, weight = training_preference(item, "mask")
    assert chosen == item["action_b"]
    assert rejected == item["action_a"]
    assert weight == 1.0


def test_v2_rescues_reverses_and_skips_zero_delta():
    rescue = row("r", "rescue_preference", 0.4)
    reverse = row("v", "reverse_preference", -0.7)
    zero = row("z", "zero_delta", 0.0)
    assert training_preference(rescue, "v2") == (
        rescue["action_b"],
        rescue["action_a"],
        0.4,
    )
    assert training_preference(reverse, "v2") == (
        reverse["action_a"],
        reverse["action_b"],
        0.7,
    )
    assert training_preference(zero, "v2") is None


def test_v3_uses_the_same_shadow_direction_and_validity_gate():
    rescue = row("r3", "rescue_preference", 0.4)
    reverse = row("v3", "reverse_preference", -0.7)
    zero = row("z3", "zero_delta", 0.0)
    assert training_preference(rescue, "v3") == (
        rescue["action_b"],
        rescue["action_a"],
        0.4,
    )
    assert training_preference(reverse, "v3") == (
        reverse["action_a"],
        reverse["action_b"],
        0.7,
    )
    assert training_preference(zero, "v3") is None


def test_completion_is_canonical_and_length_comparable():
    assert completion({"arguments": {"b": 2, "a": 1}, "tool": "x"}) == (
        '{"arguments":{"a":1,"b":2},"tool":"x"}'
    )


def test_v31_never_reverses_a_missing_required_argument():
    item = row("missing", "reverse_preference", -1.0)
    item["variant_kind"] = "missing_required_arguments"
    item["missing_parameters"] = ["body"]
    assert validity_relation(item) == "a_invalid_b_valid"
    assert training_preference(item, "v31") == (
        item["action_b"],
        item["action_a"],
        1.0,
    )
    assert preference_kind(item, "v31") == "validity_b_over_a"


def test_v31_uses_shadow_only_when_both_actions_are_executable():
    reverse = row("both", "reverse_preference", -0.7)
    reverse["variant_kind"] = "wrong_visible_candidate_value"
    assert validity_relation(reverse) == "both_valid"
    assert training_preference(reverse, "v31") == (
        reverse["action_a"],
        reverse["action_b"],
        0.7,
    )


def test_v31_explicit_validity_covers_all_four_quadrants():
    item = row("quadrants", "reverse_preference", -1.0)
    item.update(action_a_executable=True, action_b_executable=False)
    assert validity_relation(item) == "a_valid_b_invalid"
    assert training_preference(item, "v31") == (
        item["action_a"], item["action_b"], 1.0
    )
    item.update(action_a_executable=False, action_b_executable=True)
    assert validity_relation(item) == "a_invalid_b_valid"
    item.update(action_a_executable=True, action_b_executable=True)
    assert validity_relation(item) == "both_valid"
    item.update(action_a_executable=False, action_b_executable=False)
    assert validity_relation(item) == "both_invalid"
    assert training_preference(item, "v31") is None


def test_v31_partial_validity_metadata_abstains_instead_of_falling_back():
    item = row("partial", "rescue_preference", 1.0)
    item.update(
        action_a_executable=True,
        variant_kind="missing_required_arguments",
    )
    assert validity_relation(item) == "unknown"
    assert training_preference(item, "v31") is None


def test_v31_natural_ratio_sampler_is_deterministic_and_matched():
    missing = [row(f"m{i}", "zero_delta", 0.0) for i in range(3)]
    for item in missing:
        item["variant_kind"] = "missing_required_arguments"
    both = [row("both", "reverse_preference", -1.0)]
    both[0]["variant_kind"] = "wrong_visible_candidate_value"
    unknown = [row("unknown", "rescue_preference", 1.0)]
    rows = missing + both + unknown
    first = validity_first_epoch_order(rows, 42, 0, 9)
    second = validity_first_epoch_order(rows, 42, 0, 9)
    assert first == second
    assert len(first) == 9
    assert all(item["event_id"] != "unknown" for item in first)
