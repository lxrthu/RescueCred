from scripts.check_route_a_v3_expanded_gate import (
    EXPECTED_ACTIVE_PRESENTATIONS,
    EXPECTED_DECISIONS,
    EXPECTED_PRESENTATIONS_PER_EPOCH,
    MIN_ACCURACY_IMPROVEMENT,
    MIN_CAUSAL_EVENTS,
)
from scripts.freeze_route_a_v3_expanded_protocol import (
    EXPECTED_TRAIN_SHA256,
    EXPECTED_VALIDATION_SHA256,
)


def test_expanded_data_identity_is_frozen() -> None:
    assert EXPECTED_TRAIN_SHA256 == (
        "67119a7f5a6dbf0e74715f630276a908247385626b427000d273bdeec962a730"
    )
    assert EXPECTED_VALIDATION_SHA256 == (
        "fb1bec44fa8ae7ff815d93db979dbc455196c1886aba061ba77ec2436963d3e5"
    )


def test_expanded_gate_and_budget_are_frozen() -> None:
    assert MIN_CAUSAL_EVENTS == 15
    assert MIN_ACCURACY_IMPROVEMENT == 0.10
    assert EXPECTED_PRESENTATIONS_PER_EPOCH == 255
    assert EXPECTED_ACTIVE_PRESENTATIONS == 765
    assert EXPECTED_DECISIONS == {
        "rescue_preference": 384,
        "reverse_preference": 381,
    }
