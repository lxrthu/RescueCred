from scripts.audit_route_a_v3_expanded_gate_erratum import (
    expected_balanced_counts,
)


def test_odd_epoch_budget_alternates_the_extra_presentation() -> None:
    assert expected_balanced_counts(255, 1) == {
        "rescue_preference": 128,
        "reverse_preference": 127,
    }
    assert expected_balanced_counts(255, 2) == {
        "rescue_preference": 255,
        "reverse_preference": 255,
    }
    assert expected_balanced_counts(255, 3) == {
        "rescue_preference": 383,
        "reverse_preference": 382,
    }


def test_even_epoch_budget_is_exactly_balanced() -> None:
    assert expected_balanced_counts(256, 3) == {
        "rescue_preference": 384,
        "reverse_preference": 384,
    }
