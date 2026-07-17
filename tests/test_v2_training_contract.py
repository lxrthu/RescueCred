from scripts.run_train import execute_harness_action, routed_advantage, should_stop_for_budget


def test_v2_masks_prefix_instead_of_using_shadow_advantage():
    assert routed_advantage("rescuecredit_v2", 2, 0, 3.0, 99.0) == 0.0
    assert routed_advantage("rescuecredit_v2", 2, 2, 3.0, 99.0) == 0.0
    assert routed_advantage("rescuecredit_v2", 2, 3, 3.0, 99.0) == 3.0
    assert routed_advantage("mask_correction_v2", 2, 1, 3.0, 99.0) == 0.0


def test_v1_remains_available_for_ablation():
    assert routed_advantage("rescuecredit", 2, 1, 3.0, 7.0) == 7.0


def test_no_intervention_uses_assisted_advantage():
    assert routed_advantage("rescuecredit_v2", None, 0, 3.0, 99.0) == 3.0


def test_main_budget_reaches_target_then_stops_and_total_budget_reserves_batch():
    assert not should_stop_for_budget("main", 1999, 3000, 2000, 9999, 96)
    assert should_stop_for_budget("main", 2000, 3001, 2000, 9999, 96)
    assert should_stop_for_budget("total", 100, 1950, 0, 2000, 96)
    assert not should_stop_for_budget("total", 100, 1800, 0, 2000, 96)


def test_v2_dispatch_never_queries_expected_action():
    class Environment:
        def expected_action(self):
            raise AssertionError("reference action was queried")

    class Deployable:
        def execute(self, observation, proposal, receipt):
            assert "reference_actions" not in observation
            return proposal, "deployable-decision"

    executed, decision, expected, observation = execute_harness_action(
        "rescuecredit_v2",
        Environment(),
        {"user_goal": "visible", "available_tools": [], "reference_actions": ["secret"]},
        {"type": "finish"},
        None,
        None,
        Deployable(),
    )
    assert executed == {"type": "finish"}
    assert decision == "deployable-decision"
    assert expected is None
    assert "reference_actions" not in observation
