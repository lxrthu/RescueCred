from rescuecredit.accounting import BudgetCounter


def test_budget_identity():
    counter = BudgetCounter()
    counter.charge_main(10)
    counter.charge_shadow(3)
    counter.charge_failed_replay(2)
    counter.charge_evaluation(4)
    assert counter.training_steps == 15
    assert counter.total_steps == 19
    assert counter.total_steps == counter.main_steps + counter.shadow_steps + counter.failed_replay_steps + counter.evaluation_steps

