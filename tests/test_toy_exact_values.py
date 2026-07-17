from environments.rescue_mdp.exact_solver import enumerate_q_values


def select(records, state, action, condition):
    return next(record for record in records if record["state"] == state and record["action"] == action and record["condition"] == condition)


def test_h3_rescues_wrong_tool_but_h0_does_not():
    records = enumerate_q_values()
    h0 = select(records, "choose_tool", "wrong_tool", "H0")
    h3 = select(records, "choose_tool", "wrong_tool", "H3")
    assert h0["q0"] == 0.0 and h0["qh"] == 0.0
    assert h3["q0"] == 0.0 and h3["qh"] == 1.0 and h3["rescue_gain"] == 1.0


def test_correct_action_value_is_one_without_harness():
    records = enumerate_q_values()
    record = select(records, "choose_tool", "correct_tool", "H0")
    assert record["q0"] == record["qh"] == 1.0

