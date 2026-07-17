from rescuecredit.appworld_shadow_credit import (
    action_app,
    credit_decision,
    json_object,
    official_score,
    prefix_replay_failed,
    requirement_progress,
    render_compatible_action,
)


def test_json_object_accepts_fenced_noise():
    assert json_object('result={"tool":"mail__send","arguments":{}}') == {
        "tool": "mail__send",
        "arguments": {},
    }


def test_official_score_prefers_named_completion_metric():
    payload = {"irrelevant_count": 99, "task_goal_completion": 0.75}
    assert official_score(payload) == 0.75


def test_official_score_accepts_boolean_success():
    assert official_score({"success": True}) == 1.0


def test_official_score_falls_back_when_to_dict_requires_arguments():
    class Result:
        success = True

        def to_dict(self, required):
            return {"success": required}

    assert official_score(Result()) == 1.0


def test_credit_decisions_cover_rescue_reverse_and_zero():
    assert credit_decision(0.0, 1.0) == "rescue_preference"
    assert credit_decision(1.0, 0.0) == "reverse_preference"
    assert credit_decision(0.5, 0.5) == "zero_delta"


def test_frozen_bank_rest_action_renders_through_requester():
    action = {
        "tool": "post:/spotify/like_song",
        "arguments": {"song_id": 7, "access_token": "token"},
    }
    rendered = render_compatible_action(action)
    assert "requester.post('/spotify/like_song'" in rendered
    assert action_app(action) == "spotify"


def test_live_function_action_still_uses_atomic_renderer():
    action = {"tool": "spotify__like_song", "arguments": {"song_id": 7}}
    rendered = render_compatible_action(action)
    assert "apis.spotify.like_song" in rendered
    assert action_app(action) == "spotify"


def test_prefix_failure_rule_does_not_reject_visible_http_status_text():
    assert not prefix_replay_failed("Response status code is 400 but call was observed")
    assert prefix_replay_failed("Execution failed with traceback")


def test_requirement_progress_uses_only_aggregate_counts():
    report = """
    Num Passed Tests : 4
    Num Failed Tests : 2
    Failed Requirement: do not parse this text
    """
    assert requirement_progress(report) == (4, 2, 4 / 6)
