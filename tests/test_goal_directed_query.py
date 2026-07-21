from rescuecredit.goal_directed_query import (
    build_goal_directed_queries,
    build_goal_query_certificate,
    validate_action_schema,
)


SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "modify_reminder",
            "description": "Modify a reminder",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_reminder",
            "description": "Search matching reminders",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_timestamp",
            "description": "Get current timestamp",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "timestamp_to_datetime_info",
            "description": "Convert timestamp",
            "parameters": {
                "type": "object",
                "properties": {"timestamp": {"type": "integer"}},
                "required": ["timestamp"],
            },
        },
    },
]


def test_public_schema_proof_rejects_null_required_argument():
    action = {
        "tool": "timestamp_to_datetime_info",
        "arguments": {"timestamp": None},
    }
    result = validate_action_schema(action, SCHEMAS)

    assert result["valid"] is False
    assert result["violations"] == ["missing_required:timestamp"]


def test_goal_query_targets_b_referenced_reminder():
    queries = build_goal_directed_queries(
        action_a={"tool": "get_current_timestamp", "arguments": {}},
        action_b={"tool": "modify_reminder", "arguments": {"content": "buy milk"}},
        schemas=SCHEMAS,
        instruction="Remind me to buy milk",
    )

    assert len(queries) == 1
    assert queries[0].tool == "search_reminder"
    assert queries[0].arguments == {"content": "buy milk"}
    assert queries[0].expectation == "exists"


def test_query_witness_routes_only_when_b_precondition_is_refuted():
    action_a = {"tool": "get_current_timestamp", "arguments": {}}
    action_b = {"tool": "modify_reminder", "arguments": {"content": "buy milk"}}
    query = build_goal_directed_queries(
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
        instruction="Remind me to buy milk",
    )[0]
    missing_receipt = {
        "parsed": [],
        "exception": None,
    }
    certificate = build_goal_query_certificate(
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
        query=query,
        query_receipt=missing_receipt,
    )

    assert certificate["schema_only_route_to_a"] is False
    assert certificate["query_incremental_route_to_a"] is True
    assert certificate["route_to_a"] is True
    assert certificate["witness_reasons"] == ["b_hard_precondition_refuted"]


def test_unknown_query_receipt_abstains_to_b():
    action_a = {"tool": "get_current_timestamp", "arguments": {}}
    action_b = {"tool": "modify_reminder", "arguments": {"content": "buy milk"}}
    query = build_goal_directed_queries(
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
        instruction="Remind me to buy milk",
    )[0]
    certificate = build_goal_query_certificate(
        action_a=action_a,
        action_b=action_b,
        schemas=SCHEMAS,
        query=query,
        query_receipt={"parsed": None, "exception": "read failed"},
    )

    assert certificate["query_result"]["known"] is False
    assert certificate["route_to_a"] is False


def test_schema_witness_routes_without_query():
    certificate = build_goal_query_certificate(
        action_a={"tool": "get_current_timestamp", "arguments": {}},
        action_b={
            "tool": "timestamp_to_datetime_info",
            "arguments": {"timestamp": None},
        },
        schemas=SCHEMAS,
        query=None,
        query_receipt=None,
    )

    assert certificate["schema_only_route_to_a"] is True
    assert certificate["query_incremental_route_to_a"] is False
    assert certificate["route_to_a"] is True
