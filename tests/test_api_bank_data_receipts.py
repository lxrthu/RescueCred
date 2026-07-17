import json

from environments.api_bank.data import (
    candidate_tool_names_from_source,
    parse_api_catalog,
    parse_dialogue,
    reference_years_consistent,
)


def test_catalog_and_dialogue_preserve_output_schema_and_private_receipt(tmp_path):
    api_dir = tmp_path / "apis"
    api_dir.mkdir()
    (api_dir / "get_token.py").write_text(
        '''class GetToken:
    description = "Get a token."
    input_parameters = {"username": {"type": "str"}}
    output_parameters = {"token": {"type": "str"}}
''',
        encoding="utf-8",
    )
    catalog = parse_api_catalog(api_dir)
    assert "token" in catalog["GetToken"]["output_parameters"]

    dialogue = tmp_path / "sample.jsonl"
    records = [
        {"role": "User", "text": "My username is user3."},
        {
            "role": "API",
            "api_name": "GetToken",
            "param_dict": {"username": "user3"},
            "result": {"output": {"token": "runtime-token"}, "exception": None},
        },
    ]
    dialogue.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    task = parse_dialogue(dialogue, catalog)
    assert task is not None
    assert task["reference_tool_receipts"] == [
        {"status": "ok", "tool": "GetToken", "token": "runtime-token"}
    ]


def test_reference_year_mismatch_is_rejected_by_quality_filter():
    inconsistent = {
        "user_goal": "Get the stock price on January 3rd, 2023.",
        "reference_actions": [
            {"tool": "QueryStock", "arguments": {"date": "2022-01-03", "stock_code": "AAPL"}}
        ],
    }
    consistent = {
        "user_goal": "Get the stock price on January 3rd, 2023.",
        "reference_actions": [
            {"tool": "QueryStock", "arguments": {"date": "2023-01-03", "stock_code": "AAPL"}}
        ],
    }
    assert not reference_years_consistent(inconsistent)
    assert reference_years_consistent(consistent)


def test_public_tool_set_comes_from_source_and_schema_not_reference_actions(tmp_path):
    catalog = {
        "AddAlarm": {
            "name": "AddAlarm",
            "required": ["time", "token"],
            "output_parameters": {"status": {"type": "str"}},
        },
        "GetUserToken": {
            "name": "GetUserToken",
            "required": ["username", "password"],
            "output_parameters": {"token": {"type": "str"}},
        },
        "Unrelated": {
            "name": "Unrelated",
            "required": ["query"],
            "output_parameters": {"answer": {"type": "str"}},
        },
    }
    path = tmp_path / "AddAlarm-level-1-1.jsonl"
    names, provenance = candidate_tool_names_from_source(path, catalog)
    assert names == ["AddAlarm", "GetUserToken"]
    assert provenance == "source_filename_plus_schema_prerequisites"

    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"role": "User", "text": "Set an alarm at eight."},
                {
                    "role": "API",
                    "api_name": "AddAlarm",
                    "param_dict": {"time": "08:00", "token": "runtime"},
                    "result": {"output": "ok", "exception": None},
                },
            ]
        ),
        encoding="utf-8",
    )
    task = parse_dialogue(path, catalog)
    assert task is not None
    assert [tool["name"] for tool in task["available_tools"]] == names
    assert task["available_tools_reference_independent"] is True
