from __future__ import annotations

import json
import math
import re
from typing import Any


SCORE_KEYS = (
    "task_goal_completion",
    "task_goal_completion_rate",
    "task_success",
    "success",
    "passed",
)


def json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group())
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): plain(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(child) for child in value]
    for name in ("to_dict", "model_dump", "dict"):
        method = getattr(value, name, None)
        if callable(method):
            try:
                return plain(method())
            except TypeError:
                continue
    payload = {
        key: plain(getattr(value, key))
        for key in SCORE_KEYS
        if hasattr(value, key)
    }
    return payload or str(value)


def official_score(value: Any) -> float | None:
    payload = plain(value)

    def scalar(node: Any) -> float | None:
        if isinstance(node, bool):
            return float(node)
        if isinstance(node, (int, float)) and math.isfinite(float(node)):
            return min(1.0, max(0.0, float(node)))
        return None

    def search(node: Any) -> float | None:
        if isinstance(node, dict):
            for key in SCORE_KEYS:
                if key in node:
                    score = scalar(node[key])
                    if score is None:
                        score = search(node[key])
                    if score is not None:
                        return score
            for child in node.values():
                if isinstance(child, (dict, list, tuple)):
                    score = search(child)
                    if score is not None:
                        return score
        elif isinstance(node, (list, tuple)):
            for child in node:
                if isinstance(child, (dict, list, tuple)):
                    score = search(child)
                    if score is not None:
                        return score
        return None

    return search(payload)


def credit_decision(return_a: float, return_b: float, tolerance: float = 1e-9) -> str:
    delta = float(return_b) - float(return_a)
    if delta > tolerance:
        return "rescue_preference"
    if delta < -tolerance:
        return "reverse_preference"
    return "zero_delta"


def action_app(action: dict[str, Any]) -> str:
    tool = str(action.get("tool", ""))
    if ":/" in tool:
        _, url = tool.split(":", 1)
        return url.strip("/").split("/", 1)[0]
    return tool.split("__", 1)[0].split(".", 1)[0]


def render_compatible_action(action: dict[str, Any]) -> str:
    """Render frozen-bank REST actions or live function-call actions."""

    tool = str(action.get("tool", ""))
    arguments = action.get("arguments", {})
    if not isinstance(arguments, dict):
        raise TypeError("AppWorld action arguments must be an object")
    if ":/" in tool:
        method, url = tool.split(":", 1)
        method = method.lower()
        if method not in {"get", "post", "put", "patch", "delete"}:
            raise ValueError(f"unsupported AppWorld REST method: {method!r}")
        if not url.startswith("/"):
            raise ValueError("AppWorld REST URL must start with /")
        encoded = json.dumps(
            arguments, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        return (
            "import json\n"
            f"print(requester.{method}({url!r}, data=json.loads({encoded!r})))"
        )
    from environments.appworld.adapter import render_atomic_call

    return render_atomic_call(action)


def prefix_replay_failed(output: str) -> bool:
    """Match the bank builder's conservative reference-replay failure rule."""

    lowered = output.lower()
    return "execution failed" in lowered or "traceback" in lowered


def requirement_progress(report_text: str) -> tuple[int, int, float]:
    """Extract only aggregate official requirement counts from a report."""

    passed_match = re.search(r"Num Passed Tests\s*:\s*(\d+)", report_text)
    failed_match = re.search(r"Num Failed Tests\s*:\s*(\d+)", report_text)
    if passed_match is None or failed_match is None:
        raise ValueError("AppWorld report is missing aggregate pass/fail counts")
    passed = int(passed_match.group(1))
    failed = int(failed_match.group(1))
    total = passed + failed
    if total <= 0:
        raise ValueError("AppWorld report has no evaluator requirements")
    return passed, failed, passed / total
