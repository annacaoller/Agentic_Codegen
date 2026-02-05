from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


ACTION_FENCE_RE = re.compile(
    r"```action\s*(?P<body>[\s\S]*?)\s*```",
    flags=re.IGNORECASE,
)


class ActionParseError(ValueError):
    """Raised when the model response cannot be parsed into an action JSON."""


@dataclass(frozen=True)
class Action:
    tool_name: str
    args: Dict[str, Any]


def extract_action_block(text: str) -> str:
    """
    Extract the FIRST ```action ...``` fenced block from a model response.

    This mirrors the defensive strategy described in the reference material:
    don't attempt to parse arbitrary prose; enforce a structured block.
    """
    if not isinstance(text, str) or not text.strip():
        raise ActionParseError("Empty model response; expected an ```action``` block.")

    m = ACTION_FENCE_RE.search(text)
    if not m:
        raise ActionParseError("No ```action``` block found in model response.")

    body = m.group("body")
    if not body or not body.strip():
        raise ActionParseError("Found ```action``` block but it was empty.")

    return body.strip()


def parse_action_json(block_body: str) -> Dict[str, Any]:
    """
    Parse JSON from inside the action block.

    The model sometimes prefixes with 'action' or other tokens; we keep this strict:
    - body must be valid JSON object
    """
    try:
        obj = json.loads(block_body)
    except json.JSONDecodeError as e:
        raise ActionParseError(f"Invalid JSON inside action block: {e}") from e

    if not isinstance(obj, dict):
        raise ActionParseError("Action JSON must be a JSON object (dict).")

    return obj


def validate_action_obj(obj: Dict[str, Any]) -> Action:
    """
    Validate required keys: tool_name (str) and args (object/dict).
    """
    tool_name = obj.get("tool_name")
    args = obj.get("args")

    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ActionParseError("Action JSON missing valid 'tool_name' (string).")

    if args is None:
        # allow missing args -> treat as empty dict
        args = {}
    if not isinstance(args, dict):
        raise ActionParseError("Action JSON 'args' must be an object/dict.")

    return Action(tool_name=tool_name.strip(), args=args)


def parse_action(text: str) -> Action:
    """
    Convenience: extract + parse + validate.
    """
    body = extract_action_block(text)
    obj = parse_action_json(body)
    return validate_action_obj(obj)


def safe_parse_action(text: str) -> tuple[Optional[Action], Optional[str]]:
    """
    Parse action but never raise; returns (action, error_message).
    Useful for EvalGuard retry loops.
    """
    try:
        return parse_action(text), None
    except ActionParseError as e:
        return None, str(e)
