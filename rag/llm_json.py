"""Extract a JSON object from LLM output.

Even in JSON mode, models sometimes wrap the object in prose ("Sure! {...}"). Taking the
span from the first "{" to the last "}" recovers those cases without hiding real errors.
"""

import json
from typing import Any


class NoJsonObjectError(ValueError):
    pass


def extract_json_object(raw: str) -> dict[str, Any]:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise NoJsonObjectError("no JSON object in the response")
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise NoJsonObjectError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise NoJsonObjectError("the JSON value is not an object")
    return obj
