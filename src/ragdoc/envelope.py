"""Uniform tool response: status, summary, data, next_step. Default cap is about 2,000 tokens."""
from __future__ import annotations

import copy
import json
from typing import Any

BRIEF_TOKENS = 2000
FULL_TOKENS = 20000


def est_tokens(obj: Any) -> int:
    return len(json.dumps(obj, ensure_ascii=False, default=str)) // 4  # ~4 chars per token


def _shrink(o: Any, max_str: int, max_list: int, protect: frozenset[str], key: str | None = None) -> Any:
    if isinstance(o, str):
        return o if key in protect or len(o) <= max_str else o[: max_str - 1] + "…"
    if isinstance(o, list):
        out = [_shrink(x, max_str, max_list, protect) for x in o[:max_list]]
        if len(o) > max_list:
            out.append(f"…(+{len(o) - max_list} more)")
        return out
    if isinstance(o, dict):
        return {k: _shrink(v, max_str, max_list, protect, k) for k, v in o.items()}
    return o


def fit_budget(env: dict, max_tokens: int, protect: tuple[str, ...] = ()) -> dict:
    """Shorten env['data'] progressively until it fits. Never touches status, summary or next_step."""
    if est_tokens(env) <= max_tokens:
        return env
    out = env
    for ms, ml in ((600, 12), (300, 8), (160, 5), (100, 3)):
        out = copy.deepcopy(env)
        out["data"] = _shrink(env.get("data"), ms, ml, frozenset(protect))
        if est_tokens(out) <= max_tokens:
            break
    else:  # last resort: protected text is shortened too
        out["data"] = _shrink(env.get("data"), 100, 3, frozenset())
    out["truncated"] = True
    out["note"] = f"Output shortened to fit about {max_tokens} tokens. Pass detail_level='full' for everything."
    return out


def ok(summary: str, data: Any = None, next_step: str | None = None, *, status: str = "ok", **extra: Any) -> dict:
    return {"status": status, "summary": summary, "data": data if data is not None else {}, "next_step": next_step, **extra}


def err(message: str, hint: str | None = None, code: str = "ERROR") -> dict:
    hint = hint or "Check the arguments against the tool description and try again."
    return {"status": "error", "summary": message, "data": {"code": code, "hint": hint}, "next_step": hint}


def shape(env: dict, detail_level: str, protect: tuple[str, ...] = ()) -> dict:
    cap = FULL_TOKENS if detail_level == "full" else BRIEF_TOKENS
    return fit_budget(env, cap, protect)
