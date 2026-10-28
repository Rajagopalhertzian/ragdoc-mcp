"""Text helpers shared by chunkers, retrieval and diagnosis."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path

_UNITS = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_TOK = re.compile(r"\d+(?:[.,]\d+)*|\w+", re.UNICODE)
_NUM = re.compile(r"^\d+(?:[.,]\d+)*$")

STOP = frozenset(
    "a an and are as at be been by for from has have had in is it of on or that the this to was were will with "
    "what which who whom how when where why does do did can could should would about into than then there their "
    "its i we you they he she not no if but so such also these those our your".split()
)


def count_tokens(text: str) -> int:
    """Approximate token count: words plus punctuation marks. Documented as an approximation."""
    return len(_UNITS.findall(text))


def tokens(text: str) -> list[str]:
    """Normalized word tokens. Numbers keep their decimals, so '85.5' != '855'."""
    return _TOK.findall(unicodedata.normalize("NFKC", text).lower())


def content_words(text: str) -> list[str]:
    return [w for w in tokens(text) if w not in STOP and len(w) > 1]


def is_number(tok: str) -> bool:
    return bool(_NUM.match(tok))


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def sha1_hex(s: str, n: int = 16) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:n]


def find_answer(answer: str, text: str, min_cover: float = 0.85) -> dict:
    """Locate ``answer`` in ``text``.

    mode:
      exact      the answer's tokens appear contiguously and in order
      fuzzy      >= min_cover of the answer tokens appear in one window, all numbers exactly
      scattered  the tokens are all present but not close together (structure or order was lost)
      none       the tokens are not there
    Digits and units are compared exactly (tokenization keeps decimals intact).
    """
    a = tokens(answer)
    t = tokens(text)
    if not a or not t:
        return {"found": False, "mode": "none", "score": 0.0}
    if (" " + " ".join(a) + " ") in (" " + " ".join(t) + " "):
        return {"found": True, "mode": "exact", "score": 1.0}
    nums = [x for x in set(a) if is_number(x)]
    need = Counter(a)
    n = len(a)
    win = min(len(t), n + max(3, int(0.3 * n)))
    cur: Counter = Counter(t[:win])
    best = 0.0

    def consider() -> None:
        nonlocal best
        c = _cover(need, cur, n)
        if c > best and all(cur[x] > 0 for x in nums):
            best = c

    consider()
    for i in range(win, len(t)):
        cur[t[i]] += 1
        cur[t[i - win]] -= 1
        consider()
    if best >= min_cover:
        return {"found": True, "mode": "fuzzy", "score": round(best, 3)}
    whole = Counter(t)
    cov = _cover(need, whole, n)
    if cov >= min_cover and all(whole[x] > 0 for x in nums):
        return {"found": False, "mode": "scattered", "score": round(cov, 3)}
    return {"found": False, "mode": "none", "score": round(cov, 3)}


def _cover(need: Counter, have: Counter, n_need: int) -> float:
    got = sum(min(c, have.get(tok, 0)) for tok, c in need.items())
    return got / n_need if n_need else 0.0


def truncate_words(text: str, max_tokens: int) -> str:
    out: list[str] = []
    used = 0
    for w in text.split():
        c = count_tokens(w)
        if used + c > max_tokens:
            break
        out.append(w)
        used += c
    return " ".join(out)


def snippet(text: str, n: int = 240) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"
