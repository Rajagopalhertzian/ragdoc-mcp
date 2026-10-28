from ragdoc.chunkers import chunk_document
from ragdoc.parsers.base import ParsedDoc

MD = """# Manual

## Thermal
The unit has a maximum operating temperature of 92 degrees Celsius.

## Electrical
| Parameter | Value | Unit |
| --- | --- | --- |
| Supply voltage | 24 | V |
| Rated current | 3.5 | A |

## Control
The output follows a linear law.
$$ y = 3.7 \\cdot x + 0.25 $$
where the gain is 3.7.
"""


def doc(text, kind="text", **meta):
    return ParsedDoc("d/a.md", "h", "fast", kind, [{"num": 1, "text": text}], meta)


def test_heading_path_and_prefix():
    cs = chunk_document(doc(MD), {"max_tokens": 400})
    th = next(c for c in cs if "92 degrees" in c.text)
    assert th.heading_path == "Manual > Thermal"
    assert th.embed_text.startswith("Manual > Thermal")
    nop = chunk_document(doc(MD), {"max_tokens": 400, "prepend_heading": False})
    assert not next(c for c in nop if "92 degrees" in c.text).embed_text.startswith("Manual")


def test_table_whole_with_summary():
    t = next(c for c in chunk_document(doc(MD), {"max_tokens": 400}) if c.ctype == "table")
    assert "Rated current | 3.5 | A" in t.text and "columns Parameter, Value, Unit" in t.embed_text


def test_equation_never_split_even_when_over_budget():
    e = next(c for c in chunk_document(doc(MD), {"max_tokens": 12}) if c.ctype == "equation")
    assert "y = 3.7 \\cdot x + 0.25" in e.text  # 16 tokens > budget of 12: still whole


def test_equation_keeps_explanation_when_it_fits():
    e = next(c for c in chunk_document(doc(MD), {"max_tokens": 80}) if c.ctype == "equation")
    assert "y = 3.7 \\cdot x + 0.25" in e.text and "linear law" in e.text


def test_window_splits_and_overlaps():
    words = " ".join(f"w{i}" for i in range(100))
    cs = chunk_document(doc(words), {"kind": "window", "max_tokens": 20, "overlap_pct": 25})
    assert len(cs) > 4 and all(c.tokens <= 21 for c in cs)
    assert set(cs[0].text.split()[-3:]) & set(cs[1].text.split()[:6])


def test_python_ast_chunks():
    src = 'import os\n\nclass A:\n    """d"""\n    def m(self): return 1\n\ndef f(x):\n    """Do f."""\n    return x\n'
    cs = chunk_document(ParsedDoc("m.py", "h", "fast", "code", [{"num": 1, "text": src}], {"language": "python"}), {})
    names = {c.heading_path for c in cs}
    assert {"m.py::A", "m.py::f", "m.py::<module>"} <= names
    assert all(c.ctype == "code" for c in cs)


def test_regex_code_fallback_for_js():
    js = "const x = 1;\nfunction a() {\n  return 1;\n}\nfunction b() {\n  return 2;\n}\n"
    cs = chunk_document(ParsedDoc("m.js", "h", "fast", "code", [{"num": 1, "text": js}], {"language": "javascript"}), {})
    assert len(cs) >= 3
