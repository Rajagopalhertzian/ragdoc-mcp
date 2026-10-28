
import pytest

from ragdoc.parsers import parse_file
from ragdoc.parsers.base import ParsedDoc
from ragdoc.parsers.faults import apply_parse_fault


def _pdf_text(path, pages):
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    for t in pages:
        c.drawString(72, 750, t)
        c.showPage()
    c.save()


def _pdf_scanned(path):
    from PIL import Image, ImageDraw
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    img = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(img).text((10, 40), "scanned page text", fill="black")
    c = canvas.Canvas(str(path))
    c.drawString(72, 750, "Page one has real text about valves.")
    c.showPage()
    c.drawImage(ImageReader(img), 72, 600, width=400, height=100)
    c.showPage()
    c.save()


def test_text_pdf_parses(base):
    p = base / "a.pdf"
    _pdf_text(p, ["Maximum pressure is 42 bar.", "Second page text."])
    d = parse_file(p, "a.pdf", "h", "fast")
    assert d.kind == "pdf" and len(d.pages) == 2 and "42 bar" in d.pages[0]["text"] and d.meta["scanned_pages"] == []


def test_scanned_pdf_page_detected(base):
    p = base / "s.pdf"
    _pdf_scanned(p)
    d = parse_file(p, "s.pdf", "h", "fast")
    assert d.meta["scanned_pages"] == [2]
    assert "valves" in d.pages[0]["text"] and not d.pages[1]["text"].strip()


def test_reference_backend_runs_on_pdf(base):
    p = base / "a.pdf"
    _pdf_text(p, ["Maximum pressure is 42 bar."])
    assert "42 bar" in parse_file(p, "a.pdf", "h", "reference").pages[0]["text"]


def test_unknown_backend_and_unavailable_adapters(base):
    from ragdoc.errors import RagDocError
    from ragdoc.parsers import ParserUnavailable, docling_available

    (base / "a.md").write_text("# T\nbody")
    with pytest.raises(RagDocError):
        parse_file(base / "a.md", "a.md", "h", "nope")
    if not docling_available():
        with pytest.raises(ParserUnavailable):
            parse_file(base / "a.md", "a.md", "h", "docling")
    from ragdoc.errors import RagDocError as E

    with pytest.raises(E) as e:  # cloud parser without consent must refuse
        parse_file(base / "a.md", "a.md", "h", "llamaparse", manifest={"consent": {}})
    assert e.value.code == "NO_CONSENT"


def test_fault_injectors_change_text():
    t = "| P | V |\n| --- | --- |\n| a | 1 |\n| b | 2 |\n\nalpha beta gamma delta epsilon zeta\n$$ y = 3 x $$\n"
    for mode in ("blank_pages", "scramble_columns", "flatten_tables", "garble_equations"):
        d = ParsedDoc("f.md", "h", "fast", "text", [{"num": 1, "text": t}], {})
        out = apply_parse_fault(d, {"mode": mode}).pages[0]["text"]
        assert out != t
