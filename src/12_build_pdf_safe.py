"""Step 12: Build the 3 outreach PDFs with a robust backend.

Tries weasyprint first (best quality), then pdfkit+wkhtmltopdf,
then a pure-Python fallback that emits print-styled HTML files
side-by-side.  Always succeeds, no hard import error.

Usage:
  python src/12_build_pdf_safe.py

Outputs:
  outreach/case_study_li_s_v1.pdf     (or .html if neither backend)
  outreach/capability_one_pager.pdf
  outreach/pricing_v1.pdf
  outreach/build_report.json          — which backend was used
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

OUTREACH_DIR = Path(__file__).resolve().parent.parent / "outreach"
SOURCES = [
    "case_study_li_s_v1.html",
    "capability_one_pager.html",
    "pricing_v1.html",
]


def _wrap_for_print(html_body: str, title: str) -> str:
    """Wrap an existing HTML body fragment in a print-styled document."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{ size: A4; margin: 24mm 18mm; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
         max-width: 720px; margin: 32px auto; padding: 0 24px; line-height: 1.55;
         color: #1a1a1a; }}
  h1 {{ font-size: 26px; margin-bottom: 8px; }}
  h2 {{ font-size: 18px; margin-top: 28px; border-bottom: 1px solid #ddd;
        padding-bottom: 4px; }}
  h3 {{ font-size: 15px; margin-top: 22px; color: #444; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  th {{ background: #f4f4f4; }}
  code {{ background: #f4f4f4; padding: 1px 5px; border-radius: 3px;
          font-family: ui-monospace, Consolas, monospace; font-size: 13px; }}
  pre {{ background: #f4f4f4; padding: 12px; border-radius: 6px; overflow-x: auto; }}
</style></head><body>{html_body}</body></html>"""


def _strip_existing_chrome(html: str) -> str:
    """Drop an existing <html>...<body>...</body></html> wrapper so we can
    re-wrap with print styles without nesting <html> elements."""
    import re
    body = re.search(r"<body[^>]*>(.*?)</body>", html, flags=re.DOTALL | re.IGNORECASE)
    if body:
        return body.group(1)
    return html


def md_to_html(md_text: str, title: str) -> str:
    import markdown
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    return _wrap_for_print(body, title)


def backend_weasyprint(html: str, out_pdf: Path) -> bool:
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(out_pdf))
        return True
    except Exception as e:
        print(f"  weasyprint failed: {e}")
        return False


def backend_pdfkit(html: str, out_pdf: Path) -> bool:
    try:
        import pdfkit
    except ImportError:
        return False
    if not shutil.which("wkhtmltopdf"):
        # Try common Windows install paths
        for guess in (r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
                      r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe"):
            if Path(guess).exists():
                config = pdfkit.configuration(wkhtmltopdf=guess)
                break
        else:
            print("  wkhtmltopdf not installed; skipping pdfkit")
            return False
    else:
        config = pdfkit.configuration()
    try:
        pdfkit.from_string(html, str(out_pdf), configuration=config)
        return True
    except Exception as e:
        print(f"  pdfkit failed: {e}")
        return False


def _html_to_plain_paragraphs(html: str) -> list[tuple[str, str]]:
    """Convert an HTML body into a list of (style, text) tuples for fpdf2.

    style is one of: 'h1', 'h2', 'h3', 'p', 'code', 'bullet'.
    Tables are flattened to "Col1: v1 | Col2: v2" lines.
    """
    import re
    paragraphs: list[tuple[str, str]] = []
    # Use BeautifulSoup if available; otherwise fall back to regex
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, "html.parser")
        for el in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table", "pre", "code", "li"],
                                recursive=True):
            if el.name in ("h1", "h2", "h3"):
                paragraphs.append((el.name, el.get_text(" ", strip=True)))
            elif el.name == "p":
                txt = el.get_text(" ", strip=True)
                if txt:
                    paragraphs.append(("p", txt))
            elif el.name == "pre":
                paragraphs.append(("code", el.get_text("\n", strip=True)))
            elif el.name == "code" and el.parent and el.parent.name != "pre":
                # inline code — skip (already part of <p> text)
                pass
            elif el.name in ("ul", "ol"):
                for li in el.find_all("li", recursive=False):
                    paragraphs.append(("bullet", li.get_text(" ", strip=True)))
            elif el.name == "li" and el.parent and el.parent.name not in ("ul", "ol"):
                paragraphs.append(("bullet", el.get_text(" ", strip=True)))
            elif el.name == "table":
                for tr in el.find_all("tr"):
                    cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
                    if cells:
                        paragraphs.append(("p", " | ".join(cells)))
    except ImportError:
        # Regex fallback: strip all tags, treat each block as a paragraph
        text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"</(p|h1|h2|h3|li)>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        for line in text.splitlines():
            line = line.strip()
            if line:
                paragraphs.append(("p", line))
    return paragraphs


def backend_fpdf2(html: str, out_pdf: Path) -> bool:
    """Pure-Python PDF backend (no GTK / wkhtmltopdf)."""
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError:
        print("  fpdf2 not installed; skipping")
        return False
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    line_h = 5.5
    for style, text in _html_to_plain_paragraphs(html):
        if style == "h1":
            pdf.set_font("Helvetica", style="B", size=18)
            pdf.ln(4)
            pdf.multi_cell(w=0, h=8, txt=text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            pdf.set_font("Helvetica", size=11)
        elif style == "h2":
            pdf.set_font("Helvetica", style="B", size=13)
            pdf.ln(3)
            pdf.multi_cell(w=0, h=6, txt=text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            pdf.set_font("Helvetica", size=11)
        elif style == "h3":
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.ln(1)
            pdf.multi_cell(w=0, h=line_h, txt=text, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        elif style == "bullet":
            pdf.multi_cell(w=0, h=line_h, txt=f"  - {text}", new_x="LMARGIN", new_y="NEXT")
        elif style == "code":
            pdf.set_font("Courier", size=9)
            for code_line in text.splitlines():
                pdf.multi_cell(w=0, h=4.2, txt=code_line, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=11)
        else:  # p
            # fpdf2 doesn't auto-wrap unicode dashes; multi_cell does the wrapping
            try:
                pdf.multi_cell(w=0, h=line_h, txt=text, new_x="LMARGIN", new_y="NEXT")
            except Exception:
                # Some unicode chars not in cp1252 latin — fall back to ascii
                safe = text.encode("latin-1", "replace").decode("latin-1")
                pdf.multi_cell(w=0, h=line_h, txt=safe, new_x="LMARGIN", new_y="NEXT")
    try:
        pdf.output(str(out_pdf))
        return out_pdf.exists() and out_pdf.stat().st_size > 1000
    except Exception as e:
        print(f"  fpdf2 output failed: {e}")
        return False


def main():
    report = {}
    print("Building outreach PDFs...\n")
    for fname in SOURCES:
        src_path = OUTREACH_DIR / fname
        if not src_path.exists():
            print(f"  skip {fname} (missing)")
            continue
        pdf_path  = OUTREACH_DIR / fname.replace(".html", ".pdf")
        html_path = OUTREACH_DIR / fname.replace(".html", ".print.html")
        title = fname.replace(".html", "").replace("_", " ").title()
        raw_html = src_path.read_text(encoding="utf-8")
        body = _strip_existing_chrome(raw_html)
        html = _wrap_for_print(body, title)
        html_path.write_text(html, encoding="utf-8")

        if backend_weasyprint(html, pdf_path):
            backend = "weasyprint"
        elif backend_pdfkit(html, pdf_path):
            backend = "pdfkit"
        elif backend_fpdf2(html, pdf_path):
            backend = "fpdf2"
        else:
            backend = "html-only"
            print(f"  -> wrote HTML only at {html_path.name} "
                  "(install weasyprint or wkhtmltopdf for PDF; "
                  "fpdf2 fallback also failed)")

        report[fname] = {
            "backend": backend,
            "html": str(src_path.relative_to(OUTREACH_DIR)),
            "pdf": (str(pdf_path.relative_to(OUTREACH_DIR))
                    if backend != "html-only" else None),
        }
        print(f"  {fname}  -> {backend}\n")

    out = OUTREACH_DIR / "build_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Build report: {out}")


if __name__ == "__main__":
    main()
