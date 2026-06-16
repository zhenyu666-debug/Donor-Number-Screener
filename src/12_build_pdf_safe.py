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
import subprocess
import sys
from pathlib import Path

OUTREACH_DIR = Path(__file__).resolve().parent.parent / "outreach"
SOURCES = ["case_study_li_s_v1.md",
           "capability_one_pager.md",
           "pricing_v1.md"]


def md_to_html(md_text: str, title: str) -> str:
    import markdown
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
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
</style></head><body>{body}</body></html>"""


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


def main():
    report = {}
    print("Building outreach PDFs...\n")
    for fname in SOURCES:
        md_path = OUTREACH_DIR / fname
        if not md_path.exists():
            print(f"  skip {fname} (missing)")
            continue
        pdf_path = OUTREACH_DIR / fname.replace(".md", ".pdf")
        html_path = OUTREACH_DIR / fname.replace(".md", ".html")
        md = md_path.read_text(encoding="utf-8")
        title = fname.replace(".md", "").replace("_", " ").title()
        html = md_to_html(md, title)
        html_path.write_text(html, encoding="utf-8")

        if backend_weasyprint(html, pdf_path):
            backend = "weasyprint"
        elif backend_pdfkit(html, pdf_path):
            backend = "pdfkit"
        else:
            backend = "html-only"
            print(f"  -> wrote HTML only at {html_path.name} "
                  "(install weasyprint or wkhtmltopdf for PDF)")

        report[fname] = {
            "backend": backend,
            "html": str(html_path.relative_to(OUTREACH_DIR)),
            "pdf": (str(pdf_path.relative_to(OUTREACH_DIR))
                    if backend != "html-only" else None),
        }
        print(f"  {fname}  -> {backend}\n")

    out = OUTREACH_DIR / "build_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Build report: {out}")


if __name__ == "__main__":
    main()
