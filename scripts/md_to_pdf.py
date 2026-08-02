#!/usr/bin/env python
"""Render one of this repo's memos to PDF, figures and all.

Markdown -> HTML -> headless Chrome -> PDF. Chrome is used rather than a LaTeX or WeasyPrint
route because the memos are HTML-shaped documents: relative <img> paths to the committed PNGs,
GFM tables up to 13 columns wide, and a lot of unicode (µ, ρ, ≥, −, ×, superscript exponents).
Chrome resolves all three with the system fonts and needs no toolchain beyond the browser that
is already installed.

Requires the `markdown` package and a Chromium-family browser. Neither is a dependency of the
analysis environment, so this is deliberately standalone -- run it from any interpreter that has
`markdown`, and point --chrome at the browser if it is not in a default location:

    python scripts/md_to_pdf.py aux/analysis_novel_search/novel_stimulus_search_results.md

Image paths are resolved relative to the markdown file's own directory, which is what the memos
assume (`plots/foo.png` next to the memo), so the HTML is written into that directory and removed
afterwards unless --keep_html.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "google-chrome", "chromium", "chromium-browser", "microsoft-edge",
]

# Print stylesheet. Two things drive it: the memos are dense reference documents, so the base
# size is small and the leading tight; and the tables run to 13 columns, so they get their own
# much smaller size and are allowed to break across pages while figures are not.
CSS = """
@page { size: A4; margin: 13mm 10mm 13mm 10mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 9.2pt; line-height: 1.42; color: #1a1a1a; margin: 0;
}
h1 { font-size: 19pt; margin: 0 0 2mm 0; line-height: 1.2; }
h2 { font-size: 13pt; margin: 7mm 0 2mm 0; padding-top: 2mm;
     border-top: 1.2px solid #c8c8c8; break-before: page; break-after: avoid; }
h1 + h2, h2:first-of-type { break-before: auto; }
h3 { font-size: 10.6pt; margin: 5mm 0 1.5mm 0; break-after: avoid; }
h4 { font-size: 9.6pt; margin: 4mm 0 1mm 0; break-after: avoid; }
p { margin: 0 0 2.2mm 0; orphans: 3; widows: 3; }
ul, ol { margin: 0 0 2.5mm 0; padding-left: 5.5mm; }
li { margin: 0 0 1.1mm 0; }
strong { font-weight: 650; }
code {
  font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 0.87em;
  background: #f2f2f0; padding: 0.5px 2.5px; border-radius: 2px; word-break: break-word;
}
pre { background: #f6f6f4; border: 1px solid #e2e2de; border-radius: 3px;
      padding: 2.5mm 3mm; overflow-x: auto; break-inside: avoid; margin: 0 0 3mm 0; }
pre code { background: none; padding: 0; font-size: 7.8pt; line-height: 1.35; }

/* Blockquotes carry the provenance headers, which are reference material, not emphasis. */
blockquote {
  margin: 0 0 3mm 0; padding: 2mm 3mm; background: #f7f8fa;
  border-left: 2.5px solid #b8c2cc; font-size: 8.6pt; break-inside: avoid;
}
blockquote p { margin: 0 0 1.2mm 0; }
blockquote p:last-child { margin-bottom: 0; }

/* Tables: the widest is 13 columns, so they get their own size and are allowed to break. */
table {
  border-collapse: collapse; width: 100%; margin: 0 0 3.5mm 0;
  font-size: 7.4pt; line-height: 1.3; table-layout: auto;
}
thead { display: table-header-group; }        /* repeat the header on every page a table spans */
/* break-word, never `anywhere`: `anywhere` lets the layout split mid-token, which turns
   `method_1740` into "method_174 / 0" and `12.00` into "12.0 / 0" when a column is tight. */
th, td { border: 0.5px solid #d5d5d5; padding: 1.0mm 1.3mm; text-align: left;
         vertical-align: top; word-break: normal; overflow-wrap: break-word; hyphens: none; }
th { background: #eef1f4; font-weight: 650; }
tr:nth-child(even) td { background: #fafafa; }
/* Method ids and numbers are atomic: forcing them nowrap makes the auto layout give their
   column the width it actually needs instead of shredding the identifier. */
td code, th code { font-size: 0.95em; background: none; padding: 0; white-space: nowrap; }
td { font-variant-numeric: tabular-nums; }

/* Figures must not split; a half figure is worse than a page break before it. */
img { max-width: 100%; height: auto; display: block; margin: 1mm auto 2mm auto;
      break-inside: avoid; }
p:has(> img) { break-inside: avoid; text-align: center; margin-bottom: 1mm; }
hr { border: none; border-top: 0.8px solid #dcdcdc; margin: 5mm 0; }

/* Front matter */
.titleblock { margin: 0 0 4mm 0; padding: 0 0 3mm 0; border-bottom: 1.6px solid #1a1a1a; }
.titleblock .sub { font-size: 8.6pt; color: #5a5a5a; margin-top: 1.5mm; }
.toc { font-size: 8.4pt; background: #f7f8fa; border: 0.5px solid #e0e3e7;
       border-radius: 3px; padding: 3mm 4mm; margin: 0 0 4mm 0; break-inside: avoid; }
.toc > ul { padding-left: 4mm; margin: 0; }
.toc ul ul { padding-left: 4mm; }
.toc li { margin: 0.4mm 0; }
.toc a { color: #1a1a1a; text-decoration: none; }
.toc-title { font-weight: 650; font-size: 9.2pt; margin: 0 0 1.5mm 0; }
a { color: #14508c; text-decoration: none; }
"""


def find_chrome(explicit=None):
    if explicit:
        if Path(explicit).exists() or shutil.which(explicit):
            return explicit
        raise SystemExit(f"--chrome {explicit!r} not found")
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("no Chromium-family browser found; pass --chrome /path/to/browser")


def build_html(md_path, title=None):
    try:
        import markdown
    except ImportError:
        raise SystemExit("the `markdown` package is required: pip install markdown")

    text = Path(md_path).read_text(encoding="utf-8")
    md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "toc", "sane_lists"],
                           extension_configs={"toc": {"toc_depth": "2-3"}})
    body = md.convert(text)

    # The memo's own H1 becomes the title block; everything after it is the document.
    heading = title or Path(md_path).stem.replace("_", " ")
    stamp = subprocess.run(["git", "log", "-1", "--format=%h  %ad", "--date=short", "--",
                            str(md_path)], capture_output=True, text=True,
                           cwd=Path(md_path).resolve().parent).stdout.strip()
    sub = f"{md_path} &middot; rendered from markdown"
    if stamp:
        sub += f" &middot; last commit touching it: {stamp}"

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{heading}</title>
<style>{CSS}</style></head><body>
<div class="titleblock"><h1>{heading}</h1><div class="sub">{sub}</div></div>
<div class="toc"><div class="toc-title">Contents</div>{md.toc}</div>
{body}
</body></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("markdown", help="path to the .md file")
    p.add_argument("--out", help="output PDF (default: alongside the markdown, same stem)")
    p.add_argument("--title", help="title for the front matter (default: the file stem)")
    p.add_argument("--chrome", help="path to a Chromium-family browser")
    p.add_argument("--keep_html", action="store_true",
                   help="leave the intermediate HTML beside the markdown for inspection")
    p.add_argument("--timeout", type=float, default=180.0,
                   help="seconds to wait for Chrome before killing it (default 180). Headless "
                        "Chrome routinely writes the PDF and then fails to exit, so this is a "
                        "normal path, not an error: the PDF is what success is judged on")
    args = p.parse_args()

    md_path = Path(args.markdown)
    if not md_path.exists():
        raise SystemExit(f"{md_path} not found")
    out = Path(args.out) if args.out else md_path.with_suffix(".pdf")
    chrome = find_chrome(args.chrome)

    # The HTML must sit in the markdown's directory so relative image paths resolve.
    html_path = md_path.with_name(md_path.stem + ".render.html")
    html_path.write_text(build_html(md_path, args.title), encoding="utf-8")

    out.unlink(missing_ok=True)                    # so a stale PDF cannot be mistaken for success
    err = ""
    try:
        with tempfile.TemporaryDirectory() as profile:
            cmd = [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                   f"--user-data-dir={profile}",
                   "--no-pdf-header-footer",
                   "--virtual-time-budget=20000",
                   f"--print-to-pdf={out.resolve()}",
                   html_path.resolve().as_uri()]
            # Headless Chrome writes the PDF and then often does not exit, so this is bounded and
            # the PDF's existence -- not the exit code -- is what success is judged on.
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            err = r.stderr
    except subprocess.TimeoutExpired:
        err = f"Chrome did not exit within {args.timeout}s; killed after checking the output"
    finally:
        if not args.keep_html:
            html_path.unlink(missing_ok=True)
    if not out.exists() or out.stat().st_size == 0:
        raise SystemExit(f"Chrome did not produce {out}\n{err}")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")
    if args.keep_html:
        print(f"kept  {html_path}")


if __name__ == "__main__":
    main()
